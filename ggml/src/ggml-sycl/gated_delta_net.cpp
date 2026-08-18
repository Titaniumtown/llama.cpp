#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"
#include "gated_delta_net.hpp"
#include <cmath>
#include <cstdlib>


// C is how many columns of the recurrent state one warp owns. It is pure register
// blocking: every column's arithmetic and its two warp reductions are untouched, so
// the total FMA and shuffle counts across the launch do not change. What changes is
// that this token's q and k shards are loaded once per warp instead of once per
// column, cutting the redundant load traffic by C. Selected by GGML_SYCL_GDN_COLS.
template <int S_v, int C, bool KDA, bool keep_rs_t>
void gated_delta_net_sycl(
                          const float *     q,
                          const float *     k,
                          const float *     v,
                          const float *     g,
                          const float *     beta,
                          const float *     curr_state,
                          float *           dst,
                          float *           state,
                          int64_t           H,
                          int64_t           n_tokens,
                          int64_t           sq1,
                          int64_t           sq2,
                          int64_t           sq3,
                          int64_t           sv1,
                          int64_t           sv2,
                          int64_t           sv3,
                          int64_t           sb1,
                          int64_t           sb2,
                          int64_t           sb3,
                          const sycl::uint3 neqk1_magic,
                          const sycl::uint3 rq3_magic,
                          float             scale,
                          int64_t           state_slot_stride,
                          int               K,
                          bool              beta_sigmoid,
                          const int32_t *   state_row_idx,
                          int64_t           state_row_stride) {
    auto           item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const uint32_t h_idx    = item_ct1.get_group(2);
    const uint32_t sequence = item_ct1.get_group(1);
    // each warp owns C columns, using warp-level primitives to reduce across rows
    const int      lane     = item_ct1.get_local_id(2);
    const int      col0     = ((int) (item_ct1.get_group(0) * item_ct1.get_local_range(1) +
                                      item_ct1.get_local_id(1))) * C;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    float *       attn_data        = dst;

    // input state holds s0 only [S_v, S_v, H, n_seqs] -- seq stride is D = H * S_v * S_v.
    // output state layout (per-slot D * n_seqs) -- same per-(seq,head) offset as before.
    //
    // state_row_idx folds away a preceding GET_ROWS: when it is set, `curr_state` is the
    // state cache itself rather than a gathered copy of it, and this sequence's row is
    // selected here instead of by a whole kernel that copied it first. Reading the cache
    // in place is the only difference; the per-head term is untouched.
    const int64_t state_seq_base       = state_row_idx
                                             ? (int64_t) state_row_idx[sequence] * state_row_stride
                                             : sequence * H * S_v * S_v;
    const int64_t state_in_offset      = state_seq_base + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(S_v % C == 0, "S_v must be a multiple of the column block C");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    // The attn store below parks column c in lane c, so a column past the warp would be
    // dropped silently rather than caught -- the values are uniform, so nothing faults.
    static_assert(C <= warp_size, "attn store parks column c in lane c");
    float         s_shard[C][rows_per_lane];
#pragma unroll
    for (int c = 0; c < C; c++) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i   = r * warp_size + lane;
            s_shard[c][r] = curr_state[(col0 + c) * S_v + i];
        }
    }

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K only slots 0..n_tokens-1 are written; older slots are caller-owned.

    for (int t = 0; t < n_tokens; t++) {
        float out_lane = 0.0f;
        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        // beta_sigmoid folds a preceding standalone SIGMOID into this one load. The
        // expression is op_sigmoid()'s verbatim (element_wise.cpp), so the fused and
        // unfused paths are bit-identical rather than merely close. The branch is once
        // per (token, head), outside the C-column loop, and uniform across the
        // sub-group, so it costs nothing measurable.
        const float beta_raw = *beta_t;
        const float beta_val = beta_sigmoid ? 1.0f / (1.0f + sycl::exp(-beta_raw)) : beta_raw;

        // this token's q/k shard, read once and reused by all C columns -- the point of C
        float q_reg[rows_per_lane];
        float k_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            q_reg[r]    = q_t[i];
            k_reg[r]    = k_t[i];
        }

        // v for this token, issued here rather than at its first use. Its address is a
        // function of t and col alone -- nothing in the iteration computes it -- yet at
        // the point of use it sits behind C+1 warp reductions, so its global-load latency
        // is fully exposed on a loop that is already latency-bound. dst never aliases a
        // src for this op (GATED_DELTA_NET is absent from ggml_op_can_inplace's
        // whitelist), and v is read-only here, so this is a pure reordering: the values
        // and the arithmetic are unchanged, bit for bit.
        float v_reg[C];
#pragma unroll
        for (int c = 0; c < C; c++) {
            v_reg[c] = v_t[col0 + c];
        }

        if constexpr (!KDA) {
            const float g_val = sycl::native::exp(*g_t);

            // attn[col] wants the NEW state, and reducing over it directly is what used to
            // serialise this loop: that reduction could not start until delta[col] was known,
            // and delta[col] needs kv[col]'s reduction to have landed. Substituting the state
            // update removes the dependency rather than hiding it:
            //
            //   attn[col] = sum_i (g*S[i][col] + k[i]*delta[col]) * q[i]
            //             = g * sum_i S[i][col]*q[i]  +  delta[col] * sum_i k[i]*q[i]
            //             = g * qs[col]               +  delta[col] * kq
            //
            // Both sums read only the OLD state and this token's inputs, so every reduction on
            // the critical path is independent: kv and qs share one float2 shuffle loop (that
            // overload interleaves two reductions in a single pass), and kq needs one more
            // reduction per token rather than per column, since it does not depend on col.
            // Same terms in a different summation order, so results differ in rounding only.
            // kq is a function of this token's q and k alone -- it never reads the state --
            // so it can be computed before the column loop rather than between the two.
            // That is what lets those two loops fuse below, and the fusion is the point:
            // kv_col[C] and qs_col[C] existed only to carry C reductions across this
            // reduction, and at C=4 that is 8 floats/lane held live for no other reason.
            // This kernel spills ~12 registers at SIMD16/128 GRFs, and raising the GRF
            // count instead is measured at +38% (see the README); cutting live values is
            // the direction that does not cost residency.
            float kq_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kq_shard += k_reg[r] * q_reg[r];
            }
            const float kq = warp_reduce_sum<warp_size>(kq_shard);

#pragma unroll
            for (int c = 0; c < C; c++) {
                sycl::float2 acc(0.0f, 0.0f);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    acc.x() += s_shard[c][r] * k_reg[r];
                    acc.y() += s_shard[c][r] * q_reg[r];
                }
                acc = warp_reduce_sum<warp_size>(acc);

                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = (v_reg[c] - g_val * acc.x()) * beta_val;

                // S[i][col] = g * S[i][col] + k[i] * delta[col]
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                }

                // Park this column's result in the lane that will store it. Every input
                // here is warp-uniform (all three reductions broadcast; v/g/beta are
                // uniform loads), so lane c already holds column c's value -- no shuffle.
                out_lane = (lane == c) ? (g_val * acc.y() + delta_col * kq) * scale : out_lane;
            }
        } else {
            // g is per-row on this path, so it is hoisted with q/k. exp() is pure, so
            // computing it once instead of once per use is bit-identical.
            float g_reg[rows_per_lane];
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                g_reg[r]    = sycl::native::exp(g_t[i]);
            }

            // Same substitution as the !KDA branch. g is per row here, so it weights qs
            // inside the sum instead of scaling the result afterwards:
            //   attn[col] = sum_i g[i]*S[i][col]*q[i] + delta[col] * sum_i k[i]*q[i]
            float kv_col[C];
            float qs_col[C];
#pragma unroll
            for (int c = 0; c < C; c++) {
                sycl::float2 acc(0.0f, 0.0f);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    const float gs = g_reg[r] * s_shard[c][r];
                    acc.x() += gs * k_reg[r];
                    acc.y() += gs * q_reg[r];
                }
                acc       = warp_reduce_sum<warp_size>(acc);
                kv_col[c] = acc.x();
                qs_col[c] = acc.y();
            }

            float kq_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kq_shard += k_reg[r] * q_reg[r];
            }
            const float kq = warp_reduce_sum<warp_size>(kq_shard);

#pragma unroll
            for (int c = 0; c < C; c++) {
                // delta[col] = (v[col] - kv[col]) * beta
                const float delta_col = (v_reg[c] - kv_col[c]) * beta_val;

                // S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = g_reg[r] * s_shard[c][r] + k_reg[r] * delta_col;
                }

                out_lane = (lane == c) ? (qs_col[c] + delta_col * kq) * scale : out_lane;
            }
        }

        // One store instruction, lanes 0..C-1 active on consecutive addresses: the
        // hardware coalesces it into a single transaction. The wide-store variants that
        // preceded this built the same C values *in one lane* and each cost a spill --
        // C results have to stay live across the s_shard update loop. This keeps every
        // value dying in its own iteration and holds exactly one extra register.
        if (lane < C) {
            attn_data[col0 + lane] = out_lane;
        }

        attn_data += S_v * H;


    // Write state back to global memory
        if constexpr (keep_rs_t) {
            const int target_slot = (int) n_tokens - 1 - t;
            if (target_slot >= 0 && target_slot < K) {
                float * curr_state = state + target_slot * state_slot_stride;
#pragma unroll
                for (int c = 0; c < C; c++) {
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        curr_state[(col0 + c) * S_v + i] = s_shard[c][r];
                    }
                }
            }
        }
    }

    if constexpr (!keep_rs_t) {
#pragma unroll
        for (int c = 0; c < C; c++) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                state[(col0 + c) * S_v + i] = s_shard[c][r];
            }
        }
    }
}

// warps per work-group; the column block is clamped so that S_v / C never drops
// below this, which keeps every warp's columns inside the state.
//
// A group covers C * warps columns, so each (head, token)'s q_t/k_t row is re-read by
// S_v / (C * warps) groups. Raising this is therefore the direct test of whether the
// GDN row is bound by redundant q/k load traffic: it cuts exactly that redundancy and
// reaches nothing else (the kernel takes its warp index from the nd_item at runtime,
// so no registers and no s_shard change). Selected by GGML_SYCL_GDN_WARPS.
// The cols clamp below keeps its shipped meaning (C small enough for a 4-warp group);
// the warp count is clamped separately, against the columns a group actually has.
static constexpr int gdn_cols_clamp_warps = 4;

static int ggml_sycl_gdn_num_warps() {
    static const int v = [] {
        int n = 32;
        if (const char * e = std::getenv("GGML_SYCL_GDN_WARPS")) {
            const int p = std::atoi(e);
            // block is warp_size * n work-items; 32 keeps that inside every device's
            // limit and, at S_v=128/C=4, is exactly one group per head.
            if (p >= 1 && p <= 32) {
                n = p;
            }
        }
        GGML_LOG_DEBUG("%s: GDN warps per work-group = %d\n", __func__, n);
        return n;
    }();
    return v;
}

template <int C, bool KDA, bool keep_rs_t>
static void launch_gated_delta_net_cols(
                                        const float *   q_d,
                                        const float *   k_d,
                                        const float *   v_d,
                                        const float *   g_d,
                                        const float *   b_d,
                                        const float *   s_d,
                                        float *         dst_d,
                                        float *         state_d,
                                        int64_t         state_slot_stride,
                                        int64_t         S_v,
                                        int64_t         H,
                                        int64_t         n_tokens,
                                        int64_t         n_seqs,
                                        int64_t         sq1,
                                        int64_t         sq2,
                                        int64_t         sq3,
                                        int64_t         sv1,
                                        int64_t         sv2,
                                        int64_t         sv3,
                                        int64_t         sb1,
                                        int64_t         sb2,
                                        int64_t         sb3,
                                        int64_t         neqk1,
                                        int64_t         rq3,
                                        float           scale,
                                        int             K,
                                        bool            beta_sigmoid,
                                        const int32_t * state_row_idx,
                                        int64_t         state_row_stride,
                                        dpct::queue_ptr stream) {
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    // one warp now covers C columns, so C times fewer warps are needed
    // A group's warps split S_v / C column-blocks between them, so a warp count above
    // that leaves the tail warps addressing past the end of the state. Halving always
    // lands on a legal value: S_v and C are both powers of two.
    int gdn_warps = ggml_sycl_gdn_num_warps();
    while (gdn_warps > 1 && gdn_warps > S_v / C) {
        gdn_warps /= 2;
    }
    dpct::dim3 grid_dims(H, n_seqs, (S_v / C + gdn_warps - 1) / gdn_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, gdn_warps, 1);

    const sycl::uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const sycl::uint3 rq3_magic   = init_fastdiv_values(rq3);

    switch (S_v) {
        case 16:
            {
                constexpr int sv = 16;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, C, KDA, keep_rs_t>(
                                                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride);
                                     });
            }
            break;
        case 32:
            {
                constexpr int sv = 32;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, C, KDA, keep_rs_t>(
                                                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride);
                                     });
            }
            break;
        case 64:
            {
                constexpr int sv = 64;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, C, KDA, keep_rs_t>(
                                                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride);
                                     });
            }
            break;
        case 128:
            {
                constexpr int sv = 128;
                stream->parallel_for(sycl::nd_range<3>(grid_dims * block_dims, block_dims),
                                     [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                                         gated_delta_net_sycl<sv, C, KDA, keep_rs_t>(
                                                 q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, H, n_tokens, sq1, sq2, sq3, sv1, sv2, sv3,
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride);
                                     });
            }
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

template <bool KDA, bool keep_rs_t>
static void launch_gated_delta_net(
                                   const float *   q_d,
                                   const float *   k_d,
                                   const float *   v_d,
                                   const float *   g_d,
                                   const float *   b_d,
                                   const float *   s_d,
                                   float *         dst_d,
                                   float *         state_d,
                                   int64_t         S_v,
                                   int64_t         H,
                                   int64_t         n_tokens,
                                   int64_t         n_seqs,
                                   int64_t         sq1,
                                   int64_t         sq2,
                                   int64_t         sq3,
                                   int64_t         sv1,
                                   int64_t         sv2,
                                   int64_t         sv3,
                                   int64_t         sb1,
                                   int64_t         sb2,
                                   int64_t         sb3,
                                   int64_t         neqk1,
                                   int64_t         rq3,
                                   float           scale,
                                   int64_t         state_slot_stride,
                                   int             K,
                                   bool            beta_sigmoid,
                                   const int32_t * state_row_idx,
                                   int64_t         state_row_stride,
                                   dpct::queue_ptr stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    // Columns per warp. 1 reproduces the original launch exactly; 4 is the measured
    // default (+4.00% pp8192). Higher values cut the redundant per-column q/k loads
    // without changing any column's arithmetic.
    static const int cols_req = [] {
        const char * e = std::getenv("GGML_SYCL_GDN_COLS");
        if (!e) {
            return 4;
        }
        const int v = std::atoi(e);
        return (v == 1 || v == 2 || v == 4) ? v : 4;
    }();

    // Clamp so S_v / C >= gdn_num_warps: otherwise the tail warps of a group would
    // address columns past the end of the state. S_v is a power of two >= 16 and C is a
    // power of two <= 8, so halving always lands on a legal value.
    int cols = cols_req;
    while (cols > 1 && S_v / cols < gdn_cols_clamp_warps) {
        cols /= 2;
    }

    switch (cols) {
        case 1:
            launch_gated_delta_net_cols<1, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, state_slot_stride,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
            break;
        case 2:
            launch_gated_delta_net_cols<2, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, state_slot_stride,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
            break;
        default:
            launch_gated_delta_net_cols<4, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, state_slot_stride,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache, bool beta_sigmoid,
                                              const ggml_tensor * state_gather) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    // When the producing SIGMOID is folded into the kernel its output is never
    // materialised, so read the sigmoid's own input instead. Same shape and both
    // contiguous, so the shared beta/g stride triple below is unchanged -- asserted
    // below rather than assumed, because those strides also drive g's offsets.
    const ggml_tensor * beta_src = beta_sigmoid ? src_beta->src[0] : src_beta;
    const float * b_d = (const float *) beta_src->data;

    // When a preceding GET_ROWS is folded away, read the cache it would have gathered from
    // and let the kernel select the row. src[0] of that gather is the cache as a 2-D
    // [row_size, n_rows] view; src[1] is the I32 row index per sequence.
    const int32_t * state_row_idx    = nullptr;
    int64_t         state_row_stride = 0;
    const ggml_tensor * state_src    = src_state;
    if (state_gather != nullptr) {
        state_src        = state_gather->src[0];
        state_row_idx    = (const int32_t *) state_gather->src[1]->data;
        state_row_stride = (int64_t) (state_src->nb[1] / sizeof(float));
        GGML_ASSERT(state_src->type == GGML_TYPE_F32);
        GGML_ASSERT(state_gather->src[1]->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(state_src));
        // the gathered row must be exactly the state this kernel expects, or the fold
        // would silently hand it a differently-shaped buffer
        GGML_ASSERT(state_row_stride == (int64_t) ggml_nelements(src_state) / src_state->ne[3]);
    }
    const float * s_d   = (const float *) state_src->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(beta_src->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(beta_src));
    GGML_ASSERT(ggml_are_same_shape(beta_src, src_beta));
    GGML_ASSERT(ggml_are_same_stride(beta_src, src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    dpct::queue_ptr stream = ctx.stream();

    // K (snapshot slot count) is an op param; state holds s0 only [S_v, S_v, H, n_seqs].
    const int K = ggml_get_op_params_i32(dst, 0);
    const bool keep_rs = K > 1;

    // recurrent state -> dst tail (after attention scores), or the cache when fusing
    float * state_d           = dst_d + S_v * H * n_tokens * n_seqs;
    int64_t state_slot_stride = S_v * S_v * H * n_seqs;
    if (cache != nullptr) {
        state_d           = cache->data;
        state_slot_stride = cache->slot_stride;
    }

    if (kda) {
        if (keep_rs) {
            launch_gated_delta_net<true, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, state_row_idx, state_row_stride, stream);
        }
    }
}

// The "_beta" suffix is not decoration: it makes the fold countable in a
// GGML_SYCL_DEBUG dispatch census, so "is it actually firing?" is one grep rather than
// an inference from profiler row counts. A fusion that silently declines looks exactly
// like one that works, and this backend has shipped that twice.
// Suffixes are how a census answers "did it fire?" with one grep instead of an inference
// from profiler row counts. Two independent folds now, so the suffix names both.
static const char * gdn_suffix(bool beta_sigmoid, bool state_gather) {
    if (beta_sigmoid && state_gather) { return "_beta_gather"; }
    if (beta_sigmoid)                 { return "_beta"; }
    if (state_gather)                 { return "_gather"; }
    return "";
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool beta_sigmoid,
                                  const ggml_tensor * state_gather) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr, beta_sigmoid, state_gather);
}

void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool beta_sigmoid,
                               const ggml_tensor * state_gather) {
    scope_op_debug_print scope_dbg_print(__func__, gdn_suffix(beta_sigmoid, state_gather != nullptr), dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst, beta_sigmoid, state_gather);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache, bool beta_sigmoid,
                                              const ggml_tensor * state_gather) {
    scope_op_debug_print scope_dbg_print(__func__, gdn_suffix(beta_sigmoid, state_gather != nullptr), dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache, beta_sigmoid, state_gather);
}

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
                          bool              beta_sigmoid) {
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
    const int64_t state_in_offset      = sequence * H * S_v * S_v + h_idx * S_v * S_v;
    const int64_t state_out_offset     = (sequence * H + h_idx) * S_v * S_v;
    state += state_out_offset;
    curr_state += state_in_offset;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    constexpr int warp_size = ggml_sycl_get_physical_warp_size() < S_v ? ggml_sycl_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    static_assert(S_v % C == 0, "S_v must be a multiple of the column block C");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
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
            float kv_col[C];
            float qs_col[C];
#pragma unroll
            for (int c = 0; c < C; c++) {
                sycl::float2 acc(0.0f, 0.0f);
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    acc.x() += s_shard[c][r] * k_reg[r];
                    acc.y() += s_shard[c][r] * q_reg[r];
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
                // delta[col] = (v[col] - g * kv[col]) * beta
                const float delta_col = (v_reg[c] - g_val * kv_col[c]) * beta_val;

                // S[i][col] = g * S[i][col] + k[i] * delta[col]
#pragma unroll
                for (int r = 0; r < rows_per_lane; r++) {
                    s_shard[c][r] = g_val * s_shard[c][r] + k_reg[r] * delta_col;
                }

                if (lane == 0) {
                    attn_data[col0 + c] = (g_val * qs_col[c] + delta_col * kq) * scale;
                }
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

                if (lane == 0) {
                    attn_data[col0 + c] = (qs_col[c] + delta_col * kq) * scale;
                }
            }
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
static constexpr int gdn_num_warps = 4;

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
                                        dpct::queue_ptr stream) {
    const int warp_size = ggml_sycl_info().devices[ggml_sycl_get_device()].warp_size;

    // one warp now covers C columns, so C times fewer warps are needed
    dpct::dim3 grid_dims(H, n_seqs, (S_v / C + gdn_num_warps - 1) / gdn_num_warps);
    dpct::dim3 block_dims(warp_size <= S_v ? warp_size : S_v, gdn_num_warps, 1);

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
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
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
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
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
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
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
                                                 sb1, sb2, sb3, neqk1_magic, rq3_magic, scale, state_slot_stride, K, beta_sigmoid);
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
    while (cols > 1 && S_v / cols < gdn_num_warps) {
        cols /= 2;
    }

    switch (cols) {
        case 1:
            launch_gated_delta_net_cols<1, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, state_slot_stride,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, K, beta_sigmoid, stream);
            break;
        case 2:
            launch_gated_delta_net_cols<2, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, state_slot_stride,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, K, beta_sigmoid, stream);
            break;
        default:
            launch_gated_delta_net_cols<4, KDA, keep_rs_t>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d, state_slot_stride,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3, sb1, sb2, sb3, neqk1, rq3, scale, K, beta_sigmoid, stream);
            break;
    }
}

static void ggml_sycl_op_gated_delta_net_impl(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              const ggml_sycl_gated_delta_net_fused_cache * cache, bool beta_sigmoid) {
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

    const float * s_d   = (const float *) src_state->data;
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
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        } else {
            launch_gated_delta_net<true, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        }
    } else {
        if (keep_rs) {
            launch_gated_delta_net<false, true>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        } else {
            launch_gated_delta_net<false, false>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d, state_d,
                S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1, rq3, scale, state_slot_stride, K, beta_sigmoid, stream);
        }
    }
}

void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool beta_sigmoid) {
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, nullptr, beta_sigmoid);
}

// The "_beta" suffix is not decoration: it makes the fold countable in a
// GGML_SYCL_DEBUG dispatch census, so "is it actually firing?" is one grep rather than
// an inference from profiler row counts. A fusion that silently declines looks exactly
// like one that works, and this backend has shipped that twice.
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool beta_sigmoid) {
    scope_op_debug_print scope_dbg_print(__func__, beta_sigmoid ? "_beta" : "", dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net(ctx, dst, beta_sigmoid);
}

void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache, bool beta_sigmoid) {
    scope_op_debug_print scope_dbg_print(__func__, beta_sigmoid ? "_beta" : "", dst, /*num_src=*/6);
    ggml_sycl_op_gated_delta_net_impl(ctx, dst, &cache, beta_sigmoid);
}

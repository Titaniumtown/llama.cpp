// Flash attention via oneMKL GEMM (XMX-accelerated).
// Uses column_major::gemm for Q*K^T and S*V matmuls
// with an online softmax SYCL kernel.
//
// All GQA query heads sharing a KV head are batched into single
// GEMM calls, amortizing MKL launch overhead across K and V reuse.
//

#include "common.hpp"
#include "fattn-common.hpp"
#include "convert.hpp"
#include "fattn.hpp"

#include <oneapi/mkl.hpp>
#include <cstdio>
#include <climits>
#include <chrono>
#include <vector>

#define MKL_FA_CHUNK_SIZE_KV 8192

// Number of query rows processed per tile. The score buffers (KQ_f32, S_f16)
// are sized q_tile_rows * chunk_size, so this bounds their footprint
// regardless of batch size (n_query_rows = n_queries * gqa_ratio). A typical
// single-ubatch prefill (e.g. ubatch 1024 * gqa 8 = 8192 rows) is exactly one
// tile, so it runs with no extra iterations. Larger batches tile and stay
// bounded. Override with GGML_SYCL_MKL_FA_Q_TILE.
#define MKL_FA_Q_TILE 8192

#define MKL_FA_WG_SIZE 256

using oneapi::mkl::transpose;
using oneapi::mkl::blas::column_major::gemm;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Pack all GQA Q heads for one KV head into fp16, applying q_scale.
// Launches one kernel per GQA group — each kernel copies exactly
// n_queries * DKQ elements using the per-group dst offset and
// per-head source stride.
static void mkl_fa_pack_q_fp16(
    dpct::queue_ptr stream,
    sycl::half * __restrict dst,
    const float * __restrict q_src,
    int n_queries, int DKQ,
    int gqa_ratio, int kvh_base_head,
    float q_scale, int64_t q_row_stride, int64_t q_head_stride,
    int64_t wg_size) {

    for (int iqg = 0; iqg < gqa_ratio; iqg++) {
        int     iqh       = kvh_base_head + iqg;
        sycl::half * dst_g = dst + (int64_t)iqg * n_queries * DKQ;

        const int64_t n_elem = (int64_t)n_queries * DKQ;
        const int64_t wg = ((n_elem + wg_size - 1) / wg_size) * wg_size;

        stream->submit([&](sycl::handler & cgh) {
            cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                [=](sycl::nd_item<1> item) {
                    int64_t e = item.get_global_id(0);
                    if (e >= n_elem) return;

                    int64_t q = e / DKQ;
                    int64_t d = e - q * DKQ;

                    // Stride-aware source offset: handles permuted,
                    // sliced, or contiguous Q tensor layouts.
                    int64_t src_off = d
                        + q * q_row_stride
                        + (int64_t)iqh * q_head_stride;

                    dst_g[e] = sycl::half(
                        q_src[src_off] * q_scale);
                });
        });
    }
}

// Zero-initialize the online softmax state arrays.
// KQ_max → -inf, KQ_sum → 0, VKQ_accum → 0.
// Merged into one kernel to avoid per-array launch overhead.
static void mkl_fa_init_softmax_state(
    dpct::queue_ptr stream,
    float * kmax, float * ksum, float * vacc,
    int n_query_rows, int DV, int64_t wg_size) {

    const float    neg_inf   = -1e30f;
    const int64_t  n_maxsum  = n_query_rows;
    const int64_t  n_vacc    = (int64_t)n_query_rows * DV;
    const int64_t  total     = (n_vacc > n_maxsum) ? n_vacc : n_maxsum;
    const int64_t  wg = ((total + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int64_t i = item.get_global_id(0);
                if (i < n_maxsum) {
                    kmax[i] = neg_inf;
                    ksum[i] = 0.0f;
                }
                if (i < n_vacc) {
                    vacc[i] = 0.0f;
                }
            });
    });
}

// Online softmax over one KV chunk for a tile of GQA query rows.
// The tile spans absolute rows [q0, q0 + q_rows). Score buffers
// (KQ_f32/S_f16) are indexed RELATIVE to the tile; the persistent state
// (VKQ_accum/KQ_max/KQ_sum) and mask are indexed by ABSOLUTE row.
// For each row: find local max → rescale previous VKQ_accum →
// compute exp(s - max) → write S_f16 → update running max/sum.
//
// One sub-group (32 lanes) cooperates on one row: lane l handles the
// interleaved float4s (l, l+32, l+64, ...), so the loads/stores coalesce
// across the sub-group and each lane's max/sum chain is only nv/32 deep
// (16 at chunk_size 8192). The original kernel walked one row per work-item
// with a serial 8192-deep dependency chain and ran ~9x off the DRAM wall
// (73% of the FA call at depth); the per-row semantics are unchanged (the
// max/sum values are identical, the sum grouping differs by ~1e-7).
// The three mask forms (implicit causal, materialized mask, no mask) share the
// same 4-wide strided row walk; only the per-float4 mask application differs.
// Both passes are templated on it so each instantiation compiles to the same
// code as a hand-written variant (no per-iteration mask branch).
template <typename TApply>
static inline void mkl_fa_row_max4(
        const sycl::float4 * __restrict KQ4, int lane, int sg_size, int nv,
        float & m0, float & m1, float & m2, float & m3, TApply apply) {
    int i = lane;
    for (; i + 3 * sg_size < nv; i += 4 * sg_size) {
        // load all four KQ vectors before masking: the original access order
        // (batch the KQ stream, then the mask stream) measured faster than
        // interleaving them.
        sycl::float4 a = KQ4[i];
        sycl::float4 b = KQ4[i + sg_size];
        sycl::float4 c = KQ4[i + 2 * sg_size];
        sycl::float4 d = KQ4[i + 3 * sg_size];
        a = apply(i, a);
        b = apply(i + sg_size, b);
        c = apply(i + 2 * sg_size, c);
        d = apply(i + 3 * sg_size, d);
        m0 = sycl::fmax(sycl::fmax(sycl::fmax(m0, a.x()), b.x()), sycl::fmax(c.x(), d.x()));
        m1 = sycl::fmax(sycl::fmax(sycl::fmax(m1, a.y()), b.y()), sycl::fmax(c.y(), d.y()));
        m2 = sycl::fmax(sycl::fmax(sycl::fmax(m2, a.z()), b.z()), sycl::fmax(c.z(), d.z()));
        m3 = sycl::fmax(sycl::fmax(sycl::fmax(m3, a.w()), b.w()), sycl::fmax(c.w(), d.w()));
    }
    for (; i < nv; i += sg_size) {
        const sycl::float4 a = apply(i, KQ4[i]);
        m0 = sycl::fmax(m0, a.x());
        m1 = sycl::fmax(m1, a.y());
        m2 = sycl::fmax(m2, a.z());
        m3 = sycl::fmax(m3, a.w());
    }
}

static inline sycl::float4 mkl_fa_exp4(sycl::float4 v, float new_max) {
    return sycl::float4(sycl::native::exp(v.x() - new_max),
                        sycl::native::exp(v.y() - new_max),
                        sycl::native::exp(v.z() - new_max),
                        sycl::native::exp(v.w() - new_max));
}

template <typename TApply>
static inline void mkl_fa_row_exp4(
        sycl::half4 * __restrict S4, const sycl::float4 * __restrict KQ4,
        int lane, int sg_size, int nv, float new_max,
        float & s0, float & s1, float & s2, float & s3, TApply apply) {
    int i = lane;
    for (; i + 3 * sg_size < nv; i += 4 * sg_size) {
        // load all four KQ vectors before masking (see mkl_fa_row_max4).
        sycl::float4 a = KQ4[i];
        sycl::float4 b = KQ4[i + sg_size];
        sycl::float4 c = KQ4[i + 2 * sg_size];
        sycl::float4 d = KQ4[i + 3 * sg_size];
        const sycl::float4 e0 = mkl_fa_exp4(apply(i, a), new_max);
        const sycl::float4 e1 = mkl_fa_exp4(apply(i + sg_size, b), new_max);
        const sycl::float4 e2 = mkl_fa_exp4(apply(i + 2 * sg_size, c), new_max);
        const sycl::float4 e3 = mkl_fa_exp4(apply(i + 3 * sg_size, d), new_max);
        S4[i]               = e0.convert<sycl::half>();
        S4[i + sg_size]     = e1.convert<sycl::half>();
        S4[i + 2 * sg_size] = e2.convert<sycl::half>();
        S4[i + 3 * sg_size] = e3.convert<sycl::half>();
        s0 += (e0.x() + e1.x()) + (e2.x() + e3.x());
        s1 += (e0.y() + e1.y()) + (e2.y() + e3.y());
        s2 += (e0.z() + e1.z()) + (e2.z() + e3.z());
        s3 += (e0.w() + e1.w()) + (e2.w() + e3.w());
    }
    for (; i < nv; i += sg_size) {
        const sycl::float4 e = mkl_fa_exp4(apply(i, KQ4[i]), new_max);
        S4[i] = e.convert<sycl::half>();
        s0 += e.x();
        s1 += e.y();
        s2 += e.z();
        s3 += e.w();
    }
}

static void mkl_fa_online_softmax_chunk(
    dpct::queue_ptr stream,
    float * __restrict KQ_f32,
    sycl::half * __restrict S_f16,
    float * __restrict KQ_max,
    float * __restrict KQ_sum,
    float * __restrict VKQ_accum,
    int q0, int q_rows, int n_queries, int DV,
    int chunk_size, int row_ld, int chunk_start,
    int kvh_head,
    const sycl::half * mask_data, int64_t mask_head_stride,
    int64_t mask_row_stride, int mask_n_heads,
    bool implicit_causal, int n_past,
    float logit_softcap, int64_t wg_size,
    const sycl::event & dep_event) {

    // The kernel-selection gate (fattn.cpp) only routes logit_softcap == 0
    // to the MKL kernel; the vectorized body below drops the tanh path.
    GGML_ASSERT(logit_softcap == 0.0f && "MKL FA requires logit_softcap == 0 (gated in fattn.cpp)");

    // One sub-group (32 lanes) processes one row: lanes cover the row's
    // float4s strided, then tree-reduce. Launch rows * 32 items so padding
    // is whole sub-groups (the row guard is sub-group-uniform, keeping the
    // shuffles divergence-free).
    constexpr int MKL_FA_SG = 32;
    const int64_t wg = ((q_rows * MKL_FA_SG + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.depends_on(dep_event);
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
                const sycl::sub_group sg = item.get_sub_group();
                const int sg_size = sg.get_local_linear_range();
                const int lane    = sg.get_local_linear_id();
                const int jc_rel  = item.get_global_id(0) / sg_size;  // == row within the tile
                if (jc_rel >= q_rows) return;  // sub-group-uniform: only whole padded sub-groups exit

                const int jc_abs = q0 + jc_rel;
                const int gqa_group = jc_abs / n_queries;
                const int q_row     = jc_abs % n_queries;

                // row_ld (not chunk_size) is the row pitch: it is padded to a
                // multiple of 8 by the caller so every row's base stays
                // 16B-aligned for the float4/half4 accesses below. A raw
                // chunk_size pitch misaligns every odd row whenever
                // chunk_size % 4 != 0 (e.g. n_kv = 1025).
                const float * __restrict KQ_row = KQ_f32
                    + jc_rel * (int64_t) row_ld;
                sycl::half * __restrict S_row = S_f16
                    + jc_rel * (int64_t) row_ld;
                float * __restrict vkq = VKQ_accum
                    + jc_abs * (int64_t) DV;

                // implicit causal: cols >= bd are masked (-inf); the
                // materialized mask path reads the mask per element.
                const int bd = implicit_causal ? n_past + q_row + 1
                                               : INT_MAX;
                const sycl::half * mask_h = (mask_data && !implicit_causal)
                    ? mask_data
                        + (int64_t) ((mask_n_heads > 1) ? (kvh_head + gqa_group) % mask_n_heads : 0) * mask_head_stride
                        + (int64_t) q_row * mask_row_stride + chunk_start
                    : nullptr;

                const sycl::float4 neg_inf4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);

                // Each lane walks the row's float4s strided by sg_size. nv is
                // NOT guaranteed to be a multiple of sg_size -- this_chunk is
                // n_kv % MKL_FA_CHUNK_SIZE_KV on the final chunk -- so the
                // 4-way-unrolled loop is followed by a strided cleanup over the
                // leftover float4s and a lane-0 scalar tail for the
                // chunk_size % 4 leftover columns (the newest keys).
                const int nv  = chunk_size / 4;
                const int rem = chunk_size - nv * 4;
                const sycl::float4 * __restrict KQ4 = (const sycl::float4 *) KQ_row;
                // The mask pitch is the tensor's, not ours: a half4 load needs
                // the row base 4-half aligned, which an odd n_kv breaks.
                const bool mask_vec_ok = mask_h &&
                    ((mask_row_stride & 3) == 0) && ((mask_head_stride & 3) == 0);
                const sycl::half4 * __restrict M4 =
                    (const sycl::half4 *) mask_h;   // mask path: 4 halves per float4
                const auto load_mask4 = [=](int i) {
                    if (mask_vec_ok) {
                        return M4[i].convert<float>();
                    }
                    const sycl::half * __restrict p = mask_h + (int64_t) i * 4;
                    return sycl::float4((float) p[0], (float) p[1],
                                        (float) p[2], (float) p[3]);
                };

                // --- pass 1: row local maximum ---
                float m0 = -1e30f, m1 = -1e30f, m2 = -1e30f, m3 = -1e30f;
                if (implicit_causal) {
                    mkl_fa_row_max4(KQ4, lane, sg_size, nv, m0, m1, m2, m3,
                        [=](int i, sycl::float4 v) {
                            const int c0 = chunk_start + i * 4;
                            // select(a, b, c) is msbIsSet(c) ? b : a -- masked
                            // columns (key index >= bd) must take -inf.
                            return sycl::select(v, neg_inf4,
                                sycl::int4(c0, c0 + 1, c0 + 2, c0 + 3) >= bd);
                        });
                } else if (mask_h) {
                    mkl_fa_row_max4(KQ4, lane, sg_size, nv, m0, m1, m2, m3,
                        [=](int i, sycl::float4 v) { return v + load_mask4(i); });
                } else {
                    mkl_fa_row_max4(KQ4, lane, sg_size, nv, m0, m1, m2, m3,
                        [](int, sycl::float4 v) { return v; });
                }

                // Tail: chunk_size % 4 leftover columns folded in by lane 0.
                if (lane == 0) {
                    for (int t = 0; t < rem; t++) {
                        const int col = nv * 4 + t;
                        float s = KQ_row[col];
                        if (implicit_causal) {
                            if (chunk_start + col >= bd) s += -INFINITY;
                        } else if (mask_h) {
                            s += (float) mask_h[col];
                        }
                        m0 = sycl::fmax(m0, s);
                    }
                }

                float local_max = sycl::fmax(sycl::fmax(m0, m1), sycl::fmax(m2, m3));
                #pragma unroll
                for (int m = sg_size / 2; m > 0; m >>= 1) {
                    local_max = sycl::fmax(local_max,
                        dpct::permute_sub_group_by_xor(sg, local_max, m));
                }
                // --- rescale previous accumulator (every lane: DV/sg_size) ---
                const float old_max = KQ_max[jc_abs];
                const float new_max = sycl::fmax(old_max, local_max);
                const float rescale = (old_max < -1e29f) ? 1.0f
                    : sycl::native::exp(old_max - new_max);
                {
                    const int nv_v = DV / 4;   // DV % 4 == 0 (head_dim % 64 == 0)
                    sycl::float4 * vkq4 = (sycl::float4 *) vkq;
                    for (int j = lane; j < nv_v; j += sg_size) {
                        vkq4[j] *= rescale;
                    }
                }

                // --- pass 2: exp(s - max) -> S_f16, partial sums ---
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                {
                    sycl::half4 * S4 = (sycl::half4 *) S_row;
                    if (implicit_causal) {
                        mkl_fa_row_exp4(S4, KQ4, lane, sg_size, nv, new_max, s0, s1, s2, s3,
                            [=](int i, sycl::float4 v) {
                                const int c0 = chunk_start + i * 4;
                                return sycl::select(v, neg_inf4,
                                    sycl::int4(c0, c0 + 1, c0 + 2, c0 + 3) >= bd);
                            });
                    } else if (mask_h) {
                        mkl_fa_row_exp4(S4, KQ4, lane, sg_size, nv, new_max, s0, s1, s2, s3,
                            [=](int i, sycl::float4 v) { return v + load_mask4(i); });
                    } else {
                        mkl_fa_row_exp4(S4, KQ4, lane, sg_size, nv, new_max, s0, s1, s2, s3,
                            [](int, sycl::float4 v) { return v; });
                    }
                }

                if (lane == 0) {
                    for (int t = 0; t < rem; t++) {
                        const int col = nv * 4 + t;
                        float s = KQ_row[col];
                        if (implicit_causal) {
                            if (chunk_start + col >= bd) s += -INFINITY;
                        } else if (mask_h) {
                            s += (float) mask_h[col];
                        }
                        const float e = sycl::native::exp(s - new_max);
                        S_row[col] = sycl::half(e);
                        s0 += e;
                    }
                }

                float local_sum = (s0 + s1) + (s2 + s3);
                #pragma unroll
                for (int m = sg_size / 2; m > 0; m >>= 1) {
                    local_sum += dpct::permute_sub_group_by_xor(sg, local_sum, m);
                }
                if (lane == 0) {
                    KQ_sum[jc_abs] = KQ_sum[jc_abs] * rescale + local_sum;
                    KQ_max[jc_abs] = new_max;
                }
            });
    });
}
static void mkl_fa_normalize_head(
    dpct::queue_ptr stream,
    float * __restrict dst_batch,
    const float * __restrict VKQ_accum,
    const float * __restrict KQ_sum,
    int iqh, int n_queries, int DV, int n_q_heads,
    int64_t src_offset, int64_t wg_size) {

    const int64_t wg = ((n_queries + wg_size - 1) / wg_size) * wg_size;

    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
            [=](sycl::nd_item<1> item) {
                int jc = item.get_global_id(0);
                if (jc >= n_queries) return;

                int     ksum_idx = (int)(src_offset / DV) + jc;
                float   inv_sum  = 1.0f / KQ_sum[ksum_idx];
                const float * __restrict src = VKQ_accum
                    + src_offset + jc * (int64_t)DV;
                // Interleaved dst layout (matching TILE):
                // rows alternate between heads, then increment query.
                // offset = (query * n_q_heads + head) * DV
                float * __restrict dst_row = dst_batch
                    + ((int64_t)jc * n_q_heads + iqh) * (int64_t)DV;

                for (int v = 0; v < DV; v++) {
                    dst_row[v] = src[v] * inv_sum;
                }
            });
    });
}

// ---------------------------------------------------------------------------
// Per-chunk dequant
//
// Rather than dequantizing all of K/V up front (footprint scales with
// context), we dequant one KV-head chunk at a time into a dense
// [this_chunk x D] fp16 buffer (row-major, lda = D). The source address of
// element (head=ikvh, row=chunk_start+r, col=c) decomposes into independent
// linear terms head_off(ikvh) + row_off(chunk_start) + (r,c), so slicing a
// chunk is a clean pointer offset in every layout case. The true-Gemma-
// interleave vs padded-seq-view distinction is resolved once when the
// descriptor is built; slicing does not reintroduce it.
// ---------------------------------------------------------------------------
enum mkl_fa_kv_desc_mode {
    MKL_FA_KV_MODE_F16_DENSE       = 0,
    MKL_FA_KV_MODE_F16_INTERLEAVED = 1,
    MKL_FA_KV_MODE_QUANT_CONTIG    = 2,
    MKL_FA_KV_MODE_QUANT_NC        = 3,
};

struct mkl_fa_kv_desc {
    const char *         data = nullptr;
    ggml_type            type = GGML_TYPE_F16;
    int64_t              D    = 0;      // ne[0]
    int64_t              nb1  = 0;      // byte stride, seq dim
    int64_t              nb2  = 0;      // byte stride, head dim
    int64_t              nb3  = 0;      // byte stride, batch dim
    mkl_fa_kv_desc_mode  mode = MKL_FA_KV_MODE_F16_DENSE;
    int64_t              ts   = 0;      // type size (mode 3 base offset)
    int64_t              s01  = 0;      // nc row stride in blocks (mode 3)
    int64_t              s02  = 0;      // nc head stride in blocks (mode 3)
    int                  reorder_seg = 0;  // row-split cache segment elems (0 = AoS)
};

static mkl_fa_kv_desc mkl_fa_make_desc(const ggml_tensor * T, bool interleaved) {
    mkl_fa_kv_desc d;
    d.data = (const char *)T->data;
    d.type = T->type;
    d.D    = T->ne[0];
    d.nb1  = (int64_t)T->nb[1];
    d.nb2  = (int64_t)T->nb[2];
    d.nb3  = (int64_t)T->nb[3];
    d.reorder_seg = (T->type == GGML_TYPE_Q8_0 || T->type == GGML_TYPE_Q4_0 ||
                     T->type == GGML_TYPE_Q5_1) ? ggml_sycl_kv_reorder_seg(T) : 0;
    d.ts   = (int64_t)ggml_type_size(T->type);

    if (T->type == GGML_TYPE_F16) {
        // MLA's V cache is a 512-wide view of 576-wide K rows. Treat any
        // padded row stride as strided even when there is only one KV head.
        d.mode = interleaved || d.nb1 != d.D * (int64_t)sizeof(sycl::half)
            ? MKL_FA_KV_MODE_F16_INTERLEAVED : MKL_FA_KV_MODE_F16_DENSE;
    } else if (ggml_is_contiguously_allocated(T) && !interleaved) {
        d.mode = MKL_FA_KV_MODE_QUANT_CONTIG;
    } else {
        d.mode = MKL_FA_KV_MODE_QUANT_NC;
        d.s01 = d.nb1 / d.ts;
        d.s02 = d.nb2 / d.ts;
    }
    return d;
}

// Dequant one KV-head chunk into a dense [this_chunk x D] fp16 buffer.
static void mkl_fa_dequant_chunk(
    dpct::queue_ptr stream, const mkl_fa_kv_desc & d, ggml_tensor * dst_ctx,
    sycl::half * out, int ikvh, int ib, int chunk_start, int this_chunk) {

    const int64_t D = d.D;
    switch (d.mode) {
        case MKL_FA_KV_MODE_F16_DENSE: {
            // ib * nb3 selects this batch's KV (nb[3] != ne[1]*nb[1] for batched KV views)
            const char * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1 + (int64_t)ib * d.nb3;
            stream->memcpy(out, base, (size_t)this_chunk * D * sizeof(sycl::half));
            break;
        }
        case MKL_FA_KV_MODE_F16_INTERLEAVED: {
            const char * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1 + (int64_t)ib * d.nb3;
            const int64_t row_halfs = d.nb1 / (int64_t)sizeof(sycl::half);
            const sycl::half * src = (const sycl::half *)base;
            stream->parallel_for(
                sycl::range<2>((size_t)this_chunk, (size_t)D),
                [=](sycl::item<2> it) {
                    int64_t r = it.get_id(0);
                    int64_t c = it.get_id(1);
                    out[r * D + c] = src[r * row_halfs + c];
                });
            break;
        }
        case MKL_FA_KV_MODE_QUANT_CONTIG: {
            const char * base = d.data + (int64_t)ikvh * d.nb2
                + (int64_t)chunk_start * d.nb1 + (int64_t)ib * d.nb3;
            if (d.reorder_seg > 0) {
                // contiguous chunk: rows dense at natural stride -> flat walk
                if ((size_t) d.nb1 == ggml_row_size(d.type, D)) {
                    ggml_sycl_reordered_to_fp16(d.type, base, out, (int64_t) this_chunk * D, d.reorder_seg, stream);
                } else {
                    ggml_sycl_reordered_to_fp16_nc(d.type, base, out, D, this_chunk, 1, 1,
                                                   (size_t) d.nb1, 0, 0, d.reorder_seg, stream);
                }
                break;
            }
            to_fp16_sycl_t to_fp16 = ggml_get_to_fp16_sycl(d.type, dst_ctx);
            to_fp16(base, out, (int64_t)this_chunk * D, stream);
            break;
        }
        default: {  // MKL_FA_KV_MODE_QUANT_NC
            if (d.reorder_seg > 0) {
                const char * base = d.data + (int64_t)ikvh * d.nb2
                    + (int64_t)chunk_start * d.nb1 + (int64_t)ib * d.nb3;
                ggml_sycl_reordered_to_fp16_nc(d.type, base, out, D, this_chunk, 1, 1,
                                               (size_t) d.nb1, 0, 0, d.reorder_seg, stream);
                break;
            }
            to_fp16_nc_sycl_t to_fp16 = ggml_get_to_fp16_nc_sycl(d.type);
            const int64_t base_blocks = (int64_t)ikvh * d.s02
                + (int64_t)chunk_start * d.s01 + (int64_t)ib * d.nb3 / d.ts;
            const char * base = d.data + base_blocks * d.ts;
            // ne02 = ne03 = 1 → s02/s03 inert; head+chunk offset carried by base.
            to_fp16(base, out, D, this_chunk, 1, 1, d.s01, d.s02, d.s02, stream);
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// MKL Flash Attention orchestrator
//
// Pipeline: dequantize K/V → for each KV head:
//   pack GQA Q heads → MKL GEMM KQ → online softmax →
//   MKL GEMM VKQ → accumulate → normalize → scatter to dst
// ---------------------------------------------------------------------------
void ggml_sycl_flash_attn_ext_mkl(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    ggml_tensor * KQV = dst;

    GGML_ASSERT(Q->type == GGML_TYPE_F32);
    GGML_ASSERT(KQV->type == GGML_TYPE_F32);

    // --- Op params ---
    float scale = 1.0f, max_bias = 0.0f, logit_softcap = 0.0f;
    memcpy(&scale,         (const float *)KQV->op_params + 0, sizeof(float));
    memcpy(&max_bias,      (const float *)KQV->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *)KQV->op_params + 2, sizeof(float));

    const float q_scale = scale;

    // --- Dimensions ---
    const int DKQ        = (int)K->ne[0];
    const int DV         = (int)V->ne[0];
    const int n_queries  = (int)Q->ne[1];
    const int n_q_heads  = (int)Q->ne[2];
    const int n_kv_heads = (int)K->ne[2];
    const int n_batch    = (int)Q->ne[3];
    const int n_kv       = (int)K->ne[1];
    const int gqa_ratio  = n_q_heads / n_kv_heads;
    const int n_query_rows = n_queries * gqa_ratio;

    GGML_ASSERT(n_q_heads % n_kv_heads == 0);
    GGML_ASSERT(max_bias == 0.0f);  // ALiBi not supported
    GGML_ASSERT(Q->ne[3] == K->ne[3] || K->ne[3] == 1);

    const int chunk_size = std::min(MKL_FA_CHUNK_SIZE_KV, n_kv);

    // Query rows are processed in tiles of q_tile_rows so the score buffers
    // (KQ_f32/S_f16 = q_tile_rows * chunk_size) stay bounded regardless of
    // batch size. n_query_rows <= Q_TILE is a single tile (no extra work).
    static int q_tile_env = ggml_sycl_get_env("GGML_SYCL_MKL_FA_Q_TILE", MKL_FA_Q_TILE);
    const int q_tile_rows = std::max(1, std::min(q_tile_env, n_query_rows));

    const int64_t wg_size = MKL_FA_WG_SIZE;

    // --- Debug output (gated by GGML_SYCL_MKL_FA_DEBUG=1) ---
    static int mkl_call_count = 0;
    mkl_call_count++;
    static int mkl_debug = ggml_sycl_get_env("GGML_SYCL_MKL_FA_DEBUG", 0);
    const bool do_print = (mkl_debug == 1);

    const int64_t q_row_stride  = Q->nb[1] / sizeof(float);
    const int64_t q_head_stride = Q->nb[2] / sizeof(float);

    // Alias the dequantized buffers only when K and V expose the same values.
    // MLA V is a narrower view of K and needs its own strided dequantization.
    const bool V_is_K_view = V->ne[0] == K->ne[0] && V->view_src
        && (V->view_src == K || (V->view_src == K->view_src
            && V->view_offs == K->view_offs));

    // Early interleaved detection for debug output.
    // True interleaved detection happens after dequant (nb12_fp16 == nb11_fp16),
    // but we can pre-detect on the original tensor strides.
    const bool k_early_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]);
    const bool v_early_interleaved =
        !V_is_K_view && ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]);

    if (do_print) {
        GGML_LOG_INFO("[MKL-FA] #%d D=%d DV=%d n_q=%d n_kv=%d "
                "n_qh=%d n_kvh=%d gqa=%d batch=%d K=%s V=%s "
                "chunk=%d buf=%.1fMB%s%s\n",
                mkl_call_count, DKQ, DV, n_queries, n_kv,
                n_q_heads, n_kv_heads, gqa_ratio, n_batch,
                ggml_type_name(K->type), ggml_type_name(V->type),
                chunk_size,
                (double)((int64_t)n_query_rows * chunk_size * sizeof(float))
                    / (1024.0 * 1024.0),
                k_early_interleaved ? " K_ILV" : "",
                v_early_interleaved ? " V_ILV" : "");
        GGML_LOG_INFO("[MKL-FA] #%d Q-nb1=%lld Q-nb2=%lld "
                "q_rs=%lld q_hs=%lld dst_rs=%lld dst_hs=%lld\n",
                mkl_call_count,
                (long long)Q->nb[1], (long long)Q->nb[2],
                (long long)q_row_stride, (long long)q_head_stride,
                (long long)(KQV->nb[1] / sizeof(float)),
                (long long)(KQV->nb[2] / sizeof(float)));
    }

    // --- Stream and allocators ---
    dpct::queue_ptr stream = ctx.stream();

#define MKL_TAKE_TIME(t0)  auto t0 = std::chrono::steady_clock::now()
#define MKL_ACCUM(acc, t0) do { if (do_print) { \
    acc += (int64_t)std::chrono::duration_cast \
    <std::chrono::microseconds>(std::chrono::steady_clock::now() - (t0)).count(); \
} } while(0)

    int64_t gemm_kq_time_us  = 0;
    int64_t gemm_vkq_time_us = 0;
    int64_t softmax_time_us  = 0;
    int64_t dequant_time_us  = 0;

    MKL_TAKE_TIME(t_deq);

    // --- K/V dequant descriptors ---
    // Dequant is done per-chunk inside the KV loop (footprint independent of
    // context). Output is always dense row-major fp16 [this_chunk x D], lda=D.
    // Interleaved detection: ne[1]*nb[1] != nb[2] means heads are interleaved.
    const bool k_interleaved =
        ((int64_t)K->ne[1] * K->nb[1] != K->nb[2]) && K->ne[2] > 1;
    const bool v_interleaved =
        ((int64_t)V->ne[1] * V->nb[1] != V->nb[2]) && V->ne[2] > 1;

    const mkl_fa_kv_desc K_desc = mkl_fa_make_desc(K, k_interleaved);
    const mkl_fa_kv_desc V_desc = V_is_K_view
        ? K_desc : mkl_fa_make_desc(V, v_interleaved);

    MKL_ACCUM(dequant_time_us, t_deq);

    // --- Resolve mask pointers ---
    int64_t mask_head_stride = 0;
    int64_t mask_row_stride  = 0;
    int     mask_n_heads     = 0;

    if (mask) {
        // Use actual fp16 device size (2 bytes), NOT sizeof(sycl::half)
        // which may be 4 on the host in oneAPI.
        mask_head_stride = mask->nb[2] / 2;
        mask_row_stride  = mask->nb[1] / 2;
        mask_n_heads     = (int)mask->ne[2];
    }

    // Implicit causal: a sentinel (ne[0]==1) mask signals the causal boundary
    // is computed in-kernel (key index <= n_past + q_row) rather than read from
    // a materialized [n_kv, n_tokens] mask. n_past is carried in ne[1]-1 because
    // K->ne[1] is padded to a 256-multiple and n_kv - n_queries overstates it.
    const bool implicit_causal = mask && mask->ne[0] == 1;
    const int  n_past = implicit_causal ? (int)(mask->ne[1] - 1) : (n_kv - n_queries);

    // --- Allocate intermediates from pool ---
    ggml_sycl_pool & pool = ctx.pool();

    ggml_sycl_pool_alloc<float>      KQ_f32(pool);      // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<sycl::half> S_f16(pool);       // [q_tile_rows x chunk]
    ggml_sycl_pool_alloc<float>      VKQ_chunk(pool);   // [q_tile_rows x DV]
    ggml_sycl_pool_alloc<float>      VKQ_accum(pool);   // [n_query_rows x DV] (full)
    ggml_sycl_pool_alloc<float>      KQ_max(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<float>      KQ_sum(pool);      // [n_query_rows] (full)
    ggml_sycl_pool_alloc<sycl::half> Q_head_f16(pool);  // [n_query_rows x DKQ] (full)
    ggml_sycl_pool_alloc<sycl::half> K_chunk_f16(pool); // [chunk x DKQ] (per-chunk dequant)
    ggml_sycl_pool_alloc<sycl::half> V_chunk_f16(pool); // [chunk x DV] (per-chunk dequant)

    // Score-buffer row pitch is padded to a multiple of 8 so every row base is
    // 16B-aligned for the softmax's float4/half4 accesses; the GEMMs take the
    // same padded ld and simply leave the pad columns untouched/unread.
    const int chunk_ld_max = (chunk_size + 7) & ~7;
    KQ_f32.alloc((size_t)q_tile_rows * chunk_ld_max);
    S_f16.alloc((size_t)q_tile_rows * chunk_ld_max);
    VKQ_chunk.alloc((size_t)q_tile_rows * DV);
    VKQ_accum.alloc((size_t)n_query_rows * DV);
    KQ_max.alloc(n_query_rows);
    KQ_sum.alloc(n_query_rows);
    Q_head_f16.alloc((size_t)n_query_rows * DKQ);
    K_chunk_f16.alloc((size_t)chunk_size * DKQ);

    sycl::half * V_chunk_f16_ptr;
    if (V_is_K_view) {
        V_chunk_f16_ptr = K_chunk_f16.ptr;   // V aliases K (DV == DKQ)
    } else {
        V_chunk_f16.alloc((size_t)chunk_size * DV);
        V_chunk_f16_ptr = V_chunk_f16.ptr;
    }

    sycl::half * Q_head_f16_ptr  = Q_head_f16.ptr;
    float      * KQ_f32_ptr      = KQ_f32.ptr;
    sycl::half * S_f16_ptr       = S_f16.ptr;
    float      * VKQ_chunk_ptr   = VKQ_chunk.ptr;
    float      * VKQ_accum_ptr   = VKQ_accum.ptr;
    float      * KQ_max_ptr      = KQ_max.ptr;
    float      * KQ_sum_ptr      = KQ_sum.ptr;
    sycl::half * K_chunk_f16_ptr = K_chunk_f16.ptr;

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    for (int ib = 0; ib < n_batch; ib++) {
        const float * Q_batch = (const float *)Q->data
            + ib * (Q->nb[3] / sizeof(float));
        float * dst_batch = (float *)KQV->data
            + ib * (KQV->nb[3] / sizeof(float));

        const sycl::half * mask_batch = nullptr;
        if (mask) {
            int m_batch = ib % (int)mask->ne[3];
            mask_batch = (const sycl::half *)mask->data
                + m_batch * (mask->nb[3] / 2);  // 2 = actual fp16 device size
        }

        for (int ikvh = 0; ikvh < n_kv_heads; ikvh++) {
            int kvh_base_head = ikvh * gqa_ratio;

            // 1. Pack all GQA Q heads into fp16 (full n_query_rows)
            mkl_fa_pack_q_fp16(stream,
                Q_head_f16_ptr, Q_batch,
                n_queries, DKQ,
                gqa_ratio, kvh_base_head,
                q_scale, q_row_stride, q_head_stride, wg_size);

            // 2. Initialize softmax state (full n_query_rows)
            mkl_fa_init_softmax_state(stream,
                KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                n_query_rows, DV, wg_size);

            // 3. KV chunk loop (OUTER): dequant each chunk once, then tile queries.
            for (int chunk_start = 0; chunk_start < n_kv; chunk_start += chunk_size) {
                int this_chunk = std::min(chunk_size, n_kv - chunk_start);
                // Padded row pitch for the score buffers (see chunk_ld_max).
                const int chunk_ld = (this_chunk + 7) & ~7;

                // 3a. Dequant this KV chunk to dense fp16 (once per chunk)
                {
                    MKL_TAKE_TIME(t0);
                    mkl_fa_dequant_chunk(stream, K_desc, KQV,
                        K_chunk_f16_ptr, ikvh, ib, chunk_start, this_chunk);
                    if (!V_is_K_view) {
                        mkl_fa_dequant_chunk(stream, V_desc, KQV,
                            V_chunk_f16_ptr, ikvh, ib, chunk_start, this_chunk);
                    }
                    MKL_ACCUM(dequant_time_us, t0);
                }

                // 3b. Query tile loop (INNER) — bounds KQ_f32/S_f16 footprint.
                for (int q0 = 0; q0 < n_query_rows; q0 += q_tile_rows) {
                    int q_rows = std::min(q_tile_rows, n_query_rows - q0);
                    sycl::event kq_done, vkq_done;

                    // GEMM: KQ = Q_tile × K_chunk^T
                    {
                        MKL_TAKE_TIME(t0);
                        kq_done = gemm(*stream,
                            transpose::trans, transpose::nontrans,
                            this_chunk, q_rows, DKQ,
                            alpha,
                            K_chunk_f16_ptr, DKQ,
                            Q_head_f16_ptr + (int64_t)q0 * DKQ, DKQ,
                            beta,
                            KQ_f32_ptr, chunk_ld,
                            std::vector<sycl::event>{*stream->ext_oneapi_get_last_event()});
                        MKL_ACCUM(gemm_kq_time_us, t0);
                    }
                    // Online softmax over this chunk for this query tile
                    {
                        MKL_TAKE_TIME(t0);
                        mkl_fa_online_softmax_chunk(stream,
                            KQ_f32_ptr, S_f16_ptr,
                            KQ_max_ptr, KQ_sum_ptr, VKQ_accum_ptr,
                            q0, q_rows, n_queries, DV,
                            this_chunk, chunk_ld, chunk_start,
                            kvh_base_head,
                            mask_batch, mask_head_stride,
                            mask_row_stride, mask_n_heads,
                            implicit_causal, n_past,
                            logit_softcap, wg_size, kq_done);
                        MKL_ACCUM(softmax_time_us, t0);
                    }

                    // GEMM: VKQ_chunk = S × V_chunk
                    {
                        MKL_TAKE_TIME(t0);
                        vkq_done = gemm(*stream,
                            transpose::nontrans, transpose::nontrans,
                            DV, q_rows, this_chunk,
                            alpha,
                            V_chunk_f16_ptr, DV,
                            S_f16_ptr, chunk_ld,
                            beta,
                            VKQ_chunk_ptr, DV,
                            std::vector<sycl::event>{*stream->ext_oneapi_get_last_event()});
                        MKL_ACCUM(gemm_vkq_time_us, t0);
                    }
                    // VKQ_accum[q0..] += VKQ_chunk
                    {
                        const int64_t n_total = (int64_t)q_rows * DV;
                        const int64_t wg = ((n_total + wg_size - 1) / wg_size)
                            * wg_size;
                        float * accum = VKQ_accum_ptr + (int64_t)q0 * DV;
                        stream->submit([&](sycl::handler & cgh) {
                            cgh.depends_on(vkq_done);
                            cgh.parallel_for(sycl::nd_range<1>(wg, wg_size),
                                [=](sycl::nd_item<1> item) {
                                    int64_t i = item.get_global_id(0);
                                    if (i < n_total) {
                                        accum[i] += VKQ_chunk_ptr[i];
                                    }
                                });
                        });
                    }
                }
            }

            // 4. Normalize and scatter each GQA head to dst
            for (int iqg = 0; iqg < gqa_ratio; iqg++) {
                int     iqh        = kvh_base_head + iqg;
                int64_t src_offset = (int64_t)iqg * n_queries * DV;
                mkl_fa_normalize_head(stream,
                    dst_batch, VKQ_accum_ptr, KQ_sum_ptr,
                    iqh, n_queries, DV, n_q_heads,
                    src_offset, wg_size);
            }
        }
    }

#undef MKL_TAKE_TIME
#undef MKL_ACCUM

    if (do_print) {
        const int64_t v_chunk_elems = V_is_K_view ? 0 : (int64_t)chunk_size * DV;
        double total_mb = (double)(
            (int64_t)q_tile_rows * chunk_size * sizeof(float)      // KQ_f32
          + (int64_t)q_tile_rows * chunk_size * sizeof(sycl::half) // S_f16
          + (int64_t)q_tile_rows * DV * sizeof(float)              // VKQ_chunk
          + (int64_t)n_query_rows * DV * sizeof(float)             // VKQ_accum
          + (int64_t)n_query_rows * sizeof(float)                  // KQ_max
          + (int64_t)n_query_rows * sizeof(float)                  // KQ_sum
          + (int64_t)n_query_rows * DKQ * sizeof(sycl::half)       // Q_head_f16
          + (int64_t)chunk_size * DKQ * sizeof(sycl::half)         // K_chunk_f16
          + v_chunk_elems * (int64_t)sizeof(sycl::half)            // V_chunk_f16
        ) / (1024.0 * 1024.0);
        GGML_LOG_INFO("[MKL-FA] #%d n_kv=%d n_q=%d q_tile=%d time_us: "
                "dequant=%lld GEMM_KQ=%lld softmax=%lld GEMM_VKQ=%lld "
                "buf_mb=%.1f\n",
                mkl_call_count, n_kv, n_queries, q_tile_rows,
                (long long)dequant_time_us,
                (long long)gemm_kq_time_us,
                (long long)softmax_time_us,
                (long long)gemm_vkq_time_us,
                total_mb);
    }
}

#include "ssm_conv.hpp"
#include "common.hpp"

#include <cstdio>

using namespace sycl;

// DC is d_conv as a compile-time constant, or 0 to keep the runtime loop. The window
// loop reads DC *contiguous* floats per lane, which is one wide load -- but only if the
// trip count is known, and d_conv arrives as a runtime argument. Every model that reaches
// this kernel so far uses 4.
template <int DC, bool CHANNEL_FASTEST>
static void kernel_ssm_conv_impl(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs __attribute__((unused)),
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    bool apply_silu
) {
    const size_t total_work = static_cast<size_t>(d_inner) * static_cast<size_t>(n_t) * static_cast<size_t>(n_s);
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    const range<1> global_range(num_work_groups * work_group_size);
    const range<1> local_range(work_group_size);

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(global_range, local_range),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }

                // src has the tokens of one channel contiguous, dst has the channels of one
                // token contiguous, so either the loads or the store must be strided. Indexing
                // token-fastest coalesces the d_conv loads, which measured faster except for
                // short, cache-resident rows.
                // CHANNEL_FASTEST puts consecutive lanes on consecutive channels of one
                // token, so the single store per lane is contiguous. The window loads turn
                // strided instead -- but a strided *load* of DC contiguous floats touches one
                // line per lane (16 B of 64 B used, 4x), while the strided store touches one
                // line per lane for 4 B (16x), and only the store side pays read-for-ownership.
                int token;
                int channel;
                if constexpr (CHANNEL_FASTEST) {
                    channel = static_cast<int>(idx % d_inner);
                    token   = static_cast<int>((idx / d_inner) % n_t);
                } else {
                    token   = static_cast<int>(idx % n_t);
                    channel = static_cast<int>((idx / n_t) % d_inner);
                }
                const int seq     = static_cast<int>(idx / (static_cast<size_t>(n_t) * static_cast<size_t>(d_inner)));

                const float *s = src_data
                    + static_cast<size_t>(seq) * static_cast<size_t>(src_stride_seq)
                    + static_cast<size_t>(channel) * static_cast<size_t>(src_stride_inner)
                    + static_cast<size_t>(token);

                const float *c = weights + static_cast<size_t>(channel) * static_cast<size_t>(d_conv);

                float sumf = 0.0f;
                if constexpr (DC > 0) {
#pragma unroll
                    for (int i0 = 0; i0 < DC; ++i0) {
                        sumf += s[i0] * c[i0];
                    }
                } else {
                    for (int i0 = 0; i0 < d_conv; ++i0) {
                        sumf += s[i0] * c[i0];
                    }
                }

                const size_t dst_idx =
                    static_cast<size_t>(seq) * static_cast<size_t>(dst_stride_seq) +
                    static_cast<size_t>(token) * static_cast<size_t>(dst_stride_token) +
                    static_cast<size_t>(channel);

                // fused SiLU epilogue (matches element_wise op_silu: x / (1 + exp(-x)))
                dst_data[dst_idx] = apply_silu ? sumf / (1.0f + sycl::exp(-sumf)) : sumf;
            }
        );
    });
}

// MAP=2: SLM transpose tiling. The two mappings above each coalesce one side and pay
// line amplification on the other; measured they land within 4.8% of each other, which
// says neither side's coalescing is the binding constraint -- both pay for the strided
// half. This tile coalesces BOTH: a work-group owns TTxTC outputs, reads token-fastest
// (contiguous windows), transposes through SLM, and stores channel-fastest (contiguous
// rows of dst). The +1 pad makes the phase-2 stride 33, coprime with 32 banks.
template <int DC>
static void kernel_ssm_conv_tiled(
    queue &q, const float *src_data, const float *weights, float *dst_data,
    int d_inner, int n_t, int n_s, int src_stride_inner, int src_stride_seq,
    int dst_stride_token, int dst_stride_seq, bool apply_silu
) {
    constexpr int TT = 32, TC = 32, WG = 256;
    const int nt_tiles = (n_t + TT - 1) / TT;
    const int nc_tiles = d_inner / TC;
    const size_t groups = static_cast<size_t>(nt_tiles) * nc_tiles * n_s;

    q.submit([&](handler &h) {
        local_accessor<float, 1> tile(range<1>(TC * (TT + 1)), h);
        h.parallel_for(nd_range<1>(range<1>(groups * WG), range<1>(WG)), [=](nd_item<1> it) {
            const int    lid = static_cast<int>(it.get_local_id(0));
            const size_t g   = it.get_group(0);
            const int    tt  = static_cast<int>(g % nt_tiles);
            const int    ct  = static_cast<int>((g / nt_tiles) % nc_tiles);
            const int    seq = static_cast<int>(g / (static_cast<size_t>(nt_tiles) * nc_tiles));
            const int    t0 = tt * TT, c0 = ct * TC;

            const int ti = lid % TT;
            const int cj = lid / TT;
#pragma unroll
            for (int r = 0; r < TC / (WG / TT); ++r) {
                const int c   = cj + r * (WG / TT);
                const int tok = t0 + ti;
                float sumf = 0.0f;
                if (tok < n_t) {
                    const float *s = src_data + static_cast<size_t>(seq) * src_stride_seq
                                   + static_cast<size_t>(c0 + c) * src_stride_inner + tok;
                    const float *cw = weights + static_cast<size_t>(c0 + c) * DC;
#pragma unroll
                    for (int i = 0; i < DC; ++i) sumf += s[i] * cw[i];
                    if (apply_silu) sumf = sumf / (1.0f + sycl::exp(-sumf));
                }
                tile[c * (TT + 1) + ti] = sumf;
            }
            it.barrier(access::fence_space::local_space);

            const int cc = lid % TC;
            const int tj = lid / TC;
#pragma unroll
            for (int r = 0; r < TT / (WG / TC); ++r) {
                const int t   = tj + r * (WG / TC);
                const int tok = t0 + t;
                if (tok < n_t) {
                    dst_data[static_cast<size_t>(seq) * dst_stride_seq
                             + static_cast<size_t>(tok) * dst_stride_token + c0 + cc]
                        = tile[cc * (TT + 1) + t];
                }
            }
        });
    });
}

// GGML_SYCL_SSM_CONV_UNROLL=0 keeps the runtime-trip-count loop, for A/B.
static void kernel_ssm_conv(
    queue &q,
    const float *src_data,
    const float *weights,
    float *dst_data,
    int d_conv,
    int d_inner,
    int n_t,
    int n_s,
    int ncs,
    int src_stride_inner,
    int src_stride_seq,
    int dst_stride_token,
    int dst_stride_seq,
    bool apply_silu
) {
    static const bool unroll = ggml_sycl_get_env("GGML_SYCL_SSM_CONV_UNROLL", 1) != 0;
    // GGML_SYCL_SSM_CONV_MAP=0 restores the token-fastest mapping, for A/B.
    static const int  map_mode     = ggml_sycl_get_env("GGML_SYCL_SSM_CONV_MAP", 2);
    static const bool chan_fastest = map_mode == 1;

    if (map_mode == 2 && d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0) {
        kernel_ssm_conv_tiled<4>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                 src_stride_inner, src_stride_seq, dst_stride_token,
                                 dst_stride_seq, apply_silu);
        return;
    }

#define GGML_SYCL_SSM_CONV_CALL(DC_, MAP_)                                                        \
    kernel_ssm_conv_impl<DC_, MAP_>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,    \
                                    ncs, src_stride_inner, src_stride_seq, dst_stride_token,      \
                                    dst_stride_seq, apply_silu)

    if (unroll && d_conv == 4) {
        if (chan_fastest) { GGML_SYCL_SSM_CONV_CALL(4, true); } else { GGML_SYCL_SSM_CONV_CALL(4, false); }
        return;
    }
    if (chan_fastest) { GGML_SYCL_SSM_CONV_CALL(0, true); } else { GGML_SYCL_SSM_CONV_CALL(0, false); }
#undef GGML_SYCL_SSM_CONV_CALL
}

inline void ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst = nullptr) {
    ggml_tensor * src0 = dst->src[0];
    ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int d_conv   = src1->ne[0];
    const int ncs      = src0->ne[0];
    const int d_inner  = src0->ne[1];
    const int n_t      = dst->ne[1];
    const int n_s      = dst->ne[2];

    GGML_ASSERT(src0->ne[0] == d_conv - 1 + n_t);
    GGML_ASSERT(src0->ne[1] == d_inner);
    GGML_ASSERT(src1->ne[1] == d_inner);

    GGML_ASSERT(dst->ne[0] == d_inner);
    GGML_ASSERT(dst->ne[1] == n_t);
    GGML_ASSERT(dst->ne[2] == n_s);

    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));

    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const int src_stride_inner = ncs;
    const int src_stride_seq   = ncs * d_inner;
    const int dst_stride_token = d_inner;
    const int dst_stride_seq   = d_inner * n_t;

    try {
        queue *q = ctx.stream();

        const float *src_data = static_cast<const float *>(src0->data);
        const float *weights  = static_cast<const float *>(src1->data);
        const bool   apply_silu = silu_dst != nullptr;
        float *dst_data       = static_cast<float *>((silu_dst ? silu_dst : dst)->data);

        GGML_ASSERT(src_data && weights && dst_data);

        kernel_ssm_conv(
            *q,
            src_data,
            weights,
            dst_data,
            d_conv,
            d_inner,
            n_t,
            n_s,
            ncs,
            src_stride_inner,
            src_stride_seq,
            dst_stride_token,
            dst_stride_seq,
            apply_silu
        );

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

void ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_ssm_conv(ctx, dst);
}

// Fused ssm_conv + SiLU: write silu(conv) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
void ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    ggml_sycl_op_ssm_conv(ctx, dst, silu_dst);
}

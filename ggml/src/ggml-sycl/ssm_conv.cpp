#include "ssm_conv.hpp"
#include <algorithm>
#include <vector>
#include <cstdlib>
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
    bool apply_silu,
    float *state_out,
    int state_stride_seq
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

                // Fused state writeback. The next step's conv window is this row's last
                // d_conv-1 values, which is exactly s[1..d_conv-1] of the LAST token's
                // window -- already in registers, so this costs d_conv-1 stores and no
                // extra loads. It replaces a whole CPY dispatch of the same bytes.
                if (state_out != nullptr && token == n_t - 1) {
                    float *so = state_out
                        + static_cast<size_t>(seq) * static_cast<size_t>(state_stride_seq)
                        + static_cast<size_t>(channel) * static_cast<size_t>(d_conv - 1);
                    if constexpr (DC > 0) {
#pragma unroll
                        for (int j = 0; j < DC - 1; ++j) {
                            so[j] = s[j + 1];
                        }
                    } else {
                        for (int j = 0; j < d_conv - 1; ++j) {
                            so[j] = s[j + 1];
                        }
                    }
                }
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

// Whether the tiled (prefill) form would be chosen. The writeback is implemented only in
// the scalar form, so the detector asks this rather than restating the condition -- a
// detector that disagreed with the dispatch would either lose the tiled prefill kernel or
// skip a CPY nobody then performs.
bool ggml_sycl_ssm_conv_prefers_tiled(int d_conv, int d_inner, int n_t) {
    static const int map_mode = ggml_sycl_get_env("GGML_SYCL_SSM_CONV_MAP", 2);
    return map_mode == 2 && d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0;
}

// Collapsed causal-conv at a decode width of one token. The CONCAT that materialises
// [state | x] and the CPY that writes the next state both disappear: this kernel reads the
// cache row and the projection directly, convolves, and stores the shifted state.
//
// Mapping is one work-item per CHANNEL, not per (channel, seq), and that is the whole
// safety argument. Merging the cache read and the state write into one dispatch creates a
// hazard the separate CONCAT and CPY never had: the read row is rows[seq] and the write row
// is the destination slot, both runtime values, so with a per-(channel, seq) mapping
// work-item (c, s0) could write the very slice (c, s1) is reading. Per channel, every
// address either kernel touches is inside [c*(DC-1), c*DC), owned by exactly one work-item
// -- and the two phases below keep all reads ahead of all writes within it. No barrier
// exists that could have ordered the other mapping.
template <int DC, int MAX_NS>
static void kernel_ssm_conv_collapsed_t1(
    queue &q,
    const float *cache, const int32_t *rows, int cache_stride_seq,
    const float *x, int x_stride_inner, int x_stride_seq,
    const float *weights,
    float *dst_data, int dst_stride_seq,
    float *state_out, int state_stride_seq,
    int d_inner, int n_s, bool apply_silu
) {
    const size_t work_group_size = 256;
    const size_t total_work      = static_cast<size_t>(d_inner);
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(range<1>(num_work_groups * work_group_size), range<1>(work_group_size)),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }
                const int channel = static_cast<int>(idx);

                const float *c = weights + static_cast<size_t>(channel) * DC;

                // phase 1: every load, before any store
                float win[MAX_NS][DC];
                for (int s = 0; s < n_s; ++s) {
                    const float *st = cache
                        + static_cast<size_t>(rows[s]) * static_cast<size_t>(cache_stride_seq)
                        + static_cast<size_t>(channel) * (DC - 1);
#pragma unroll
                    for (int k = 0; k < DC - 1; ++k) {
                        win[s][k] = st[k];
                    }
                    win[s][DC - 1] = x[static_cast<size_t>(channel) * static_cast<size_t>(x_stride_inner)
                                       + static_cast<size_t>(s) * static_cast<size_t>(x_stride_seq)];
                }

                // phase 2: every store
                for (int s = 0; s < n_s; ++s) {
                    float sumf = 0.0f;
#pragma unroll
                    for (int i = 0; i < DC; ++i) {
                        sumf += win[s][i] * c[i];
                    }
                    dst_data[static_cast<size_t>(s) * static_cast<size_t>(dst_stride_seq) + channel] =
                        apply_silu ? sumf / (1.0f + sycl::exp(-sumf)) : sumf;

                    float *so = state_out
                        + static_cast<size_t>(s) * static_cast<size_t>(state_stride_seq)
                        + static_cast<size_t>(channel) * (DC - 1);
#pragma unroll
                    for (int k = 0; k < DC - 1; ++k) {
                        so[k] = win[s][k + 1];
                    }
                }
            }
        );
    });
}

// GGML_SYCL_SSM_CONV_UNROLL=0 keeps the runtime-trip-count loop, for A/B.
static bool kernel_ssm_conv(
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
    bool apply_silu,
    float *state_out,
    int state_stride_seq
) {
    static const bool unroll = ggml_sycl_get_env("GGML_SYCL_SSM_CONV_UNROLL", 1) != 0;
    // GGML_SYCL_SSM_CONV_MAP=0 restores the token-fastest mapping, for A/B.
    static const int  map_mode     = ggml_sycl_get_env("GGML_SYCL_SSM_CONV_MAP", 2);
    static const bool chan_fastest = map_mode == 1;

    // The tiled form is prefill-only (n_t >= 32) and does not implement the writeback;
    // there the CPY is once per layer per ubatch rather than per token, so it is not worth
    // a second code path. Reporting false leaves the caller to run the CPY normally.
    if (map_mode == 2 && d_conv == 4 && n_t >= 32 && (d_inner % 32) == 0 && state_out == nullptr) {
        kernel_ssm_conv_tiled<4>(q, src_data, weights, dst_data, d_inner, n_t, n_s,
                                 src_stride_inner, src_stride_seq, dst_stride_token,
                                 dst_stride_seq, apply_silu);
        return false;
    }

#define GGML_SYCL_SSM_CONV_CALL(DC_, MAP_)                                                        \
    kernel_ssm_conv_impl<DC_, MAP_>(q, src_data, weights, dst_data, d_conv, d_inner, n_t, n_s,    \
                                    ncs, src_stride_inner, src_stride_seq, dst_stride_token,      \
                                    dst_stride_seq, apply_silu, state_out, state_stride_seq)

    if (unroll && d_conv == 4) {
        if (chan_fastest) { GGML_SYCL_SSM_CONV_CALL(4, true); } else { GGML_SYCL_SSM_CONV_CALL(4, false); }
        return state_out != nullptr;
    }
    if (chan_fastest) { GGML_SYCL_SSM_CONV_CALL(0, true); } else { GGML_SYCL_SSM_CONV_CALL(0, false); }
#undef GGML_SYCL_SSM_CONV_CALL
    return state_out != nullptr;
}

// GGML_SYCL_CONV_COLLAPSE=0 disables the collapsed one-token conv (same-binary control).
bool ggml_sycl_conv_collapse_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_SYCL_CONV_COLLAPSE");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

// GGML_SYCL_CONV_COLLAPSE_DIAG=1 leaves the CONCAT in the graph and byte-compares the two
// sources this kernel reads against the buffer the CONCAT materialised.
bool ggml_sycl_conv_collapse_diag() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_SYCL_CONV_COLLAPSE_DIAG");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

// GGML_SYCL_CONV_WB=0 disables the fused conv-state writeback (same-binary control).
bool ggml_sycl_conv_wb_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_SYCL_CONV_WB");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
}

// GGML_SYCL_CONV_WB_DIAG=1 byte-compares what the fused writeback stored against the
// bytes the CPY it replaces would have copied. Generation output cannot gate a state
// write on this box -- see the nondeterminism control recorded for the gather fold --
// and test-backend-ops cannot see a graph-level fusion at all.
bool ggml_sycl_conv_wb_diag() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_SYCL_CONV_WB_DIAG");
        return e != nullptr && std::atoi(e) != 0;
    }();
    return on;
}

// src0 is [d_conv-1+n_t, d_inner, n_s] contiguous; the state the CPY would write is its
// last d_conv-1 columns, laid out flat as channel*(d_conv-1) + k per sequence row.
static void ssm_conv_wb_verify(ggml_backend_sycl_context & ctx, const ggml_tensor * src0,
                               const ggml_tensor * dstv, int d_conv, int d_inner, int n_t, int n_s) {
    queue_ptr q = ctx.stream();

    const size_t       ncs = (size_t) (d_conv - 1 + n_t);
    std::vector<float> in(ncs * d_inner * n_s);
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(in.data(), src0->data, in.size() * sizeof(float)).wait()));

    const size_t       row = (size_t) (d_conv - 1) * d_inner;
    const size_t       rs  = dstv->nb[1] / sizeof(float);
    std::vector<float> out(rs * n_s);
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(out.data(), dstv->data, out.size() * sizeof(float)).wait()));

    long bad = 0;
    for (int sq = 0; sq < n_s; ++sq) {
        for (int c = 0; c < d_inner; ++c) {
            for (int k = 0; k < d_conv - 1; ++k) {
                const float want = in[(size_t) sq * ncs * d_inner + (size_t) c * ncs + n_t + k];
                const float got  = out[(size_t) sq * rs + (size_t) c * (d_conv - 1) + k];
                bad += (want != got);
            }
        }
    }
    fprintf(stderr, "[CONVWB] n_s=%d n_t=%d d_conv=%d elems=%zu mismatches=%ld\n", n_s, n_t, d_conv,
            row * n_s, bad);
}

static bool ggml_sycl_op_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                  ggml_tensor * silu_dst = nullptr, const ggml_tensor * wb = nullptr) {
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

        // wb is the CPY that copies src0's last d_conv-1 columns into the conv-state
        // cache. Its destination view is contiguous per sequence with nb[1] the cache's
        // row stride, and ggml_cpy's flat element order puts channel c at c*(d_conv-1).
        float * state_out        = nullptr;
        int     state_stride_seq = 0;
        if (wb != nullptr) {
            const ggml_tensor * dv = wb->src[1];
            GGML_ASSERT(dv->type == GGML_TYPE_F32 && dv->nb[0] == sizeof(float));
            GGML_ASSERT(dv->ne[0] == (int64_t) (d_conv - 1) * d_inner && dv->ne[1] == n_s);
            state_out        = (float *) dv->data;
            state_stride_seq = (int) (dv->nb[1] / sizeof(float));
        }

        const bool wrote = kernel_ssm_conv(
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
            apply_silu,
            state_out,
            state_stride_seq
        );

        // the detector rejects any shape the tiled form would claim, so a requested
        // writeback must always be taken -- the CPY has already been skipped by then
        GGML_ASSERT(wrote == (wb != nullptr));

        if (wrote && ggml_sycl_conv_wb_diag()) {
            ssm_conv_wb_verify(ctx, src0, wb->src[1], d_conv, d_inner, n_t, n_s);
        }
        return wrote;

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
}

// Proves the two-source gather reproduces the buffer the CONCAT materialises. Under the
// diag the CONCAT is left in the graph, so conv_input exists to compare against; the
// convolution itself is the unchanged arithmetic on those same values.
static void conv_collapse_verify(ggml_backend_sycl_context & ctx, const ggml_tensor * concat,
                                 const ggml_tensor * gather, const ggml_tensor * xsrc,
                                 int d_conv, int d_inner, int n_s) {
    queue_ptr q = ctx.stream();

    std::vector<int32_t> rows(n_s);
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(rows.data(), gather->src[1]->data, n_s * sizeof(int32_t)).wait()));

    const size_t       ncs = (size_t) d_conv;  // n_t == 1
    std::vector<float> ci(ncs * d_inner * n_s);
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(ci.data(), concat->data, ci.size() * sizeof(float)).wait()));

    const size_t       crow = gather->src[0]->nb[1] / sizeof(float);
    std::vector<float> cache(crow * (rows.empty() ? 0 : (*std::max_element(rows.begin(), rows.end()) + 1)));
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(cache.data(), gather->src[0]->data, cache.size() * sizeof(float)).wait()));

    const size_t       xsi = xsrc->nb[1] / sizeof(float), xss = xsrc->nb[2] / sizeof(float);
    std::vector<float> xh((size_t) d_inner * xsi + (size_t) n_s * xss + 1);
    SYCL_CHECK(CHECK_TRY_ERROR(q->memcpy(xh.data(), xsrc->data, xh.size() * sizeof(float)).wait()));

    long bad_state = 0, bad_x = 0;
    for (int sq = 0; sq < n_s; ++sq) {
        for (int c = 0; c < d_inner; ++c) {
            for (int k = 0; k < d_conv - 1; ++k) {
                bad_state += (cache[(size_t) rows[sq] * crow + (size_t) c * (d_conv - 1) + k] !=
                              ci[(size_t) sq * ncs * d_inner + (size_t) c * ncs + k]);
            }
            bad_x += (xh[(size_t) c * xsi + (size_t) sq * xss] !=
                      ci[(size_t) sq * ncs * d_inner + (size_t) c * ncs + (d_conv - 1)]);
        }
    }
    fprintf(stderr, "[CONVCOL] n_s=%d d_inner=%d state_mismatches=%ld x_mismatches=%ld\n", n_s, d_inner,
            bad_state, bad_x);
}

// One dispatch for what was GET_ROWS + CONCAT + CPY + SSM_CONV (+ SiLU). Runs at the
// CONCAT's position in the graph, so both reads happen where they already did and only the
// writes move -- see find_conv_collapse for why the mirror of that does not work.
bool ggml_sycl_ssm_conv_collapsed(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                                  const ggml_tensor * gather, const ggml_tensor * concat, const ggml_tensor * wb) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * src1 = dst->src[1];
    const int           d_conv  = (int) src1->ne[0];
    const int           d_inner = (int) dst->ne[0];
    const int           n_t     = (int) dst->ne[1];
    const int           n_s     = (int) dst->ne[2];

    // the detector admits exactly this; anything else must not have skipped the CONCAT
    GGML_ASSERT(n_t == 1 && d_conv == 4 && n_s >= 1 && n_s <= 4);

    const ggml_tensor * xsrc = concat->src[1];
    const ggml_tensor * cch  = gather->src[0];
    const ggml_tensor * dv   = wb->src[1];

    if (ggml_sycl_conv_collapse_diag()) {
        conv_collapse_verify(ctx, concat, gather, xsrc, d_conv, d_inner, n_s);
    }

    // a fused SiLU is written in place of the conv's own output, never alongside it
    const ggml_tensor * outT = silu_dst != nullptr ? silu_dst : dst;
    kernel_ssm_conv_collapsed_t1<4, 4>(
        *ctx.stream(),
        (const float *) cch->data, (const int32_t *) gather->src[1]->data, (int) (cch->nb[1] / sizeof(float)),
        (const float *) xsrc->data, (int) (xsrc->nb[1] / sizeof(float)), (int) (xsrc->nb[2] / sizeof(float)),
        (const float *) src1->data,
        (float *) outT->data, (int) (outT->nb[2] / sizeof(float)),
        (float *) dv->data, (int) (dv->nb[1] / sizeof(float)),
        d_inner, n_s, silu_dst != nullptr);
    return true;
}

bool ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * wb) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    return ggml_sycl_op_ssm_conv(ctx, dst, nullptr, wb);
}

// Fused ssm_conv + SiLU: write silu(conv) straight into silu_dst, eliding the
// standalone SiLU launch and its HBM round-trip of the conv output.
bool ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                              const ggml_tensor * wb) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    GGML_ASSERT(silu_dst && ggml_are_same_shape(dst, silu_dst) && silu_dst->type == GGML_TYPE_F32);
    return ggml_sycl_op_ssm_conv(ctx, dst, silu_dst, wb);
}

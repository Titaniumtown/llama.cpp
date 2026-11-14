#include "ssm_conv.hpp"
#include <cstdlib>
#include "common.hpp"
#include "element_wise.hpp"
#include "fusion.hpp"

#include <algorithm>
#include <climits>
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

// Prefill fold: one dispatch for CONCAT + K state CPYs + SSM_CONV + SiLU.
//
// The concat this replaces spends its whole 545 us/call materialising
// ggml_transpose(qkv_mixed): the source has channels of one token contiguous, the
// concat wants tokens of one channel contiguous. This kernel never builds that
// layout. It maps consecutive lanes to consecutive CHANNELS of one token, so the
// x reads (x is addressed through its own strides, i.e. the *untransposed* buffer)
// and the dst store are both contiguous -- the coalescing the tiled kernel buys
// with an SLM transpose falls out of reading the source in its native layout.
// Window positions p < DC-1 come from the gathered state instead; only the first
// DC-1 output tokens ever touch it.
//
// The state writeback rides the same registers as the single-slot absorb above:
// the CPY for snapshot slot k copies concat columns [ne0-(DC-1)-k, ..), which are
// exactly window[1..DC-1] of the work-item computing token n_t-1-k. wb_stride is
// the (finder-validated) uniform byte distance between consecutive slots' rows.
// Accumulation order and the SiLU expression are verbatim kernel_ssm_conv_impl,
// so the fold is bit-identical to the unfused chain.
template <int DC>
static void kernel_ssm_conv_prefill_fold(
    queue &q,
    const float *state, int st_s2,                 // state (p, c, s): p + c*(DC-1) + s*st_s2
    const float *x, int64_t x_s0, int64_t x_s1, int64_t x_s2,   // x (t, c, s), float strides
    const float *weights,
    float *dst_data, int64_t dst_s1, int64_t dst_s2,            // dst (c, t, s), float strides
    float *wb_base, int64_t wb_stride,             // slot 0 row; float stride between slots
    int64_t wb_s2,                                 // float stride between sequences of one slot
    int wb_count,
    int d_inner, int n_t, int n_s, bool apply_silu
) {
    // One work-item per (channel, token-chunk, seq), walking its chunk serially with
    // the DC-1 window tail in registers: every x element is loaded exactly once, and
    // both the loads and the stores are contiguous across lanes (consecutive channels
    // of one token). The first form of this kernel mapped one WI per output and paid
    // the window overlap as 4x strided re-reads 40 KB apart -- 84 MB x 4 passes
    // through an 18 MB L2 is DRAM, and it measured 917 us/call against the unfused
    // chain's 955 (CONCAT 541 + conv 414): the concat's token-fastest layout hands
    // the conv all four window taps in one cache line, so beating the pair needs the
    // 1x-read walk, not just transpose avoidance.
    // Chunking exists only for occupancy: C*S alone is ~10k lanes on a 2.5k-lane
    // device; chunks multiply that without changing any read count beyond the DC-1
    // halo per chunk seam. n_t/256 was exactly backwards: it gave the LONGEST serial
    // walks (fewest lanes) at the widest prefill -- chunks=2, 80 work-groups, and the
    // x stride (one token = 40 KiB, page-aligned, which aliases on this part: +64 B
    // on the same stride doubles achievable bandwidth) left the kernel latency-bound
    // at 287 GB/s. Standalone sweep at the production shape (d_inner 10240, n_t 512,
    // includes the writeback, bit-identical output across all rows):
    //   chunks   2 (shipped)  148.5 us   288 GB/s
    //   chunks  16            117.8 us   363 GB/s
    //   chunks  32             78.3 us   545 GB/s
    //   chunks  64             77.4 us   552 GB/s   <- plateau == the aliased-stride
    //   chunks 128             77.0 us   555 GB/s      ceiling this read can reach
    //   (wg 512 and vec4-per-lane variants: no better than their scalar rows)
    // At n_t=64 the old formula's chunks=1 ran 42.6 us; n_t/8 -> chunks=8 runs 18.4.
    // chunk_len >= 8 keeps the halo re-read <= 3/8 of x; it is invisible next to the
    // occupancy cliff (c64 >= c32 above despite +18% reads).
    int chunks = n_t / 8;
    if (chunks < 1) { chunks = 1; }
    if (chunks > 64) { chunks = 64; }
    const int chunk_len = (n_t + chunks - 1) / chunks;

    const size_t total_work = (size_t) d_inner * (size_t) chunks * (size_t) n_s;
    const size_t work_group_size = 256;
    const size_t num_work_groups = (total_work + work_group_size - 1) / work_group_size;

    q.submit([&](handler &h) {
        h.parallel_for(
            nd_range<1>(range<1>(num_work_groups * work_group_size), range<1>(work_group_size)),
            [=](nd_item<1> item) {
                const size_t idx = item.get_global_id(0);
                if (idx >= total_work) {
                    return;
                }
                const int channel = (int) (idx % (size_t) d_inner);
                const int chunk   = (int) ((idx / (size_t) d_inner) % (size_t) chunks);
                const int seq     = (int) (idx / ((size_t) d_inner * (size_t) chunks));

                const int t0 = chunk * chunk_len;
                int       t1 = t0 + chunk_len;
                if (t1 > n_t) { t1 = n_t; }
                if (t0 >= t1) { return; }

                const float *c = weights + (size_t) channel * DC;
                float w0 = c[0], w1 = c[1], w2 = c[2], w3 = c[3];
                static_assert(DC == 4, "sliding window is written for DC == 4");

                // in(p) for concat position p, without the concat
                const float *xb = x + (int64_t) channel * x_s1 + (int64_t) seq * x_s2;
                const float *sb = state + (size_t) channel * (DC - 1) + (size_t) seq * st_s2;
                auto in_at = [&](int p) -> float {
                    return p < DC - 1 ? sb[p] : xb[(int64_t) (p - (DC - 1)) * x_s0];
                };

                // prime the window for output t0: positions t0, t0+1, t0+2
                float a0 = in_at(t0);
                float a1 = in_at(t0 + 1);
                float a2 = in_at(t0 + 2);

                float *db = dst_data + (size_t) channel + (int64_t) seq * dst_s2;

                for (int t = t0; t < t1; ++t) {
                    const float a3 = xb[(int64_t) t * x_s0];   // position t+3 is always x
                    // accumulation order matches kernel_ssm_conv_impl: j ascending
                    float sumf = a0 * w0;
                    sumf += a1 * w1;
                    sumf += a2 * w2;
                    sumf += a3 * w3;
                    db[(int64_t) t * dst_s1] = apply_silu ? sumf / (1.0f + sycl::exp(-sumf)) : sumf;

                    // absorbed snapshot CPYs: token n_t-1-k owns slot k's bytes,
                    // which are window positions t+1..t+3 = [a1, a2, a3]
                    const int k = (n_t - 1) - t;
                    if (wb_base != nullptr && k < wb_count) {
                        float *so = wb_base + (int64_t) k * wb_stride
                            + (int64_t) seq * wb_s2
                            + (size_t) channel * (DC - 1);
                        so[0] = a1;
                        so[1] = a2;
                        so[2] = a3;
                    }

                    a0 = a1;
                    a1 = a2;
                    a2 = a3;
                }
            }
        );
    });
}

// GGML_SYCL_CONV_PREFILL_FOLD=0 disables the prefill conv fold (same-binary control).
bool ggml_sycl_conv_prefill_fold_enabled() {
    static const int v = [] {
        const char * e = std::getenv("GGML_SYCL_CONV_PREFILL_FOLD");
        return e == nullptr ? 1 : std::atoi(e);
    }();
    return v != 0;
}

// One dispatch for what was CONCAT + K snapshot CPYs + SSM_CONV + SiLU at prefill
// widths. Runs at the CONCAT's position; find_conv_prefill_fold proves every
// consumer of the concat is inside this kernel before it is allowed to run.
void ggml_sycl_ssm_conv_prefill_fold(ggml_backend_sycl_context & ctx, const ggml_tensor * cat,
                                     const ggml_tensor * conv, ggml_tensor * silu,
                                     float * wb_base, int64_t wb_stride, int64_t wb_s2, int wb_count) {
    const ggml_tensor * st = cat->src[0];
    const ggml_tensor * x  = cat->src[1];
    const ggml_tensor * w  = conv->src[1];

    const int d_conv  = (int) w->ne[0];
    const int d_inner = (int) conv->ne[0];
    const int n_t     = (int) conv->ne[1];
    const int n_s     = (int) conv->ne[2];
    GGML_ASSERT(d_conv == 4);

    dpct::queue_ptr stream = ctx.stream();
    kernel_ssm_conv_prefill_fold<4>(
        *stream,
        (const float *) st->data, (int) (st->nb[2] / sizeof(float)),
        (const float *) x->data,
        (int64_t) (x->nb[0] / sizeof(float)), (int64_t) (x->nb[1] / sizeof(float)), (int64_t) (x->nb[2] / sizeof(float)),
        (const float *) w->data,
        (float *) silu->data,
        (int64_t) (silu->nb[1] / sizeof(float)), (int64_t) (silu->nb[2] / sizeof(float)),
        wb_base, wb_stride, wb_s2, wb_count,
        d_inner, n_t, n_s, /*apply_silu=*/true);
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

// GGML_SYCL_CONV_WB=0 disables the fused conv-state writeback (same-binary control).
//
// This one looks like dead code and is not. Three mechanisms now cover the conv chain
// and they partition the shapes rather than overlapping, so each looks unreachable from
// the others' regime. Counted by their own diags, one llama-bench run each:
//
//   shape          [CONVFOLD]   [CONVCOL]   [CONVWB]   what covers it
//   -p 2048          96 rej         0          0       the prefill fold
//   -p 4 (verify)    96 + 1 acc     0         96       THIS, and nothing else
//   -n 8 (decode)   432            432         0       the chain collapse
//
// So the writeback absorb is live in exactly the MTP-verify regime, where neither the
// prefill fold (which suppresses this absorb when it fires) nor the decode collapse
// applies. Its measured worth there is small and under the harness floor -- pp4 d16384
// 93.75 / 93.58 default against 93.44 / 93.42 with it off, on +-1.4 error bars -- so
// the honest bound is "fires 96 times per verify pass, worth at most 0.3%".
//
// Recorded because the tempting move is to delete it: it is invisible at prefill and at
// decode, which are the two shapes anyone screens first. For scale, the same sweep
// prices its neighbours at 2.0% (prefill fold, pp2048), 1.0% (gather fold, tg64) and
// 0.7% (chain collapse, tg64).
bool ggml_sycl_conv_wb_enabled() {
    static const bool on = [] {
        const char * e = std::getenv("GGML_SYCL_CONV_WB");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return on;
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

    // The three strides above are dense arithmetic, not reads of nb[]. The
    // asserts above pin src0's first two dims but say nothing about dim 2,
    // nor about src1's row stride, nor about dst at all -- so a tensor
    // strided in any of those is read as if contiguous. The CPU reference
    // has no such gap because it reads the real strides instead of deriving
    // them (ggml-cpu/ops.cpp:9735-9737). The fused SYCL paths in this same
    // file already read nb[2] from every tensor they touch (lines 425, 427,
    // 430, 681, 728); this is the one that does not.
    GGML_ASSERT(src0->nb[2] == src0->ne[0] * src0->ne[1] * sizeof(float));
    GGML_ASSERT(src1->nb[1] == src1->ne[0] * sizeof(float));
    GGML_ASSERT(dst->nb[0]  == sizeof(float));
    GGML_ASSERT(dst->nb[1]  == dst->ne[0]  * sizeof(float));
    GGML_ASSERT(dst->nb[2]  == dst->ne[0]  * dst->ne[1] * sizeof(float));

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

        return wrote;

    } catch (const std::exception &e) {
        std::fprintf(stderr, "[SYCL-SSM_CONV] ERROR: %s\n", e.what());
        throw;
    }
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

static const ggml_tensor * conv_window_base(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

static bool conv_window_f32_strides(const ggml_tensor * t) {
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (t->nb[d] % sizeof(float) != 0) {
            return false;
        }
    }
    return true;
}

bool ggml_sycl_match_conv_window(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_conv_window_match & m) {
    if (!g_ggml_sycl_fuse_conv_window || !g_ggml_sycl_enable_fusion) {
        return false;
    }

    const ggml_tensor * concat = cgraph->nodes[node_idx];
    if (concat->op != GGML_OP_CONCAT || ggml_get_op_params_i32(concat, 0) != 0 || concat->type != GGML_TYPE_F32 ||
        (concat->flags & GGML_TENSOR_FLAG_OUTPUT) || concat->ne[3] != 1 || !ggml_is_contiguous(concat)) {
        return false;
    }

    const ggml_tensor * state = concat->src[0];
    const ggml_tensor * xt    = concat->src[1];
    if (state->type != GGML_TYPE_F32 || xt->type != GGML_TYPE_F32 || !conv_window_f32_strides(state) ||
        !conv_window_f32_strides(xt)) {
        return false;
    }

    // the kernel is written for d_conv == 4: three state columns
    const int64_t state_cols = state->ne[0];
    const int64_t channels   = concat->ne[1];
    const int64_t n_seqs     = concat->ne[2];
    if (state_cols != 3 || channels <= 0 || channels > INT_MAX - 255 ||
        xt->ne[0] <= 0 || xt->ne[0] > INT_MAX - 7 || n_seqs <= 0 || n_seqs > INT_MAX) {
        return false;
    }

    ggml_sycl_conv_window_match r;
    r.concat = concat;

    const ggml_tensor * cache   = nullptr;
    const ggml_tensor * pending = nullptr;  // a cont still waiting for its cpy
    int                 n_views = 0;

    // Check intervening nodes after the cache is known.
    const int end = std::min(cgraph->n_nodes, node_idx + 64);
    for (int j = node_idx + 1; j < end && r.conv == nullptr; ++j) {
        ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_VIEW && n->src[0] == concat) {
            if ((n->flags & GGML_TENSOR_FLAG_OUTPUT) || n->ne[0] != state_cols || n->ne[1] != channels || n->ne[2] != n_seqs || n->ne[3] != 1 ||
                n->nb[1] != concat->nb[1] || n->nb[2] != concat->nb[2] || n->view_offs % sizeof(float) != 0 ||
                ggml_node_get_use_count(cgraph, j) != 1) {
                return false;
            }
            n_views++;
        } else if (n->op == GGML_OP_CONT && n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == concat) {
            if (pending != nullptr || r.n_slots == GGML_SYCL_CONV_WINDOW_MAX_SLOTS ||
                !ggml_node_has_n_uses(cgraph, j, 1)) {
                return false;
            }
            pending = n;
        } else if (n->op == GGML_OP_CPY && ((pending != nullptr && n->src[0] == pending) ||
                   (pending == nullptr && n->src[0]->op == GGML_OP_VIEW && n->src[0]->src[0] == concat))) {
            if (r.n_slots == GGML_SYCL_CONV_WINDOW_MAX_SLOTS) {
                return false;
            }
            // dst is a [state_cols * channels, n_seqs] view of the cache rows
            const ggml_tensor * dst = n->src[1];
            if (dst->type != GGML_TYPE_F32 || dst->ne[0] != state_cols * channels || dst->ne[1] != n_seqs ||
                dst->ne[2] != 1 || dst->ne[3] != 1 || dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[1] < dst->ne[0] * sizeof(float)) {
                return false;
            }
            if (cache == nullptr) {
                cache = conv_window_base(dst);
            }
            // one seq stride for all slots, and the tail must stay inside one channel row
            const ggml_tensor * tail = pending != nullptr ? pending->src[0] : n->src[0];
            const int64_t s_idx = tail->view_offs / sizeof(float);
            if (conv_window_base(dst) != cache || (r.n_slots > 0 && dst->nb[1] != r.cpy[0]->src[1]->nb[1]) ||
                s_idx + state_cols > concat->ne[0]) {
                return false;
            }
            for (int p = 0; p < r.n_slots; ++p) {
                const ggml_tensor * prev = r.cpy[p]->src[1];
                if (dst->view_offs >= prev->view_offs ? dst->view_offs - prev->view_offs < ggml_nbytes(prev) :
                                                      prev->view_offs - dst->view_offs < ggml_nbytes(dst)) {
                    return false;
                }
            }
            r.cont[r.n_slots] = pending;
            r.cpy[r.n_slots]  = n;
            r.n_slots++;
            pending = nullptr;
        } else if (n->op == GGML_OP_SSM_CONV && n->src[0] == concat) {
            r.conv     = n;
            r.conv_idx = j;
        }
    }

    if (r.conv == nullptr || r.n_slots == 0 || pending != nullptr || n_views != r.n_slots ||
        ggml_node_get_use_count(cgraph, node_idx) != n_views + 1) {
        return false;
    }

    const ggml_tensor * weights = r.conv->src[1];
    if (weights->type != GGML_TYPE_F32 || weights->ne[0] != state_cols + 1 || !ggml_is_contiguous(weights) ||
        r.conv->type != GGML_TYPE_F32 || conv_window_base(weights) == cache) {
        return false;
    }

    // the window reads state and x, the kernel writes the cache: they must not overlap
    const ggml_tensor * state_base = conv_window_base(state);
    const ggml_tensor * x_base     = conv_window_base(xt);
    if (state_base == cache || x_base == cache) {
        return false;
    }

    // Delayed cache copies must not cross a reader or an input write.
    for (int j = node_idx + 1; j < r.conv_idx; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || n->op == GGML_OP_VIEW || n->op == GGML_OP_RESHAPE ||
            n->op == GGML_OP_TRANSPOSE || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_NONE) {
            continue;
        }
        if (std::find(r.cont, r.cont + r.n_slots, n) != r.cont + r.n_slots ||
            std::find(r.cpy, r.cpy + r.n_slots, n) != r.cpy + r.n_slots) {
            continue;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * src = n->src[s];
            if (src == nullptr) {
                continue;
            }
            const ggml_tensor * b = conv_window_base(src);
            if (b == cache || b == concat || std::find(r.cont, r.cont + r.n_slots, b) != r.cont + r.n_slots) {
                return false;
            }
        }
        const ggml_tensor * b = conv_window_base(n);
        if (b == cache || b == state_base || b == x_base) {
            return false;
        }
    }

    if (ggml_sycl_can_fuse(cgraph, r.conv_idx, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        r.add      = cgraph->nodes[r.conv_idx + 1];
        r.silu     = cgraph->nodes[r.conv_idx + 2];
        r.last_idx = r.conv_idx + 2;
    } else if (ggml_sycl_can_fuse(cgraph, r.conv_idx, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        r.silu     = cgraph->nodes[r.conv_idx + 1];
        r.last_idx = r.conv_idx + 1;
    } else {
        return false;
    }
    if (r.silu->type != GGML_TYPE_F32 || !ggml_is_contiguous(r.silu)) {
        return false;
    }

    if (r.add != nullptr) {
        const ggml_tensor * bias = r.add->src[0] == r.conv ? r.add->src[1] : r.add->src[0];
        if (conv_window_base(bias) == cache) {
            return false;
        }
    }

    m = r;
    return true;
}

struct conv_window_slots {
    float * dst[GGML_SYCL_CONV_WINDOW_MAX_SLOTS];
    int     s_idx[GGML_SYCL_CONV_WINDOW_MAX_SLOTS];
    int     n;
};

// Each work-item slides a four-value window over up to eight tokens.
// Token max(0, w - 3) writes cache column w, so each cache value has one writer.
static void kernel_ssm_conv_window(queue & q, const float * state, const float * x, const float * weights,
                                   const float * bias, float * dst, conv_window_slots slots, int d_inner, int n_t,
                                   int n_s, int64_t st_col, int64_t st_ch, int64_t st_seq, int64_t x_tok,
                                   int64_t x_ch, int64_t x_seq, int64_t cache_seq) {
    constexpr int DC = 4;
    constexpr int NC = DC - 1;
    constexpr int TT = 8;

    // a 3D range: 64-bit div/mod of a flat index is emulated and costs a lot at the prefill shape
    const int wg    = 256;
    const int c_pad = (d_inner + wg - 1) / wg * wg;
    const int n_tb  = (n_t + TT - 1) / TT;

    q.parallel_for(nd_range<3>(range<3>(n_s, n_tb, c_pad), range<3>(1, 1, wg)), [=](nd_item<3> item) {
        const int c  = static_cast<int>(item.get_global_id(2));
        const int t0 = static_cast<int>(item.get_global_id(1)) * TT;
        const int s  = static_cast<int>(item.get_global_id(0));
        if (c >= d_inner) {
            return;
        }
        const int t1 = sycl::min(t0 + TT, n_t);

        const float * sp = state + s * st_seq + c * st_ch;
        const float * xp = x + s * x_seq + c * x_ch;

        float win[DC];
#pragma unroll
        for (int k = 0; k < NC; ++k) {
            const int w = t0 + k;
            win[k + 1] = w < NC ? sp[w * st_col] : xp[(w - NC) * x_tok];
        }

        float cw[DC];
#pragma unroll
        for (int k = 0; k < DC; ++k) {
            cw[k] = weights[static_cast<size_t>(c) * DC + k];
        }
        const float b = bias != nullptr ? bias[c] : 0.0f;

        for (int t = t0; t < t1; ++t) {
#pragma unroll
            for (int k = 0; k < NC; ++k) {
                win[k] = win[k + 1];
            }
            win[NC] = xp[t * x_tok];

            float sumf = 0.0f;
#pragma unroll
            for (int k = 0; k < DC; ++k) {
                sumf += win[k] * cw[k];
            }
            if (bias != nullptr) {
                sumf += b;
            }
            dst[(static_cast<size_t>(s) * n_t + t) * d_inner + c] = op_silu(sumf);

            // constant win[] indices keep the window in registers
            for (int p = 0; p < slots.n; ++p) {
                float * cd = slots.dst[p] + s * cache_seq + static_cast<size_t>(c) * NC;
#pragma unroll
                for (int k = 0; k < DC; ++k) {
                    const int w = t + k;
                    const int j = w - slots.s_idx[p];
                    if (j >= 0 && j < NC && sycl::max(w - NC, 0) == t) {
                        cd[j] = win[k];
                    }
                }
            }
        }
    });
}

void ggml_sycl_ssm_conv_window(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_window_match & m) {
    const ggml_tensor * state = m.concat->src[0];
    const ggml_tensor * xt    = m.concat->src[1];

    const int d_inner = m.concat->ne[1];
    const int n_t     = xt->ne[0];
    const int n_s     = m.concat->ne[2];

    GGML_ASSERT(m.conv->ne[0] == d_inner && m.conv->ne[1] == n_t && m.conv->ne[2] == n_s);

    conv_window_slots slots = {};
    slots.n = m.n_slots;
    for (int p = 0; p < m.n_slots; ++p) {
        const ggml_tensor * tail = m.cpy[p]->src[0];
        if (tail->op == GGML_OP_CONT) {
            tail = tail->src[0];
        }
        slots.dst[p]   = static_cast<float *>(m.cpy[p]->src[1]->data);
        slots.s_idx[p] = static_cast<int>(tail->view_offs / sizeof(float));
    }

    const float * bias = nullptr;
    if (m.add != nullptr) {
        const ggml_tensor * bias_t = (m.add->src[0] == m.conv) ? m.add->src[1] : m.add->src[0];
        bias = static_cast<const float *>(bias_t->data);
    }

    constexpr int64_t fs = sizeof(float);
    kernel_ssm_conv_window(*ctx.stream(), static_cast<const float *>(state->data),
                           static_cast<const float *>(xt->data), static_cast<const float *>(m.conv->src[1]->data),
                           bias, static_cast<float *>(m.silu->data), slots, d_inner, n_t, n_s, state->nb[0] / fs,
                           state->nb[1] / fs, state->nb[2] / fs, xt->nb[0] / fs, xt->nb[1] / fs, xt->nb[2] / fs,
                           m.cpy[0]->src[1]->nb[1] / fs);
}

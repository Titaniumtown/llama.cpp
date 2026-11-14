// A defective DPAS EU corrupts two output channels of every oneDNN f16/bf16 GEMM tile it runs.
// The series' own XMX kernels repair this in place; oneDNN's jit kernels cannot, so the rows
// those channels land on are found by checksum and recomputed here. The recompute is plain
// f32 FMAs: a DPAS repair would land on the same EU.

#include "dpas-eu-repair.hpp"

#include "common.hpp"
#include "dpas-check.hpp"

#include <algorithm>
#include <vector>

template <typename HT>
static void k_col_sums(const HT * B, float * v, float * w, int k, int n, int64_t sb_k, int64_t sb_col,
                       const sycl::nd_item<1> & it) {
    const int l = it.get_global_id(0);
    if (l >= k) {
        return;
    }
    float s1 = 0, s2 = 0;
    for (int j = 0; j < n; j++) {
        const float b = (float) B[(int64_t) l * sb_k + (int64_t) j * sb_col];
        s1 += b;
        s2 += (j & 1) ? -b : b;
    }
    v[l] = s1;
    w[l] = s2;
}

// One sub-group per row. Candidate rows (row % 16 has its bit set in bad_mask) compare the
// GEMM output against the column-sum checksum; every lane of a non-candidate returns together
// so the reductions below stay full-width.
template <typename HT>
static void k_check_rows(const HT * A, const float * C, const float * v, const float * w, int * flags, int m, int n,
                         int k, uint32_t bad_mask, int64_t sa_k, int64_t sa_row, int64_t sc_row, int64_t sc_col,
                         const sycl::nd_item<1> & it) {
    const int row  = it.get_group(0);
    const int lane = it.get_local_id(0);
    if (row >= m || !(bad_mask & (1u << (row & 15)))) {
        return;
    }
    float ref1 = 0, ref2 = 0;
    for (int l = lane; l < k; l += WARP_SIZE) {
        const float a = (float) A[(int64_t) row * sa_row + (int64_t) l * sa_k];
        ref1 += a * v[l];
        ref2 += a * w[l];
    }
    float got1 = 0, got2 = 0;
    for (int j = lane; j < n; j += WARP_SIZE) {
        const float c = C[(int64_t) row * sc_row + (int64_t) j * sc_col];
        got1 += c;
        got2 += (j & 1) ? -c : c;
    }
    const auto sg = it.get_sub_group();
    ref1 = sycl::reduce_over_group(sg, ref1, sycl::plus<float>());
    ref2 = sycl::reduce_over_group(sg, ref2, sycl::plus<float>());
    got1 = sycl::reduce_over_group(sg, got1, sycl::plus<float>());
    got2 = sycl::reduce_over_group(sg, got2, sycl::plus<float>());
    if (lane == 0) {
        // f16 rounding of the K*N-term sum is a few ulps of the sum; a defective channel is off by
        // whole units, so the gap between the two is the tolerance
        const float tol = sycl::max(1.0f, 0.01f * sycl::max(sycl::fabs(ref1), sycl::fabs(ref2)));
        flags[row]      = (sycl::fabs(ref1 - got1) > tol || sycl::fabs(ref2 - got2) > tol) ? 1 : 0;
    }
}

// 16 flagged rows x 16 columns, K streamed in chunks of 32. 256 threads, one output each.
// Sub-group sg owns row-local sg (A, contiguous in K) and column-local sg (B, contiguous in K);
// thread (sg, lane) accumulates column-local lane. SLM is 4 KB. No DPAS.
template <typename HT>
static void k_repair_tiled(const HT * A, const HT * B, float * C, const int * flagged, int n_flagged, int n, int k,
                           int64_t sa_k, int64_t sa_row, int64_t sb_k, int64_t sb_col, int64_t sc_row, int64_t sc_col,
                           float * As, float * Bs, const sycl::nd_item<1> & it) {
    constexpr int RM = 16;
    constexpr int BN = 16;
    constexpr int BK = 32;
    const int     tid  = it.get_local_id(0);
    const int     sg   = tid / WARP_SIZE;
    const int     lane = tid % WARP_SIZE;
    const int     cols = (n + BN - 1) / BN;
    const int     gx   = it.get_group(0) % cols;
    const int     gy   = it.get_group(0) / cols;
    const int     ridx = gy * RM + sg;
    const int     col  = gx * BN + lane;

    float acc = 0;
    for (int l0 = 0; l0 < k; l0 += BK) {
        for (int t = 0; t < 2; t++) {
            const int kk   = lane + t * WARP_SIZE;
            const int l    = l0 + kk;
            const int ar   = gy * RM + sg;
            float     aval = 0;
            if (ar < n_flagged && l < k) {
                const int row = flagged[ar];
                aval          = (float) A[(int64_t) row * sa_row + (int64_t) l * sa_k];
            }
            As[sg * BK + kk] = aval;

            const int bc   = gx * BN + sg;
            float     bval = 0;
            if (l < k && bc < n) {
                bval = (float) B[(int64_t) l * sb_k + (int64_t) bc * sb_col];
            }
            Bs[kk * BN + sg] = bval;
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (ridx < n_flagged && col < n) {
#pragma unroll
            for (int kk = 0; kk < BK; kk++) {
                acc += As[sg * BK + kk] * Bs[kk * BN + lane];
            }
        }
        it.barrier(sycl::access::fence_space::local_space);
    }
    if (ridx < n_flagged && col < n) {
        const int row = flagged[ridx];
        C[(int64_t) row * sc_row + (int64_t) col * sc_col] = acc;
    }
}

template <typename HT>
static int repair_batch(ggml_backend_sycl_context & ctx, sycl::queue & q, const HT * A, const HT * B, float * C, int m,
                        int n, int k, int64_t sa_k, int64_t sa_row, int64_t sb_k, int64_t sb_col, int64_t sc_row,
                        int64_t sc_col, uint32_t bad_mask) {
    ggml_sycl_pool_alloc<float> vw(ctx.pool(), (size_t) k * 2);
    float *                     v = vw.get();
    float *                     w = v + k;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((k + 255) & ~255), sycl::range<1>(256)),
                   [=](sycl::nd_item<1> it) { k_col_sums<HT>(B, v, w, k, n, sb_k, sb_col, it); });

    ggml_sycl_pool_alloc<int> flags_alloc(ctx.pool(), m);
    int *                     flags = flags_alloc.get();
    q.memset(flags, 0, (size_t) m * sizeof(int));
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) m * WARP_SIZE), sycl::range<1>(WARP_SIZE)),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                       k_check_rows<HT>(A, C, v, w, flags, m, n, k, bad_mask, sa_k, sa_row, sc_row, sc_col, it);
                   });

    std::vector<int> hflags(m), flagged;
    q.memcpy(hflags.data(), flags, (size_t) m * sizeof(int)).wait();
    for (int i = 0; i < m; i++) {
        if (hflags[i]) {
            flagged.push_back(i);
        }
    }
    if (flagged.empty()) {
        return 0;
    }

    ggml_sycl_pool_alloc<int> flagged_alloc(ctx.pool(), flagged.size());
    q.memcpy(flagged_alloc.get(), flagged.data(), flagged.size() * sizeof(int)).wait();
    const int   nf          = (int) flagged.size();
    const int * flagged_ptr = flagged_alloc.get();
    constexpr int RM = 16;
    constexpr int BN = 16;
    const int     row_tiles = (nf + RM - 1) / RM;
    const int     col_tiles = (n + BN - 1) / BN;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> As(RM * 32, cgh);
        sycl::local_accessor<float, 1> Bs(32 * BN, cgh);
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) row_tiles * col_tiles * 256), sycl::range<1>(256)),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             k_repair_tiled<HT>(A, B, C, flagged_ptr, nf, n, k, sa_k, sa_row, sb_k, sb_col, sc_row,
                                                sc_col, get_pointer(As), get_pointer(Bs), it);
                         });
    }).wait();
    return nf;
}

void ggml_sycl_dpas_repair_gemm(ggml_backend_sycl_context & ctx, const void * a, bool a_bf16, int64_t sa_k,
                                int64_t sa_row, int64_t sa_batch, const void * b, bool b_bf16, int64_t sb_k,
                                int64_t sb_col, int64_t sb_batch, float * c, int64_t sc_row, int64_t sc_col,
                                int64_t sc_batch, int m, int n, int k, int batches_a, int batches_b) {
    if (m <= 0 || n <= 0 || k <= 0 || a_bf16 != b_bf16) {
        return;
    }
    // the rows are found by checksum, so the lanes of every defective EU are repaired the same way
    uint32_t mask = 0;
    for (const ggml_sycl_dpas_fault & f : ggml_sycl_dpas_get_health(get_current_device_id()).faults) {
        mask |= f.lanes_chain;
    }
    if (mask == 0) {
        return;
    }

    sycl::queue & q       = *ctx.stream();
    const int     n_batch = std::max(batches_a, batches_b);
    for (int batch = 0; batch < n_batch; batch++) {
        const int64_t oa = (int64_t) (batches_a == 1 ? 0 : batch) * sa_batch;
        const int64_t ob = (int64_t) (batches_b == 1 ? 0 : batch) * sb_batch;
        const int64_t oc = (int64_t) batch * sc_batch;
        if (a_bf16) {
#ifdef GGML_SYCL_HAS_BF16
            using HT = sycl::ext::oneapi::bfloat16;
            repair_batch<HT>(ctx, q, (const HT *) a + oa, (const HT *) b + ob, c + oc, m, n, k, sa_k, sa_row, sb_k,
                             sb_col, sc_row, sc_col, mask);
#else
            return;
#endif
        } else {
            repair_batch<sycl::half>(ctx, q, (const sycl::half *) a + oa, (const sycl::half *) b + ob, c + oc, m, n,
                                     k, sa_k, sa_row, sb_k, sb_col, sc_row, sc_col, mask);
        }
    }
}

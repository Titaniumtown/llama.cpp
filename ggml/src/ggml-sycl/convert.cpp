#include "convert.hpp"
#include "dequantize.hpp"
#include "presets.hpp"

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t k,
                             const sycl::nd_item<3> &item_ct1) {
    const int64_t i = 2 * (item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                       item_ct1.get_local_id(2));

    if (i >= k) {
        return;
    }

    const int64_t ib = i/qk; // block index
    const int64_t iqs = (i%qk)/qr; // quant index
    const int64_t iybs = i - i%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0] = v.x();
    y[iybs + iqs + y_offset] = v.y();
}

template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_sycl(const void *__restrict__ vx,
                                  dst_t *__restrict__ y, const int64_t k,
                                  dpct::queue_ptr stream) {
    const int64_t num_blocks = (k + 2*SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2*SYCL_DEQUANTIZE_BLOCK_SIZE);
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});
        stream->parallel_for(
            sycl::nd_range<3>(
                sycl::range<3>(1, 1, num_blocks) *
                    sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
            [=](sycl::nd_item<3> item_ct1) {
                dequantize_block<qk, qr, dequantize_kernel>(vx, y, k, item_ct1);
            });
    }
}

template <typename dst_t>
static void dequantize_row_q2_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q2_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q3_K(vx, y, item_ct1);
                             });
    }
#endif
}

template <typename dst_t>
static void dequantize_row_q3_K_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                             dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 64), sycl::range<3>(1, 1, 64)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q3_K_reorder(vx, y, item_ct1, nb);
        });
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_0(vx, y, nb32, item_ct1);
                             });
    }
}

template <typename dst_t>
static void dequantize_row_q4_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    int constexpr WARP_K = WARP_SIZE * QK4_0;
    const int n_warp = (k + WARP_K - 1) / WARP_K;
    GGML_ASSERT(k % 2 == 0);
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_warp) *
        sycl::range<3>(1, 1, WARP_SIZE),
        sycl::range<3>(1, 1, WARP_SIZE)),
        [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]]{
            dequantize_block_q4_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q8_0_sycl_reorder(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {

    dpct::has_capability_or_fail(stream->get_device(),
                                    {sycl::aspect::fp16});

    GGML_ASSERT(k % QK8_0 == 0);
    // One contiguous chunk per work-item, ordinary work-group, no forced sub-group size: the
    // previous wg=32 + reqd_sub_group_size(32) pairing was the slowest arrangement measured.
    constexpr int EPT      = GGML_SYCL_Q8_0_DEQ_EPT;
    constexpr int wg       = 256;
    const int64_t n_items  = (k + EPT - 1) / EPT;
    const int64_t n_groups = (n_items + wg - 1) / wg;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, n_groups) *
        sycl::range<3>(1, 1, wg),
        sycl::range<3>(1, 1, wg)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_q8_0_reorder(vx, y, k, item_ct1);
        });

}

template <typename dst_t>
static void dequantize_row_q4_1_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb32 = k / 32;
    const int64_t nb = (k + 255) / 256;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q4_1(vx, y, nb32, item_ct1);
                             });
    }
}


// Work-group width for the wide K-quant dequant kernels (q4_K/q5_K, both layouts).
// 16 threads cover one 256-weight block; the launcher packs WG/16 blocks per group.
// Swept standalone on the hot 5120x17408 shape (see dequantize.hpp): 64 -> 526 GB/s,
// 128 -> 896, 256 -> 1124, 512 -> 1133. 256 is the default; 512 is within noise of it
// but halves the group count, which matters for the small tail shapes.
static inline int deqk_wg() {
    static const int v = [] {
        const int w = ggml_sycl_get_env("GGML_SYCL_DEQK_WG", 256);
        return (w >= 16 && w <= 1024 && w % 16 == 0) ? w : 256;
    }();
    return v;
}

// Elements per thread in the q6_K reorder dequant. 8 measured best (620 GB/s) against the
// previous shape's 434-465, with 4 within noise of it; see the table on the kernel.
static inline int q6k_deq_m() {
    static const int v = [] {
        const int m = ggml_sycl_get_env("GGML_SYCL_Q6K_DEQ_M", 8);
        return (m == 2 || m == 4 || m == 8 || m == 16) ? m : 8;
    }();
    return v;
}

#define GGML_SYCL_LAUNCH_DEQK_WIDE(KERNEL)                                                        \
    do {                                                                                          \
        const int     wg     = deqk_wg();                                                         \
        const int64_t groups = (nb * 16 + wg - 1) / wg;                                           \
        stream->parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) {       \
            KERNEL<dst_t>(vx, y, nb, it);                                                         \
        });                                                                                       \
    } while (0)

template <typename dst_t>
static void dequantize_row_q4_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    GGML_SYCL_LAUNCH_DEQK_WIDE(dequantize_block_q4_K_wide);
}

template <typename dst_t>
static void dequantize_row_q4_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    GGML_SYCL_LAUNCH_DEQK_WIDE(dequantize_block_q4_K_reorder_wide);
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    GGML_SYCL_LAUNCH_DEQK_WIDE(dequantize_block_q5_K_wide);
}

template <typename dst_t>
static void dequantize_row_q5_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });
    GGML_SYCL_LAUNCH_DEQK_WIDE(dequantize_block_q5_K_reorder_wide);
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl(const void *vx, dst_t *y, const int64_t k,
                                     dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
#if QK_K == 256
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 64),
                                               sycl::range<3>(1, 1, 64)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }
#else
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_q6_K(vx, y, item_ct1);
                             });
    }

#endif
}

template <typename dst_t>
static void dequantize_row_q6_K_sycl_reorder(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;

    dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

    const auto launch = [&]<int m>() {
        constexpr int    per_ip = 32 / m;
        constexpr size_t wg     = 256;
        const int64_t    chunks = nb * 2 * per_ip;
        const int64_t    ngroup = (chunks + wg - 1) / wg;
        stream->parallel_for(sycl::nd_range<1>(sycl::range<1>(ngroup * wg), sycl::range<1>(wg)),
                             [=](sycl::nd_item<1> item_ct1) {
                                 dequantize_block_q6_K_reorder<dst_t, m>(vx, y, item_ct1, nb);
                             });
    };
    switch (q6k_deq_m()) {
        case 2:  launch.template operator()<2>();  break;
        case 4:  launch.template operator()<4>();  break;
        case 16: launch.template operator()<16>(); break;
        default: launch.template operator()<8>();  break;
    }
}

template <typename dst_t>
static void dequantize_row_iq1_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_s(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq1_m_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq1_m(
                                     vx, y, item_ct1, iq1s_grid_gpu
                                     );
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xxs(
                                     vx, y, item_ct1, iq2xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_xs(
                                     vx, y, item_ct1, iq2xs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq2_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                      dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq2_s(vx, y, item_ct1);
                             });
        });
    }
}


template <typename dst_t>
static void dequantize_row_iq3_xxs_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_xxs(
                                     vx, y, item_ct1, iq3xxs_grid,
                                     ksigns_iq2xs, kmask_iq2xs);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq3_s_sycl(const void *vx, dst_t *y, const int64_t k,
                                        dpct::queue_ptr stream) {
    const int64_t nb = k / QK_K;
    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->submit([&](sycl::handler &cgh) {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                                   sycl::range<3>(1, 1, 32),
                                               sycl::range<3>(1, 1, 32)),
                             [=](sycl::nd_item<3> item_ct1) {
                                 dequantize_block_iq3_s(
                                     vx, y, item_ct1, kmask_iq2xs, iq3s_grid);
                             });
        });
    }
}

template <typename dst_t>
static void dequantize_row_iq4_xs_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
#if QK_K == 64
    dequantize_row_iq4_nl_sycl(vx, y, k, stream);
#else
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_xs(vx, y, item_ct1);
                      });
            });
      }
#endif
}

template <typename dst_t>
static void dequantize_row_iq4_nl_sycl(const void *vx, dst_t *y, const int64_t k,
                                       dpct::queue_ptr stream) {
    const int64_t nb = (k + QK_K - 1) / QK_K;
      {
            dpct::has_capability_or_fail(stream->get_device(),
                                         {sycl::aspect::fp16});

            stream->submit([&](sycl::handler &cgh) {
                  cgh.parallel_for(
                      sycl::nd_range<3>(sycl::range<3>(1, 1, nb) *
                                            sycl::range<3>(1, 1, 32),
                                        sycl::range<3>(1, 1, 32)),
                      [=](sycl::nd_item<3> item_ct1) {
                            dequantize_block_iq4_nl(vx, y, item_ct1);
                      });
            });
      }
}

template <typename dst_t>
static void dequantize_row_mxfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    const int nb = (k + QK_K - 1) / QK_K;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> item_ct1) {
            dequantize_block_mxfp4(vx, y, item_ct1);
        });
}

template <typename dst_t>
static void dequantize_row_nvfp4_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr stream) {
    GGML_ASSERT(k % QK_NVFP4 == 0);
    const int nb = k / QK_NVFP4;
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, nb) * sycl::range<3>(1, 1, 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_item<3> /*item_ct1*/) {
            dequantize_block_nvfp4(vx, y, k);
        });
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc(const void * __restrict__ vx, dst_t * __restrict__ y,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t s01, const int64_t s02, const int64_t s03) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i00 = 2 * (int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2));

    if (i00 >= ne00) {
        return;
    }

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    const int64_t ibx0 = i03*s03 + i02*s02 + i01*s01;

    const int64_t ib = ibx0 + i00/qk; // block index
    const int64_t iqs = (i00%qk)/qr; // quant index
    const int64_t iybs = i00 - i00%qk; // y block start index
    const int64_t y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    #ifdef GGML_SYCL_F16
        sycl::half2 v;
    #else
        sycl::float2 v;
    #endif

    dequantize_kernel(vx, ib, iqs, v);

    const int64_t iy0 = ((i03*ne02 + i02)*ne01 + i01)*ne00 + iybs + iqs;
    y[iy0 + 0]        = ggml_sycl_cast<dst_t>(v.x());
    y[iy0 + y_offset] = ggml_sycl_cast<dst_t>(v.y());
}


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static void dequantize_block_nc_sycl(const void *    vx,
                                  dst_t *         y,
                                  const int64_t   ne00,
                                  const int64_t   ne01,
                                  const int64_t   ne02,
                                  const int64_t   ne03,
                                  const int64_t   s01,
                                  const int64_t   s02,
                                  const int64_t   s03,
                                  dpct::queue_ptr stream) {
    const dpct::dim3 num_blocks((ne00 + 2 * SYCL_DEQUANTIZE_BLOCK_SIZE - 1) / (2 * SYCL_DEQUANTIZE_BLOCK_SIZE), ne01,
                                ne02 * ne03);
    stream->parallel_for(sycl::nd_range<3>(num_blocks * sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_DEQUANTIZE_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) {
                             GGML_UNUSED(item_ct1);
                             dequantize_block_nc<qk, qr, dequantize_kernel>(vx, y, ne00, ne01, ne02, s01, s02, s03);
                         });
}
// `w` contiguous elements per work-item. With w == 1 this is the original one-element-per-item
// grid-stride loop, whose successive iterations are a whole global range apart -- so each item
// issues one narrow load and one narrow store, and nothing can merge them. Giving an item a
// contiguous run instead lets the compiler widen the accesses, while a subgroup still covers a
// contiguous span so coalescing is unchanged. No sycl::vec, so bf16 needs no special case.
template <typename src_t, typename dst_t, int w>
static void convert_unary_nc(const void * __restrict__ vx, dst_t * __restrict__ y, const int64_t ne00, const int64_t ne01,
                          const int64_t ne02, const int64_t s01, const int64_t s02, const int64_t s03,
                          const sycl::nd_item<3> & item_ct1) {

    const int64_t work_group_size = item_ct1.get_local_range(2);
    const int64_t global_id       = item_ct1.get_local_id(2) + work_group_size * item_ct1.get_group(2);

    const int64_t i01 = item_ct1.get_group(1);
    const int64_t i02 = item_ct1.get_group(0) % ne02;
    const int64_t i03 = item_ct1.get_group(0) / ne02;

    // make each work-item deal with more elements since sycl global range can not exceed max int
    const src_t * x = static_cast<const src_t *>(vx);
    const int64_t ix = i03 * s03 + i02 * s02 + i01 * s01;
    const int64_t iy = ((i03 * ne02 + i02) * ne01 + i01) * ne00;

    const int64_t stride = work_group_size * item_ct1.get_group_range(2) * w;

    for (int64_t base = global_id * w; base < ne00; base += stride) {
        if (base + w <= ne00) {
#pragma unroll
            for (int j = 0; j < w; ++j) {
                y[iy + base + j] = static_cast<dst_t>(x[ix + base + j]);
            }
        } else {
            // ragged tail: only the last item of the last group can land here
            for (int64_t i00 = base; i00 < ne00; ++i00) {
                y[iy + i00] = static_cast<dst_t>(x[ix + i00]);
            }
        }
    }
}

// Elements per work-item for the unary convert. 1 is the historical kernel, which gives an
// item a single element and so cannot use a wide access. Measured CONVERT rate on this
// model, prefill at ub 2048: w=1 340 GB/s, w=2 450, w=4 347, w=8 323 -- so the peak is at 2
// and it is not monotone, which is why this is a swept knob and not a derived constant.
static inline int convert_unary_w() {
    static const int v = [] {
        const int w = ggml_sycl_get_env("GGML_SYCL_CONVERT_W", 2);
        return (w == 1 || w == 2 || w == 4 || w == 8) ? w : 2;
    }();
    return v;
}

template <typename src_t, typename dst_t, int w>
static void convert_unary_nc_launch(const void * __restrict__ vx, dst_t * __restrict__ y,
                                    const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
                                    const int64_t s01, const int64_t s02, const int64_t s03, dpct::queue_ptr queue) {
    sycl::range<3> global_size(ne02 * ne03, ne01, ceil_div(ne00, SYCL_DEQUANTIZE_BLOCK_SIZE * (int64_t) w));

    // decrease global range when it exceeds the max int
    // TODO: Downsample logic is separated from the kernel, a rewrite is desirable
    int64_t        downsized_workgroup = downsample_sycl_global_range(global_size[0], SYCL_DEQUANTIZE_BLOCK_SIZE);
    sycl::range<3> workgroup_size(1, 1, downsized_workgroup);

    queue->parallel_for(sycl::nd_range<3>(global_size * workgroup_size, workgroup_size), [=](sycl::nd_item<3> item_ct1) {
        convert_unary_nc<src_t, dst_t, w>(vx, y, ne00, ne01, ne02, s01, s02, s03, item_ct1);
    });
}

template <typename src_t, typename dst_t>
static void convert_unary_nc_sycl(const void * __restrict__ vx, dst_t * __restrict__ y,
                                  const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
                                  const int64_t s01, const int64_t s02, const int64_t s03, dpct::queue_ptr queue) {
    dpct::has_capability_or_fail(queue->get_device(), { sycl::aspect::fp16 });

    // A wide access has to be aligned to its own width, and every row start must be too, so
    // the row strides are part of the condition and not just the row length. Anything that
    // fails it keeps the one-element-per-item kernel, which has no alignment requirement.
    const int w = convert_unary_w();
    const bool aligned = (ne00 % w == 0) && (s01 % w == 0) && (s02 % w == 0) && (s03 % w == 0) &&
                         (reinterpret_cast<uintptr_t>(vx) % (w * sizeof(src_t)) == 0) &&
                         (reinterpret_cast<uintptr_t>(y) % (w * sizeof(dst_t)) == 0);

    if (aligned) {
        switch (w) {
            case 2: convert_unary_nc_launch<src_t, dst_t, 2>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, queue); return;
            case 4: convert_unary_nc_launch<src_t, dst_t, 4>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, queue); return;
            case 8: convert_unary_nc_launch<src_t, dst_t, 8>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, queue); return;
            default: break;
        }
    }
    convert_unary_nc_launch<src_t, dst_t, 1>(vx, y, ne00, ne01, ne02, ne03, s01, s02, s03, queue);
}

template <typename src_t, typename dst_t>
static void convert_unary_sycl(const void * vx, dst_t * y, const int64_t k, dpct::queue_ptr queue) {
    convert_unary_nc_sycl<src_t>(vx, y, k, 1, 1, 1, k, k, k, queue);
}


to_fp16_sycl_t ggml_get_to_fp16_sycl(ggml_type type, ggml_tensor * dst) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK4_0, QR4_0, dequantize_q4_0>;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_block_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q8_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
            }
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q3_K_sycl_reorder;
            } else {
                return dequantize_row_q3_K_sycl;
            }
        case GGML_TYPE_Q4_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q5_K_sycl_reorder;
            } else {
                return dequantize_row_q5_K_sycl;
            }
        case GGML_TYPE_Q6_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_sycl;
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}

to_fp32_sycl_t ggml_get_to_fp32_sycl(ggml_type type, ggml_tensor *dst) {
    switch (type) {
        case GGML_TYPE_Q1_0:
            return dequantize_block_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q2_0:
            return dequantize_block_sycl<QK2_0, QR2_0, dequantize_q2_0>;
        case GGML_TYPE_Q4_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_0_sycl_reorder;
            } else {
                return dequantize_row_q4_0_sycl;
            }
        case GGML_TYPE_Q4_1:
            return dequantize_row_q4_1_sycl;
        case GGML_TYPE_Q5_0:
            return dequantize_block_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q8_0_sycl_reorder;
            } else {
                return dequantize_block_sycl<QK8_0, QR8_0, dequantize_q8_0>;
            }
        case GGML_TYPE_Q2_K:
            return dequantize_row_q2_K_sycl;
        case GGML_TYPE_Q3_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q3_K_sycl_reorder;
            } else {
                return dequantize_row_q3_K_sycl;
            }
        case GGML_TYPE_Q4_K:
            if (dst->src[0]->extra &&
                ((ggml_tensor_extra_gpu*)dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q4_K_sycl_reorder;
            } else {
                return dequantize_row_q4_K_sycl;
            }
        case GGML_TYPE_Q5_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q5_K_sycl_reorder;
            } else {
                return dequantize_row_q5_K_sycl;
            }
        case GGML_TYPE_Q6_K:
            if (dst->src[0]->extra && ((ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder) {
                return dequantize_row_q6_K_sycl_reorder;
            } else {
                return dequantize_row_q6_K_sycl;
            }
        case GGML_TYPE_IQ1_S:
            return dequantize_row_iq1_s_sycl;
        case GGML_TYPE_IQ1_M:
            return dequantize_row_iq1_m_sycl;
        case GGML_TYPE_IQ2_XXS:
            return dequantize_row_iq2_xxs_sycl;
        case GGML_TYPE_IQ2_XS:
            return dequantize_row_iq2_xs_sycl;
        case GGML_TYPE_IQ2_S:
            return dequantize_row_iq2_s_sycl;
        case GGML_TYPE_IQ3_XXS:
            return dequantize_row_iq3_xxs_sycl;
        case GGML_TYPE_IQ3_S:
            return dequantize_row_iq3_s_sycl;
        case GGML_TYPE_IQ4_XS:
            return dequantize_row_iq4_xs_sycl;
        case GGML_TYPE_IQ4_NL:
            return dequantize_row_iq4_nl_sycl;
        case GGML_TYPE_MXFP4:
            return dequantize_row_mxfp4_sycl;
        case GGML_TYPE_NVFP4:
            return dequantize_row_nvfp4_sycl;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}


#ifdef GGML_SYCL_HAS_BF16
to_bf16_sycl_t ggml_get_to_bf16_sycl(ggml_type type, ggml_tensor * /*dst*/) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_sycl<float>;
        case GGML_TYPE_F16:
            return convert_unary_sycl<sycl::half>;
        case GGML_TYPE_BF16:
            return convert_unary_sycl<sycl::ext::oneapi::bfloat16>;
        default:
            GGML_ABORT("fatal error: unsupport data type=%s\n", ggml_type_name(type));
            return nullptr;
    }
}
#endif

to_fp16_nc_sycl_t ggml_get_to_fp16_nc_sycl(ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
            return convert_unary_nc_sycl<float>;
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            return convert_unary_nc_sycl<sycl::ext::oneapi::bfloat16>;
#endif
        case GGML_TYPE_Q1_0:
            return dequantize_block_nc_sycl<QK1_0, QR1_0, dequantize_q1_0>;
        case GGML_TYPE_Q4_0:
            return dequantize_block_nc_sycl<QK4_0, QR4_0, dequantize_q4_0>;
        case GGML_TYPE_Q4_1:
            return dequantize_block_nc_sycl<QK4_1, QR4_1, dequantize_q4_1>;
        case GGML_TYPE_Q5_0:
            return dequantize_block_nc_sycl<QK5_0, QR5_0, dequantize_q5_0>;
        case GGML_TYPE_Q5_1:
            return dequantize_block_nc_sycl<QK5_1, QR5_1, dequantize_q5_1>;
        case GGML_TYPE_Q8_0:
            return dequantize_block_nc_sycl<QK8_0, QR8_0, dequantize_q8_0>;
        default:
            return nullptr;
    }
}

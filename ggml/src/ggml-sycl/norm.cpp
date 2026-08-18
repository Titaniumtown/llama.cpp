#include <type_traits>

#include "norm.hpp"
#include "quantize.hpp"
#include "presets.hpp"
#include "ggml-sycl/common.hpp"
#include "ggml-sycl/presets.hpp"

static void norm_f32(const float* x, float* dst, const int ncols,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, const sycl::nd_item<3>& item_ct1, sycl::float2* s_sum, int block_size) {

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int nthreads = item_ct1.get_local_range(2);
    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x += src_offset;
    dst += dst_offset;

    sycl::float2 mean_var = sycl::float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col * src_stride_col];
        mean_var.x() += xi;
        mean_var.y() += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var, item_ct1);
    if  (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = mean_var;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);
        mean_var = 0.f;
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        for (size_t i = 0; i < nreduce; i += 1)
        {
            mean_var += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        mean_var = warp_reduce_sum(mean_var, item_ct1);
    }

    const float mean = mean_var.x() / ncols;
    const float var = mean_var.y() / ncols - mean * mean;
    const float inv_std = sycl::rsqrt(var + eps);

    for (int col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = (x[col * src_stride_col] - mean) * inv_std;
    }
}

static void group_norm_f32(const float* x, float* dst, const int group_size, const int ne_elements, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size) {
    int start = item_ct1.get_group(2) * group_size;
    int end = start + group_size;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;
    start += item_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_SIZE;

    if (end >= ne_elements) {
        end = ne_elements;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int j = start; j < end; j += block_size) {
        tmp += x[j];
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float mean = tmp / group_size;
    tmp = 0.0f;

    for (int j = start; j < end; j += block_size) {
        float xi = x[j] - mean;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {

        int warp_id = item_ct1.get_local_id(2) / WARP_SIZE;
        int lane_id = item_ct1.get_local_id(2) % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        item_ct1.barrier();
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[lane_id + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    float variance = tmp / group_size;
    float scale = sycl::rsqrt(variance + eps);
    for (int j = start; j < end; j += block_size) {
        dst[j] *= scale;
    }
}

// do_add folds the residual add that FEEDS this norm into it. The sum is still a real
// output -- the next block's residual reads it -- so it is written out here, but the norm
// then keeps it rather than re-reading it from DRAM through a second kernel launch.
// add_b/add_out are required to be the same shape and contiguity as x, so all three share
// x's strides and no second stride set is needed.
template <bool do_multiply = false, bool do_add = false, bool do_mirror = false>
static void rms_norm_f32(const float* x, float* dst, const int ncols,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, const sycl::nd_item<3>& item_ct1, float* s_sum, int block_size,
    const float* mul = nullptr, const int64_t mul_stride_row = 0, const int64_t mul_stride_channel = 0,
    const int64_t mul_stride_sample = 0, const int mul_nrows = 0, const int mul_nchannels = 0, const int mul_nsamples = 0,
    const float* add_b = nullptr, float* add_out = nullptr, sycl::half * mir = nullptr,
    char * q8_out = nullptr, int q8_kx = 0, int q8_row_stride = 0) {

    const int nrows = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int sample  = item_ct1.get_group(0);
    const int channel = item_ct1.get_group(1);
    const int row     = item_ct1.get_group(2);

    const int nthreads = item_ct1.get_local_range(2);

    const int tid = item_ct1.get_local_id(2);
    const int nwarps = nthreads / WARP_SIZE;

    const auto src_offset = calculate_offset<3>({src_stride_sample, src_stride_channel, src_stride_row}, {sample, channel, row});
    const auto dst_offset = calculate_offset<3>({dst_stride_sample, dst_stride_channel, dst_stride_row}, {sample, channel, row});

    x   += src_offset;
    dst += dst_offset;
    if constexpr (do_mirror) {
        // only enabled for a contiguous dst, so dst's own offset and stride index it too
        mir += dst_offset;
    }
    if (q8_out != nullptr) {
        // q8_1 rows are padded to MATRIX_ROW_PADDING, so this stride is NOT dst's -- it is the
        // one ggml_sycl_op_mul_mat derives from src1_padded_col_size, and the two differ
        // whenever ne00 is not a multiple of 512.
        q8_out += (size_t) row * (size_t) q8_row_stride;
    }

    if constexpr (do_multiply) {
        const int mul_row     = row     % mul_nrows;
        const int mul_channel = channel % mul_nchannels;
        const int mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_row * mul_stride_row;
    }

    if constexpr (do_add) {
        add_b   += src_offset;
        add_out += src_offset;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        float xi;
        if constexpr (do_add) {
            xi = x[col * src_stride_col] + add_b[col * src_stride_col];
            add_out[col * src_stride_col] = xi;
        } else {
            xi = x[col * src_stride_col];
        }
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp, item_ct1);
    if (block_size > WARP_SIZE) {
        const auto sub_group = item_ct1.get_sub_group();
        const auto sg_id = sub_group.get_group_linear_id();
        const auto wi_in_sg = sub_group.get_local_linear_id();
        if (wi_in_sg == 0) {
            s_sum[sg_id] = tmp;
        }

        item_ct1.barrier(sycl::access::fence_space::local_space);
        const size_t nreduce = ceil_div(nwarps, WARP_SIZE);
        tmp = 0.f;
        for (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_sum[wi_in_sg + i * WARP_SIZE];
        }
        tmp = warp_reduce_sum(tmp, item_ct1);
    }

    const float mean = tmp / ncols;
    const float scale = sycl::rsqrt(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        float xi;
        if constexpr (do_add) {
            // written above by this same work-group, so this read is L1/L2-hot
            xi = add_out[col * src_stride_col];
        } else {
            xi = x[col * src_stride_col];
        }
        float v;
        if constexpr (do_multiply) {
            v = scale * xi * mul[col];
        } else {
            v = scale * xi;
        }
        dst[col * dst_stride_col] = v;
        if constexpr (do_mirror) {
            mir[col * dst_stride_col] = static_cast<sycl::half>(v);
        }
    }

    // Emit the q8_1 copy that the consuming mat-vec would otherwise launch a separate kernel
    // for. QUANTIZE/src1 is 259 launches/token at decode, 5.03 us each and 8.5 GB/s against a
    // 598 GB/s wall -- essentially pure launch latency. 0122's work-group sweep proved the
    // kernel itself is not the problem (1.7% on the kernel, exactly zero end-to-end, md5
    // identical at BPG=1/8/32), so the only way to remove the cost is to not launch it.
    //
    // One q8_1 block per SUB-GROUP, ElementsPerWI columns per lane -- the identical mapping and
    // arithmetic to quantize_and_reorder_q8_1_soa, which is what makes the bytes bit-identical
    // and the generation md5 immovable.
    //
    // The obvious formulation -- fold this into the store loop above, one column per lane -- is
    // WRONG here and silently so. GGML_SYCL_WARP_SIZE is 16 on Intel (ggml-sycl/CMakeLists.txt),
    // so a sub-group spans 16 columns while a q8_1 block spans 32: reduce_over_group then
    // computes d and sum over HALF a block. Nothing faults, every quant stays in range, and
    // generation drifts by a stable amount (5f9f7ae3a2a2e43c -> f9eacfcbf9979260, 3/3 reps).
    //
    // Iterating over blocks rather than columns also keeps the reduction legal: a sub-group
    // enters or skips an iteration as a unit, so every reduce_over_group is full-width, which a
    // column-strided loop cannot guarantee (ncols=5120 is not a multiple of 2*block_size).
    if (q8_out != nullptr) {
        constexpr int EPW = QK8_1 / WARP_SIZE;
        // dst is read across sub-group boundaries here -- block b was written by the threads
        // holding columns [32b,32b+32), which are not this sub-group's.
        item_ct1.barrier(sycl::access::fence_space::global_and_local);

        const auto sg   = item_ct1.get_sub_group();
        const int  lane = tid % WARP_SIZE;
        const int  nsg  = block_size / WARP_SIZE;
        const int  nblk = ncols / QK8_1;

        for (int b = tid / WARP_SIZE; b < nblk; b += nsg) {
            const int c0 = b * QK8_1 + lane * EPW;

            float vv[EPW];
            float sum  = 0.0f;
            float amax = 0.0f;
#pragma unroll
            for (int i = 0; i < EPW; i++) {
                vv[i] = dst[(c0 + i) * dst_stride_col];
                sum  += vv[i];
                amax  = sycl::fmax(amax, sycl::fabs(vv[i]));
            }
            sum  = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());

            const float d = amax == 0.0f ? 1.0f : amax / 127.0f;
#pragma unroll
            for (int i = 0; i < EPW; i++) {
                ((int8_t *) q8_out)[c0 + i] = static_cast<int8_t>(sycl::round(vv[i] / d));
            }
            if (lane == 0) {
                *(sycl::half2 *) (q8_out + q8_kx + b * (int) sizeof(sycl::half2)) =
                    sycl::half2(sycl::half(amax == 0.0f ? 0.0f : d), sycl::half(sum));
            }
        }
    }
}

template<int warp_size>
static void l2_norm_f32(const float * x, float * dst, const int ncols,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel,
    const int64_t src_stride_sample, const int64_t dst_stride_col, const int64_t dst_stride_row,
    const int64_t dst_stride_channel, const int64_t dst_stride_sample, const float eps,
    const sycl::nd_item<3>& item_ct1, float* s_sum, const int block_size) {
    const int nrows     = item_ct1.get_group_range(2);
    const int nchannels = item_ct1.get_group_range(1);

    const int row     = item_ct1.get_group(2);
    const int channel = item_ct1.get_group(1);
    const int sample  = item_ct1.get_group(0);
    const int tid     = item_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + row*src_stride_row;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + row*dst_stride_row;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[col * src_stride_col];
        tmp += xi * xi;
    }

    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, s_sum, block_size);
    const float scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    for (int col = tid; col < ncols; col += block_size) {
        dst[col * dst_stride_col] = scale * x[col * src_stride_col];
    }
}

static void norm_f32_sycl(const float * x, float * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const float eps, queue_ptr stream, int device) {

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_sum_acc_ct1(
                            sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void group_norm_f32_sycl(const float* x, float* dst,
    const int num_groups, const float eps, const int group_size,
    const int ne_elements, queue_ptr stream, int device) {
    if (group_size < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            const float eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(
                        x, dst, group_size, ne_elements, eps_ct4, item_ct1,
                        nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);

            const float eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, num_groups) * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    group_norm_f32(x, dst, group_size, ne_elements,
                        eps_ct4, item_ct1,
                        get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

static void rms_norm_f32_sycl(const float* x, float* dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
    const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
    const float eps, queue_ptr stream, int device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

template <bool do_add = false>
static void rms_norm_mul_f32_sycl(const float* x, const float* mul, float* dst, const int ncols, const int nrows,
        const int nchannels, const int nsamples,
        const int64_t src_stride_col, const int64_t src_stride_row, const int64_t src_stride_channel, const int64_t src_stride_sample,
        const int64_t dst_stride_col, const int64_t dst_stride_row, const int64_t dst_stride_channel, const int64_t dst_stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const float eps, queue_ptr stream, int device,
        const float * add_b = nullptr, float * add_out = nullptr, sycl::half * mir = nullptr,
        char * q8_out = nullptr, int q8_kx = 0, int q8_row_stride = 0) {
    const sycl::range<3> global_dims(nsamples, nchannels, nrows);
    const auto body = [&](auto mirror_tag) {
    constexpr bool MIR = decltype(mirror_tag)::value;
    if (ncols < 1024) {
        const sycl::range<3> block_dims(1, 1, WARP_SIZE);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<true, do_add, MIR>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, nullptr, WARP_SIZE,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add_b, add_out, mir, q8_out, q8_kx, q8_row_stride);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (WARP_SIZE * WARP_SIZE) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(work_group_size / WARP_SIZE), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * block_dims, block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    rms_norm_f32<true, do_add, MIR>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size,
                        mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        add_b, add_out, mir, q8_out, q8_kx, q8_row_stride);
                });
            });
    }
    };
    if (mir != nullptr) { body(std::true_type{}); } else { body(std::false_type{}); }
}

template<int warp_size>
static void l2_norm_f32_sycl(const float *   x,
                             float *         dst,
                             const int       ncols,
                             const int       nrows,
                             const int       nchannels,
                             const int       nsamples,
                             const int64_t   src_stride_col,
                             const int64_t   src_stride_row,
                             const int64_t   src_stride_channel,
                             const int64_t   src_stride_sample,
                             const int64_t   dst_stride_col,
                             const int64_t   dst_stride_row,
                             const int64_t   dst_stride_channel,
                             const int64_t   dst_stride_sample,
                             const float     eps,
                             queue_ptr       stream,
                             int             device) {
    const dpct::dim3 blocks_num(nrows, nchannels, nsamples);

    if (ncols < 1024) {
        const dpct::dim3 block_dims(warp_size, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(blocks_num * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(warp_size)]] {
                    l2_norm_f32<warp_size>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1,
                        nullptr, warp_size);
                });
            });
    }
    else {
        const int work_group_size = ggml_sycl_info().max_work_group_sizes[device];
        assert(work_group_size % (warp_size * warp_size) == 0);
        const sycl::range<3> block_dims(1, 1, work_group_size);
        int lsm_size =  block_dims[2] > warp_size ? work_group_size / warp_size * sizeof(float): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<float, 1> s_sum_acc_ct1(sycl::range<1>(lsm_size),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(blocks_num * block_dims,
                    block_dims),
                [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_group_size(warp_size)]] {
                    l2_norm_f32<warp_size>(x, dst, ncols,
                        src_stride_col, src_stride_row, src_stride_channel, src_stride_sample,
                        dst_stride_col, dst_stride_row, dst_stride_channel, dst_stride_sample,
                        eps, item_ct1, get_pointer(s_sum_acc_ct1), work_group_size);
                });
            });
    }
}

// Batched L2 norm: N independent, same-shape, contiguous F32 tensors normalized in
// one launch. The GDN layer emits q/k (and, across adjacent GDN layers, more)
// l2_norms as consecutive independent siblings; on the latency-bound decode path
// collapsing N launches into 1 removes per-launch overhead. The tensor index is
// folded into grid dim0; each row's reduction is identical to the single-tensor
// kernel, so the result is bit-exact.
struct l2_batch_ptrs {
    const float * src[GGML_SYCL_L2_BATCH_MAX];
    float *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// The batch shares one stride set: the caller only groups tensors whose nb[] all match,
// so the per-tensor state stays two pointers and the kernel argument block stays small.
// Strides are mandatory rather than an extra feature -- the GDN q/k norms this exists for
// read strided views of the fused qkv buffer, never contiguous memory.
struct l2_batch_strides {
    int     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

template <int warp_size>
static void l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, const int ncols, const float eps,
                              const sycl::nd_item<3> & item_ct1) {
    const int t   = item_ct1.get_group(0);  // tensor index
    const int r   = item_ct1.get_group(2);  // flattened row over ne1*ne2*ne3
    const int tid = item_ct1.get_local_id(2);

    const int i1 = r % st.ne1;
    const int i2 = (r / st.ne1) % st.ne2;
    const int i3 = r / (st.ne1 * st.ne2);

    const float * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    float *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += warp_size) {
        const float xi = x[col * st.ss0];
        tmp += xi * xi;
    }
    tmp = block_reduce<block_reduce_method::SUM, warp_size>(tmp, (float *) nullptr, warp_size);
    const float scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    for (int col = tid; col < ncols; col += warp_size) {
        dst[col * st.ds0] = scale * x[col * st.ss0];
    }
}

template <int warp_size>
static void l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, const int n_tensors,
                                   const int ncols, const int nrows_total, const float eps,
                                   queue_ptr stream) {
    const dpct::dim3 blocks_num(nrows_total, 1, n_tensors);
    const dpct::dim3 block_dims(warp_size, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(blocks_num * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(warp_size)]] {
                l2_norm_f32_batch<warp_size>(p, st, ncols, eps, item_ct1);
            });
    });
}

void ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);
    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = nb0 / tdst;
    const int64_t ds1 = nb1 / tdst;
    const int64_t ds2 = nb2 / tdst;
    const int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

void ggml_sycl_op_group_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    int num_groups = dst->op_params[0];
    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params + 1, sizeof(float));

    int group_size = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + num_groups - 1) / num_groups);
    group_norm_f32_sycl(src0_dd, dst_dd, num_groups, eps, group_size, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_stream, ctx.device);
}

void ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    const ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const float * src0_dd = static_cast<const float *>(dst->src[0]->data);
    float *       dst_dd  = static_cast<float *>(dst->data);

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));

    GGML_TENSOR_UNARY_OP_LOCALS
    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = nb0 / tdst;
    const int64_t ds1 = nb1 / tdst;
    const int64_t ds2 = nb2 / tdst;
    const int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_stream, ctx.device);
}

// Gate + register for the producer-side q8_1 emission. Returns the buffer to write into, or
// nullptr with a reason on stderr under GGML_SYCL_NORM_EMIT_Q8_TRACE=1 -- an attempt that
// yields exactly zero has to be able to say which condition rejected it.
static char * norm_q8_emit_target(ggml_backend_sycl_context & ctx, const ggml_tensor * out,
                                  int64_t ne00, int64_t ne01, int64_t ne02, int64_t ne03,
                                  int64_t d00, const void * strm, const char * site, int * q8_kx,
                                  int * q8_row_stride) {
    static const int emit_q8 = ggml_sycl_get_env("GGML_SYCL_NORM_EMIT_Q8", 1);
    static const int trace   = ggml_sycl_get_env("GGML_SYCL_NORM_EMIT_Q8_TRACE", 0);
    *q8_kx         = 0;
    *q8_row_stride = 0;
    if (emit_q8 != 1) { return nullptr; }

    const char * why = nullptr;
    int bs = 0;
    if (ne02 != 1 || ne03 != 1)               { why = "multi-plane"; }
    // Emit only where a consumer could possibly use it. can_use_mul_mat_vec_q requires
    // src1->ne[1] <= MMVQ_MAX_BATCH_SIZE; wider than that the mat-mul takes the f16 GEMM path,
    // which never looks up the q8_1 cache, so the emission is pure wasted traffic. Measured:
    // without this bound a `-d 8192` run emits ne=[5120,512] and ne=[5120,16] during its
    // prefill and single-token decode goes 26.29 -> 26.23 (3 pairs), turning a win into a loss.
    else if (ne01 > MMVQ_MAX_BATCH_SIZE)      { why = "gemm-width"; }
    else if (ne00 % QK8_1 != 0)               { why = "ne00%QK8_1"; }
    else if (!ggml_is_contiguous(out))        { why = "noncontig"; }
    else if (d00 != 1)                        { why = "d00!=1"; }
    else {
        bs = ne00 < 1024 ? WARP_SIZE : ggml_sycl_info().max_work_group_sizes[ctx.device];
        if (ne00 % bs != 0) { why = "ne00%bs"; }
    }
    if (why != nullptr) {
        if (trace) {
            fprintf(stderr, "[NQ8] %s decline=%s ne=[%ld,%ld,%ld,%ld] bs=%d\n", site, why,
                    (long) ne00, (long) ne01, (long) ne02, (long) ne03, bs);
        }
        return nullptr;
    }

    const int64_t padded  = GGML_PAD(ne00, MATRIX_ROW_PADDING);
    const size_t  rstride = (size_t) padded * sizeof(block_q8_1) / QK8_1;
    const size_t  bytes   = rstride * (size_t) ne01;
    const size_t  src_ne = (size_t) ggml_nelements(out);
    char * p = ggml_sycl_src1_q8_store_ext(ctx, (const void *) out->data, strm, bytes, src_ne, /*SoA=*/1, out);
    if (trace) {
        fprintf(stderr, "[NQ8] %s EMIT ne=[%ld,%ld] bytes=%zu src_ne=%zu key=%p\n", site,
                (long) ne00, (long) ne01, bytes, src_ne, (const void *) out->data);
    }
    *q8_kx         = (int) ne00;
    *q8_row_stride = (int) rstride;
    return p;
}

void ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    const ggml_tensor * rms_norm_src = dst->src[0];
    float eps = 0.0f;
    memcpy(&eps, dst->op_params, sizeof(float));

    const float *       src0_dd = static_cast<const float *>(rms_norm_src->data);
    const float *       mul_dd  = nullptr;
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }
    float * dst_dd = static_cast<float *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    const int64_t ne00 = rms_norm_src->ne[0];
    const int64_t ne01 = rms_norm_src->ne[1];
    const int64_t ne02 = rms_norm_src->ne[2];
    const int64_t ne03 = rms_norm_src->ne[3];

    const size_t ts0 = ggml_type_size(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    const int64_t s00 = rms_norm_src->nb[0] / ts0;
    const int64_t s01 = rms_norm_src->nb[1] / ts0;
    const int64_t s02 = rms_norm_src->nb[2] / ts0;
    const int64_t s03 = rms_norm_src->nb[3] / ts0;

    const size_t tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    const int64_t d00 = mul_tensor->nb[0] / tdst;
    const int64_t d01 = mul_tensor->nb[1] / tdst;
    const int64_t d02 = mul_tensor->nb[2] / tdst;
    const int64_t d03 = mul_tensor->nb[3] / tdst;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    int    q8_kx         = 0;
    int    q8_row_stride = 0;
    char * q8_out = norm_q8_emit_target(ctx, mul_tensor, ne00, ne01, ne02, ne03, d00,
                                        (const void *) main_stream, "fused", &q8_kx,
                                        &q8_row_stride);

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device,
        nullptr, nullptr, nullptr, q8_out, q8_kx, q8_row_stride);
}

void ggml_sycl_op_rms_norm_fused_add(ggml_backend_sycl_context & ctx, ggml_tensor * rms_norm,
                                     ggml_tensor * mul_tensor, ggml_tensor * add_tensor) {
    // add_tensor is the residual sum feeding rms_norm; it stays a written output because a
    // later residual consumes it. Folding it here removes one kernel launch and one full
    // DRAM round-trip of the sum per norm.
    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));

    const ggml_tensor * add_a = add_tensor->src[0];
    const ggml_tensor * add_b = add_tensor->src[1];

    const float * src0_dd = static_cast<const float *>(add_a->data);
    const float * add_b_dd = static_cast<const float *>(add_b->data);
    float *       add_dd   = static_cast<float *>(add_tensor->data);

    const float *       mul_dd  = nullptr;
    const ggml_tensor * mul_src = nullptr;
    if (mul_tensor->src[0] == rms_norm) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[1]->data);
        mul_src = mul_tensor->src[1];
    } else if (mul_tensor->src[1] == rms_norm) {
        mul_dd  = static_cast<const float *>(mul_tensor->src[0]->data);
        mul_src = mul_tensor->src[0];
    } else {
        GGML_ASSERT(false);
    }
    float * dst_dd = static_cast<float *>(mul_tensor->data);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_ASSERT(add_a->type == GGML_TYPE_F32);
    GGML_ASSERT(add_b->type == GGML_TYPE_F32);
    GGML_ASSERT(add_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    // the kernel indexes all three add operands with the sum's strides
    GGML_ASSERT(ggml_are_same_shape(add_a, add_b) && ggml_are_same_shape(add_a, add_tensor));
    GGML_ASSERT(ggml_is_contiguous(add_a) && ggml_is_contiguous(add_b) && ggml_is_contiguous(add_tensor));

    const int64_t ne00 = add_tensor->ne[0];
    const int64_t ne01 = add_tensor->ne[1];
    const int64_t ne02 = add_tensor->ne[2];
    const int64_t ne03 = add_tensor->ne[3];

    const size_t ts0 = ggml_type_size(add_tensor->type);
    const int64_t s00 = add_tensor->nb[0] / ts0;
    const int64_t s01 = add_tensor->nb[1] / ts0;
    const int64_t s02 = add_tensor->nb[2] / ts0;
    const int64_t s03 = add_tensor->nb[3] / ts0;

    const size_t tdst = ggml_type_size(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    const int64_t d00 = mul_tensor->nb[0] / tdst;
    const int64_t d01 = mul_tensor->nb[1] / tdst;
    const int64_t d02 = mul_tensor->nb[2] / tdst;
    const int64_t d03 = mul_tensor->nb[3] / tdst;

    const size_t ts_mul = ggml_type_size(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    const int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    const int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    const int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    const int mul_nrows     = mul_src->ne[1];
    const int mul_nchannels = mul_src->ne[2];
    const int mul_nsamples  = mul_src->ne[3];

    // The span's output is the next f16 GEMM's activation in every transformer block here.
    // Emit its f16 copy from the register rather than letting that GEMM re-read the whole
    // tensor to make one. Contiguity is what lets the mirror share dst's own indexing.
    sycl::half * mir = nullptr;
    if (ggml_is_contiguous(mul_tensor)) {
        mir = ggml_sycl_f16_mirror_for_next_matmul(ctx, mul_tensor,
                                                   (size_t) ggml_nelements(mul_tensor));
    }

    int    q8_kx         = 0;
    int    q8_row_stride = 0;
    char * q8_out = norm_q8_emit_target(ctx, mul_tensor, ne00, ne01, ne02, ne03, d00,
                                        (const void *) main_stream, "fused_add", &q8_kx,
                                        &q8_row_stride);

    rms_norm_mul_f32_sycl<true>(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_stream, ctx.device,
        add_b_dd, add_dd, mir, q8_out, q8_kx, q8_row_stride);
}

void ggml_sycl_op_rms_norm_back(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    float eps = 1e-5f;
    std::memcpy(&eps, dst->op_params, sizeof(float));
    if (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    const float * g_base  = static_cast<const float *>(dst->src[0]->data); // dz
    const float * x_base  = static_cast<const float *>(dst->src[1]->data); // x
          float * dx_base = static_cast<      float *>(dst->data);

    const int64_t D  = dst->ne[0];
    const int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (void) n3;
    const int64_t N  = ggml_nrows(dst);
    if (D == 0 || N == 0) return;

    const ggml_tensor *G = dst->src[0];
    const ggml_tensor *X = dst->src[1];
    const int ts = (int) ggml_type_size(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    const int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    const int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    const int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::queue_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    const int device_max_wg = ggml_sycl_info().max_work_group_sizes[ctx.device];
    auto roundup = [](int v, int m) { return ((v + m - 1) / m) * m; };
    int wg_cap = 256;
    if (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    int WG = std::max(WARP_SIZE, std::min(roundup((int)std::min<int64_t>(D, wg_cap), WARP_SIZE), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        const int nwarps_loc = std::max(1, WG / WARP_SIZE);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        auto l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        auto l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                const int row = item_ct1.get_group(2);
                const int tid = item_ct1.get_local_id(2);

                const int64_t i1 = row % n1;
                const int64_t i2 = (row / n1) % n2;
                const int64_t i3 = row / (n1 * n2);

                const float *__restrict x_row = x_base + i3 * xs3 + i2 * xs2 + i1 * xs1;
                const float *__restrict g_row = g_base + i3 * gs3 + i2 * gs2 + i1 * gs1;
                float *__restrict d_row       = dx_base + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                float sum_xx = 0.f, sum_xg = 0.f;
#ifndef GGML_SYCL_RMS_BACK_FAST
                float c_xx = 0.f, c_xg = 0.f;
#endif
                for (int64_t col = tid; col < D; col += WG) {
                    const float xv = x_row[col];
                    const float gv = g_row[col];
#ifdef GGML_SYCL_RMS_BACK_FAST
                    sum_xx += xv * xv;
                    sum_xg += xv * gv;
#else
                    float y1 = xv * xv - c_xx;
                    float t1 = sum_xx + y1;
                    c_xx = (t1 - sum_xx) - y1;
                    sum_xx = t1;

                    float y2 = xv * gv - c_xg;
                    float t2 = sum_xg + y2;
                    c_xg = (t2 - sum_xg) - y2;
                    sum_xg = t2;
#endif
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(sum_xx,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xx
#else
                    0.f
#endif
                );
                sycl::float2 xg = sycl::float2(sum_xg,
#ifndef GGML_SYCL_RMS_BACK_FAST
                    c_xg
#else
                    0.f
#endif
                );
                xx = warp_reduce_sum(xx, item_ct1);
                xg = warp_reduce_sum(xg, item_ct1);

                // cross-warp reduction using local memory (single barrier)
                const auto sub_group = item_ct1.get_sub_group();
                const auto sg_id     = sub_group.get_group_linear_id();
                const auto wi_in_sg  = sub_group.get_local_linear_id();
                const int nthreads   = item_ct1.get_local_range(2);
                const int nwarps     = nthreads / WARP_SIZE;

                sycl::float2 xx_total = xx;
                sycl::float2 xg_total = xg;
                if (nwarps > 1) {
                    if (wi_in_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    item_ct1.barrier(sycl::access::fence_space::local_space);

                    if (sg_id == 0) {
                        const unsigned wi_u = wi_in_sg;
                        sycl::float2 xx_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_first = (wi_u < static_cast<unsigned>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_total = warp_reduce_sum(xx_first, item_ct1);
                        xg_total = warp_reduce_sum(xg_first, item_ct1);
                    } else {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_total = xx;
                        xg_total = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                float inv_r = 0.f;
                float coeff = 0.f;
                if (tid == 0) {
                    const float sum_xx_f  = xx_total.x() + xx_total.y();
                    const float sum_xdz_f = xg_total.x() + xg_total.y();
                    const float mean_eps  = sum_xx_f / (float) D + eps;
                    const float sum_eps   = sum_xx_f + eps * (float) D;
                    inv_r = sycl::rsqrt(mean_eps);
                    coeff = -sum_xdz_f / sum_eps;
                }
                inv_r = sycl::group_broadcast(item_ct1.get_group(), inv_r);
                coeff = sycl::group_broadcast(item_ct1.get_group(), coeff);

                for (int64_t col = tid; col < D; col += WG) {
                    d_row[col] = (g_row[col] + coeff * x_row[col]) * inv_r;
                }
            });
    });

}

void ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *) src0->data;
    float * dst_d = (float *) dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    float eps;
    memcpy(&eps, dst->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    const size_t ts0 = ggml_type_size(src0->type);
    const size_t tdst = ggml_type_size(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    const int64_t ss0 = nb00 / ts0;
    const int64_t ss1 = nb01 / ts0;
    const int64_t ss2 = nb02 / ts0;
    const int64_t ss3 = nb03 / ts0;
    const int64_t ds0 = nb0 / tdst;
    const int64_t ds1 = nb1 / tdst;
    const int64_t ds2 = nb2 / tdst;
    const int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_SIZE>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// Batched entry: nodes[0..count) are independent, same-shape, contiguous F32 L2_NORM
// ops with identical eps (validated by the caller). Requires ncols < 1024 (the warp
// reduction path); the caller must not batch wider rows.
void ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, int count) {
    const ggml_tensor * s0 = nodes[0]->src[0];
    const int ncols       = (int) s0->ne[0];
    const int nrows_total = (int) ggml_nrows(s0);
    float eps;
    memcpy(&eps, nodes[0]->op_params, sizeof(float));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    for (int t = 0; t < count; ++t) {
        p.src[t] = (const float *) nodes[t]->src[0]->data;
        p.dst[t] = (float *) nodes[t]->data;
    }

    const ggml_tensor * d0 = nodes[0];
    const size_t        ts = ggml_type_size(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (int) s0->ne[1];
    st.ne2 = (int) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_SIZE>(p, st, count, ncols, nrows_total, eps, ctx.stream());
}

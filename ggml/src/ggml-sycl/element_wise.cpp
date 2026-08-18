#include "common.hpp"
#include "ggml-sycl/presets.hpp"
#include "ggml.h"
#include "element_wise.hpp"

#define SYCL_GLOBAL_ID_LOOP(K, ITEM) \
    for (auto i = ITEM.get_global_id(0); i < (size_t)K; i += ITEM.get_global_range(0))

#define SYCL_LOCAL_ID_CALC(ITEM, IDX) \
    (ITEM.get_local_range(IDX) * ITEM.get_group(IDX) + ITEM.get_local_id(IDX))

static void acc_f32(const char * x, const char * y, float * dst, const int64_t ne,
    const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
    const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
    const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
    const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
    const int64_t s11, const int64_t s12, const int64_t s13, const int64_t offset) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    const int64_t i = SYCL_LOCAL_ID_CALC(item_ct1, 2);

    if (i >= ne) {
        return;
    }

    int64_t src1_idx = i - offset;

    int64_t tmp = src1_idx;
    const int64_t i13 = tmp / s13;
    tmp -= i13 * s13;
    const int64_t i12 = tmp / s12;
    tmp -= i12 * s12;
    const int64_t i11 = tmp / s11;
    tmp -= i11 * s11;
    const int64_t i10 = tmp;

    int64_t tmp_dst = i;
    const int64_t i3 = tmp_dst / (ne2*ne1*ne0);
    tmp_dst -= i3 * (ne2*ne1*ne0);
    const int64_t i2 = tmp_dst / (ne1*ne0);
    tmp_dst -= i2 * (ne1*ne0);
    const int64_t i1 = tmp_dst / ne0;
    tmp_dst -= i1 * ne0;
    const int64_t i0 = tmp_dst;

    float val = *(const float *) (x + i0*nb00 + i1*nb01 + i2*nb02 + i3*nb03);
    if (src1_idx >= 0 && i10 < ne10 && i11 < ne11 && i12 < ne12 && i13 < ne13) {
        val += *(const float *) (y + i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13);
    }
    dst[i] = val;
}

/* Unary OP funcs */
template<typename T>
static __dpct_inline__ T op_sgn(T x) {
    return x > static_cast<T>(0.f) ? static_cast<T>(1.f) : ((x < static_cast<T>(0.f) ? static_cast<T>(-1.f) : static_cast<T>(0.f)));
}


template<typename T>
static __dpct_inline__ T op_abs(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::fabs(x);  // or experimental namespace if needed
    } else {
        return sycl::fabs(x);
    }
}

template<typename T>
static __dpct_inline__ T op_expm1(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::expm1(static_cast<float>(x))
        );
    } else {
        return sycl::expm1(x);
    }
}

template<typename T>
static __dpct_inline__ T op_elu(T x) {
    return (x > static_cast<T>(0.f)) ? x : op_expm1(x);
}

template<typename T>
static __dpct_inline__ T op_erf(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::erf(static_cast<float>(x))
        );
    } else {
        return sycl::erf(x);
    }
}

template<typename T>
static __dpct_inline__ T op_gelu_erf(T x) {
    const T SQRT_2_INV = static_cast<T>(0.70710678118654752440084436210484f);
    return static_cast<T>(0.5f) * x * (static_cast<T>(1.0f) + op_erf(x * SQRT_2_INV));
}

template<typename T>
static __dpct_inline__ T op_gelu_quick(T x) {
    const T GELU_QUICK_COEF_LOCAL = static_cast<T>(-1.702f);
    return x * (static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(GELU_QUICK_COEF_LOCAL * x)));
}

template<typename T>
static __dpct_inline__ T op_relu(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0));
    } else {
        return sycl::fmax(x, static_cast<T>(0));
    }
}

template<typename T>
static __dpct_inline__ T op_sigmoid(T x) {
    return static_cast<T>(1.0f) / (static_cast<T>(1.0f) + op_exp(-x));
}

template<typename T>
static __dpct_inline__ T op_sqrt(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::sqrt(x);
    } else {
        return sycl::sqrt(x);
    }
}

template<typename T>
static __dpct_inline__ T op_sin(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::sin(x);
    } else {
        return sycl::sin(x);
    }
}

template<typename T>
static __dpct_inline__ T op_cos(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::cos(x);
    } else {
        return sycl::cos(x);
    }
}

template<typename T>
static __dpct_inline__ T op_hardsigmoid(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::fmin(
            static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(
                                      static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } else {
        return sycl::fmin(static_cast<T>(1.0f),
                          sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
static __dpct_inline__ T op_hardswish(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return x * sycl::ext::oneapi::experimental::fmin(static_cast<T>(1.0f), sycl::ext::oneapi::experimental::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    } else {
        return x * sycl::fmin(static_cast<T>(1.0f), sycl::fmax(static_cast<T>(0.0f), (x + static_cast<T>(3.0f)) / static_cast<T>(6.0f)));
    }
}

template<typename T>
static __dpct_inline__ T op_log(T x) {
    if (x <= static_cast<T>(0)) {
        return neg_infinity<T>();
    }
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::log(x);
    } else {
        return sycl::log(x);
    }
}

template<typename T>
static __dpct_inline__ T op_softplus(T x) {
    const float xf = (float) x;
    const float ax = op_abs(xf);
    const float m  = sycl::fmax(xf, 0.0f);
    const float y  = m + sycl::log1p(sycl::exp(-ax));
    return (T) y;
}

template<typename T>
static __dpct_inline__ T op_neg(T x) {
    return -x;
}

template<typename T>
static __dpct_inline__ T op_step(T x) {
    return (x > static_cast<T>(0.0f)) ? static_cast<T>(1.0f) : static_cast<T>(0.0f);
}

template<typename T>
static __dpct_inline__ T op_leaky_relu(T x, float negative_slope) {
    T neg_slope_T = static_cast<T>(negative_slope);
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::fmax(x, static_cast<T>(0)) +
           sycl::ext::oneapi::experimental::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;

    } else {
        return sycl::fmax(x, static_cast<T>(0)) +
           sycl::fmin(x, static_cast<T>(0.0f)) * neg_slope_T;
    }
}

template<typename T>
static __dpct_inline__ T op_xielu(T x, float alpha_n, float alpha_p, float beta, float eps) {
    const float xi        = static_cast<float>(x);
    const float gate_pos  = (xi > 0.0f);
    const float y_pos     = alpha_p * xi * xi + beta * xi;
    const float min_v_eps = sycl::fmin(xi, eps);
    const float y_neg     = (sycl::expm1(min_v_eps) - xi) * alpha_n + beta * xi;
    const float out       = gate_pos * y_pos + (1.0f - gate_pos) * y_neg;
    return static_cast<T>(out);
}

template<typename T>
static __dpct_inline__ T op_sqr(T x) {
    return x * x;
}

template<typename T>
static __dpct_inline__ T op_clamp(T x, float min_val, float max_val) {
    return x < static_cast<T>(min_val) ? static_cast<T>(min_val) : (x > static_cast<T>(max_val) ? static_cast<T>(max_val) : x);
}

template<typename T>
static __dpct_inline__ T op_floor(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::floor(x);
    } else {
        return sycl::floor(x);
    }
}

template<typename T>
static __dpct_inline__ T op_ceil(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::ceil(x);
    } else {
        return sycl::ceil(x);
    }
}

template<typename T>
static __dpct_inline__ T op_round(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return static_cast<sycl::ext::oneapi::bfloat16>(
            sycl::round(static_cast<float>(x))
        );
    } else {
        return sycl::round(x);
    }
}

template<typename T>
static __dpct_inline__ T op_trunc(T x) {
    if constexpr (std::is_same_v<T, sycl::ext::oneapi::bfloat16>) {
        return sycl::ext::oneapi::experimental::trunc(x);
    } else {
        return sycl::trunc(x);
    }
}

template<typename T, typename F>
static void unary_op_flat_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = func(x[i]);
    }
}

template<typename T, typename F>
static void unary_op_generic_kernel(
        const T * x,
        T * dst,
        const int k,
        const sycl::uint3 ne0_fd, const sycl::uint3 ne1_fd, const sycl::uint3 ne2_fd,
        const size_t nb0,  const size_t nb1,  const size_t nb2,  const size_t nb3,
        const size_t nbd0, const size_t nbd1, const size_t nbd2, const size_t nbd3,
        const sycl::nd_item<1> & item_ct1,
        F func) {

    // 32-bit index math: k is int, so every logical index fits u32. 64-bit integer div/mod is
    // emulated on Xe and dominates this kernel otherwise, and even the 32-bit divide is worth
    // avoiding -- the divisors are launch-invariant, so the magic numbers are precomputed
    // host-side and each division becomes a multiply-high plus a shift.
    // Byte offsets are widened back to size_t only for the final address math.
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        sycl::uint2 dm = fast_div_modulo((uint32_t) i, ne0_fd);
        const uint32_t i0 = dm.y();
        dm = fast_div_modulo(dm.x(), ne1_fd);
        const uint32_t i1 = dm.y();
        dm = fast_div_modulo(dm.x(), ne2_fd);
        const uint32_t i2 = dm.y();
        const uint32_t i3 = dm.x();

        const char * src_base = (const char *) x;
        char       * dst_base = (char *) dst;

        const T * srcp = (const T *)(src_base + (size_t) i0*nb0  + (size_t) i1*nb1  + (size_t) i2*nb2  + (size_t) i3*nb3 );
        T *       dstp = (T *)(dst_base + (size_t) i0*nbd0 + (size_t) i1*nbd1 + (size_t) i2*nbd2 + (size_t) i3*nbd3);

        *dstp = func(*srcp);
    }
}

template<typename T>
static void unary_op_sqrt_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqrt(x[i]);
    }
}

template<typename T>
static void unary_op_sin_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sin(x[i]);
    }
}

template<typename T>
static void unary_op_cos_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_cos(x[i]);
    }
}

template<typename T>
static void unary_op_log_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_log(x[i]);
    }
}


template<typename T>
static void unary_op_leaky_relu_kernel(const T * x, T * dst, const int k, float negative_slope, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_leaky_relu(x[i], negative_slope);
    }
}

template<typename T>
static void unary_op_xielu_kernel(const T * x, T * dst, const int k, float alpha_n, float alpha_p, float beta, float eps, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_xielu(x[i], alpha_n, alpha_p, beta, eps);
    }
}

template<typename T>
static void unary_op_sqr_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_sqr(x[i]);
    }
}

template<typename T>
static void unary_op_clamp_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1, float min_val, float max_val) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_clamp(x[i], min_val, max_val);
    }
}

template<typename T>
static void unary_op_ceil_kernel(const T * x, T * dst, const int k, const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = op_ceil(x[i]);
    }
}

template<typename T>
static void clamp(const T * x, T * dst, const float min, const float max, const int k,
                      const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = x[i] < static_cast<T>(min) ? static_cast<T>(min) : (x[i] > static_cast<T>(max) ? static_cast<T>(max) : x[i]);
    }
}

// MIRROR additionally writes an f16 copy of each result. The value is already in a register,
// so the mirror costs one narrow store and no extra read; it exists to delete the standalone
// f32->f16 pass the consuming f16 GEMM would otherwise run over the whole tensor.
// Q8 additionally emits the q8_1 SoA copy of the result -- the dual of MIRROR for the
// decode side, where the consuming mat-vec would otherwise launch a standalone
// QUANTIZE/src1 kernel (5.3 us of pure launch latency at decode, 0123's numbers). Same
// block-per-sub-group recipe as the norm's emission in 0123: the value is re-read from
// dst after a work-group barrier (proven md5-identical to the register in 0123), and a
// sub-group enters or skips a whole block as a unit so every reduce_over_group is
// full-width -- the one-column-per-lane form silently reduces over HALF a block on
// Intel's 16-wide sub-groups.
template<typename T>
static __dpct_inline__ void producer_q8_emit(const T * dst, char * q8, const int64_t k,
                                        const int kx, const sycl::nd_item<1> & it) {
    if constexpr (std::is_same_v<T, float>) {
        it.barrier(sycl::access::fence_space::global_and_local);
        const auto    sg   = it.get_sub_group();
        const int     lane = (int) it.get_local_id(0) % WARP_SIZE;
        const int     sgid = (int) it.get_local_id(0) / WARP_SIZE;
        const int     nsg  = (int) it.get_local_range(0) / WARP_SIZE;
        constexpr int EPW  = QK8_1 / WARP_SIZE;
        const int64_t base = (int64_t) it.get_group(0) * it.get_local_range(0);
        const int64_t lim  = base + (int64_t) it.get_local_range(0) < k ? base + (int64_t) it.get_local_range(0) : k;
        const int64_t rowbytes = kx + (kx / QK8_1) * (int) sizeof(sycl::half2);
        for (int64_t b32 = base + (int64_t) sgid * QK8_1; b32 + QK8_1 <= lim; b32 += (int64_t) nsg * QK8_1) {
            float vv[EPW];
            float sum = 0.0f, amax = 0.0f;
#pragma unroll
            for (int e = 0; e < EPW; e++) {
                vv[e] = (float) dst[b32 + lane * EPW + e];
                sum  += vv[e];
                amax  = sycl::fmax(amax, sycl::fabs(vv[e]));
            }
            sum  = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
            amax = sycl::reduce_over_group(sg, amax, sycl::maximum<float>());
            const float   d    = amax == 0.0f ? 1.0f : amax / 127.0f;
            const int64_t row  = b32 / kx;
            const int     colb = (int) (b32 - row * kx);
            char *        rowp = q8 + row * rowbytes;
#pragma unroll
            for (int e = 0; e < EPW; e++) {
                ((int8_t *) rowp)[colb + lane * EPW + e] = static_cast<int8_t>(sycl::round(vv[e] / d));
            }
            if (lane == 0) {
                *(sycl::half2 *) (rowp + kx + (colb / QK8_1) * (int) sizeof(sycl::half2)) =
                    sycl::half2(sycl::half(amax == 0.0f ? 0.0f : d), sycl::half(sum));
            }
        }
    }
}

template<bool MIRROR, bool Q8, typename T, typename F>
static void unary_gated_op_flat_kernel(const T * x, const T * g, T * dst, sycl::half * mir,
                                       char * q8, const int q8_kx,
                                       const uint64_t k, const sycl::nd_item<1> & item_ct1, F func) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const T v = func(x[i]) * g[i];
        dst[i]    = v;
        if constexpr (MIRROR) {
            mir[i] = static_cast<sycl::half>(v);
        }
    }
    if constexpr (Q8) {
        producer_q8_emit(dst, q8, (int64_t) k, q8_kx, item_ct1);
    }
}

template<bool MIRROR, bool Q8, typename T, typename F>
static void unary_gated_op_generic_kernel(
        const T * x,
        const T * g,
        T * dst,
        sycl::half * mir,
        char * q8,
        const int q8_kx,
        const uint64_t k,
        const sycl::uint3 n_fd,
        const uint64_t o0,
        const uint64_t o1,
        const sycl::nd_item<1> & item_ct1,
        F func) {

    // rows of n columns at strides o0 and o1: two halves of one fused tensor, or two tensors
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        const int64_t j0 = rc.x() * o0 + rc.y();
        const int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        const T v = func(x[j0]) * g[j1];
        dst[i]    = v;
        if constexpr (MIRROR) {
            mir[i] = static_cast<sycl::half>(v);
        }
    }
    if constexpr (Q8) {
        producer_q8_emit(dst, q8, (int64_t) k, q8_kx, item_ct1);
    }
}

// Fused UNARY + MUL. Unlike the gated ops above, `x` and `g` are separate tensors of the
// same shape; `o0`/`o1` are their row strides in elements, so a half-view needs no repack.
// `dst` is contiguous and indexed flat. Math is done in f32, as the CPU and CUDA references do.
template<bool Q8, typename T, typename F>
static void unary_mul_flat_kernel(const T * x, const T * g, T * dst, char * q8, const int q8_kx,
                                  const int64_t k, const sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = (T) (op((float) x[i]) * (float) g[i]);
    }
    if constexpr (Q8) {
        producer_q8_emit(dst, q8, k, q8_kx, item_ct1);
    }
}

template<bool Q8, typename T, typename F>
static void unary_mul_strided_kernel(const T * x, const T * g, T * dst, char * q8, const int q8_kx,
                                     const int64_t k, const sycl::uint3 n_fd, const int64_t o0, const int64_t o1, const sycl::nd_item<1> &item_ct1, F op) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        const sycl::uint2 rc = fast_div_modulo((uint32_t) i, n_fd);
        const int64_t j0 = rc.x() * o0 + rc.y();
        const int64_t j1 = o0 == o1 ? j0 : rc.x() * o1 + rc.y();
        dst[i] = (T) (op((float) x[j0]) * (float) g[j1]);
    }
    if constexpr (Q8) {
        producer_q8_emit(dst, q8, k, q8_kx, item_ct1);
    }
}

template<typename T, typename F>
static void unary_mul_sycl(const T * x, const T * g, T * dst, const int64_t k, const int64_t n, const int64_t o0, const int64_t o1, queue_ptr main_stream, F op,
                           char * q8 = nullptr, int q8_kx = 0) {
    const size_t            num_blocks = ceil_div((size_t) k, (size_t) SYCL_GLU_BLOCK_SIZE);
    const sycl::nd_range<1> range(num_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

    // o0 == o1 == n makes (i/n)*o0 + (i%n) == i, so the strided kernel degenerates to the flat one
    if (o0 == n && o1 == n) {
        auto launch_flat = [&](auto q8_tag) {
            constexpr bool Q8 = decltype(q8_tag)::value;
            main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                unary_mul_flat_kernel<Q8>(x, g, dst, q8, q8_kx, k, item_ct1, op);
            });
        };
        if (q8 != nullptr) { launch_flat(std::true_type{}); } else { launch_flat(std::false_type{}); }
        return;
    }

    // 32-bit fastdiv, exact only below 2^31; ggml_sycl_can_fuse() already declined past that
    GGML_ASSERT(k < ((int64_t) 1 << 31));
    const sycl::uint3 n_fd = init_fastdiv_values((uint32_t) n);
    auto launch_strided = [&](auto q8_tag) {
        constexpr bool Q8 = decltype(q8_tag)::value;
        main_stream->parallel_for(range, [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            unary_mul_strided_kernel<Q8>(x, g, dst, q8, q8_kx, k, n_fd, o0, o1, item_ct1, op);
        });
    };
    if (q8 != nullptr) { launch_strided(std::true_type{}); } else { launch_strided(std::false_type{}); }
}

namespace ggml_sycl_detail {
static void acc_f32_sycl(const char *x, const char *y, float *dst,
                         const int64_t n_elements,
                         const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
                         const int64_t nb00, const int64_t nb01, const int64_t nb02, const int64_t nb03,
                         const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
                         const int64_t nb10, const int64_t nb11, const int64_t nb12, const int64_t nb13,
                         const int64_t s1, const int64_t s2, const int64_t s3,
                         const int64_t offset, queue_ptr stream) {
    const int num_blocks = (n_elements + SYCL_ACC_BLOCK_SIZE - 1) / SYCL_ACC_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_ACC_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> /*item_ct1*/) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                            acc_f32(x, y, dst, n_elements,
                                ne0, ne1, ne2, ne3,
                                nb00, nb01, nb02, nb03,
                                ne10, ne11, ne12, ne13,
                                nb10, nb11, nb12, nb13,
                                s1, s2, s3, offset);
                         });
}

template<typename T>
static void arange_kernel(T * dst, const int k, T start, T step,
                         const sycl::nd_item<1> &item_ct1) {
    SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
        dst[i] = start + static_cast<T>(i) * step;
    }
}

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_unary(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_BF16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    switch (dst->type) {
        case GGML_TYPE_F16:
            {
                auto data_pts = cast_data<sycl::half>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                break;
            }
#ifdef GGML_SYCL_HAS_BF16
        case GGML_TYPE_BF16:
            {
                auto data_pts = cast_data<sycl::ext::oneapi::bfloat16>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                break;
            }
#endif
        case GGML_TYPE_F32:
            {
                auto data_pts = cast_data<float>(dst);
                kernel_invoker(data_pts.src, data_pts.dst, (int)ggml_nelements(dst->src[0]), main_stream, std::forward<Args>(args)...);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename KernelInvoker, typename... Args>
static inline void dispatch_ggml_sycl_op_fused_glu(ggml_backend_sycl_context & ctx, ggml_tensor * dst, KernelInvoker kernel_invoker, Args&&... args) {
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->src[0]->type == dst->type);

    dpct::queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;;
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_is_contiguous_1(dst->src[0]));
    GGML_ASSERT(ggml_is_contiguous(dst));
    // The fused GLU kernels index with 32-bit fastdiv, which is exact only for indices below
    // 2^31. A dst that large is ~8 GB at f32, and the grid sizing already narrows to 32 bits,
    // so assert the bound rather than carry a second code path for it.
    GGML_ASSERT(ggml_nelements(dst) < ((int64_t) 1 << 31));
    const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }
    switch (dst->type) {
        case GGML_TYPE_F16:
            {
                sycl::half * src0_p = (sycl::half *) src0_d;
                sycl::half * src1_p = (sycl::half *) src1_d;

                    if (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }
                kernel_invoker(src0_p,
                               src1_p,
                               (sycl::half *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / sizeof(sycl::half),
                               src1_o / sizeof(sycl::half),
                               main_stream,
                               std::forward<Args>(args)...);
                break;
            }
        case GGML_TYPE_F32:
            {
                float * src0_p = (float *) src0_d;
                float * src1_p = (float *) src1_d;

                    if (!src1) {
                        src0_p += swapped ? nc : 0;
                        src1_p += swapped ? 0 : nc;
                    }

                kernel_invoker(src0_p,
                               src1_p,
                               (float *) dst_d,
                               ggml_nelements(dst),
                               nc,
                               src0_o / sizeof(float),
                               src1_o / sizeof(float),
                               main_stream,
                               std::forward<Args>(args)...);
                break;
            }
        default:
            GGML_ABORT("GGML tensor type not supported!\n");
    }
}

template<typename F>
static inline void ggml_sycl_op_unary(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    ggml_tensor * src0 = dst->src[0];

    const int64_t ne0  = dst->ne[0];
    const int64_t ne1  = dst->ne[1];
    const int64_t ne2  = dst->ne[2];

    const size_t  nb0  = src0->nb[0];
    const size_t  nb1  = src0->nb[1];
    const size_t  nb2  = src0->nb[2];
    const size_t  nb3  = src0->nb[3];

    const size_t  nbd0 = dst->nb[0];
    const size_t  nbd1 = dst->nb[1];
    const size_t  nbd2 = dst->nb[2];
    const size_t  nbd3 = dst->nb[3];

    // Hot unary ops (FFN/GDN silu, sigmoid, ...) run on contiguous tensors;
    // skip the strided index math entirely for them.
    const bool contiguous = ggml_is_contiguous(src0) && ggml_is_contiguous(dst);

    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [=](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {

            const int num_blocks = ceil_div(k_elements, 256);

            if (contiguous) {
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_flat_kernel(src, dst_ptr, k_elements, item_ct1, func);
                    });
            } else {
                // Launch-invariant divisors: compute the magic numbers once on the host so the
                // kernel never issues an integer divide. Only the strided path needs them.
                const sycl::uint3 ne0_fd = init_fastdiv_values((uint32_t) ne0);
                const sycl::uint3 ne1_fd = init_fastdiv_values((uint32_t) ne1);
                const sycl::uint3 ne2_fd = init_fastdiv_values((uint32_t) ne2);
                stream->parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(256),
                                      sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                        unary_op_generic_kernel(
                            src, dst_ptr, k_elements,
                            ne0_fd, ne1_fd, ne2_fd,
                            nb0, nb1, nb2, nb3,
                            nbd0, nbd1, nbd2, nbd3,
                            item_ct1,
                            func
                        );
                    });
            }
        });
}

template<typename F>
static inline void ggml_sycl_op_unary_gated(
        ggml_backend_sycl_context & ctx, ggml_tensor * dst, F func) {

    dispatch_ggml_sycl_op_fused_glu(ctx, dst,
        [func, &ctx, dst](const auto * x_ptr, const auto * g_ptr, auto * dst_ptr, uint64_t k, uint64_t n, uint64_t o0, uint64_t o1, queue_ptr main_stream) {

            const uint32_t num_blocks = (uint32_t) ceil_div(k, SYCL_GLU_BLOCK_SIZE);
            const sycl::nd_range<1> launch_range(num_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE),
                                                 sycl::range<1>(SYCL_GLU_BLOCK_SIZE));

            // An f32 result whose only consumer is an f16 GEMM is about to be re-read in full
            // and rewritten as f16. Emit that copy here instead, from the register.
            using dst_t = std::remove_cv_t<std::remove_pointer_t<decltype(dst_ptr)>>;
            sycl::half * mir    = nullptr;
            char *       q8_out = nullptr;
            int          q8_kx  = 0;
            if constexpr (std::is_same_v<dst_t, float>) {
                mir = ggml_sycl_f16_mirror_for_next_matmul(ctx, dst, (size_t) k);
                // Mutually exclusive by construction: the f16 mirror fires only for an f16 GEMM
                // consumer (prefill), the q8_1 emission only for a mat-vec consumer (decode).
                if (mir == nullptr) {
                    q8_out = ggml_sycl_q8_emit_for_next_matvec(ctx, dst, &q8_kx);
                }
            }

            // o0 == n and o1 == n make the index math the identity, so index flat
            // note: not ggml_is_contiguous - a fused [gate|up] src0 is contiguous with o0 == 2n
            auto launch = [&](auto mirror_tag, auto q8_tag) {
                constexpr bool MIR = decltype(mirror_tag)::value;
                constexpr bool Q8  = decltype(q8_tag)::value;
                if (o0 == n && o1 == n) {
                    main_stream->parallel_for(launch_range,
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                            unary_gated_op_flat_kernel<MIR, Q8>(x_ptr, g_ptr, dst_ptr, mir, q8_out, q8_kx, k, item_ct1, func);
                        });
                } else {
                    // launch-invariant divisor, and only this path needs it
                    const sycl::uint3 n_fd = init_fastdiv_values((uint32_t) n);
                    main_stream->parallel_for(launch_range,
                        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                            unary_gated_op_generic_kernel<MIR, Q8>(x_ptr, g_ptr, dst_ptr, mir, q8_out, q8_kx, k, n_fd, o0, o1, item_ct1, func);
                        });
                }
            };
            if (mir != nullptr)         { launch(std::true_type{},  std::false_type{}); }
            else if (q8_out != nullptr) { launch(std::false_type{}, std::true_type{}); }
            else                        { launch(std::false_type{}, std::false_type{}); }
        });
}


static inline void ggml_sycl_op_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    float start, stop, step;
    memcpy(&start, dst->op_params, sizeof(float));
    memcpy(&stop, (float *) dst->op_params + 1, sizeof(float));
    memcpy(&step, (float *) dst->op_params + 2, sizeof(float));
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));
    float * dst_ptr = (float *)dst->data;
    const int k = (int)ggml_nelements(dst);
    const int num_blocks = ceil_div(k, SYCL_ARANGE_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE),
                          sycl::range<1>(SYCL_ARANGE_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            arange_kernel(dst_ptr, k, start, step, item_ct1);
        });
}

} // namespace ggml_sycl_detail



static inline void ggml_sycl_op_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_sgn(x);
    });
}


static inline void ggml_sycl_op_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_abs(x);
    });
}

static inline void ggml_sycl_op_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_elu(x);
    });
}
static inline void ggml_sycl_op_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_silu(x);
    });
}

static inline void ggml_sycl_op_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_gelu(x);
    });
}

static inline void ggml_sycl_op_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_gelu_quick(x);
    });
}

static inline void ggml_sycl_op_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_gelu_erf(x);
    });
}

static inline void ggml_sycl_op_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_tanh(x);
    });
}

static inline void ggml_sycl_op_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_relu(x);
    });
}

static inline void ggml_sycl_op_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_hardsigmoid(x);
    });
}

static inline void ggml_sycl_op_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_hardswish(x);
    });
}

static inline void ggml_sycl_op_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_exp(x);
    });
}

static inline void ggml_sycl_op_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_expm1(x);
    });
}

static inline void ggml_sycl_op_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_EXP_BLOCK_SIZE); // Using EXP block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_EXP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_EXP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_log_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_softplus(x);
    });
}

static inline void ggml_sycl_op_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_neg(x);
    });
}


static inline void ggml_sycl_op_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_step(x);
    });
}

static inline void ggml_sycl_op_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_sigmoid(x);
    });
}

static inline void ggml_sycl_op_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQRT_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQRT_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQRT_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqrt_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sin_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SIN_BLOCK_SIZE); // Using SIN block size
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SIN_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SIN_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_cos_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    float negative_slope;
    memcpy(&negative_slope, dst->op_params, sizeof(float));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream, float slope) {
            const int num_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_leaky_relu_kernel(src, dst_ptr, k_elements, slope, item_ct1);
                });
        }, negative_slope);
}

static inline void ggml_sycl_op_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream) {
            const int num_blocks = ceil_div(k_elements, SYCL_SQR_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_SQR_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_SQR_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_sqr_kernel(src, dst_ptr, k_elements, item_ct1);
                });
        });
}

static inline void ggml_sycl_op_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    float min_val;
    float max_val;
    memcpy(&min_val, dst->op_params, sizeof(float));
    memcpy(&max_val, (float *) dst->op_params + 1, sizeof(float));
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream, float min_arg, float max_arg) {
            const int num_blocks = ceil_div(k_elements, SYCL_CLAMP_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_CLAMP_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    clamp(src, dst_ptr, min_arg, max_arg, k_elements, item_ct1);
                });
        }, min_val, max_val);
}

static inline void ggml_sycl_op_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const float alpha_n = ggml_get_op_params_f32(dst, 1);
    const float alpha_p = ggml_get_op_params_f32(dst, 2);
    const float beta    = ggml_get_op_params_f32(dst, 3);
    const float eps     = ggml_get_op_params_f32(dst, 4);
    ggml_sycl_detail::dispatch_ggml_sycl_op_unary(ctx, dst,
        [](const auto* src, auto* dst_ptr, int k_elements, queue_ptr stream, float alpha_n_arg, float alpha_p_arg, float beta_arg, float eps_arg) {
            const int num_blocks = ceil_div(k_elements, SYCL_RELU_BLOCK_SIZE);
            stream->parallel_for(
                sycl::nd_range<1>(sycl::range<1>(num_blocks) * sycl::range<1>(SYCL_RELU_BLOCK_SIZE),
                                  sycl::range<1>(SYCL_RELU_BLOCK_SIZE)),
                [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                    unary_op_xielu_kernel(src, dst_ptr, k_elements, alpha_n_arg, alpha_p_arg, beta_arg, eps_arg, item_ct1);
                });
        }, alpha_n, alpha_p, beta, eps);
}

static inline void ggml_sycl_op_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_floor(x);
    });
}

static inline void ggml_sycl_op_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_ceil(x);
    });
}

static inline void ggml_sycl_op_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_round(x);
    });
}

static inline void ggml_sycl_op_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary(ctx, dst, [](auto x) {
        return op_trunc(x);
    });
}

static inline void ggml_sycl_op_acc(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    const char  * src0_d = (const char  *) src0->data;
    const char  * src1_d = (const char  *) src1->data;
    float       * dst_d  = (float       *)  dst->data;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(dst->nb[0] == ggml_element_size(dst));
    GGML_ASSERT(ggml_is_contiguously_allocated(dst));
    GGML_ASSERT(ggml_are_same_shape(src0, dst));

    const int64_t s1     = (int64_t) ((const int32_t *) dst->op_params)[0] / (int64_t) sizeof(float);
    const int64_t s2     = (int64_t) ((const int32_t *) dst->op_params)[1] / (int64_t) sizeof(float);
    const int64_t s3     = (int64_t) ((const int32_t *) dst->op_params)[2] / (int64_t) sizeof(float);
    const int64_t offset = (int64_t) ((const int32_t *) dst->op_params)[3] / (int64_t) sizeof(float);

    ggml_sycl_detail::acc_f32_sycl(src0_d, src1_d, dst_d, ggml_nelements(dst),
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3],
        src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
        src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3],
        s1, s2, s3, offset, stream);
}

static inline void ggml_sycl_op_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        return op_gelu(x);
    });
}

static inline void ggml_sycl_op_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        return op_relu(x);
    });
}

static inline void ggml_sycl_op_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        return op_silu(x);
    });
}

// dst = op(unary_node->src[0]) * other, written straight to the MUL output, saving the
// standalone unary launch. Preconditions come from ggml_sycl_can_fuse(); re-asserted here.
void ggml_sycl_op_unary_mul_fused(ggml_backend_sycl_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    scope_op_debug_print scope_dbg_print(__func__, mul_node, /*num_src=*/2);

    const ggml_tensor * x = unary_node->src[0];
    const ggml_tensor * g = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    // g is picked by elimination; ggml_can_fuse()'s single-use rule rules out MUL(unary, unary)
    GGML_ASSERT(g != unary_node);
    GGML_ASSERT(x->type == g->type && x->type == mul_node->type);
    GGML_ASSERT(ggml_are_same_shape(x, g) && ggml_are_same_shape(x, mul_node));
    GGML_ASSERT(ggml_is_contiguous_1(x) && ggml_is_contiguous_1(g));
    // dst is indexed flat
    GGML_ASSERT(ggml_is_contiguous(mul_node));

    queue_ptr main_stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    const int64_t k = ggml_nelements(mul_node);
    const int64_t n = mul_node->ne[0];

    // Same producer-side q8_1 emission as the GLU (0143): attn_gated is ssm_out /
    // attn_output's src1, whose consumer otherwise launches a standalone quantize.
    int    q8_kx  = 0;
    char * q8_out = ggml_sycl_q8_emit_for_next_matvec(ctx, mul_node, &q8_kx);

    const auto dispatch_type = [&](auto op) {
        switch (mul_node->type) {
            case GGML_TYPE_F32:
                unary_mul_sycl((const float *) x->data, (const float *) g->data, (float *) mul_node->data,
                               k, n, x->nb[1] / sizeof(float), g->nb[1] / sizeof(float), main_stream, op,
                               q8_out, q8_kx);
                break;
            case GGML_TYPE_F16:
                unary_mul_sycl((const sycl::half *) x->data, (const sycl::half *) g->data, (sycl::half *) mul_node->data,
                               k, n, x->nb[1] / sizeof(sycl::half), g->nb[1] / sizeof(sycl::half), main_stream, op);
                break;
            default:
                GGML_ABORT("fused unary+mul: unsupported type %s", ggml_type_name(mul_node->type));
        }
    };

    switch (ggml_get_unary_op(unary_node)) {
        case GGML_UNARY_OP_SILU:     dispatch_type([](float v) { return op_silu(v); });     break;
        case GGML_UNARY_OP_SIGMOID:  dispatch_type([](float v) { return op_sigmoid(v); });  break;
        case GGML_UNARY_OP_SOFTPLUS: dispatch_type([](float v) { return op_softplus(v); }); break;
        default:
            GGML_ABORT("fused unary+mul: unsupported unary op %s", ggml_unary_op_name(ggml_get_unary_op(unary_node)));
    }
}

// Fused UNARY(silu|sigmoid|softplus) + MUL: dst = op(unary_src) * other_src, with
// unary_src and other_src separate contiguous same-shape tensors. Elides the
// standalone unary launch and its HBM round-trip of the unary output. Mirrors
// the fused unary+mul path in ggml-cuda.
template <bool MIRROR, bool Q8, typename T, typename Op>
static void fused_unary_mul_launch(const T * a, const T * b, T * dst, sycl::half * mir,
                                   char * q8, const int q8_kx,
                                   const int64_t k, queue_ptr stream, Op op) {
    const uint32_t num_blocks = ceil_div((uint32_t) k, SYCL_GLU_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
            SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
                const T v = op(a[i]) * b[i];
                dst[i]    = v;
                if constexpr (MIRROR) {
                    mir[i] = static_cast<sycl::half>(v);
                }
            }
            if constexpr (Q8) {
                producer_q8_emit(dst, q8, k, q8_kx, item_ct1);
            }
        });
}

// Runtime choice, compile-time store: with no mirror the extra write is not branched over,
// it is not instantiated. The q8_1 emission is the mirror's decode-side dual and the two
// are mutually exclusive (GEMM consumer vs mat-vec consumer).
template <typename T, typename Op>
static void fused_unary_mul_sycl(const T * a, const T * b, T * dst, const int64_t k, queue_ptr stream, Op op,
                                 sycl::half * mir = nullptr, char * q8 = nullptr, int q8_kx = 0) {
    if (mir != nullptr)     { fused_unary_mul_launch<true,  false>(a, b, dst, mir, q8, q8_kx, k, stream, op); }
    else if (q8 != nullptr) { fused_unary_mul_launch<false, true >(a, b, dst, mir, q8, q8_kx, k, stream, op); }
    else                    { fused_unary_mul_launch<false, false>(a, b, dst, mir, q8, q8_kx, k, stream, op); }
}

// Fused ADD(bias) + UNARY + MUL(scale) with per-ne0 broadcast of bias and scale:
//   dst[i] = op(a[i] + bias[i % ne0]) * scale[i % ne0]
// bias and scale are ne0-length vectors broadcast over the outer dims. This is the
// delta-net alpha-gate softplus(alpha + dt) * a_coeff. Once n_tokens>1 (MTP verify)
// makes the operands broadcast, the same-shape unary+mul path stops firing, so this
// recovers that fusion and additionally folds in the standalone bias add.
template <typename T, typename Op>
static void fused_add_unary_mul_sycl(const T * a, const T * bias, const T * scale, T * dst,
                                     const int64_t k, const int ne0, queue_ptr stream, Op op) {
    const uint32_t num_blocks = ceil_div((uint32_t) k, SYCL_GLU_BLOCK_SIZE);
    stream->parallel_for(
        sycl::nd_range<1>(num_blocks * sycl::range<1>(SYCL_GLU_BLOCK_SIZE), sycl::range<1>(SYCL_GLU_BLOCK_SIZE)),
        [=](sycl::nd_item<1> item_ct1) {
            SYCL_GLOBAL_ID_LOOP(k, item_ct1) {
                const int h = (int) (i % ne0);
                dst[i] = op((T) (a[i] + bias[h])) * scale[h];
            }
        });
}

static inline void ggml_sycl_op_fused_unary_mul(ggml_backend_sycl_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    const ggml_tensor * unary_src = unary_node->src[0];
    const ggml_tensor * other_src = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous(unary_src) && ggml_is_contiguous(other_src) && ggml_is_contiguous(mul_node));
    GGML_ASSERT(ggml_are_same_shape(unary_src, other_src) && ggml_are_same_shape(unary_src, mul_node));

    const int64_t       k      = ggml_nelements(mul_node);
    queue_ptr           stream = ctx.stream();
    const ggml_unary_op uop    = ggml_get_unary_op(unary_node);

    switch (mul_node->type) {
        case GGML_TYPE_F32: {
            const float * a = (const float *) unary_src->data;
            const float * b = (const float *) other_src->data;
            float *       d = (float *) mul_node->data;
            // The gated linear-attention output is consumed by the next f16 GEMM; emit its
            // f16 copy here rather than letting that GEMM re-read the tensor to build one.
            // At decode the consumer is a mat-vec instead: emit the q8_1 copy (0143's dual).
            sycl::half * mir    = ggml_sycl_f16_mirror_for_next_matmul(ctx, mul_node, (size_t) k);
            int          q8_kx  = 0;
            char *       q8_out = mir == nullptr ? ggml_sycl_q8_emit_for_next_matvec(ctx, mul_node, &q8_kx) : nullptr;
            switch (uop) {
                case GGML_UNARY_OP_SILU:     fused_unary_mul_sycl<float>(a, b, d, k, stream, [](auto x) { return op_silu(x); }, mir, q8_out, q8_kx); break;
                case GGML_UNARY_OP_SIGMOID:  fused_unary_mul_sycl<float>(a, b, d, k, stream, [](auto x) { return op_sigmoid(x); }, mir, q8_out, q8_kx); break;
                case GGML_UNARY_OP_SOFTPLUS: fused_unary_mul_sycl<float>(a, b, d, k, stream, [](auto x) { return op_softplus(x); }, mir, q8_out, q8_kx); break;
                default: GGML_ABORT("fused unary+mul: unsupported unary op");
            }
            break;
        }
        case GGML_TYPE_F16: {
            const sycl::half * a = (const sycl::half *) unary_src->data;
            const sycl::half * b = (const sycl::half *) other_src->data;
            sycl::half *       d = (sycl::half *) mul_node->data;
            switch (uop) {
                case GGML_UNARY_OP_SILU:     fused_unary_mul_sycl<sycl::half>(a, b, d, k, stream, [](auto x) { return op_silu(x); }); break;
                case GGML_UNARY_OP_SIGMOID:  fused_unary_mul_sycl<sycl::half>(a, b, d, k, stream, [](auto x) { return op_sigmoid(x); }); break;
                case GGML_UNARY_OP_SOFTPLUS: fused_unary_mul_sycl<sycl::half>(a, b, d, k, stream, [](auto x) { return op_softplus(x); }); break;
                default: GGML_ABORT("fused unary+mul: unsupported unary op");
            }
            break;
        }
        default:
            GGML_ABORT("fused unary+mul: unsupported type");
    }
}

static inline void ggml_sycl_op_fused_add_unary_mul(ggml_backend_sycl_context & ctx, ggml_tensor * add_node,
                                                    ggml_tensor * unary_node, ggml_tensor * mul_node) {
    const ggml_tensor * a     = add_node->src[0];
    const ggml_tensor * bias  = add_node->src[1];
    const ggml_tensor * scale = (mul_node->src[0] == unary_node) ? mul_node->src[1] : mul_node->src[0];

    GGML_ASSERT(ggml_is_contiguous(a) && ggml_is_contiguous(mul_node));
    GGML_ASSERT(bias->ne[0] == a->ne[0] && scale->ne[0] == a->ne[0]);

    const int64_t       k      = ggml_nelements(mul_node);
    const int           ne0    = (int) a->ne[0];
    queue_ptr           stream = ctx.stream();
    const ggml_unary_op uop    = ggml_get_unary_op(unary_node);

    GGML_ASSERT(mul_node->type == GGML_TYPE_F32);
    const float * pa = (const float *) a->data;
    const float * pb = (const float *) bias->data;
    const float * ps = (const float *) scale->data;
    float *       d  = (float *) mul_node->data;
    switch (uop) {
        case GGML_UNARY_OP_SILU:     fused_add_unary_mul_sycl<float>(pa, pb, ps, d, k, ne0, stream, [](auto x) { return op_silu(x); }); break;
        case GGML_UNARY_OP_SIGMOID:  fused_add_unary_mul_sycl<float>(pa, pb, ps, d, k, ne0, stream, [](auto x) { return op_sigmoid(x); }); break;
        case GGML_UNARY_OP_SOFTPLUS: fused_add_unary_mul_sycl<float>(pa, pb, ps, d, k, ne0, stream, [](auto x) { return op_softplus(x); }); break;
        default: GGML_ABORT("fused add+unary+mul: unsupported unary op");
    }
}

__dpct_inline__ float ggml_sycl_op_swiglu_oai_single(float x, float g, float alpha = 1.702f, float limit = 7.0f) {
    x = sycl::fmin(x, limit);
    g = sycl::fmax(sycl::fmin(g, limit), -limit);

    float out_glu = x / (1.0f + sycl::native::exp(-x * alpha));
    out_glu = out_glu * (1.0f + g);
    return out_glu;
}

template <typename T>
static void swiglu_oai_kernel(const T * x, const T * g, T * dst, const int64_t k,
                              const int64_t n, const int64_t o0, const int64_t o1,
                              float alpha, float limit, sycl::nd_item<3> item_ct1) {
    const int64_t i = int64_t(item_ct1.get_local_range(2)) * item_ct1.get_group(2) + item_ct1.get_local_id(2);

    if (i >= k) {
        return;
    }

    const int64_t j0 = (i / n) * o0 + (i % n);
    const int64_t j1 = o0 == o1 ? j0 : (i / n) * o1 + (i % n);

    float xi = x[j0];
    float gi = g[j1];

    dst[i] = ggml_sycl_op_swiglu_oai_single(xi, gi, alpha, limit);
}

template <typename T>
static void swiglu_oai_sycl(const T *       x,
                            const T *       g,
                            T *             dst,
                            const int64_t   k,
                            const int64_t   n,
                            const int64_t   o0,
                            const int64_t   o1,
                            const float     alpha,
                            const float     limit,
                            dpct::queue_ptr stream) {
    const int64_t num_blocks = (k + SYCL_GLU_BLOCK_SIZE - 1) / SYCL_GLU_BLOCK_SIZE;
    stream->parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, num_blocks) * sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE),
                                           sycl::range<3>(1, 1, SYCL_GLU_BLOCK_SIZE)),
                         [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_group_size(WARP_SIZE)]] {
                             swiglu_oai_kernel(x, g, dst, k, n, o0, o1, alpha, limit, item_ct1);
                         });
}

void ggml_sycl_op_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    void * src0_d = src0->data;
    void * src1_d = src1 ? src1->data : src0->data;
    const int64_t src0_o = src0->nb[1];
    const int64_t src1_o = src1 ? src1->nb[1] : src0->nb[1];
    void * dst_d = dst->data;
    const int64_t nc = src1 ? src0->ne[0] : src0->ne[0] / 2;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(ggml_is_contiguous_1(src0));
    GGML_ASSERT(src0->nb[0] == ggml_element_size(src0));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(dst->ne[0] == nc);
    GGML_ASSERT(ggml_nrows(dst) == ggml_nrows(src0));

    if (src1) {
        GGML_ASSERT(ggml_is_contiguous_1(src1));
        GGML_ASSERT(src1->nb[0] == ggml_element_size(src1));
        GGML_ASSERT(src1->ne[0] == nc);
        GGML_ASSERT(src0->type == src1->type);
    }

    //const int32_t swapped = ((const int32_t *) dst->op_params)[1];
    const int32_t swapped = ggml_get_op_params_i32(dst, 1);
    const float alpha = ggml_get_op_params_f32(dst, 2);
    const float limit = ggml_get_op_params_f32(dst, 3);

    float * src0_p = (float *) src0_d;
    float * src1_p = (float *) src1_d;

    if (!src1) {
        src0_p += swapped ? nc : 0;
        src1_p += swapped ? 0 : nc;
    }

    swiglu_oai_sycl(src0_p, src1_p, (float *)dst_d, ggml_nelements(dst), nc, src0_o / sizeof(float), src1_o / sizeof(float), alpha, limit, stream);
}

static inline void ggml_sycl_op_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        return op_gelu_erf(x);
    });
}

static inline void ggml_sycl_op_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ggml_sycl_detail::ggml_sycl_op_unary_gated(ctx, dst, [](auto x) {
        return op_gelu_quick(x);
    });
}


void ggml_sycl_sqrt(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqrt(ctx, dst);
}

void ggml_sycl_sin(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sin(ctx, dst);
}

void ggml_sycl_cos(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cos(ctx, dst);
}

void ggml_sycl_acc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_acc(ctx, dst);
}

void ggml_sycl_gelu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu(ctx, dst);
}

void ggml_sycl_silu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_silu(ctx, dst);
}

void ggml_sycl_gelu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_quick(ctx, dst);
}

void ggml_sycl_gelu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_gelu_erf(ctx, dst);
}

void ggml_sycl_tanh(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_tanh(ctx, dst);
}

void ggml_sycl_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_relu(ctx, dst);
}

void ggml_sycl_sigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sigmoid(ctx, dst);
}

void ggml_sycl_hardsigmoid(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardsigmoid(ctx, dst);
}

void ggml_sycl_hardswish(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_hardswish(ctx, dst);
}

void ggml_sycl_exp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_exp(ctx, dst);
}

void ggml_sycl_expm1(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_expm1(ctx, dst);
}

void ggml_sycl_log(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_log(ctx, dst);
}

void ggml_sycl_softplus(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_softplus(ctx, dst);
}

void ggml_sycl_neg(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_neg(ctx, dst);
}

void ggml_sycl_step(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_step(ctx, dst);
}

void ggml_sycl_leaky_relu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_leaky_relu(ctx, dst);
}

void ggml_sycl_sqr(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sqr(ctx, dst);
}

void ggml_sycl_clamp(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_clamp(ctx, dst);
}

void ggml_sycl_xielu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_xielu(ctx, dst);
}

void ggml_sycl_sgn(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_sgn(ctx, dst);
}

void ggml_sycl_abs(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_abs(ctx, dst);
}

void ggml_sycl_elu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_elu(ctx, dst);
}

void ggml_sycl_geglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu(ctx, dst);
}

void ggml_sycl_reglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_reglu(ctx, dst);
}

void ggml_sycl_swiglu(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu(ctx, dst);
}

void ggml_sycl_fused_unary_mul(ggml_backend_sycl_context & ctx, ggml_tensor * unary_node, ggml_tensor * mul_node) {
    scope_op_debug_print scope_dbg_print(__func__, mul_node, /*num_src=*/2);
    ggml_sycl_op_fused_unary_mul(ctx, unary_node, mul_node);
}

void ggml_sycl_fused_add_unary_mul(ggml_backend_sycl_context & ctx, ggml_tensor * add_node,
                                   ggml_tensor * unary_node, ggml_tensor * mul_node) {
    scope_op_debug_print scope_dbg_print(__func__, mul_node, /*num_src=*/3);
    ggml_sycl_op_fused_add_unary_mul(ctx, add_node, unary_node, mul_node);
}

void ggml_sycl_swiglu_oai(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_swiglu_oai(ctx, dst);
}

void ggml_sycl_geglu_erf(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_erf(ctx, dst);
}

void ggml_sycl_geglu_quick(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_geglu_quick(ctx, dst);
}

void ggml_sycl_arange(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_detail::ggml_sycl_op_arange(ctx, dst);
}

void ggml_sycl_floor(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_floor(ctx, dst);
}

void ggml_sycl_ceil(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_ceil(ctx, dst);
}

void ggml_sycl_round(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_round(ctx, dst);
}

void ggml_sycl_trunc(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_trunc(ctx, dst);
}

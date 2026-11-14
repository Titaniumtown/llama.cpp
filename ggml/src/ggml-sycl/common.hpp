//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ifndef GGML_SYCL_COMMON_HPP
#define GGML_SYCL_COMMON_HPP

#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <string>
#include <unordered_set>
#include <unordered_map>

#include "base.hpp"
#include "dpct/helper.hpp"
#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-sycl.h"
#include "presets.hpp"
#include "type.hpp"
#include "sycl_hw.hpp"
#include "memtrace.hpp"

namespace syclexp = sycl::ext::oneapi::experimental;

#if defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/oneapi/bfloat16.hpp>)
    #include <sycl/ext/oneapi/bfloat16.hpp>
    #ifndef GGML_SYCL_HAS_BF16
        #define GGML_SYCL_HAS_BF16
    #endif
#endif

#if defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif

#if GGML_SYCL_DNNL
#include "dnnl.hpp"
#include "dnnl_sycl.hpp"
#include "oneapi/dnnl/dnnl_graph.hpp"
#endif

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#define SYCL_FLASH_ATTN //remove it to disable FLASH_ATTENTION in building.
#define SYCL_FAST_FP16  //don't change. remove it will break fattn-tile.hpp building

/* suppress warning spam */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnested-anon-types"
#include "ggml-common.h"
#pragma clang diagnostic pop
#include "ggml-impl.h"

void* ggml_sycl_host_malloc(size_t size);
void ggml_sycl_host_free(void* ptr);


extern int g_ggml_sycl_debug;
extern int g_ggml_sycl_enable_optimize;
extern int g_ggml_sycl_enable_fusion;
extern int g_ggml_sycl_fuse_conv_window;
extern int g_ggml_sycl_enable_esimd;
extern int g_ggml_sycl_mmid_esimd;
extern int g_ggml_sycl_mmvq_wide;
extern int g_ggml_sycl_mmvq_max_cols;
extern int g_ggml_sycl_f32mm;
extern int g_ggml_sycl_enable_xmx_esimd;
extern int g_ggml_sycl_xmx_max_cols;
extern int g_ggml_sycl_xmx_glu_max_cols;
extern int g_ggml_sycl_prioritize_dmmv;

// Which quantized weight formats may take the XMX dequant-GEMM paths. A bitmask rather than one
// flag per path, so a format can be enabled or measured on its own and adding a format is one bit.
enum ggml_sycl_xmx_gather_type {
    GGML_SYCL_XMX_GATHER_IQ4_NL   = 1 << 0,
    GGML_SYCL_XMX_GATHER_IQ3_S    = 1 << 1,
    GGML_SYCL_XMX_GATHER_IQ4_XS   = 1 << 2,
    GGML_SYCL_XMX_GATHER_IQ3_XXS  = 1 << 3,
    GGML_SYCL_XMX_GATHER_IQ2_XXS  = 1 << 4,
    GGML_SYCL_XMX_GATHER_IQ2_XS   = 1 << 5,
    GGML_SYCL_XMX_GATHER_IQ2_S    = 1 << 6,
    GGML_SYCL_XMX_GATHER_IQ1_S    = 1 << 7,
    GGML_SYCL_XMX_GATHER_IQ1_M    = 1 << 8,
    GGML_SYCL_XMX_GATHER_Q8_0     = 1 << 9,
    GGML_SYCL_XMX_GATHER_Q4_K     = 1 << 10,
    GGML_SYCL_XMX_GATHER_Q5_K     = 1 << 11,
    GGML_SYCL_XMX_GATHER_Q6_K     = 1 << 12,
    GGML_SYCL_XMX_GATHER_Q2_K     = 1 << 13,
    GGML_SYCL_XMX_GATHER_Q3_K     = 1 << 14,
};
static constexpr int GGML_SYCL_XMX_GATHER_TYPES_DEFAULT = ~0;
extern int g_ggml_sycl_xmx_gather_types;

enum ggml_sycl_mmid_sched_bit {
    GGML_SYCL_MMID_SCHED_DEVICE    = 1 << 0,
    GGML_SYCL_MMID_SCHED_UNBOUNDED = 1 << 1,
};
extern int g_ggml_sycl_mmid_sched;
// Which joint_matrix combinations the XMX dequant-GEMM paths may use, one bit each (see fused-gemm.cpp).
// GGML_SYCL_DYNAMIC_PRECISION picks the operand type, this mask the combinations of that type.
static constexpr int GGML_SYCL_XMX_GATHER_SHAPES_DEFAULT = 0xff;
extern int g_ggml_sycl_xmx_gather_shapes;

// GGML_SYCL_DYNAMIC_PRECISION: operand type of the XMX dequant-GEMM paths. F32 turns them off and
// keeps the library GEMM in f32. A src1 precision request of an op [TAG_GGML_PREC] is always met.
enum ggml_sycl_dynamic_precision {
    GGML_SYCL_DYNAMIC_PRECISION_F16,
    GGML_SYCL_DYNAMIC_PRECISION_BF16,
    GGML_SYCL_DYNAMIC_PRECISION_TF32,
    GGML_SYCL_DYNAMIC_PRECISION_F32,
};
#ifdef GGML_SYCL_F16
static constexpr int GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F16;
#else
static constexpr int GGML_SYCL_DYNAMIC_PRECISION_DEFAULT = GGML_SYCL_DYNAMIC_PRECISION_F32;
#endif
extern int g_ggml_sycl_dynamic_precision;
// GGML_SYCL_DYNAMIC_REQUIRED_PRECISION: the XMX type an F32 src1 request may run on instead of f32
// (TF32, or BF16 which also allows tf32). F32 (default): none. F16: src1 requests are ignored.
extern int g_ggml_sycl_dynamic_required_precision;

// [TAG_GGML_PREC] src1 precision request of the MUL_MAT/MUL_MAT_ID op dst
static inline int32_t ggml_sycl_src1_prec(const ggml_tensor * dst) {
    return g_ggml_sycl_dynamic_required_precision == GGML_SYCL_DYNAMIC_PRECISION_F16 ? GGML_PREC_UNDEFINED :
                                                                                       ggml_prec_get_src(dst, 1);
}

// [TAG_GGML_PREC] the library GEMM and dmmv may convert src1 of the MUL_MAT/MUL_MAT_ID op dst to f16
static inline bool ggml_sycl_src1_f16_ok(const ggml_tensor * dst) {
    const int32_t src1_prec = ggml_sycl_src1_prec(dst);
    return g_ggml_sycl_dynamic_precision != GGML_SYCL_DYNAMIC_PRECISION_F32 &&
           (src1_prec == GGML_PREC_UNDEFINED || src1_prec >= GGML_PREC_F16);
}
extern int g_ggml_sycl_enable_flash_attention;
extern int g_ggml_sycl_dev2dev_memcpy;
extern int g_ggml_sycl_fa_onednn;
extern int g_ggml_sycl_fa_onednn_max_kv;
extern int g_ggml_sycl_enable_mkl_fa;
extern int g_ggml_sycl_fa_dpas;
extern int g_ggml_sycl_dpas_check;
extern int g_ggml_sycl_memtrace;
extern int g_ggml_sycl_memtrace_step;
extern int g_ggml_sycl_graph_eviction_timeout;


#define CHECK_TRY_ERROR(expr)                                            \
  [&]() {                                                                \
    try {                                                                \
      expr;                                                              \
      return dpct::success;                                              \
    } catch (std::exception const& e) {                                  \
      std::cerr << e.what() << "\nException caught at file:" << __FILE__ \
                << ", line:" << __LINE__ << ", func:" << __func__        \
                << std::endl;                                            \
      return dpct::default_error;                                        \
    }                                                                    \
  }()


#define __SYCL_ARCH__ DPCT_COMPATIBILITY_TEMP
#define VER_4VEC 610 // todo for hardware optimize.
#define VER_GEN9 700 // todo for hardware optimize.
#define VER_GEN12 1000000 // todo for hardware optimize.
#define VER_GEN13 (VER_GEN12 + 1030) // todo for hardware optimize.

#define GGML_SYCL_MAX_NODES 8192 // TODO: adapt to hardwares

// max batch size to use MMQ kernels when tensor cores are available
#define MMQ_MAX_BATCH_SIZE 32

// dmmv = dequantize_mul_mat_vec
#ifndef GGML_SYCL_DMMV_X
#define GGML_SYCL_DMMV_X 32
#endif
#ifndef GGML_SYCL_MMV_Y
#define GGML_SYCL_MMV_Y 1
#endif

typedef sycl::queue *queue_ptr;

enum ggml_sycl_backend_gpu_mode {
  SYCL_UNSET_GPU_MODE = -1,
  SYCL_SINGLE_GPU_MODE = 0,
  SYCL_MUL_GPU_MODE
};

enum ggml_sycl_dev2dev_memcpy_mode {
  DEV2DEV_MEMCPY_SYCL = 0,
  DEV2DEV_MEMCPY_L0 = 1,
  DEV2DEV_MEMCPY_FORWARD = 2
};

static_assert(sizeof(sycl::half) == sizeof(ggml_fp16_t), "wrong fp16 size");

static void crash() {
  int* ptr = NULL;
  *ptr = 0;
}

[[noreturn]] static void ggml_sycl_error(
    const char* stmt,
    const char* func,
    const char* file,
    const int line,
    const char* msg) {
  fprintf(stderr, "SYCL error: %s: %s\n", stmt, msg);
  fprintf(stderr, "  in function %s at %s:%d\n", func, file, line);
  GGML_ABORT("SYCL error");
}

#define SYCL_CHECK(err)                                                                                    \
    do {                                                                                                   \
        auto err_ = (err);                                                                                 \
        if (err_ != 0)                                                                                     \
            ggml_sycl_error(#err, __func__, __FILE__, __LINE__, "Exception caught in this line of code."); \
    } while (0)

#if DPCT_COMPAT_RT_VERSION >= 11100
#define GGML_SYCL_ASSUME(x) __builtin_assume(x)
#else
#define GGML_SYCL_ASSUME(x)
#endif // DPCT_COMPAT_RT_VERSION >= 11100

#ifdef GGML_SYCL_F16
typedef sycl::half dfloat; // dequantize float
typedef sycl::half2 dfloat2;
#else
typedef float dfloat; // dequantize float
typedef sycl::float2 dfloat2;
#endif // GGML_SYCL_F16

#define MMVQ_MAX_BATCH_SIZE  8
// default of GGML_SYCL_MMVQ_MAX_COLS: -1 keeps the per-type widths mul_mat_vec_q runs as several launches
#define GGML_SYCL_MMVQ_MAX_COLS_DEFAULT -1
// widest of the per-type widths (ggml_sycl_mmvq_type_max_cols returns no larger value), the bound for producers
// that pre-quantize a mat-vec's src1
#define GGML_SYCL_MMVQ_TYPE_MAX_COLS 32

#if defined(GGML_SYCL_SUPPORT_XMX) && defined(__INTEL_LLVM_COMPILER) && __has_include(<sycl/ext/intel/esimd/xmx/dpas.hpp>)
#    define GGML_SYCL_MMVQ_HAS_XMX
#endif // GGML_SYCL_SUPPORT_XMX && __INTEL_LLVM_COMPILER

// default of the most columns the XMX mul_mat_vec_q handles (GGML_SYCL_XMX_MAX_COLS), wider batches use other kernels
#define GGML_SYCL_XMX_MAX_COLS_DEFAULT 80

// most columns the fused gate and up XMX kernel is built for, GGML_SYCL_XMX_GLU_MAX_COLS can only lower this limit
#define GGML_SYCL_XMX_GLU_MAX_COLS 16

static inline bool ggml_sycl_xmx_supports_type(ggml_type type) {
    switch (type) {
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

// the DPAS shapes and 2D block loads used by the XMX kernel are those of Xe2
static inline bool ggml_sycl_xmx_supports_arch(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31 ||
           arch == gpu_arch::intel_gpu_lnl_m;
}

// fewest columns for which the XMX kernel is faster than the other kernels
static inline int ggml_sycl_xmx_min_cols(ggml_type type) {
    return type == GGML_TYPE_Q2_K || type == GGML_TYPE_Q3_K || type == GGML_TYPE_Q4_K ? 1 : 2;
}

static int g_all_sycl_device_count = -1;
static bool g_ggml_backend_sycl_buffer_type_initialized = false;

static ggml_sycl_backend_gpu_mode g_ggml_sycl_backend_gpu_mode =
    SYCL_UNSET_GPU_MODE;

static void* g_scratch_buffer = nullptr;
static size_t g_scratch_size = 0; // disabled by default
static size_t g_scratch_offset = 0;

[[noreturn]] static inline void bad_arch(const sycl::stream& stream_ct1) {
  stream_ct1 << "ERROR: ggml-sycl was compiled without support for the "
                "current GPU architecture.\n";
  // __trap();
  std::exit(1);

  (void)bad_arch; // suppress unused function warning
}

// thread-local copy of dpct's per-thread selected device, -1 until first read.
// Select devices only through ggml_sycl_set_device so the copy stays in sync.
extern thread_local int g_sycl_current_device_id;

int get_current_device_id();

inline int ggml_sycl_get_device() {
    return get_current_device_id();
}

inline dpct::err0 ggml_sycl_set_device(const int device) try {
  if (device == g_sycl_current_device_id) {
    return 0;
  }
  const dpct::err0 err = CHECK_TRY_ERROR(dpct::select_device(device));
  if (err == dpct::success) {
    g_sycl_current_device_id = device;
  }
  return err;
} catch (sycl::exception const& exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  crash();
  std::exit(1);
}

//////////////////////
struct optimize_feature {
    bool reorder=false;
    bool onednn_optimized_gemm=false;
};

struct sycl_device_info {
    bool active; //the device is activated/used in application.
    int cc;  // compute capability
    int nsm; // number of streaming multiprocessors (CUDA) maps to the maximum
             // number of compute units on a SYCL device.
    // size_t  smpb;               // max. shared memory per block
    size_t  smpbo;              // max. shared memory per block (with opt-in)
    int warp_size;     // WARP_SIZE(16)|WARP_32_SIZE(32)|WARP_16_SIZE(16). For Intel GPU, 16 is better in most cases. Some OP support 32 only.
    int max_wg_per_cu; // max work groups per compute unit - refer to
                       // cudaOccupancyMaxActiveBlocksPerMultiprocessor
    bool    vmm;                // virtual memory support
    bool    l0_device_type_valid;
    bool    l0_discrete_gpu;    // Level Zero backend and not an integrated GPU
    size_t  vmm_granularity;    // granularity of virtual memory
    size_t  total_vram;
    sycl_hw_info hw_info;
    optimize_feature opt_feature;
    bool    usm_system_support; // support for USM system allocations
#ifdef GGML_SYCL_GRAPH
    bool    graph_support;        // command graphs can be recorded and replayed
    bool    graph_update_support; // a finalized command graph can be updated
#endif
};


struct ggml_sycl_device_info {
    int active_device_count; //the number of active devices in the application
    int device_count;

    sycl_device_info devices[GGML_SYCL_MAX_DEVICES] = {};

    std::array<float, GGML_SYCL_MAX_DEVICES> default_tensor_split = {};

    int max_work_group_sizes[GGML_SYCL_MAX_DEVICES] = {0};

    bool ext_oneapi_level_zero = true; // sycl::backend::ext_oneapi_level_zero used by all enumerated GPU devices
};

ggml_sycl_device_info & ggml_sycl_info();

// DPAS self-test of the device (dpas-check.cpp), once, at backend init
void ggml_sycl_dpas_check(int device);
// the self-test found wrong DPAS results on the device and GGML_SYCL_DPAS_CHECK=1 keeps the integer DPAS kernels off it
bool ggml_sycl_dpas_faulty(int device);

static inline bool ggml_sycl_xmx_enabled(int device) {
    return g_ggml_sycl_enable_xmx_esimd && ggml_sycl_xmx_supports_arch(ggml_sycl_info().devices[device].hw_info.arch) &&
           !ggml_sycl_dpas_faulty(device);
}

static constexpr size_t SYCL_BUFFER_ALIGNMENT = 128;

struct ggml_sycl_pool {
    virtual ~ggml_sycl_pool() = default;

    virtual void * alloc(size_t size, size_t * actual_size) = 0;
    virtual void free(void * ptr, size_t size) = 0;
};

template<typename T>
struct ggml_sycl_pool_alloc {
    ggml_sycl_pool * pool = nullptr;
    T * ptr = nullptr;
    size_t actual_size = 0;

    explicit ggml_sycl_pool_alloc(ggml_sycl_pool & pool) : pool(&pool) {
    }

    ggml_sycl_pool_alloc(ggml_sycl_pool & pool, size_t size) : pool(&pool) {
        alloc(size);
    }

    ~ggml_sycl_pool_alloc() {
        if (ptr != nullptr) {
            pool->free(ptr, actual_size);
        }
    }

    T * realloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        if (ptr)
            pool->free(ptr, actual_size);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    // size is in number of elements
    T * alloc(size_t size) {
        GGML_ASSERT(pool != nullptr);
        GGML_ASSERT(ptr == nullptr);
        ptr = (T *) pool->alloc(size * sizeof(T), &this->actual_size);
        return ptr;
    }

    T * alloc(ggml_sycl_pool & pool, size_t size) {
        this->pool = &pool;
        return alloc(size);
    }

    T * get() {
        return ptr;
    }

    ggml_sycl_pool_alloc() = default;
    ggml_sycl_pool_alloc(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc(ggml_sycl_pool_alloc &&) = delete;
    ggml_sycl_pool_alloc& operator=(const ggml_sycl_pool_alloc &) = delete;
    ggml_sycl_pool_alloc& operator=(ggml_sycl_pool_alloc &&) = delete;
};

// backend interface

struct ggml_tensor_extra_gpu {
  void* data_device[GGML_SYCL_MAX_DEVICES]; // 1 pointer for each device for split
                                       // tensors
  dpct::event_ptr events[GGML_SYCL_MAX_DEVICES]
                        [GGML_SYCL_MAX_STREAMS]; // events for synchronizing multiple GPUs
  optimize_feature optimized_feature;
};

extern int g_ggml_sycl_use_level_zero_api;
extern int g_ggml_sycl_bmg_vram_check;
bool sycl_get_mem_info(int device, size_t * free, size_t * total);
void * ggml_sycl_malloc_device(size_t size, sycl::queue &q,
                               ggml_sycl_mem_type type = GGML_SYCL_MEM_DIRECT);
void ggml_sycl_free_device(void *ptr, sycl::queue &q);

void release_extra_gpu(ggml_tensor_extra_gpu * extra, std::vector<queue_ptr> streams={});

struct mmid_row_mapping {
    int32_t i1;
    int32_t i2;
};

struct ggml_sycl_grouped_gemm_tile {
    int32_t expert;
    int32_t n0;
    int32_t n1;
};

namespace sycl_ex = sycl::ext::oneapi::experimental;

#ifdef GGML_SYCL_GRAPH
struct ggml_sycl_graph {
    // src data/ne/nb are kept next to the node copy: the scheduler can hand back the same src
    // pointer with different contents or shape, see https://github.com/ggml-org/llama.cpp/pull/21736
    struct node_properties {
        ggml_tensor node;
        void *      node_src_data_ptrs[GGML_MAX_SRC];
        int64_t     node_src_ne[GGML_MAX_SRC][GGML_MAX_DIMS];
        size_t      node_src_nb[GGML_MAX_SRC][GGML_MAX_DIMS];
    };

    std::unique_ptr<sycl_ex::command_graph<sycl_ex::graph_state::executable>> exec_graph = nullptr;
    std::vector<node_properties> node_props;
    bool     warmup_complete = false;
    uint64_t uid             = 0;
    int64_t  last_used_time  = 0;

    // result of check_graph_compatibility() and the uid it was made for
    bool     compatible     = false;
    uint64_t compatible_uid = 0;
};

static_assert(std::is_trivial<ggml_sycl_graph::node_properties>::value, "node_properties must be trivial");
#endif

#if GGML_SYCL_DNNL
struct sdpa_partition {
    dnnl::graph::compiled_partition          cp;
    std::vector<dnnl::graph::logical_tensor> ins;
    dnnl::graph::logical_tensor              out;
    size_t id_q = 0, id_k = 0, id_v = 0, id_scale = 0, id_mask = 0, id_kvlen = 0, id_qlen = 0, id_ninf = 0;
    // implicit-causal offset scalars, owned here so the host pointers handed to
    // tensor::make_scalar_tensor outlive the in-flight execute.
    int32_t kvlen_val = 0, qlen_val = 0;
    float   ninf_val  = -INFINITY;
    dnnl::graph::logical_tensor lt_kvlen, lt_qlen, lt_ninf;   // original lts: partition ports drop the host_scalar property
    bool   has_mask = false;
    bool   ok = false;
};
#endif

struct ggml_backend_sycl_context {
    int device;
    std::string name;
    optimize_feature opt_feature;

    // Set while dispatching a mat-vec whose per-row epilogue has been folded in:
    //     out[row] = softplus(dot(row) + bias[row]) * scale[row]
    // This is the GDN decay chain (ssm_alpha -> +ssm_dt -> softplus -> *ssm_a). Its
    // three ggml nodes move 48 floats and cost a full dispatch each; the mat-vec that
    // produces their input already has the value in a register at the store.
    // Set while dispatching the SECOND mat-vec of a (gate-mm, up-mm, SWIGLU) triple whose
    // quant types differ (the equal-type triple takes the direct GLU fusion hook): the first
    // mat-vec ran normally, and this dispatch's store reads its row from DRAM and writes
    // silu(gate)*up straight into the GLU output -- the standalone SWIGLU kernel never runs.
    const ggml_tensor * mmvq_glu_epi_out          = nullptr;  // the GLU node (written here)
    const ggml_tensor * mmvq_glu_epi_other        = nullptr;  // the already-dispatched mat-vec
    bool                mmvq_glu_epi_anchor_is_up = false;    // anchor is the linear (up) half
    const ggml_tensor * mmvq_dsa_out   = nullptr;
    const ggml_tensor * mmvq_dsa_bias  = nullptr;
    const ggml_tensor * mmvq_dsa_scale = nullptr;
    // Set alongside mmvq_dsa_* when a SECOND, independent mat-vec over the same activation
    // is folded into this dispatch. mmvq_pair_other is that mat-vec, whose weight is read
    // here; mmvq_pair_anchor_is_dsa says whether the decay epilogue belongs to the anchor
    // or to it, since ggml's order between the two is not fixed.
    const ggml_tensor * mmvq_pair_other = nullptr;
    bool                mmvq_pair_anchor_is_dsa = false;

    // f32->f16 conversion cache for a matmul's src1. Several projections inside one block
    // read the SAME activation tensor -- Qwen3Next builds wq/wk/wv from one `cur` and
    // wqkv/wqkv_gate from one `input` -- but ggml_sycl_op_mul_mat_sycl converts src1 into a
    // per-call pool allocation, so the identical bytes are converted 2-3 times per block.
    //
    // Reuse is valid only while nothing has overwritten the source. That is checked against
    // the GRAPH rather than the executed node sequence, so the fusion paths that advance `i`
    // past several nodes are covered without each one having to remember to invalidate.
    struct src1_f16_cache {
        const ggml_cgraph * graph  = nullptr;
        const void *        key    = nullptr;  // src1_ddf_i of the cached slice
        const void *        stream = nullptr;  // separates the multi-device split path
        size_t              ne     = 0;        // elements converted
        int                 node   = -1;       // graph node index it was produced at
        std::unique_ptr<ggml_sycl_pool_alloc<sycl::half>> buf;
    } src1_f16;

    // Dead f32 store elision. A producer that has just written the f16 mirror of an f32
    // activation whose ONLY readers in this graph are f16 GEMMs taking it as src1 skips
    // the f32 store itself: at pp2048 the GLU's is 142.6 MB of writes per call that
    // nothing reads, the fused ADD+RMS_NORM+MUL's 41.94 MB x2 consumers and the gated
    // UNARY+MUL's 50.33 MB. The bytes are then never written, and ggml-alloc hands that
    // buffer to a later tensor, so a consumer that missed the mirror and fell back to
    // converting the f32 would read garbage silently -- the failure mode this campaign
    // has already met once through a recycled buffer. This registry turns every such
    // miss into an abort: the producer records the range; every consumer must hit the
    // mirror and each hit retires one of `remaining`; the graph loop aborts if the range
    // is still registered past its LAST consumer.
    struct elide_f32_reg {
        const ggml_cgraph * graph = nullptr;
        const void *        key   = nullptr;  // dst->data of the elided tensor
        size_t              bytes = 0;
        int                 node  = -1;       // graph index of the producer
        int                 last  = -1;       // graph index of the last consumer
        int                 remaining = 0;    // consumers still to hit before the range retires
    } elide_f32;
    // per-graph census: producer tensors every reference of which is a MUL_MAT src1 within
    // the f16 cache's lookup window -- n_consumers of them, the last at node `last` -- and
    // not flagged as graph outputs
    struct elide_f32_cand {
        int n_consumers = 0;
        int last        = -1;
    };
    std::unordered_map<const ggml_tensor *, elide_f32_cand> elide_f32_ok;

    // The graph the two per-graph censuses last ran on. Both are pure functions
    // of the node list, and llama_context::process_ubatch reuses the SAME ggml_cgraph
    // object when it can, so a repeat of this pointer means the answer is already in
    // place.
    const ggml_cgraph * census_graph = nullptr;

    // The same redundancy on the DECODE side, which the f16 cache above structurally cannot
    // reach: the mat-VEC path quantizes src1 to q8_1 rather than converting it to f16, once
    // per mat-mul. A transformer block feeds ONE normed activation to every projection that
    // hangs off it -- Q/K/V and the GDN gate all read attn_norm, up and gate both read
    // ffn_norm -- so the same rows are re-quantized 2-4x per block. Counted on this model
    // from a GGML_SYCL_DEBUG trace: 510 mat-muls per graph over 266 distinct src1, i.e.
    // 1.92x, stable to +/-0.01 across five consecutive graphs.
    struct src1_q8_cache {
        const ggml_cgraph * graph   = nullptr;
        const void *        key     = nullptr;  // src1_ddf of the cached tensor
        const void *        stream  = nullptr;  // separates the multi-device split path
        size_t              bytes   = 0;        // size of the q8_1 buffer
        size_t              src_ne  = 0;        // f32 elements the key spans, for the overlap test
        int                 variant = -1;       // which quantize functor wrote this layout
        int                 node    = -1;       // graph node index it was produced at
        std::unique_ptr<ggml_sycl_pool_alloc<char>> buf;
        // Producer-emitted entries (the glu-epi anchor) live in the ctx-owned side
        // buffer below, NOT in buf: the anchor kernel still READS the previous entry's
        // buf as its own activation, so the producer can neither free buf now (that
        // reader has not finished) nor free it before that reader retires. Not because
        // the pool would reject the free -- ggml_sycl_pool_vmm's free list takes a free
        // in any order and hands the range to the next allocation that fits, whatever
        // may still be reading it -- but because the
        // caller has to supply the quiescence the free list cannot: it has no view of
        // submitted work, and the pool never unmaps during runtime, so the range would
        // come back either way. When ext is set it is the entry's storage and buf is
        // just the previous allocation, kept until its reader is done; the next
        // consumer store frees it then.
        char * ext = nullptr;
    } src1_q8;

    // Grow-only device buffer for producer-emitted q8_1 mirrors, outside every pool.
    // ~157 KB worst case on this model (8 cols x 19.6 KB rows); freed in the dtor.
    char * src1_q8_ext_buf = nullptr;
    size_t src1_q8_ext_cap = 0;

    // set by the graph walker so the matmul op can locate itself in the graph
    const ggml_cgraph * cur_graph = nullptr;
    int                 cur_node  = -1;

    // The general device pool is shared and short-lived, so a buffer held across nodes
    // cannot live there: it would sit inside a pool whose other tenants are freed and
    // re-handed out around it, and the free list has no view of submitted work, so such
    // a free could not prove that no reader of the buffer is still in flight. Give the
    // cache its own pool -- it is the only allocation in it, so the only free it ever
    // sees is its own previous entry. Mirrors the existing host_pools / fattn_bufs
    // pattern.
    std::unique_ptr<ggml_sycl_pool> src1_pools[GGML_SYCL_MAX_DEVICES];

    ggml_sycl_pool & src1_pool(int device) {
        if (src1_pools[device] == nullptr) {
            src1_pools[device] = new_pool_for_device(stream(device, 0), device);
        }
        return *src1_pools[device];
    }

    ggml_sycl_pool & src1_pool() { return src1_pool(device); }

    // A SEPARATE pool, not a second tenant of src1_pools: that pool holds exactly one
    // allocation, so the only free it ever sees is its own previous entry. Two caches
    // replacing their buffers in either order would hand a live reader the other's
    // bytes.
    std::unique_ptr<ggml_sycl_pool> src1q_pools[GGML_SYCL_MAX_DEVICES];

    ggml_sycl_pool & src1q_pool(int device) {
        if (src1q_pools[device] == nullptr) {
            src1q_pools[device] = new_pool_for_device(stream(device, 0), device);
        }
        return *src1q_pools[device];
    }

    ggml_sycl_pool & src1q_pool() { return src1q_pool(device); }

    void src1_f16_reset() {
        src1_f16.buf.reset();
        src1_f16.graph = nullptr;
        src1_f16.key   = nullptr;
        src1_f16.node  = -1;
        // The census is memoised on this stamp, and this is the only place that can
        // invalidate it: the two maps cleared below are what the stamp stands in
        // for, so a stamp that survives here suppresses the refill that rebuilds
        // them. Both consumers run after this and before it is reassigned --
        // the KV-reorder scan at ggml-sycl.cpp:3374 and the F32-elision census at
        // ggml-sycl.cpp:6028.
        census_graph = nullptr;
        elide_f32 = {};
        elide_f32_ok.clear();
    }

    void src1_q8_reset() {
        src1_q8.buf.reset();
        src1_q8.ext     = nullptr;
        src1_q8.graph   = nullptr;
        src1_q8.key     = nullptr;
        src1_q8.node    = -1;
        src1_q8.variant = -1;
    }

    queue_ptr qptrs[GGML_SYCL_MAX_DEVICES][GGML_SYCL_MAX_STREAMS] = { { nullptr } };

    explicit ggml_backend_sycl_context(int device) :
        device(device),
        name(GGML_SYCL_NAME + std::to_string(device)) {
        opt_feature = ggml_sycl_info().devices[device].opt_feature;
    }

    // src1_f16.buf is owned by src1_pools, and members are destroyed in REVERSE declaration
    // order -- so the pools die first and ggml_sycl_pool_alloc's destructor then calls free()
    // on a dead pool. Releasing the cache here works regardless of declaration order, because
    // a destructor body runs before any member is destroyed.
    //
    // This is not theoretical: it segfaulted at exit on every process that ran a prefill, and
    // on no decode-only one -- decode's ne1 == 1 takes mmvq, never converts src1, and so never
    // populates the cache. The results had already been printed by then, which is exactly what
    // makes it easy to miss.
    ~ggml_backend_sycl_context() {
        src1_f16_reset();
        src1_q8_reset();
        if (src1_q8_ext_buf != nullptr) {
            sycl::free(src1_q8_ext_buf, *stream());
        }
    }

    queue_ptr stream(int device, int stream) {
        if (qptrs[device][stream] == nullptr) {
            qptrs[device][stream] = &(dpct::get_device(device).default_queue());
        }
        return qptrs[device][stream];
    }

    queue_ptr stream() {
        return stream(device, 0);
    }

#if GGML_SYCL_DNNL
    dnnl::engine make_engine(sycl::queue* q) {
        // Get the device associated with the queue
        sycl::device dev = q->get_device();
        // Get the context associated with the queue
        sycl::context ctx = q->get_context();
        const dnnl::engine eng = dnnl::sycl_interop::make_engine(dev, ctx);
        return eng;
    }

    std::unordered_map<sycl::queue*, dnnl::stream> stream_map;
    std::unordered_map<sycl::queue*, dnnl::engine> engine_map;
    dnnl::stream stream_dnnl(int device, int _stream) {
        auto q = stream(device, _stream);
        return stream_dnnl(q);
    }
    dnnl::engine engine_dnnl(sycl::queue* qptr) {
        auto it = engine_map.find(qptr);
        if (it == engine_map.end()) {
            auto eng = make_engine(qptr);
            engine_map[qptr] = eng;
            return eng;
        }
        else
        {
            return it->second;
        }
    }
    dnnl::stream stream_dnnl(sycl::queue* qptr) {
        auto it = stream_map.find(qptr);
        if (it == stream_map.end()) {
            auto eng = engine_dnnl(qptr);
            auto stream = dnnl::sycl_interop::make_stream(eng, *qptr);
            stream_map[qptr] = stream;
            return stream;
        }
        else
        {
            return it->second;
        }
    }
    dnnl::stream stream_dnnl() {
        return stream_dnnl(device, 0);
    }

    std::unordered_map<std::string, sdpa_partition> sdpa_partitions;
#endif

    // pool
    std::unique_ptr<ggml_sycl_pool> pools[GGML_SYCL_MAX_DEVICES];

    std::unique_ptr<ggml_sycl_pool> host_pools[GGML_SYCL_MAX_DEVICES];

    std::vector<mmid_row_mapping> mmid_row_mapping_host;
    std::vector<ggml_sycl_grouped_gemm_tile> mmid_tile_schedule_host;

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_device(queue_ptr qptr, int device);

    static std::unique_ptr<ggml_sycl_pool> new_pool_for_host(queue_ptr qptr, int device);

    ggml_sycl_pool & pool(int device) {
        if (pools[device] == nullptr) {
            pools[device] = new_pool_for_device(stream(device,0), device);
        }
        return *pools[device];
    }

    ggml_sycl_pool & pool() {
        return pool(device);
    }

#ifdef GGML_SYCL_GRAPH
    // Map from first node pointer to graph - allows multiple graphs per context when the
    // computation is split across CPU/GPU (e.g. with --n-cpu-moe)
    std::unordered_map<const void *, std::unique_ptr<ggml_sycl_graph>> sycl_graphs;

    int64_t last_graph_eviction_sweep = 0;
    std::optional<bool> graph_compatible;

    ggml_sycl_graph * sycl_graph(const void * first_node_ptr) {
        const int64_t time_now = ggml_time_us();
        const int64_t eviction_timeout_us = g_ggml_sycl_graph_eviction_timeout * 1'000'000LL;

        // sweep every half the eviction timeout, evicting graphs unused for >= the eviction timeout
        if (time_now - last_graph_eviction_sweep >= eviction_timeout_us / 2) {
            last_graph_eviction_sweep = time_now;
            for (auto it = sycl_graphs.begin(); it != sycl_graphs.end(); ) {
                if (time_now - it->second->last_used_time >= eviction_timeout_us) {
                    it = sycl_graphs.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto it = sycl_graphs.find(first_node_ptr);
        if (it == sycl_graphs.end()) {
            it = sycl_graphs.emplace(first_node_ptr, std::make_unique<ggml_sycl_graph>()).first;
        }
        it->second->last_used_time = time_now;
        return it->second.get();
    }
#endif

    ggml_sycl_pool & host_pool(int device) {
        if (host_pools[device] == nullptr) {
            host_pools[device] = new_pool_for_host(stream(device, 0), device);
        }
        return *host_pools[device];
    }

    ggml_sycl_pool & host_pool() { return host_pool(device); }
};

// common device functions

static __dpct_inline__ float warp_reduce_sum(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), x, mask);
    }
    return x;
}

static __dpct_inline__ sycl::float2
warp_reduce_sum(sycl::float2 a, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        a.x() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.x(),
            mask);
        a.y() += dpct::permute_sub_group_by_xor(item_ct1.get_sub_group(), a.y(),
            mask);
    }
    return a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ int warp_reduce_sum(int x) {
  return sycl::reduce_over_group(
      sycl::ext::oneapi::this_work_item::get_sub_group(), x, sycl::plus<>());
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), x, offset, width);
  }
  return x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ float warp_reduce_sum(float x, const sycl::nd_item<3>& item_ct1) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += dpct::permute_sub_group_by_xor(
        item_ct1.get_sub_group(), x, offset);
  }
  return x;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ sycl::float2 warp_reduce_sum(sycl::float2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a.x() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.x(), offset,
        width);
    a.y() += dpct::permute_sub_group_by_xor(
        sycl::ext::oneapi::this_work_item::get_sub_group(), a.y(), offset,
        width);
  }
  return a;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ sycl::half2 warp_reduce_sum(sycl::half2 a) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    a = a + dpct::permute_sub_group_by_xor(
                sycl::ext::oneapi::this_work_item::get_sub_group(), a, offset,
                width);
  }
  return a;
}

static constexpr int ggml_sycl_get_physical_warp_size() {
  // todo: for old iGPU + dGPU case, need to be changed.
  return WARP_SIZE;
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ int warp_reduce_all(int x) {
    if (width == ggml_sycl_get_physical_warp_size()) {
        return sycl::all_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (~0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) ||
                x);
    } else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) &&
                x;
        }
        return x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ int warp_reduce_any(int x) {
    if (width == ggml_sycl_get_physical_warp_size()) {
        return sycl::any_of_group(
            sycl::ext::oneapi::this_work_item::get_sub_group(),
            (0xffffffff &
             (0x1 << sycl::ext::oneapi::this_work_item::get_sub_group()
                         .get_local_linear_id())) &&
                x);
    } else {
#pragma unroll
        for (int offset = width / 2; offset > 0; offset >>= 1) {
            x = dpct::permute_sub_group_by_xor(
                    sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                    offset, width) ||
                x;
        }
        return x;
    }
}

/* use WARP_SIZE or WARP_32_SIZE*/
template <int width>
static __dpct_inline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
                          sycl::ext::oneapi::this_work_item::get_sub_group(), x,
                          offset, width));
  }
  return x;
}

static __dpct_inline__ float warp_reduce_max(float x,
    const sycl::nd_item<3>& item_ct1) {
#pragma unroll
    for (int mask = WARP_SIZE / 2; mask > 0; mask >>= 1) {
        x = sycl::fmax(x, dpct::permute_sub_group_by_xor(
            item_ct1.get_sub_group(), x, mask));
    }
    return x;
}

/* Helper for Computing the linear offset of a ggml_tensor given
per-dimension sizes, strides, and indices */
template<int N>
__dpct_inline__ size_t calculate_offset(const std::array<int, N> & strides, const std::array<int, N> & indices) {
    size_t offset = 0;
#pragma unroll
    for (int i = 0; i < N; i++) {
        auto index_i = indices[i];
        offset += strides[i] * index_i;
    }
    return offset;
}

// Helper for vec loading aligned data
template <typename Tp, int n>
inline sycl::vec<Tp, n> vec_aligned_load(const Tp* aligned_ptr) {
    return *reinterpret_cast<const sycl::vec<Tp, n>*>(aligned_ptr);
}

// Helper for accessing pointers with no warnings
template <typename Tp, int dim>
static __dpct_inline__ Tp* get_pointer(sycl::local_accessor<Tp, dim> acc) {
    return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int64_t downsample_sycl_global_range(int64_t accumulate_block_num, int64_t block_size);

constexpr size_t ceil_div(const size_t m, const size_t n) {
    return (m + n - 1) / n;
}

bool gpu_has_xmx(sycl::device &dev);

#if GGML_SYCL_DNNL
// oneDNN builds JIT kernels only for some GPU architectures. On the rest it falls back to
// reference kernels, which are much slower than the SYCL kernels here.
inline bool ggml_sycl_dnnl_has_optimized_gemm(int device) {
    return ggml_sycl_info().devices[device].opt_feature.onednn_optimized_gemm;
}
#endif

int ggml_sycl_get_env(const char *env_name, int default_val);

// Widest SINGLE mat-vec launch. The plain reorder switches for Q4_K/Q5_K/Q6_K also
// instantiate ncols_dst == 16; every other type, and the GLU-epilogue and glu-fused
// twins, stop at MMVQ_MAX_BATCH_SIZE.
//
// Why 16 is worth its own instantiation: a group's cost is dominated by reading the
// weight, and one 16-column launch reads it ONCE where two 8-column launches read it
// twice. Measured per decode step at COLS=32, the cost is exactly linear in the number
// of groups -- 55.9 / 109.4 / 163.3 / 217.6 ms for 1/2/3/4 groups of 8, i.e. increments
// of 53.5 / 53.9 / 54.3 -- and the 16.85 GiB weight read alone is 30.2 ms of each 53.9.
// So halving the group count should remove ~30 ms per pair of groups, while the marginal
// per-column term (~2.7 ms, itself ~10x its own arithmetic floor) is paid either way.
//
// The risk is register pressure: the kernels hold `float partial[ncols_dst]` live across
// the whole block loop, so 16 columns is 16 accumulators per lane against a 32-float
// SIMD32 budget. GGML_SYCL_MMVQ_WIDE16=0 backs the whole thing out to groups of 8, which
// is the A/B control -- one binary, both arms.
static inline int ggml_sycl_mmvq_wide_batch_size() {
    static const int on = ggml_sycl_get_env("GGML_SYCL_MMVQ_WIDE16", 1);
    return on ? 16 : MMVQ_MAX_BATCH_SIZE;
}

// Which plain reorder switch serves a wide group. One table: the wide path reads the
// launch width from it and the partial-wide launch reads the kernel from it, so a type
// that reaches the wide path always reaches the kernel that reads its own layout.
enum ggml_sycl_mmvq_wide_kind {
    GGML_SYCL_MMVQ_WIDE_NONE = 0,
    GGML_SYCL_MMVQ_WIDE_Q4_K,
    GGML_SYCL_MMVQ_WIDE_Q5_K,
    GGML_SYCL_MMVQ_WIDE_Q6_K,
    GGML_SYCL_MMVQ_WIDE_IQ4_XS,
};

static inline enum ggml_sycl_mmvq_wide_kind ggml_sycl_mmvq_type_wide_kind(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_Q4_K:   return GGML_SYCL_MMVQ_WIDE_Q4_K;
        case GGML_TYPE_Q5_K:   return GGML_SYCL_MMVQ_WIDE_Q5_K;
        case GGML_TYPE_Q6_K:   return GGML_SYCL_MMVQ_WIDE_Q6_K;
        case GGML_TYPE_IQ4_XS: return GGML_SYCL_MMVQ_WIDE_IQ4_XS;
        default:               return GGML_SYCL_MMVQ_WIDE_NONE;
    }
}

// The widest group the type's plain reorder switch is instantiated for. The Q4_K/Q5_K/Q6_K
// switches carry the 16-column case; the IQ4_XS one carries every width from 9 to 16, so a
// partial remainder is served by an exact instantiation instead of a 16-wide kernel with a
// runtime act count.
static inline int ggml_sycl_mmvq_type_max_group_cols(enum ggml_type t) {
    return ggml_sycl_mmvq_type_wide_kind(t) == GGML_SYCL_MMVQ_WIDE_NONE
               ? MMVQ_MAX_BATCH_SIZE
               : ggml_sycl_mmvq_wide_batch_size();
}

// Interleaved activation staging for the 16-column reorder mat-vec.
//
// The instruction budget put the 16-wide kernels at their source-level floor:
// ~55 instructions per column for 8 dp4a, the rest being three strided per-lane
// loads per column (columns sit stride_col_y_bytes ~ 5.7 KB apart, so every column
// is its own gather with its own address). The fix is a layout, not a rewrite: a
// tiny repack kernel stages the 16 columns' q8_1 into [block][lane-slot][col]
// order, which makes a lane's u0||u1 for column c 32 contiguous bytes at
// col*32, and its (ds0,ds1) pair 8 bytes at col*8. Same bytes, permuted; the
// column loop's loads become base + small-immediate addresses.
//
// Applies to Q4_K/Q5_K plain and SPLIT wide kernels at ncols_dst == 16 only (Q6_K
// has a different lane map and keeps the strided path). The repack traffic is
// 16 columns x ~K*9/8 bytes vs the ~nrows x that the kernel reads back -- three
// orders down. GGML_SYCL_MMVQ_INTERLEAVE=0 restores the strided reads exactly.
static inline bool ggml_sycl_mmvq_interleave() {
    static const int on = ggml_sycl_get_env("GGML_SYCL_MMVQ_INTERLEAVE", 1);
    return on != 0;
}


char * ggml_sycl_src1_q8_store_ext(ggml_backend_sycl_context & ctx, const void * key, const void * strm,
                                   size_t bytes, size_t src_ne, int variant, const ggml_tensor * out,
                                   bool self_reading = false);

// Attribute a device-side pass that is NOT a graph node to its own profiler row
// (GGML_SYCL_PROF=1). No-op when profiling is off. Without this, work submitted inside an
// op -- notably launch_fattn's K/V dequant -- is invisible: it shows up in no row at all.
// A C string, so an unsampled graph builds no std::string; a caller that composes its
// row key gates the composition on ggml_sycl_prof_active().
void ggml_sycl_prof_mark_sub(const char * name, uint64_t bytes);
// True only while a sampled graph is being instrumented.
bool ggml_sycl_prof_active();

// Bytes rms_norm_f32's q8_1 emission actually moved, keyed by the tensor it was
// emitted for. The emission writes into a side buffer that is not a graph tensor, so
// span_bytes cannot see it; on the decode widths it is bounded for, that is a 90%
// under-count weighted across the five fused-norm rows. The map is file-scope in
// ggml-sycl.cpp, beside ggml_sycl_prof_mark_sub and the rest of the profiler's free
// functions -- producer and reader must share ONE object, which is the whole reason
// this is a function and not a context field. A second copy on the context would
// compile, apply, and charge nothing.
void ggml_sycl_q8_emit_record(const ggml_tensor * out, uint64_t bytes);

template <int N, class T> std::string debug_get_array_str(const std::string & prefix, const T array[N]) {
    if (LIKELY(!g_ggml_sycl_debug)) {
        return "";
    }
    std::stringstream ss;
    ss << prefix << "=[";
    for (std::size_t i = 0; i < N - 1; ++i) {
        ss << array[i] << ", ";
    }
    if constexpr (N > 0) {
        ss << array[N - 1];
    }
    ss << "]";
    return ss.str();
}

inline std::string debug_get_tensor_str(const std::string &prefix,
        const ggml_tensor *tensor, const std::string &suffix = "") {
    std::stringstream ss;
    if (LIKELY(!g_ggml_sycl_debug)) { return ss.str(); }
    ss << prefix.c_str() << "=";
    if (tensor) {
        ss << "'" << tensor->name << "':type=" << ggml_type_name(tensor->type);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";ne", tensor->ne);
        ss << debug_get_array_str<GGML_MAX_DIMS>(";nb", tensor->nb);

        if (!ggml_is_contiguous(tensor)) { ss << ";strided"; }
        if (ggml_is_permuted(tensor)) { ss << ";permuted"; }
    } else {
        ss << "nullptr";
    }
    ss << suffix;
    return ss.str();
}

// Use scope_op_debug_print to log operations coming from running a model
struct scope_op_debug_print {
    // Use string_views to avoid the cost of creating a string and concatenating them
    // string_views must be alive for as long as the object is alive
    // scope_op_debug_print are used with string literals in practice which are stored in constant space so always accessible
    scope_op_debug_print(const std::string_view & func, const std::string_view & func_suffix, const ggml_tensor * dst,
                         std::size_t num_src, const std::string_view & suffix = "") :
        func(func),
        func_suffix(func_suffix) {
        if (LIKELY(!g_ggml_sycl_debug)) {
            return;
        }
        GGML_SYCL_DEBUG("[SYCL][OP] call %s%s:", func.data(), func_suffix.data());
        GGML_SYCL_DEBUG("%s", debug_get_tensor_str(" dst", dst).c_str());
        if (dst) {
            for (std::size_t i = 0; i < num_src; ++i) {
                GGML_SYCL_DEBUG("%s", debug_get_tensor_str("\tsrc" + std::to_string(i), dst->src[i]).c_str());
            }
        }
        GGML_SYCL_DEBUG("%s\n", suffix.data());
    }

    scope_op_debug_print(const std::string_view & func, const ggml_tensor * dst, std::size_t num_src,
                         const std::string_view & suffix = "") :
        scope_op_debug_print(func, "", dst, num_src, suffix) {}

    ~scope_op_debug_print() { GGML_SYCL_DEBUG("[SYCL][OP] call %s%s done\n", func.data(), func_suffix.data()); }

  private:
    std::string_view func;
    std::string_view func_suffix;
};

static __dpct_inline__ float get_alibi_slope(const float    max_bias,
                                             const uint32_t h,
                                             const uint32_t n_head_log2,
                                             const float    m0,
                                             const float    m1) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return dpct::pow(base, exph);
}

static const sycl::uint3 init_fastdiv_values(uint64_t d_64) {
    GGML_ASSERT(d_64 != 0);
    GGML_ASSERT(d_64 <= std::numeric_limits<uint32_t>::max());

    const uint32_t d = (uint32_t) d_64;

    uint32_t L = 0;
    while (L < 32 && (uint32_t{ 1 } << L) < d) {
        L++;
    }

    uint32_t mp = (uint32_t) ((uint64_t{ 1 } << 32) * ((uint64_t{ 1 } << L) - d) / d + 1);
    return sycl::uint3(mp, L, d);
}

// Maximum number of bytes that can be copied in a single instruction.
// Set by test result.
static constexpr int ggml_sycl_get_max_cpy_bytes() {
    return 16;
}

// If the node after `dst` is a mul_mat that will convert `dst` to f16 as its src1, hand back a
// buffer for the producing kernel to write that f16 copy into while the value is still in a
// register, and register it in the src1 conversion cache so the mul_mat finds it. Returns
// nullptr when the conversion would not happen (mat-vec paths, f16 dst, cache disabled), so a
// caller can template the extra store away entirely. Defined in ggml-sycl.cpp, where the
// cache and the mul_mat path predicates live.
sycl::half * ggml_sycl_f16_mirror_for_next_matmul(ggml_backend_sycl_context & ctx,
                                                  const ggml_tensor * dst, size_t ne);

// Companion to the mirror: true iff the f32 store of `dst` may be skipped because the
// per-graph census says every reader of the tensor is one of the GEMM consumers that
// will hit this mirror. Registers the range so that any f32 read of it before the LAST
// consumer clears it aborts. Call only after ggml_sycl_f16_mirror_for_next_matmul
// returned non-null for the same dst.
bool ggml_sycl_elide_f32_arm(ggml_backend_sycl_context & ctx, const ggml_tensor * dst, size_t bytes);

// Per-producer sub-switches under GGML_SYCL_ELIDE_F32, so each new elision site can be
// isolated in the same-binary A/B. 0 keeps the producer's f32 store.
static inline bool ggml_sycl_elide_f32_norm_enabled() {
    static const int e = ggml_sycl_get_env("GGML_SYCL_ELIDE_F32_NORM", 1);
    return e != 0;
}
static inline bool ggml_sycl_elide_f32_unary_enabled() {
    static const int e = ggml_sycl_get_env("GGML_SYCL_ELIDE_F32_UNARY", 1);
    return e != 0;
}

// Decode-side dual of the f16 mirror: if the node after `dst` is a mat-vec that will quantize
// `dst` to q8_1 as its src1, hand back the SoA q8_1 buffer for the producing kernel to fill and
// register it in the src1 q8_1 cache (the rendezvous -- no consumer detection).
// Returns nullptr when the consumer would not quantize, so the caller can template the
// emission away. *kx receives ne00. Defined in ggml-sycl.cpp beside the cache.
char * ggml_sycl_q8_emit_for_next_matvec(ggml_backend_sycl_context & ctx,
                                         const ggml_tensor * dst, int * kx,
                                         bool self_reading = false);

// Binary GGML_OP_MUL fast path: same-shape contiguous f32 product fused with the q8_1
// emission above. Returns false (and launches nothing) when the emission gate declines;
// the caller then takes the generic bin_bcast path. Defined in element_wise.cpp.
bool ggml_sycl_mul_emit_q8_fast(ggml_backend_sycl_context & ctx, ggml_tensor * dst);

// ---- q8_0 KV cache row-split reorder ("KV reorder") ----
// block_q8_0's 34-byte AoS stride byte-gathers on Intel: the fattn-vec K dot measured
// 23-27.8 GB/s against 240 GB/s for the same bytes laid out as a per-segment qs plane
// (D int8s) followed by the d halves (D/32 of them) -- same 34*D/32 bytes per segment,
// same row stride, nothing outside the segment moves. A segment is one head's D
// elements: FA reads K/V at nb[2] == row_size(q8_0, D), so every reader's row pointer
// is a segment base. Registration happens per graph before any node runs (every graph
// that writes KV also holds the FLASH_ATTN_EXT that reads it); the registry is keyed
// by the cache tensor's base allocation and purged on buffer free.
// Off switch: GGML_SYCL_KV_REORDER=0.
bool ggml_sycl_kv_reorder_enabled();
void ggml_sycl_kv_reorder_scan_graph(ggml_backend_sycl_context & ctx, const ggml_cgraph * cgraph);
void ggml_sycl_kv_reorder_try_register(const ggml_tensor * t);
void ggml_sycl_kv_reorder_purge_range(const void * base, size_t bytes);
// segment length in elements when `t`'s base allocation holds reordered rows, else 0
// Bytes one row of a row-split KV segment occupies, per quant type. q8_0 is one
// byte per element plus one half per 32; q4_0 a nibble per element plus one half
// per 32; q5_1 a nibble per element plus a 5th-bit plane and a half2 (d,m) per 32.
// All three stay 16-aligned at seg=256 (272 / 144 / 192).
static inline size_t segbytes_q8(const int seg, const int segblk) {
    return (size_t) seg + (size_t) segblk * sizeof(sycl::half);
}
static inline size_t segbytes_q4(const int seg, const int segblk) {
    return (size_t) seg/2 + (size_t) segblk * sizeof(sycl::half);
}
static inline size_t segbytes_q5(const int seg, const int segblk) {
    return (size_t) seg/2 + (size_t) segblk*4 + (size_t) segblk*4;
}

int  ggml_sycl_kv_reorder_seg(const ggml_tensor * t);
// Aligned memory transfers of 8/16 bytes can be faster than 2 transfers with 4 bytes.
template <int nbytes, int alignment = 0>
static __dpct_inline__ void ggml_sycl_memcpy_1(void * dst, const void * src) {
    if constexpr (alignment != 0) {
        static_assert(nbytes % alignment == 0, "bad alignment");
    }
    constexpr int nb_per_cpy = alignment == 0 ? nbytes : alignment;

#pragma unroll
    for (int i = 0; i < nbytes/nb_per_cpy; ++i) {
        if constexpr (nb_per_cpy == 1) {
            ((char *) dst)[i] = ((const char *) src)[i];
        } else if constexpr (nb_per_cpy == 2) {
            ((short *) dst)[i] = ((const short *) src)[i];
        } else if constexpr (nb_per_cpy == 4) {
            ((int *) dst)[i] = ((const int *) src)[i];
        } else if constexpr (nb_per_cpy == 8) {
            ((sycl::int2 *) dst)[i] = ((const sycl::int2 *) src)[i];
        } else if constexpr (nb_per_cpy == 16) {
            ((sycl::int4 *) dst)[i] = ((const sycl::int4 *) src)[i];
        } else {
            static_assert(nbytes == 0 && nbytes == -1, "bad nbytes");
        }
    }
}
template <typename T>
sycl::half2 __dpct_inline__ make_half2( T x, T y) {
    sycl::half2 res(static_cast<sycl::half>(x),static_cast<sycl::half>(y));
    return res;
}

static __dpct_inline__ uint32_t fastdiv(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t hi = sycl::mul_hi<unsigned>(n, fastdiv_values.x());
    return (hi + n) >> fastdiv_values.y();
}


template <typename T>
sycl::float2 __dpct_inline__ make_float2( T x, T y) {
    sycl::float2 res(static_cast<float>(x),static_cast<float>(y));
    return res;
}

sycl::float2 __dpct_inline__ __half22float2(sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    return float2_value;
}

static __dpct_inline__ sycl::uint2 fast_div_modulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    const uint32_t div_val = fastdiv(n, fastdiv_values);
    const uint32_t mod_val = n - div_val * fastdiv_values.z();
    return sycl::uint2(div_val, mod_val);
}

static __dpct_inline__ int ggml_sycl_dp4a(const int a, const int b, int c) {
    return dpct::dp4a(a, b, c);
}

static __dpct_inline__ float ggml_sycl_e8m0_to_fp32(uint8_t x) {
    uint32_t bits;
    if (x == 0) {
        bits = 0x00400000;
    } else {
        bits = (uint32_t) x << 23;
    }

    float result;
    memcpy(&result, &bits, sizeof(float));
    return result;
}

sycl::float2 __dpct_inline__ __half22float2(const sycl::half2 &H) {
    sycl::float2 float2_value(static_cast<float>(H.x()), static_cast<float>(H.y()));
    return float2_value;
}

float __dpct_inline__ __half2float(sycl::half H) {
    return static_cast<float>(H);
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const float v, const float u) {
    acc += v*u;
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::float2 v, const sycl::float2 u) {
    acc += v.x() * u.x();
    acc += v.y() * u.y();
}

static __dpct_inline__ void ggml_sycl_mad(float & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16
    const sycl::float2 tmp = (v * u).template convert<float, sycl::rounding_mode::automatic>();
    acc += tmp.x() + tmp.y();
#else
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    acc += tmpv.x() * tmpu.x();
    acc += tmpv.y() * tmpu.y();
#endif // GGML_SYCL_F16
}

static __dpct_inline__ void ggml_sycl_mad(sycl::half2 & acc, const sycl::half2 v, const sycl::half2 u) {
#ifdef GGML_SYCL_F16
    acc += v*u;
#else
    const sycl::float2 tmpv = __half22float2(v);
    const sycl::float2 tmpu = __half22float2(u);
    sycl::float2 tmpacc = __half22float2(acc);
    // tmpacc.x += tmpv.x() * tmpu.x();
    // tmpacc.y += tmpv.y() * tmpu.y();
    sycl::float2 tmp1(tmpacc.x() + tmpv.x() * tmpu.x(), tmpacc.y() + tmpv.y() * tmpu.y());
    acc = make_half2(tmp1.x(), tmp1.y());
#endif // GGML_SYCL_F16
}

template <int n>
struct ggml_sycl_unroll {
    template <typename Func, typename... Args>
    void operator()(const Func & f, Args... args) const {
        f(n - 1, args...);
        ggml_sycl_unroll<n - 1>{}(f, args...);
    }
};

template <>
struct ggml_sycl_unroll<1> {
    template <typename Func, typename... Args>
    void operator()(const Func & f, Args... args) const {
        f(0, args...);
    }
};

static __dpct_inline__ sycl::half2 ggml_sycl_hmax2(const sycl::half2 a, const sycl::half2 b) {
    sycl::half2 ret;
    reinterpret_cast<sycl::half &>(ret.x()) =
        sycl::vec<float, 1>(sycl::fmax(a[0], b[0])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    reinterpret_cast<sycl::half &>(ret.y()) =
        sycl::vec<float, 1>(sycl::fmax(a[1], b[1])).convert<sycl::half, sycl::rounding_mode::automatic>()[0];
    return ret;
}

static __dpct_inline__ sycl::half ggml_sycl_hmax(const sycl::half a, const sycl::half b) {
    return sycl::vec<float, 1>(
               sycl::fmax(sycl::vec<sycl::half, 1>(a).convert<float, sycl::rounding_mode::automatic>()[0],
                          sycl::vec<sycl::half, 1>(b).convert<float, sycl::rounding_mode::automatic>()[0]))
        .convert<sycl::half, sycl::rounding_mode::automatic>()[0];
}

static __dpct_inline__ uint32_t __hgt2_mask(const sycl::half2 a, const sycl::half2 b) {
    const uint32_t mask_low  = 0x0000FFFF * (float(a[0]) > float(b[0]));
    const uint32_t mask_high = 0xFFFF0000 * (float(a[1]) > float(b[1]));
    return mask_low | mask_high;
}

static __dpct_inline__ uint32_t fastmodulo(uint32_t n, const sycl::uint3 fastdiv_values) {
    // expects  fastdiv_values to contain <mp, L, divisor> in <x, y, z> (see init_fastdiv_values)
    return n - fastdiv(n, fastdiv_values) * fastdiv_values.z();
}

static bool fast_fp16_available(const int cc) {
    GGML_UNUSED(cc);
    return true;   //Intel GPUs always support FP16.
}

enum class block_reduce_method {
    MAX,
    SUM,
};

template<block_reduce_method method_t, typename T, int warp_size>
struct block_reduce_policy;

template <typename T, typename... Ts>
inline constexpr bool is_any = (std::is_same_v<T, Ts> || ...);

template<typename...>
inline constexpr bool ggml_sycl_dependent_false_v = false;

#define WARP_32_SIZE 32

template <typename T, int warp_size> struct block_reduce_policy<block_reduce_method::SUM, T, warp_size> {
    static T reduce(T val) {
        if constexpr (is_any<T, float, sycl::float2, sycl::half2, int>) {
            return warp_reduce_sum<warp_size>(val);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }

    static T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return 0.0f;
        } else if constexpr (std::is_same_v<T, sycl::float2>) {
            return sycl::float2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, sycl::half2>) {
            return sycl::half2(0.0f, 0.0f);
        } else if constexpr (std::is_same_v<T, int>) {
            return 0;
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce sum");
        }
    }
};

template <typename T, int warp_size> struct block_reduce_policy<block_reduce_method::MAX, T, warp_size> {
    static T reduce(T val) {
        if constexpr (is_any<T, float, sycl::half2>) {
            return warp_reduce_max<warp_size>(val);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }

    static T sentinel() {
        if constexpr (std::is_same_v<T, float>) {
            return -INFINITY;
        } else if constexpr (std::is_same_v<T, sycl::half2>) {
            return sycl::half2(-INFINITY, -INFINITY);
        } else {
            static_assert(ggml_sycl_dependent_false_v<T>, "Unsupported type for block reduce max");
        }
    }
};


template <block_reduce_method reduce_method_t, int warp_size, typename T>
static T block_reduce(T val, T * shared_vals, int block_size_template) {
    auto item_ct1                 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    val                           = block_reduce_policy<reduce_method_t, T,warp_size>::reduce(val);
    const int block_size = block_size_template == 0 ? item_ct1.get_local_range(2) : block_size_template;
    const int nthreads = item_ct1.get_local_range(2);
    const int nwarps = nthreads / WARP_SIZE;

    if (block_size > warp_size) {
        assert((block_size <= 1024) && (block_size % warp_size) == 0);
        const int warp_id = item_ct1.get_local_id(2) / warp_size;
        const int lane_id = item_ct1.get_local_id(2) % warp_size;
        if (lane_id == 0) {
            shared_vals[warp_id] = val;
        }
        item_ct1.barrier(sycl::access::fence_space::local_space);

        size_t nreduce = nwarps / WARP_SIZE;
        float tmp = 0.f;
        if (lane_id < (static_cast<int>(block_size) / warp_size)) {
            for (size_t i = 0; i < nreduce; i += 1)
            {
                tmp += shared_vals[lane_id + i * WARP_SIZE];
            }
        }
        return block_reduce_policy<reduce_method_t, T, warp_size>::reduce(tmp);
    }
    return val;
}

static __dpct_inline__ float ggml_sycl_ue4m3_to_fp32(uint8_t x) {
    // UE4M3 is unsigned: 4 exp bits (bias 7), 3 mantissa bits, no sign, no NaN.
    // exp == 0xF is a valid exponent (256-448 range), not NaN.
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = man * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } else {
        raw = (1.0f + man / 8.0f) * sycl::pow(2.0f, (float) exp - 7.0f);
    }
    return raw * 0.5f;
}

#endif // GGML_SYCL_COMMON_HPP

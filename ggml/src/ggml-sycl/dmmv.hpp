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

#ifndef GGML_SYCL_DMMV_HPP
#define GGML_SYCL_DMMV_HPP

#include "common.hpp"

// The reordered-ESIMD K-quant DMMV kernels exist only on the Intel compiler: esimd.hpp
// pulls in sycl/ext/intel/esimd.hpp, so dmmv.cpp guards both the include and the whole
// ESIMD section on this. It lives here, not in dmmv.cpp, because ggml-sycl.cpp needs the
// same answer when it reports the knob at startup.
#if defined(__INTEL_LLVM_COMPILER)
    #define GGML_SYCL_DMMV_HAS_ESIMD
#endif


void ggml_sycl_op_dequantize_mul_mat_vec(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

// Fused dense-FFN mat-vec: writes glu(gate . y, up . y) for same-type reordered
// weights (Q2_K..Q6_K, Q8_0) and a single F32 activation column, using the ESIMD DMMV kernels.
// vx is the up weight, vgate the gate weight. Only defined when GGML_SYCL_DMMV_HAS_ESIMD;
// returns false if the type is unhandled.
bool ggml_sycl_dequantize_mul_mat_vec_glu_reorder_esimd(
    enum ggml_type src0_type, enum ggml_glu_op glu_op,
    const void * vx, const void * vgate, const float * y,
    float * dst, int ncols, int nrows, dpct::queue_ptr stream);

// Plain f32 mat-vec. Same op signature as the dequantise path so it plugs into
// ggml_sycl_op_mul_mat unchanged. Requires ne00 % (WARP_SIZE*4) == 0; callers must
// check the same condition before selecting it.
void ggml_sycl_op_mul_mat_vec_f32(
    ggml_backend_sycl_context & ctx,
    const ggml_tensor *src0, const ggml_tensor *src1, ggml_tensor *dst,
    const char *src0_dd_i, const float *src1_ddf_i, const char *src1_ddq_i,
    float *dst_dd_i, const int64_t row_low, const int64_t row_high,
    const int64_t src1_ncols, const int64_t src1_padded_row_size,
    const dpct::queue_ptr &stream);

#ifdef GGML_SYCL_DMMV_HAS_ESIMD
// Echo the resolved reordered-ESIMD DMMV work-group width into ggml_check_sycl()'s
// "Running with Environment Variables:" block. Without it a run that never reaches an
// ESIMD K-quant DMMV launch -- GGML_SYCL_ENABLE_ESIMD=0, a non-K-quant model, a graph
// that never decodes -- says nothing at all about the knob, so a sweep arm that
// silently did nothing is indistinguishable from one that ran. Defined in dmmv.cpp
// because esimd.hpp, and the resolver behind it, is only included there.
void ggml_sycl_dmmv_esimd_wg_size_report();
#endif // GGML_SYCL_DMMV_HAS_ESIMD

#endif // GGML_SYCL_DMMV_HPP

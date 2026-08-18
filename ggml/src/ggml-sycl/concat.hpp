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

#ifndef GGML_SYCL_CONCAT_HPP
#define GGML_SYCL_CONCAT_HPP

#include "common.hpp"

void ggml_sycl_op_concat(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                         const ggml_tensor * gather = nullptr);

// GGML_SYCL_CONV_FOLD=0 disables the conv-state gather fold. A same-binary control is
// not a convenience here: the closure holding the previous binary is GC-able, and this
// A/B has already been lost to a garbage collection once.
bool ggml_sycl_conv_fold_enabled();

// GGML_SYCL_CONV_FOLD_DIAG=1 runs the gather anyway and byte-compares the address the
// fold reads against the copy the gather produced. Generation md5 is nondeterministic on
// this hardware, so this is the only exact gate available for a fold whose failure mode
// is one sequence of two silently reading the other's convolution state.
bool ggml_sycl_conv_fold_diag();

#endif // GGML_SYCL_CONCAT_HPP

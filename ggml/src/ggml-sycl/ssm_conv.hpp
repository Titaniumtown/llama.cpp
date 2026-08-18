#pragma once

#include "common.hpp"

// returns true if the conv-state writeback CPY (wb) was folded into the dispatch
bool ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * wb = nullptr);
bool ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                              const ggml_tensor * wb = nullptr);

bool ggml_sycl_conv_wb_enabled();
bool ggml_sycl_ssm_conv_prefers_tiled(int d_conv, int d_inner, int n_t);
bool ggml_sycl_conv_wb_diag();

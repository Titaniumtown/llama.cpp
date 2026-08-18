#pragma once

#include "common.hpp"

// returns true if the conv-state writeback CPY (wb) was folded into the dispatch
bool ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * wb = nullptr);
bool ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                              const ggml_tensor * wb = nullptr);

bool ggml_sycl_conv_wb_enabled();
bool ggml_sycl_conv_collapse_enabled();
bool ggml_sycl_conv_collapse_diag();

// Collapses GET_ROWS + CONCAT + SSM_CONV + CPY into one dispatch at a decode width of one
// token. gather supplies the cache and its row indices, concat the projection (src[1]),
// wb the destination state view.
bool ggml_sycl_ssm_conv_collapsed(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                                  const ggml_tensor * gather, const ggml_tensor * concat, const ggml_tensor * wb);
bool ggml_sycl_ssm_conv_prefers_tiled(int d_conv, int d_inner, int n_t);
bool ggml_sycl_conv_wb_diag();

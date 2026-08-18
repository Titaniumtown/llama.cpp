#pragma once

#include <sycl/sycl.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_sycl_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

// beta_sigmoid: apply sigmoid to src[4] at the kernel's single beta load, so a preceding
// standalone SIGMOID node can be skipped entirely. Defaults off; the caller opts in only
// after checking that the sigmoid feeds this GDN and nothing else.
void ggml_sycl_op_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool beta_sigmoid = false,
                                  const ggml_tensor * state_gather = nullptr);
void ggml_sycl_gated_delta_net(ggml_backend_sycl_context & ctx, ggml_tensor * dst, bool beta_sigmoid = false,
                               const ggml_tensor * state_gather = nullptr);

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_sycl_try_gdn_cache_fusion)
void ggml_sycl_op_gated_delta_net_fused_cache(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                              ggml_sycl_gated_delta_net_fused_cache cache, bool beta_sigmoid = false,
                                              const ggml_tensor * state_gather = nullptr);

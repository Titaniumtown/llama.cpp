#pragma once

#include "ggml-sycl.h"

#include <cstdint>

struct ggml_backend_sycl_context;
// Recompute the rows of an f16/bf16 oneDNN GEMM that a defective DPAS EU wrote wrong.
// A[batch, row, k] at a + batch*sa_batch + row*sa_row + k*sa_k, same for B (k, col) and C (row, col),
// C is f32. No effect when the device's DPAS self-test found no bad output channel.
void ggml_sycl_dpas_repair_gemm(ggml_backend_sycl_context & ctx,
                                const void * a, bool a_bf16, int64_t sa_k, int64_t sa_row, int64_t sa_batch,
                                const void * b, bool b_bf16, int64_t sb_k, int64_t sb_col, int64_t sb_batch,
                                float * c, int64_t sc_row, int64_t sc_col, int64_t sc_batch,
                                int m, int n, int k, int batches_a, int batches_b);

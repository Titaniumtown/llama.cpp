#pragma once

#include "common.hpp"

// returns true if the conv-state writeback CPY (wb) was folded into the dispatch
bool ggml_sycl_ssm_conv(ggml_backend_sycl_context & ctx, ggml_tensor * dst, const ggml_tensor * wb = nullptr);
bool ggml_sycl_ssm_conv_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                              const ggml_tensor * wb = nullptr);

bool ggml_sycl_conv_wb_enabled();
bool ggml_sycl_conv_collapse_enabled();

// Collapses GET_ROWS + CONCAT + SSM_CONV + CPY into one dispatch at a decode width of one
// token. gather supplies the cache and its row indices, concat the projection (src[1]),
// wb the destination state view.
bool ggml_sycl_conv_prefill_fold_enabled();
void ggml_sycl_ssm_conv_prefill_fold(ggml_backend_sycl_context & ctx, const ggml_tensor * cat,
                                     const ggml_tensor * conv, ggml_tensor * silu,
                                     float * wb_base, int64_t wb_stride, int64_t wb_s2, int wb_count);
bool ggml_sycl_ssm_conv_collapsed(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * silu_dst,
                                  const ggml_tensor * gather, const ggml_tensor * concat, const ggml_tensor * wb);
bool ggml_sycl_ssm_conv_prefers_tiled(int d_conv, int d_inner, int n_t);

#define GGML_SYCL_CONV_WINDOW_MAX_SLOTS 8

// Match the recurrent window, rollback cache copies, convolution and SiLU.
struct ggml_sycl_conv_window_match {
    const ggml_tensor * concat   = nullptr;
    ggml_tensor *       conv     = nullptr;
    ggml_tensor *       add      = nullptr;
    ggml_tensor *       silu     = nullptr;
    int                 conv_idx = -1;
    int                 last_idx = -1;  // the silu node: the last node the kernel writes
    int                 n_slots  = 0;
    const ggml_tensor * cont[GGML_SYCL_CONV_WINDOW_MAX_SLOTS] = {};
    const ggml_tensor * cpy[GGML_SYCL_CONV_WINDOW_MAX_SLOTS]  = {};
};

// Purely structural, so graph_optimize and the compute loop see the same answer.
bool ggml_sycl_match_conv_window(const ggml_cgraph * cgraph, int node_idx, ggml_sycl_conv_window_match & m);
void ggml_sycl_ssm_conv_window(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_window_match & m);

// The rs_replay conv chain at d_conv 4: GET_ROWS(cache, rows) -> GET_ROWS(hist, win) ->
// CONCAT(window, x, 1) -> CONT(TRANSPOSE) -> SSM_CONV (+ SILU), plus the CPY of the history tail
// back to the cache, as one dispatch. Strides and offsets are in elements.
#define GGML_SYCL_CONV_REPLAY_MAX_T  8
#define GGML_SYCL_CONV_REPLAY_MAX_NS 4
struct ggml_sycl_conv_replay_args {
    const float *   cache;      // the gathered tensor's src0
    int64_t         row_stride; // between cache rows
    const int32_t * rows;       // cache row of each seq
    int64_t         hist_off;   // history start inside a row
    int64_t         col_stride; // between history columns
    const int32_t * win;        // window columns [3, n_s]
    int64_t         win_stride;
    const float *   x;          // new tokens [channels, n_t, n_s]
    int64_t         x_t, x_s;
    const float *   w;          // conv weights [4, channels], contiguous
    float *         out;        // conv (or silu) output [channels, n_t, n_s]
    int64_t         out_t, out_s;
    float *         hd;         // history destination [channels, n_dst, n_s]
    int64_t         hd_col, hd_s;
    int             channels, n_t, n_s;
    int             j0, n_dst;  // the tail is columns [j0, j0 + n_dst) of [window | x]
    bool            silu;
};
bool ggml_sycl_conv_replay_fold_enabled();
void ggml_sycl_ssm_conv_replay(ggml_backend_sycl_context & ctx, const ggml_sycl_conv_replay_args & a);

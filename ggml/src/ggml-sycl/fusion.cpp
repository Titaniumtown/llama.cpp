#include "fusion.hpp"
#include "binbcast.hpp"

#include <algorithm>
#include <cstdlib>

// Two INDEPENDENT mat-vecs over one activation, written to two destinations. Same weight
// geometry requirement as the GLU pair -- identical shape and row stride in the reorder
// (SoA) layout, so the second reuses the first's per-block offsets -- but with no combining
// op, because these two feed unrelated consumers. The delta-net beta and alpha projections
// are exactly this: two build_lora_mm calls on the same `cur`, q4_K ne=[n_embd, n_v_heads].
bool ggml_sycl_should_fuse_mul_mat_pair(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * wa = a->src[0];
    const ggml_tensor * wb = b->src[0];
    if (wa == nullptr || wb == nullptr || wa == wb) {
        return false;
    }
    if (wa->type != GGML_TYPE_Q4_K || wb->type != GGML_TYPE_Q4_K) {
        return false;
    }
    if (!ggml_are_same_shape(wa, wb) || wa->nb[1] != wb->nb[1]) {
        return false;
    }
    if (a->src[1] == nullptr || a->src[1] != b->src[1] || a->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (a->src[1]->ne[1] != 1) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, b)) {
        return false;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return false;
    }
    const auto * ea = (const ggml_tensor_extra_gpu *) wa->extra;
    const auto * eb = (const ggml_tensor_extra_gpu *) wb->extra;
    if (!ea || !eb || !ea->optimized_feature.reorder || !eb->optimized_feature.reorder) {
        return false;
    }
    if (ggml_sycl_info().device_count > 1) {
        return false;
    }
    return true;
}

// The grouped (segmented-dispatch) form of the pair: two independent mat-vecs over
// one activation whose weights need NOT have the same shape -- only the same quant
// type, the same ncols and the reorder layout. This is the Q/K projection pair and
// the GDN qkv/gate pair in qwen3.5/3.6: same src1, 12288-vs-1024 and 10240-vs-6144
// rows respectively. Kernel: mul_mat_vec_q_reorder_grouped2 (segmented row space).
bool ggml_sycl_should_fuse_mul_mat_pair_grouped(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT) {
        return false;
    }
    const ggml_tensor * wa = a->src[0];
    const ggml_tensor * wb = b->src[0];
    if (wa == nullptr || wb == nullptr || wa == wb) {
        return false;
    }
    // Q4_K/Q5_K only: qwen3.5/3.6's Q/K and qkv/gate bandwidth pairs. q8_0 was
    // tried for qwen3.8's ssm_alpha/ssm_beta and attn V and engages ZERO times:
    // beta's dst arena slot is reused by alpha's own decay-chain scratch (the
    // region check correctly rejects), and Vcur is the trio's last mat-vec so a
    // forward scan finds only mismatched types ahead of it. See the series ledger.
    if (wa->type != wb->type || (wa->type != GGML_TYPE_Q4_K && wa->type != GGML_TYPE_Q5_K)) {
        return false;
    }
    // shared activation implies equal ncols; rows may differ
    if (wa->ne[0] != wb->ne[0] || wa->ne[0] % QK_K != 0) {
        return false;
    }
    if (wa->ne[2] != 1 || wa->ne[3] != 1 || wb->ne[2] != 1 || wb->ne[3] != 1) {
        return false;
    }
    if (a->src[1] == nullptr || a->src[1] != b->src[1] || a->src[1]->type != GGML_TYPE_F32) {
        return false;
    }
    if (a->src[1]->ne[1] != 1 || a->src[1]->ne[2] != 1 || a->src[1]->ne[3] != 1) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    if (!ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return false;
    }
    // a row-limited mat-vec (the MTP draft head) must keep its own dispatch
    if (ggml_mul_mat_get_row_limit(a) > 0 || ggml_mul_mat_get_row_limit(b) > 0) {
        return false;
    }
    const auto * ea = (const ggml_tensor_extra_gpu *) wa->extra;
    const auto * eb = (const ggml_tensor_extra_gpu *) wb->extra;
    if (!ea || !eb || !ea->optimized_feature.reorder || !eb->optimized_feature.reorder) {
        return false;
    }
    if (ggml_sycl_info().device_count > 1) {
        return false;
    }
    return true;
}

static bool ggml_sycl_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {
    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 is not handled by the fused write path
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }
    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }
    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }
    // the view must flatten two rope dims into one contiguous dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }
    // Which rope kernels carry the fused set_rows write. VISION is the one that still
    // does not: rope_vision() has no row_indices/set_rows_stride plumbing, and unlike
    // rope_multi() it is not otherwise identical to rope_neox(), so adding it there is
    // a separate change. Keep this an allowlist rather than "not VISION" so a new mode
    // is excluded until someone has looked at its kernel.
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX &&
        mode != GGML_ROPE_TYPE_MROPE && mode != GGML_ROPE_TYPE_IMROPE) {
        return false;
    }
    return true;
}

// mul_mat(gate) + mul_mat(up) + GLU: graph shape and tensor properties only. Backend state
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_op_mul_mat_glu_fused().
static bool ggml_sycl_should_fuse_mul_mat_glu(const ggml_tensor * gate, const ggml_tensor * up,
                                              const ggml_tensor * glu) {
    // the fused epilogue implements these two; the rest fall back to the standalone GLU kernels
    const ggml_glu_op glu_op = ggml_get_glu_op(glu);
    if (glu_op != GGML_GLU_OP_SWIGLU && glu_op != GGML_GLU_OP_GEGLU) {
        return false;
    }

    // the kernel always treats src[0] as the activated operand and src[1] as the multiplier
    if (ggml_get_op_params_i32(glu, 1) /* swapped */) {
        return false;
    }

    const ggml_tensor * wu  = up->src[0];
    const ggml_tensor * wg  = gate->src[0];
    const ggml_tensor * act = up->src[1];

    // one activation and one output indexing must serve both weights; the block types
    // may differ, since the plain-layout fused kernel runs each operand's own vec_dot
    // (different types then imply different byte strides, so only the shape must agree)
    if (!ggml_are_same_shape(wu, wg)) {
        return false;
    }
    if (act != gate->src[1]) {
        return false;
    }

    // fused K-quant GEMVs walk whole QK_K super-blocks: the reorder kernel covers same-type
    // q4_K, the plain-layout kernel covers q5_K / iq4_xs pairs incl. mixed gate/up types
    const bool reorder_pair = wu->type == GGML_TYPE_Q4_K && wg->type == GGML_TYPE_Q4_K;
    const bool plain_pair   = (wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_IQ4_XS) &&
                            (wg->type == GGML_TYPE_Q5_K || wg->type == GGML_TYPE_IQ4_XS);
#ifdef GGML_SYCL_MMVQ_HAS_XMX
    // same-type pairs of the types of the XMX kernels, which also take a few more columns
    // (the XMX kernels walk whole QK_K super-blocks, so they need rows of a multiple of QK_K)
    const bool xmx_pair = wu->type == wg->type && ggml_sycl_xmx_supports_type(wu->type) && wu->ne[0] % QK_K == 0;
#else
    const bool xmx_pair = false;
#endif // GGML_SYCL_MMVQ_HAS_XMX
    // the ESIMD kernel covers same-type q2_K..q6_K and q8_0, single column only (as unfused DMMV)
#ifdef GGML_SYCL_DMMV_HAS_ESIMD
    const bool esimd_pair   = g_ggml_sycl_enable_esimd && wu->type == wg->type && act->ne[1] == 1 &&
                            (wu->type == GGML_TYPE_Q2_K || wu->type == GGML_TYPE_Q3_K || wu->type == GGML_TYPE_Q4_K ||
                             wu->type == GGML_TYPE_Q5_K || wu->type == GGML_TYPE_Q6_K || wu->type == GGML_TYPE_Q8_0);
#else
    const bool esimd_pair   = false;
#endif
    // q8_0 rows are walked in 32-wide blocks, so they only need to be a multiple of QK8_0
    const int64_t row_align = esimd_pair && wu->type == GGML_TYPE_Q8_0 ? QK8_0 : QK_K;
    if ((!reorder_pair && !plain_pair && !xmx_pair && !esimd_pair) || wu->ne[0] % row_align != 0) {
        return false;
    }

    // one 2D reorder-layout matrix in, a plain column stride out: no broadcast or padding
    if (!ggml_is_contiguous(wu) || !ggml_is_contiguous(wg) || !ggml_is_contiguous(act) ||
        !ggml_is_contiguous(glu)) {
        return false;
    }
    if (act->type != GGML_TYPE_F32 || glu->type != GGML_TYPE_F32) {
        return false;
    }
    if (act->ne[2] != 1 || act->ne[3] != 1 || wu->ne[2] != 1 || wu->ne[3] != 1) {
        return false;
    }
    // the kernel writes rows [0, wu->ne[1]) of each glu column, strided by glu->ne[0]
    if (glu->ne[0] != wu->ne[1] || glu->ne[1] != act->ne[1]) {
        return false;
    }
    // mat-vec only: one column per decoded token, up to the batch the reorder kernels cover
    const int max_cols = xmx_pair ? std::max(MMVQ_MAX_BATCH_SIZE, g_ggml_sycl_xmx_glu_max_cols) : MMVQ_MAX_BATCH_SIZE;
    if (act->ne[1] > max_cols) {
        return false;
    }

    return true;
}

// the unary ops the fused unary chains in element_wise.cpp have a functor for
static bool ggml_sycl_fused_unary_has_kernel(ggml_unary_op unary_op) {
    return unary_op == GGML_UNARY_OP_SILU || unary_op == GGML_UNARY_OP_SIGMOID ||
           unary_op == GGML_UNARY_OP_SOFTPLUS;
}

bool ggml_sycl_can_fuse(const ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops,
                        std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    if (!g_ggml_sycl_enable_fusion) {
        return false;
    }

    // gate and up are siblings, not a chain, so ggml_can_fuse cannot express this: use the
    // subgraph form with the GLU as the only materialised output.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_MUL_MAT &&
        ops.begin()[2] == GGML_OP_GLU) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * glu  = cgraph->nodes[node_idx + 2];
        const ggml_tensor * gate = glu->src[0];
        const ggml_tensor * up   = glu->src[1];

        // don't assume which of the two mat-muls is the gate; infer it from the GLU's operands
        const bool ok = (gate == cgraph->nodes[node_idx] && up == cgraph->nodes[node_idx + 1]) ||
                        (gate == cgraph->nodes[node_idx + 1] && up == cgraph->nodes[node_idx]);
        if (!ok) {
            return false;
        }

        return ggml_sycl_should_fuse_mul_mat_glu(gate, up, glu);
    }

    // the residual sum stays materialised (a later residual add reads it), so ggml_can_fuse cannot
    // express this either: the add and the mul are the subgraph's outputs
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM &&
        ops.begin()[2] == GGML_OP_MUL) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx, node_idx + 2 })) {
            return false;
        }

        const ggml_tensor * add = cgraph->nodes[node_idx];
        if (cgraph->nodes[node_idx + 1]->src[0] != add) {
            return false;
        }

        // the fused kernel does not broadcast the add
        if (add->src[0]->type != GGML_TYPE_F32 || add->src[1]->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(add->src[0], add) || !ggml_are_same_shape(add->src[1], add)) {
            return false;
        }

        return ggml_sycl_can_fuse(cgraph, node_idx + 1, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {});
    }

    // rope + view + set_rows: the view feeds set_rows at node_idx + 2, so use the
    // subgraph check (plain ggml_can_fuse only matches a linear chain)
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ROPE && ops.begin()[1] == GGML_OP_VIEW &&
        ops.begin()[2] == GGML_OP_SET_ROWS) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }
        return ggml_sycl_should_fuse_rope_set_rows(cgraph->nodes[node_idx], cgraph->nodes[node_idx + 1],
                                                   cgraph->nodes[node_idx + 2]);
    }

    // mul_mat + reshape + add(bias) + softplus + mul(scale): the GDN decay chain.
    // These are disjoint DFS subtrees from the delta-net op, so the two 5120x48 mat-vecs
    // (alpha, beta) can never be adjacent -- but each IS adjacent to its own epilogue,
    // which is what this fuses. The reshape is a view and costs no dispatch; it is listed
    // only because it occupies a graph node slot. The mat-vec output is elided (nothing
    // else reads it), so the single declared output is the final mul.
    if (ops.size() == 5 && ops.begin()[0] == GGML_OP_MUL_MAT && ops.begin()[1] == GGML_OP_RESHAPE &&
        ops.begin()[2] == GGML_OP_ADD && ops.begin()[3] == GGML_OP_UNARY && ops.begin()[4] == GGML_OP_MUL) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
            return false;
        }
        const ggml_tensor * mm  = cgraph->nodes[node_idx];
        const ggml_tensor * rsh = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add = cgraph->nodes[node_idx + 2];
        const ggml_tensor * sp  = cgraph->nodes[node_idx + 3];
        const ggml_tensor * mul = cgraph->nodes[node_idx + 4];
        if (ggml_get_unary_op(sp) != GGML_UNARY_OP_SOFTPLUS) { return false; }
        // strict chain: anything else means an operand we would silently drop
        if (rsh->src[0] != mm || add->src[0] != rsh || sp->src[0] != add || mul->src[0] != sp) { return false; }
        const ggml_tensor * bias  = add->src[1];
        const ggml_tensor * scale = mul->src[1];
        if (bias == nullptr || scale == nullptr) { return false; }
        // the epilogue is applied per output row, so both operands must be exactly one
        // value per row -- a broadcast over anything else is a different computation
        if (bias->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) { return false; }
        if (!ggml_is_contiguous(bias) || !ggml_is_contiguous(scale))     { return false; }
        if (ggml_nelements(bias) != mm->ne[0] || ggml_nelements(scale) != mm->ne[0]) { return false; }
        if (mm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32)     { return false; }
        if (ggml_nelements(mm) != mm->ne[0] || ggml_nelements(mul) != mm->ne[0]) { return false; }
        return true;
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        if (ops.size() == 3 && ops.begin()[2] != GGML_OP_ADD) {
            return false;
        }

        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * add      = ops.size() == 3 ? cgraph->nodes[node_idx + 2] : nullptr;

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        // if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        const ggml_tensor * mul_w = (mul->src[0] == rms_norm) ? mul->src[1] : mul->src[0];
        // the fused kernel indexes the weight as mul[col], so it must span ncols contiguously
        if (mul_w->ne[0] != rms_norm->ne[0] || mul_w->nb[0] != ggml_type_size(mul_w->type)) {
            return false;
        }

        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add != nullptr) {
            if (add->src[0]->type != GGML_TYPE_F32 ||
                add->src[1]->type != GGML_TYPE_F32 ||
                add->type != GGML_TYPE_F32) {
                return false;
            }

            // the fused kernel indexes the residual as add[col] and does not broadcast it
            const ggml_tensor * add_w = (add->src[0] == mul) ? add->src[1] : add->src[0];
            if (!ggml_are_same_shape(add_w, add)) {
                return false;
            }

            if (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1])) {
                return false;
            }
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_ADD) {
        const ggml_tensor * add0 = cgraph->nodes[node_idx];
        const ggml_tensor * add1 = cgraph->nodes[node_idx + 1];
        // ggml_can_fuse already guarantees add1 consumes add0 and that add0 has a single use.
        // Keep the CUDA association: the running sum is src0 of the next ADD so the fused
        // float fold matches two sequential add() launches.
        if (add1->src[0] != add0) {
            return false;
        }

        const ggml_tensor * c = add1->src[1];
        if (!ggml_sycl_add_kernel_supports(add0->src[0]->type, add0->src[1]->type, add0->type) ||
            !ggml_sycl_add_kernel_supports(add0->type, c->type, add1->type)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        if (!ggml_sycl_fused_unary_has_kernel(unary_op)) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }

        // the kernel indexes both sources and the destination FLAT (a[i], b[i], dst[i]
        // over nelements), so all three must be fully contiguous and equally shaped.
        // contiguous_1 is not enough: a permuted tensor keeps contiguous rows while its
        // higher-dim strides are shuffled, and the flat walk then reads the wrong bytes
        // -- accepted here, it died on the kernel's assert instead of declining
        // (test-backend-ops SILU_MUL f16 ne=[5,7,11,13] swap=1).
        if (!ggml_is_contiguous(unary->src[0]) || !ggml_is_contiguous(other) ||
            !ggml_are_same_shape(other, unary) || !ggml_is_contiguous(mul)) {
            return false;
        }

        // the 32-bit fastdiv is inexact past 2^31; decline, the unfused path handles it
        if (ggml_nelements(mul) >= ((int64_t) 1 << 31)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY &&
        unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx + 1];

        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }
        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel writes the SiLU output with dense strides, so it must be contiguous
        if (!ggml_is_contiguous(silu)) {
            return false;
        }

        return true;
    }


    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_SCALE) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * scale    = cgraph->nodes[node_idx + 1];
        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);
        if (scale->src[0]->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32) {
            return false;
        }
        // the fused kernel reads/writes rows flat like the unfused pair
        if (!ggml_is_contiguous_rows(rms_norm) || !ggml_is_contiguous_rows(scale)) {
            return false;
        }
        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1 &&
        (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID ||
         unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }
        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }
        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        // fused kernel indexes all three tensors flat: require full contiguity
        if (!ggml_is_contiguous(other) || !ggml_is_contiguous(unary->src[0]) ||
            !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    // add(bias) + unary + mul(scale): the delta-net alpha-gate softplus(alpha+dt)*a.
    // Linear chain, but bias and scale are ne0 vectors broadcast over the outer dims,
    // which stops the same-shape unary+mul fusion firing once n_tokens>1 (MTP verify).
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_UNARY &&
        ops.begin()[2] == GGML_OP_MUL && unary_ops.size() == 1 &&
        (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID ||
         unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
            return false;
        }
        const ggml_tensor * add   = cgraph->nodes[node_idx];
        const ggml_tensor * unary = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 2];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }
        // strict linear chain: unary consumes add, mul consumes unary
        if (unary->src[0] != add || (mul->src[0] != unary && mul->src[1] != unary)) {
            return false;
        }
        const ggml_tensor * a     = add->src[0];
        const ggml_tensor * bias  = add->src[1];
        const ggml_tensor * scale = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (a->type != GGML_TYPE_F32 || bias->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }
        // full activation contiguous, output same shape; bias/scale are ne0 vectors
        // broadcast over the outer dims (a single contiguous row each)
        if (!ggml_is_contiguous(a) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(a, mul)) {
            return false;
        }
        if (bias->ne[0] != a->ne[0] || scale->ne[0] != a->ne[0] ||
            ggml_nrows(bias) != 1 || ggml_nrows(scale) != 1 ||
            !ggml_is_contiguous(bias) || !ggml_is_contiguous(scale)) {
            return false;
        }
        return true;
    }

    return false;
}

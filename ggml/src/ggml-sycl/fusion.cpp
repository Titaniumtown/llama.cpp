#include "fusion.hpp"

#include <algorithm>
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
// (weight layout, split buffers, DMMV) is checked by ggml_sycl_mul_mat_glu_mmvq_fused().
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

    // one set of block offsets and one quantized activation must serve both weights
    if (wu->type != wg->type || !ggml_are_same_shape(wu, wg) || !ggml_are_same_stride(wu, wg)) {
        return false;
    }
    if (act != gate->src[1]) {
        return false;
    }

    // only q4_K has a fused reorder GEMV so far, and it walks whole super-blocks
    if (wu->type != GGML_TYPE_Q4_K || wu->ne[0] % QK_K != 0) {
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
    if (act->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }

    return true;
}

static bool ggml_sycl_should_fuse_rms_norm_mul(const ggml_tensor * rms_norm, const ggml_tensor * mul) {
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

    return true;
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

    // add(residual) + rms_norm + mul(weight). Every transformer block ends with a residual
    // add whose sum is read twice: once by the norm that follows it, and once by the NEXT
    // block's residual. That second use makes the sum a real output, so plain ggml_can_fuse
    // -- which requires single-use intermediates -- rejects the chain. The subgraph check
    // exempts the listed outputs from that requirement, which is exactly the shape here:
    // the add (node_idx) and the mul (node_idx + 2) are written, the rms_norm is elided.
    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_ADD && ops.begin()[1] == GGML_OP_RMS_NORM &&
        ops.begin()[2] == GGML_OP_MUL) {
        if (!ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx, node_idx + 2 })) {
            return false;
        }
        const ggml_tensor * add      = cgraph->nodes[node_idx];
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx + 1];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 2];

        if (rms_norm->src[0] != add) {
            return false;
        }
        if (add->type != GGML_TYPE_F32 || add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32) {
            return false;
        }
        // the kernel walks all three add operands with the sum's strides, so no broadcast
        // and no strided rows -- residual adds are always plain same-shape contiguous
        if (!ggml_are_same_shape(add->src[0], add->src[1]) || !ggml_are_same_shape(add->src[0], add)) {
            return false;
        }
        if (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous(add->src[1]) ||
            !ggml_is_contiguous(add)) {
            return false;
        }
        return ggml_sycl_should_fuse_rms_norm_mul(rms_norm, mul);
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        return ggml_sycl_should_fuse_rms_norm_mul(cgraph->nodes[node_idx], cgraph->nodes[node_idx + 1]);
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL &&
        unary_ops.size() == 1) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx + 1];

        const ggml_unary_op unary_op = ggml_get_unary_op(unary);
        if (unary_op != unary_ops.begin()[0]) {
            return false;
        }

        // the ops ggml_sycl_op_unary_mul_fused() has a kernel for
        if (unary_op != GGML_UNARY_OP_SILU && unary_op != GGML_UNARY_OP_SIGMOID &&
            unary_op != GGML_UNARY_OP_SOFTPLUS) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }

        // one row stride per source comes from nb[1], so rows must be contiguous and equally
        // shaped; the destination is written flat, so it must be fully contiguous
        if (!ggml_is_contiguous_1(unary->src[0]) || !ggml_is_contiguous_1(other) ||
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

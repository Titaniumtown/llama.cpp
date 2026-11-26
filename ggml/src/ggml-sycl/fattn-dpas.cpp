#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include "dpct/helper.hpp"
#include "common.hpp"
#include "fattn-common.hpp"
#include "fattn-dpas.hpp"
#include "dpas-check.hpp"

// GGML_SYCL_SUPPORT_XMX=OFF builds skip the DPAS below, as do spir64_gen builds for sub-group 8
// targets (Xe-HPG) or mixed target lists, which cannot compile it
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 16)
#define GGML_SYCL_FA_DPAS_BUILD 0
#else
#define GGML_SYCL_FA_DPAS_BUILD 1
#endif

#if GGML_SYCL_FA_DPAS_BUILD
static constexpr int   FA_DPAS_WG    = 256;      // work-group size
static constexpr int   FA_DPAS_NSG   = 16;       // sub-groups of 16 lanes
static constexpr int   FA_DPAS_MAX_Q = 31;       // larger batches go to the prefill kernels
static constexpr int   FA_DPAS_MAX_G = 32;       // query heads per KV head
static constexpr float FA_DPAS_MINF  = -1.0e30f; // finite start value of the running max

static bool fattn_dpas_support_arch(gpu_arch arch) {
    return arch == gpu_arch::intel_gpu_bmg_g21 || arch == gpu_arch::intel_gpu_bmg_g31 ||
           arch == gpu_arch::intel_gpu_lnl_m   || arch == gpu_arch::intel_gpu_ptl_h   ||
           arch == gpu_arch::intel_gpu_ptl_u   || arch == gpu_arch::intel_gpu_wcl;
}

// head sizes with a compiled kernel (multiples of the 32-dim q4_0 / q8_0 block)
static bool fattn_dpas_head_ok(int64_t d) {
    return d == 64 || d == 96 || d == 128 || d == 256 || d == 512;
}
#endif // GGML_SYCL_FA_DPAS_BUILD

bool ggml_sycl_flash_attn_ext_dpas_supported(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_FA_DPAS_BUILD
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    if (!g_ggml_sycl_fa_dpas || !fattn_dpas_support_arch(ggml_sycl_info().devices[device].hw_info.arch) ||
        ggml_sycl_dpas_faulty(device)) {
        return false;
    }

    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (Q->type != GGML_TYPE_F32 || (K->type != GGML_TYPE_Q4_0 && K->type != GGML_TYPE_Q8_0) || V->type != K->type ||
        (sinks && sinks->type != GGML_TYPE_F32)) {
        return false;
    }
    if (ggml_sycl_kv_reorder_seg(K) > 0 || ggml_sycl_kv_reorder_seg(V) > 0) {
        return false;
    }
    if (!fattn_dpas_head_ok(K->ne[0]) || V->ne[0] != K->ne[0] || Q->ne[0] != K->ne[0]) {
        return false;
    }
    if (Q->ne[1] < 1 || Q->ne[1] > FA_DPAS_MAX_Q || K->ne[1] < 1) {
        return false;
    }
    if (K->ne[2] < 1 || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0 || Q->ne[2] / K->ne[2] > FA_DPAS_MAX_G) {
        return false;
    }
    if (Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3]) {
        return false;
    }

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }

    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 || mask->nb[0] != sizeof(sycl::half) ||
                 mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])) {
        return false;
    }
    if (sinks && (sinks->nb[0] != sizeof(float) || sinks->ne[0] < Q->ne[2])) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // q4_0 / q8_0 rows are read as dwords from the dword at or before the row start: rows must start on 2-byte bounds
    // (head sizes with an odd block count, e.g. 96, put every other head 2 bytes into a dword)
    // so a row's first/last dword may hold 2 bytes outside it; aligned dwords never cross the 128-byte buffer alignment
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[0] != ggml_type_size(t->type) || t->nb[1] % 2 != 0 || t->nb[2] % 2 != 0 || t->nb[3] % 2 != 0 ||
            (uintptr_t) t->data % 2 != 0) {
            return false;
        }
    }
    return true;
#endif
}

bool ggml_sycl_flash_attn_ext_dpas_r_supported(int device, const ggml_tensor * dst) {
#if !GGML_SYCL_FA_DPAS_BUILD
    GGML_UNUSED(device);
    GGML_UNUSED(dst);
    return false;
#else
    static const int enabled = ggml_sycl_get_env("GGML_SYCL_FA_DPAS_R", 1);
    uint32_t bad_eu, bad_mask;
    if (!enabled || !fattn_dpas_support_arch(ggml_sycl_info().devices[device].hw_info.arch) ||
        !ggml_sycl_dpas_pair_repair(ggml_sycl_dpas_get_health(device), bad_eu, bad_mask)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    const bool types_ok = (K->type == GGML_TYPE_Q5_1 && V->type == GGML_TYPE_Q4_0) ||
                          (K->type == GGML_TYPE_Q4_0 && V->type == GGML_TYPE_Q4_0) ||
                          (K->type == GGML_TYPE_Q8_0 && V->type == GGML_TYPE_Q8_0);
    if (Q->type != GGML_TYPE_F32 || !types_ok || Q->ne[0] != 256 || K->ne[0] != 256 || V->ne[0] != 256 ||
        ggml_sycl_kv_reorder_seg(K) != 256 || ggml_sycl_kv_reorder_seg(V) != 256) {
        return false;
    }
    // decode widths: every query row of a KV head in one work-group (verify 5 x GQA 6 = 30)
    if (Q->ne[1] < 1 || K->ne[1] < 1 || K->ne[2] < 1 || V->ne[2] != K->ne[2] || Q->ne[2] % K->ne[2] != 0 ||
        Q->ne[1] * (Q->ne[2] / K->ne[2]) > 32 || Q->ne[3] != K->ne[3] || V->ne[3] != K->ne[3] ||
        V->ne[1] != K->ne[1]) {
        return false;
    }
    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f || (sinks && (sinks->type != GGML_TYPE_F32 || sinks->ne[0] < Q->ne[2]))) {
        return false;
    }
    // a full mask, or the implicit-causal sentinel [1, n_past + 1]
    if (mask && (mask->type != GGML_TYPE_F16 || mask->ne[2] != 1 ||
                 (mask->ne[0] != 1 && (mask->ne[0] < K->ne[1] || mask->ne[1] < Q->ne[1])))) {
        return false;
    }
    if (Q->nb[0] != sizeof(float)) {
        return false;
    }
    // the readers are 2D block loads: a 64-byte aligned surface per stream, a pitch of a multiple of 16 bytes,
    // a head segment of whole dwords, at most 2^24 rows
    for (const ggml_tensor * t : { K, V }) {
        if (t->nb[1] % 16 != 0 || t->nb[2] % 4 != 0 || t->nb[3] % 64 != 0 || (uintptr_t) t->data % 64 != 0 ||
            t->ne[1] > (1 << 24)) {
            return false;
        }
    }
    return true;
#endif
}

#if GGML_SYCL_FA_DPAS_BUILD

typedef short fa_dpas_short8 __attribute__((ext_vector_type(8)));
typedef int   fa_dpas_int4   __attribute__((ext_vector_type(4)));
typedef int   fa_dpas_int8   __attribute__((ext_vector_type(8)));
typedef float fa_dpas_float8 __attribute__((ext_vector_type(8)));

// Xe2 DPAS (SIMD16, systolic depth 8, repeat count 8) as inline vISA: joint_matrix has no s4 type.
// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s4, 16 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s4(const fa_dpas_short8 & a, const fa_dpas_int4 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=64 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s4.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// d = a x b. a: 8 rows x 32 s8, 2 per lane per row. b: 32 x 16 s8, 32 bytes per lane, one column per lane.
static inline void fa_dpas_s8_s8(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    d = 0;
    GGML_UNUSED(a);
    GGML_UNUSED(b);
#endif
}

// acc += a x b. a: 8 rows x 16 f16, one per lane per row. b: 16 x 16 f16 as pairs, 8 dwords per lane, one column per lane.
static inline void fa_dpas_f16(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED(a);
    GGML_UNUSED(b);
    GGML_UNUSED(acc);
#endif
}

// Block b of a row = f16 scale + NQ dwords of quants (q4_0: 18 bytes, NQ 4; q8_0: 34 bytes, NQ 8), at byte (2 + 4 NQ) b.
// w = the NQ + 1 dwords from that byte / 4. An odd block starts 2 bytes into w[0].
template <int NQ> static inline void fa_dpas_split_block(const uint32_t * w, bool odd, uint32_t & sc, uint32_t * q) {
    if (!odd) {
        sc = w[0] & 0xFFFFu;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = (w[i] >> 16) | (w[i + 1] << 16);
        }
    } else {
        sc = w[0] >> 16;
#pragma unroll
        for (int i = 0; i < NQ; ++i) {
            q[i] = w[i + 1];
        }
    }
}

struct fa_dpas_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;
    float *        dst;
    float *        parts;  // per-split partial results, nsplit > 1 only
    sycl::float2 * meta;   // per-split (max, sum), nsplit > 1 only
    float          scale;
    const float *  sinks;  // per query head softmax sink logit, or null
    int            G;      // query heads per KV head
    int            nq;     // query tokens per work-group
    int            nqc;    // token chunks
    int            ne01;
    int            ne02;
    int            ne11;
    int            ne33;
    int            nsplit;
    int            chunk;  // keys per split
    int64_t        nb01, nb02, nb03;
    int64_t        nb11, nb12, nb13;
    int64_t        nb21, nb22, nb23;
    int64_t        nb31, nb33;
};

// One work-group = one KV head x nq query tokens x one slice of the keys: S = Q K^T by s8 x s4 / s8 x s8 DPAS on the
// raw K blocks, online softmax, O^T += V^T P^T by f16 DPAS. ST: stage K/V tiles in SLM (1: 16-byte, 2: dword loads).
// LG: 256-register file (no spills at 32 rows, same occupancy). Q8: K/V are q8_0, else q4_0.
template <int D, int NR16, int KG, int TC, int ST, bool LG, bool Q8>
static void fattn_dpas_kernel(const fa_dpas_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr int RMAX  = 16 * NR16;
    constexpr int RTMAX = RMAX / 8;                         // 8-row tiles in Q.K
    constexpr int TK    = 16 * KG;                          // keys per tile
    static_assert(QK4_0 == QK8_0, "q4_0 and q8_0 blocks hold the same number of dims");
    constexpr int BB    = Q8 ? (int) sizeof(block_q8_0) : (int) sizeof(block_q4_0); // bytes per block
    constexpr int NQ    = (BB - 2) / 4;                     // quant dwords per block
    constexpr int NB    = D / QK4_0;                        // blocks per row
    constexpr int DPS   = D <= 128 ? 8 : D <= 256 ? 16 : 32; // head dims per sub-group in P.V, inside one 32-dim block
    constexpr int NPV   = D / DPS;                          // sub-groups doing P.V, the rest only reach its barrier
    constexpr bool ALLPV = NPV == FA_DPAS_NSG;              // every sub-group does P.V: no runtime check in the loops
    constexpr int XT    = DPS / 8;                          // 8-dim tiles per sub-group in P.V
    constexpr int PS    = TK + 16;                          // f16 row stride of P
    constexpr int RW    = NB * BB / 4;                      // dwords per row
    constexpr bool ODD  = (NB * BB) % 4 != 0;               // odd block count: rows may start 2 bytes into a dword
    constexpr int SW    = RW + 1;                           // odd SLM row stride: no bank conflicts for lane = key
    static_assert(D % QK4_0 == 0 && D % DPS == 0 && NPV <= FA_DPAS_NSG && QK4_0 % DPS == 0, "unsupported head size");
    static_assert(ST != 1 || RW % 4 == 0, "16-byte staging needs rows of 16-byte multiples");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<fa_dpas_short8, 1> sQ8(sycl::range<1>(RTMAX * NB * 16), cgh); // Q s8 [tile][block][lane]
        sycl::local_accessor<fa_dpas_float8, 1> sDQ(sycl::range<1>(RTMAX * NB), cgh);      // Q scales [tile][block]
        sycl::local_accessor<float, 1>          sS(sycl::range<1>(RMAX * TK), cgh);        // S [row][key]
        sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RMAX * PS / 16), cgh);   // P f16 [row][key]
        sycl::local_accessor<float, 1>          sM(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sL(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<float, 1>          sA(sycl::range<1>(RMAX), cgh);
        sycl::local_accessor<uint32_t, 1>       sKV(sycl::range<1>(ST ? TK * SW : 1), cgh);
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03 * p.nqc, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                    sycl::range<3>(1, 1, FA_DPAS_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
                const int tid   = it.get_local_id(2);
                const int split = it.get_group(2);
                const int kvh   = it.get_group(1);
                const int seq   = it.get_group(0) / p.nqc;
                const int c0    = (it.get_group(0) % p.nqc) * p.nq;
                const int G     = p.G;
                const int R     = p.nq * G;
                const int RT    = (R + 7) / 8;
                auto      sg    = it.get_sub_group();
                const int w     = tid / 16;
                const int lane  = tid % 16;
                short *   sQs   = (short *) &sQ8[0];
                short *   sPs   = (short *) &sP[0];

                // Q -> s8, one scale per row and block. Lane l holds the 2 dims of K column elements 2 l, 2 l + 1:
                // q4_0 packs dims l and l + 16 into byte l, q8_0 stores dims in order
                for (int e = tid; e < RT * 8 * NB; e += FA_DPAS_WG) {
                    const int  r    = e / NB;
                    const int  b    = e % NB;
                    const int  c    = c0 + r / G;
                    const bool real = r < R && c < p.ne01;
                    const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) (real ? c : 0) * p.nb01 +
                                                        (int64_t) (kvh * G + (real ? r % G : 0)) * p.nb02) + b * QK4_0;
                    float x[QK4_0];
                    float amax = 0.0f;
#pragma unroll
                    for (int i = 0; i < QK4_0; ++i) {
                        x[i] = real ? qr[i] * p.scale : 0.0f;
                        amax = sycl::fmax(amax, sycl::fabs(x[i]));
                    }
                    const float dq = amax / 127.0f;
                    const float iq = amax > 0.0f ? 127.0f / amax : 0.0f;
                    const int   t  = r / 8;
                    const int   m  = r % 8;
                    ((float *) &sDQ[0])[(t * NB + b) * 8 + m] = dq;
#pragma unroll
                    for (int l = 0; l < 16; ++l) {
                        const int lo = (int) sycl::rint(x[Q8 ? 2 * l : l] * iq);
                        const int hi = (int) sycl::rint(x[Q8 ? 2 * l + 1 : l + 16] * iq);
                        sQs[((t * NB + b) * 16 + l) * 8 + m] = (short) ((lo & 0xFF) | ((hi & 0xFF) << 8));
                    }
                }
                // rows without a query keep P = 0
                for (int e = tid; e < RMAX * PS; e += FA_DPAS_WG) {
                    sPs[e] = 0;
                }
                if (tid < RMAX) {
                    // the sink is one more key with logit sinks[head] and no value; the first key slice owns it
                    const bool sink = p.sinks && split == 0 && tid < R;
                    sM[tid] = sink ? p.sinks[kvh * G + tid % G] : FA_DPAS_MINF;
                    sL[tid] = sink ? 1.0f : 0.0f;
                    sA[tid] = 1.0f;
                }
                it.barrier(sycl::access::fence_space::local_space);

                const int    k_begin = split * p.chunk;
                const int    k_end   = sycl::min(k_begin + p.chunk, p.ne11);
                const char * Kh      = p.K + seq * p.nb13 + kvh * p.nb12;
                const char * Vh      = p.V + seq * p.nb23 + kvh * p.nb22;
                const sycl::half * mrow = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;

                // O^T: lane = query row, element m = head dim
                fa_dpas_float8 o[NR16][XT];
#pragma unroll
                for (int t = 0; t < NR16; ++t) {
#pragma unroll
                    for (int x = 0; x < XT; ++x) {
                        o[t][x] = 0.0f;
                    }
                }
                // the dims of this sub-group lie in one V block
                const int vb  = (DPS * w) / QK4_0;
                const int vd0 = (DPS * w) % QK4_0;

                // keys past k_end repeat key k_begin, their scores are -inf
                auto stage = [&](const char * base, int64_t nb, int t0) {
                    if constexpr (ST == 2) {
                        for (int e = tid; e < TK * RW; e += FA_DPAS_WG) {
                            const int key = e / RW;
                            const int q   = e - key * RW;
                            const int j   = t0 + key < k_end ? t0 + key : k_begin;
                            sKV[key * SW + q] = ((const uint32_t *) (base + (int64_t) j * nb))[q];
                        }
                        return;
                    }
                    constexpr int RV  = ST == 1 ? RW / 4 : 1;
                    constexpr int NV  = TK * RV;
                    constexpr int PER = (NV + FA_DPAS_WG - 1) / FA_DPAS_WG;
                    sycl::uint4 v[PER];
#pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int j   = t0 + key < k_end ? t0 + key : k_begin;
                            v[i] = ((const sycl::uint4 *) (base + (int64_t) j * nb))[q];
                        }
                    }
#pragma unroll
                    for (int i = 0; i < PER; ++i) {
                        const int e = tid + i * FA_DPAS_WG;
                        if (e < NV) {
                            const int key = e / RV;
                            const int q   = e - key * RV;
                            const int o   = key * SW + 4 * q;
                            sKV[o + 0] = v[i][0];
                            sKV[o + 1] = v[i][1];
                            sKV[o + 2] = v[i][2];
                            sKV[o + 3] = v[i][3];
                        }
                    }
                };

                // the NQ + 1 dwords of block b: of tile key kk in SLM, or of a row that starts 2 ph bytes into dword row[0]
                auto load_block = [&](const uint32_t * row, int ph, int kk, int b, uint32_t * wv) {
                    const int wd = (2 * ph + BB * b) / 4;
                    if constexpr (ST) {
                        GGML_UNUSED(row);
#pragma unroll
                        for (int i = 0; i < NQ + 1; ++i) {
                            wv[i] = sKV[kk * SW + wd + i];
                        }
                    } else {
                        GGML_UNUSED(kk);
#pragma unroll
                        for (int i = 0; i < NQ + 1; ++i) {
                            wv[i] = row[wd + i];
                        }
                    }
                };

                // start loading the rows of keys t0 .. t0 + TK - 1 into L1 while the current phase computes
                auto prefetch_rows = [&](const char * base, int64_t nb, int t0) {
                    constexpr int NL = RW * 4 / 64 + 1; // cache lines per row
                    for (int e = tid; e < TK * NL; e += FA_DPAS_WG) {
                        const int j = t0 + e / NL;
                        if (j < k_end) {
                            char * pa = const_cast<char *>(base) + (int64_t) j * nb + (e % NL) * 64;
                            sycl::ext::oneapi::experimental::prefetch(pa, sycl::ext::oneapi::experimental::properties{ sycl::ext::oneapi::experimental::prefetch_hint_L1 });
                        }
                    }
                };

                for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                    if constexpr (ST) {
                        stage(Kh, p.nb11, t0);
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                    prefetch_rows(Vh, p.nb21, t0);

                    // S = Q K^T, job = (key group, TC row tiles)
                    const int nch = (RT + TC - 1) / TC;
                    for (int job = w; job < KG * nch; job += FA_DPAS_NSG) {
                        const int  kg = job % KG;
                        const int  ch = job / KG;
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const char *     kb = Kh + (int64_t) (jv ? j : k_begin) * p.nb11;
                        const int        kp = (ST || !ODD) ? 0 : (int) (((uintptr_t) kb >> 1) & 1); // row starts 2 bytes into a dword
                        const uint32_t * kr = (const uint32_t *) (kb - 2 * kp);
                        fa_dpas_float8 s[TC];
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            s[tt] = 0.0f;
                        }
#pragma unroll
                        for (int b = 0; b < NB; ++b) {
                            uint32_t wv[NQ + 1];
                            uint32_t sc;
                            uint32_t q[NQ];
                            load_block(kr, kp, kg * 16 + lane, b, wv);
                            fa_dpas_split_block<NQ>(wv, (b + kp) % 2 != 0, sc, q);
                            const float dk = static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc));
#pragma unroll
                            for (int tt = 0; tt < TC; ++tt) {
                                const int t = ch * TC + tt;
                                if (t < RT) {
                                    fa_dpas_int8 ia;
                                    if constexpr (Q8) {
                                        const fa_dpas_int8 bv = { (int) q[0], (int) q[1], (int) q[2], (int) q[3],
                                                                  (int) q[4], (int) q[5], (int) q[6], (int) q[7] };
                                        fa_dpas_s8_s8(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    } else {
                                        // u4 (n + 8) -> s4 n
                                        const fa_dpas_int4 bv = { (int) (q[0] ^ 0x88888888u), (int) (q[1] ^ 0x88888888u),
                                                                  (int) (q[2] ^ 0x88888888u), (int) (q[3] ^ 0x88888888u) };
                                        fa_dpas_s8_s4(sQ8[(t * NB + b) * 16 + lane], bv, ia);
                                    }
                                    const fa_dpas_float8 ds = sDQ[t * NB + b] * dk;
                                    s[tt] += __builtin_convertvector(ia, fa_dpas_float8) * ds;
                                }
                            }
                        }
#pragma unroll
                        for (int tt = 0; tt < TC; ++tt) {
                            const int t = ch * TC + tt;
                            if (t < RT) {
#pragma unroll
                                for (int m = 0; m < 8; ++m) {
                                    const int r = t * 8 + m;
                                    const int c = c0 + r / G;
                                    if (r < R) {
                                        const bool  ok = jv && c < p.ne01;
                                        const float mv = (ok && mrow) ? static_cast<float>(mrow[((int64_t) c * p.nb31) / (int64_t) sizeof(sycl::half) + j]) : 0.0f;
                                        sS[r * TK + kg * 16 + lane] = ok ? s[tt][m] + mv : -INFINITY;
                                    }
                                }
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr (ST) {
                        stage(Vh, p.nb21, t0); // the softmax does not use sKV
                    }

                    // online softmax per row -> P
                    for (int r = w; r < R; r += FA_DPAS_NSG) {
                        float sv[KG];
                        float mt = -INFINITY;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            sv[i] = sS[r * TK + i * 16 + lane];
                            mt    = sycl::fmax(mt, sv[i]);
                        }
                        mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                        const float m_old = sM[r];
                        const float m_new = sycl::fmax(m_old, mt);
                        float       lt    = 0.0f;
#pragma unroll
                        for (int i = 0; i < KG; ++i) {
                            const float pv = sycl::native::exp(sv[i] - m_new);
                            sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                            lt += pv;
                        }
                        lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                        if (lane == 0) {
                            const float a = sycl::native::exp(m_old - m_new);
                            sA[r] = a;
                            sL[r] = sL[r] * a + lt;
                            sM[r] = m_new;
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);

                    prefetch_rows(Kh, p.nb11, t0 + TK);

                    // O^T = diag(alpha) O^T + V^T P^T. A = V^T (8 dims x 16 keys, lane = key), B = P^T (16 keys x 16 rows)
#pragma unroll
                    for (int t = 0; t < NR16; ++t) {
                        const float a = sA[t * 16 + lane];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
                            o[t][x] *= a;
                        }
                    }
#pragma unroll 1
                    for (int kg = 0; kg < ((ALLPV || w < NPV) ? KG : 0); ++kg) {
                        const int  j  = t0 + kg * 16 + lane;
                        const bool jv = j < k_end;
                        const char *     vbp = Vh + (int64_t) (jv ? j : k_begin) * p.nb21;
                        const int        vp  = (ST || !ODD) ? 0 : (int) (((uintptr_t) vbp >> 1) & 1);
                        const uint32_t * vr  = (const uint32_t *) (vbp - 2 * vp);
                        uint32_t wv[NQ + 1];
                        uint32_t sc;
                        uint32_t q[NQ];
                        load_block(vr, vp, kg * 16 + lane, vb, wv);
                        fa_dpas_split_block<NQ>(wv, (vb + vp) % 2 != 0, sc, q);
                        const float dv = jv ? static_cast<float>(sycl::bit_cast<sycl::half>((uint16_t) sc)) : 0.0f;
                        fa_dpas_short8 va[XT];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = vd0 + x * 8 + m;
                                int       v;
                                if constexpr (Q8) {
                                    v = (int8_t) ((q[d / 4] >> (8 * (d % 4))) & 0xFF);
                                } else {
                                    const int byte = d % 16;
                                    v = (int) ((q[byte / 4] >> (8 * (byte % 4) + 4 * (d / 16))) & 0xF) - 8;
                                }
                                va[x][m] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) v * dv));
                            }
                        }
#pragma unroll
                        for (int t = 0; t < NR16; ++t) {
                            const fa_dpas_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
#pragma unroll
                            for (int x = 0; x < XT; ++x) {
                                fa_dpas_f16(va[x], pb, o[t][x]);
                            }
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                }

#pragma unroll
                for (int t = 0; t < NR16; ++t) {
                    const int r = t * 16 + lane;
                    const int c = c0 + r / G;
                    if ((ALLPV || w < NPV) && r < R && c < p.ne01) {
                        const int64_t jdu = ((int64_t) seq * p.ne01 + c) * p.ne02 + kvh * G + r % G;
                        const float   l   = sL[r];
#pragma unroll
                        for (int x = 0; x < XT; ++x) {
#pragma unroll
                            for (int m = 0; m < 8; ++m) {
                                const int d = w * DPS + x * 8 + m;
                                if (p.nsplit == 1) {
                                    p.dst[jdu * D + d] = l > 0.0f ? o[t][x][m] / l : 0.0f;
                                } else {
                                    p.parts[(jdu * p.nsplit + split) * D + d] = o[t][x][m];
                                }
                            }
                        }
                    }
                }
                if (p.nsplit > 1 && tid < R && c0 + tid / G < p.ne01) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + c0 + tid / G) * p.ne02 + kvh * G + tid % G;
                    p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
                }
            };
        if constexpr (LG) {
            cgh.parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
        } else {
            cgh.parallel_for(ndr, kern);
        }
    });
}

template <int D, int NR16, int KG, int TC, int ST, bool Q8>
static void fattn_dpas_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    constexpr int TK = 16 * KG;

    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int ne01 = Q->ne[1];
    const int ne11 = K->ne[1];
    const int nkvh = K->ne[2];
    const int ne03 = Q->ne[3];

    fa_dpas_params p;
    p.Q    = (const char *) Q->data;
    p.K    = (const char *) K->data;
    p.V    = (const char *) V->data;
    p.mask = mask ? (const char *) mask->data : nullptr;
    p.sinks = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst  = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.G    = Q->ne[2] / K->ne[2];
    p.nq   = std::max(1, std::min(ne01, 16 * NR16 / p.G));
    p.nqc  = (ne01 + p.nq - 1) / p.nq;
    p.ne01 = ne01;
    p.ne02 = Q->ne[2];
    p.ne11 = ne11;
    p.ne33 = mask ? mask->ne[3] : 1;
    p.nb01 = Q->nb[1];
    p.nb02 = Q->nb[2];
    p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];
    p.nb12 = K->nb[2];
    p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];
    p.nb22 = V->nb[2];
    p.nb23 = V->nb[3];
    p.nb31 = mask ? mask->nb[1] : 0;
    p.nb33 = mask ? mask->nb[3] : 0;

    // split the keys so that all work-groups run in one wave
    // SLM fits about 4 work-groups per Xe2 core with 16 rows, 2 with 32. Xe2 core = 8 EUs, nsm = EUs / 16.
    const int xe_cores = std::max(1, ggml_sycl_info().devices[ctx.device].nsm * 2);
    const int wave     = xe_cores * (NR16 == 1 ? 4 : 2);
    const int ntiles   = (ne11 + TK - 1) / TK;
    int nsplit = std::max(1, std::min(wave / (nkvh * ne03 * p.nqc), ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * TK;
    nsplit     = (ne11 + p.chunk - 1) / p.chunk;
    p.nsplit   = nsplit;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    // the large register file pays off for 32-row launches from about 4K keys
    if (NR16 == 2 && ne11 >= 4096) {
        fattn_dpas_kernel<D, NR16, KG, TC, ST, true, Q8>(p, nkvh, ne03, stream);
    } else {
        fattn_dpas_kernel<D, NR16, KG, TC, ST, false, Q8>(p, nkvh, ne03, stream);
    }

    // copied into a non-static local: a runtime-initialized static const is not
    // constant-initialized, and oneAPI 2026 rejects it inside a kernel.
    static const int combine_unroll_env = ggml_sycl_get_env("GGML_SYCL_FA_COMBINE_UNROLL", 4);
    const int        combine_unroll     = combine_unroll_env;
    if (nsplit > 1) {
        float *        parts_p = parts.get();
        sycl::float2 * meta_p  = meta.get();
        float *        dst_p   = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nsplit * sizeof(sycl::float2)), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(Q->ne[3], Q->ne[2], (size_t) ne01 * D), sycl::range<3>(1, 1, D)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_combine_results<D>(parts_p, meta_p, dst_p, nsplit,
                                                 combine_unroll,
                                                  lm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

template <bool Q8> static void fattn_dpas_dispatch(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];

    const int D    = K->ne[0];
    const int G    = Q->ne[2] / K->ne[2];
    const int ne01 = Q->ne[1];

    bool aligned16 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned16 = aligned16 && t->nb[1] % 16 == 0 && t->nb[2] % 16 == 0 && t->nb[3] % 16 == 0 &&
                    (uintptr_t) t->data % 16 == 0;
    }

    // single token decode: staged K/V tiles are faster at long context
    if (D == 256 && ne01 == 1 && G <= 16 && aligned16) {
        fattn_dpas_case<256, 1, 8, 1, 1, Q8>(ctx, dst);
        return;
    }
    const bool rows16 = ne01 * G <= 16;
    // 256-key tiles for one row tile; 128 at head size 512 so S and the Q copy stay in SLM for the planned waves
#define FATTN_DPAS_D(D_, KG16_)                                                                        \
    case D_:                                                                                           \
        if (rows16) {                                                                                  \
            fattn_dpas_case<D_, 1, KG16_, 1, 0, Q8>(ctx, dst);                                     \
        } else {                                                                                       \
            fattn_dpas_case<D_, 2, 8, 2, 0, Q8>(ctx, dst);                                         \
        }                                                                                              \
        break;
    // q8_0 at head size 128: its 136-byte rows gather poorly per lane, stage them in SLM by coalesced dword loads
    bool aligned4 = true;
    for (const ggml_tensor * t : { K, V }) {
        aligned4 = aligned4 && t->nb[1] % 4 == 0 && t->nb[2] % 4 == 0 && t->nb[3] % 4 == 0 && (uintptr_t) t->data % 4 == 0;
    }
    if (Q8 && D == 128 && aligned4) {
        if (rows16) {
            fattn_dpas_case<128, 1, 16, 1, 2, Q8>(ctx, dst);
        } else {
            fattn_dpas_case<128, 2, 8, 2, 2, Q8>(ctx, dst);
        }
        return;
    }
    switch (D) {
        FATTN_DPAS_D( 64, 16)
        FATTN_DPAS_D( 96, 16)
        FATTN_DPAS_D(128, 16)
        FATTN_DPAS_D(256, 16)
        FATTN_DPAS_D(512, 8)
        default:
            GGML_ABORT("fattn-dpas: unsupported head size %d", D);
    }
#undef FATTN_DPAS_D
}

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    if (dst->src[1]->type == GGML_TYPE_Q8_0) {
        fattn_dpas_dispatch<true>(ctx, dst);
    } else {
        fattn_dpas_dispatch<false>(ctx, dst);
    }
}

// ---- row-split ("reordered") KV caches, D 256 -------------------------------------------------------------
// A segment row is element-major: q4_0 [128 nibble bytes][8 f16 d], q5_1 [128 nibble bytes][32 B 5th-bit plane]
// [8 half2 (d, m)], q8_0 [256 s8][8 f16 d]; byte l of a nibble plane holds elements 2l (low) and 2l+1 (high).
// Both products are f16 DPAS on dequantized operands, so K and V are each dequantized once per work-group:
// S = Q K^T (A = Q, lane = dim; B = K^T, lane = key) and O^T += V^T P^T (A = V^T, lane = key; B = P^T, lane = row).
enum { FA_R_Q4_0 = 0, FA_R_Q5_1 = 1, FA_R_Q8_0 = 2 };

static constexpr int FA_R_D  = 256;
static constexpr int FA_R_TK = 256;  // keys per tile: one 16-key group per sub-group

struct fa_dpas_r_params {
    const char *   Q;
    const char *   K;
    const char *   V;
    const char *   mask;   // full mask, null for the implicit-causal sentinel
    float *        dst;
    float *        parts;
    sycl::float2 * meta;
    const float *  sinks;
    float          scale;
    int            G, ne01, ne02, ne33, nsplit, chunk;
    int            k_lim;   // keys past this are masked for every query (ne11, or n_past + ne01 when implicit)
    int            n_past;  // implicit causal: key j is visible to query column c iff j <= n_past + c; -1 = full mask
    uint32_t       bad_eu;   // sr0 & eu_mask of the EU whose dependent DPAS get output lanes bad_mask wrong
    uint32_t       bad_mask;
    uint32_t       eu_mask;
    int64_t        nb01, nb02, nb03;
    int64_t        nb11, nb12, nb13;
    int64_t        nb21, nb22, nb23;
    int64_t        nb31, nb33;
};

static inline uint32_t fa_dpas_sr0() {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t v;
    __asm__ volatile("mov (M1_NM, 16) %0(0,0)<1> %%sr0(0,0)<0;1,0>" : "=rw"(v));
    return v;
#else
    return 0;
#endif
}

static inline int fa_r_pack(float x, float y) {
    return (int) ((uint32_t) sycl::bit_cast<uint16_t>(sycl::half(x)) |
                  ((uint32_t) sycl::bit_cast<uint16_t>(sycl::half(y)) << 16));
}

// acc += a x b; on the defective EU its wrong output channels are recomputed by a second DPAS with the B columns
// swapped pairwise, which puts each of those columns on its neighbouring channel (the self-test requires it correct)
static inline void fa_r_dpas(const fa_dpas_short8 & a, const fa_dpas_int8 & b, fa_dpas_float8 & acc, const bool fix,
                             const uint32_t bad_mask, const sycl::sub_group & sg) {
    if (fix) {
        fa_dpas_int8 b2;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            b2[k] = sycl::permute_group_by_xor(sg, (int) b[k], 1);
        }
        fa_dpas_float8 t = 0.0f;
        fa_dpas_f16(a, b2, t);
        const fa_dpas_float8 keep = acc;
        fa_dpas_f16(a, b, acc);
        const bool mine = (bad_mask >> sg.get_local_linear_id()) & 1u;
#pragma unroll
        for (int m = 0; m < 8; ++m) {
            const float tm = sycl::permute_group_by_xor(sg, (float) t[m], 1);
            acc[m] = mine ? keep[m] + tm : acc[m];
        }
    } else {
        fa_dpas_f16(a, b, acc);
    }
}

// B = K^T for dims 16 h .. 16 h + 15 of one 32-dim block: lane = key, dword i = dims (2 i, 2 i + 1)
template <int KT>
static inline fa_dpas_int8 fa_r_k_operand(const sycl::uint4 * q, const uint32_t qh, const float d, const float m,
                                          const int h) {
    fa_dpas_int8 b;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if constexpr (KT == FA_R_Q8_0) {
            const uint32_t w  = q[h][i / 2];
            const int      sh = 16 * (i % 2);
            b[i] = fa_r_pack((float) (int8_t) (w >> sh) * d, (float) (int8_t) (w >> (sh + 8)) * d);
        } else {
            const uint32_t byte = (q[0][(8 * h + i) / 4] >> (8 * ((8 * h + i) % 4))) & 0xFFu;
            if constexpr (KT == FA_R_Q5_1) {
                const uint32_t pr = qh >> (16 * h + 2 * i);
                const int      lo = (int) ((byte & 0xFu) | ((pr & 1u) << 4));
                const int      hi = (int) ((byte >> 4) | ((pr & 2u) << 3));
                b[i] = fa_r_pack((float) lo * d + m, (float) hi * d + m);
            } else {
                b[i] = fa_r_pack((float) ((int) (byte & 0xFu) - 8) * d, (float) ((int) (byte >> 4) - 8) * d);
            }
        }
    }
    return b;
}

#ifdef __SYCL_DEVICE_ONLY__
extern "C" SYCL_EXTERNAL int * __builtin_IB_subgroup_createBlock2DAddressPayload(long base, int width_minus_one,
                                                                                int height_minus_one, int pitch_minus_one,
                                                                                int block_x, int block_y, int block_width,
                                                                                int block_height, int num_blocks);
#endif
typedef int fa_dpas_int2 __attribute__((ext_vector_type(2)));

// Transposed 2D block load from a surface of h rows of pitch bytes at base (64-byte aligned): lane l receives
// dwords x .. x + W - 1 of row y + l, zero past the last row.
template <int W, typename T> static inline T fa_r_ld2d(const char * base, const int pitch, const int h, const int x, const int y) {
    T v;
#ifdef __SYCL_DEVICE_ONLY__
    int * pl = __builtin_IB_subgroup_createBlock2DAddressPayload((long) base, pitch - 1, h - 1, pitch - 1, x, y, W, 16, 1);
    __asm__("lsc_load_block2d.ugm (M1, 1) %0:d32.%2x16tn flat[%1+(0,0)]" : "=rw"(v) : "rw.u"(pl), "P"(W));
#else
    v = T{};
    GGML_UNUSED_VARS(base, pitch, h, x, y);
#endif
    return v;
}

// Prefetch of dwords x .. x + n - 1 of rows y .. y + 15 into the caches, in 64-byte wide blocks
static inline void fa_r_pf2d(const char * base, const int pitch, const int h, const int x, const int y, const int n) {
#ifdef __SYCL_DEVICE_ONLY__
    for (int i = 0; i < n; i += 16) {
        int * pl = __builtin_IB_subgroup_createBlock2DAddressPayload((long) base, pitch - 1, h - 1, pitch - 1, x + i, y, 16, 16, 1);
        __asm__ volatile("lsc_load_block2d.ugm.ca.ca (M1, 1) %%null:d32.16x16nn flat[%0+(0,0)]" ::"rw.u"(pl));
    }
#else
    GGML_UNUSED_VARS(base, pitch, h, x, y, n);
#endif
}

// One work-group = one KV head x all query rows (R = n_q x GQA <= 32) x one slice of the keys, in tiles of 256
// keys: sub-group w computes S for keys 16 w .. 16 w + 15 and P V for head dims 16 w .. 16 w + 15.
template <int KT, int VT, int NRQ, int NRP>
static void fattn_dpas_r_kernel(const fa_dpas_r_params p, const int nkvh, const int ne03, dpct::queue_ptr stream) {
    constexpr int D   = FA_R_D;
    constexpr int TK  = FA_R_TK;
    constexpr int KG  = TK / 16;      // key groups per tile, one per sub-group in S
    constexpr int NKS = D / 16;       // 16-dim steps of S
    constexpr int RQ  = 8 * NRQ;      // rows covered by S
    constexpr int RP  = 16 * NRP;     // rows covered by P V
    constexpr int PS  = TK + 16;      // f16 row stride of P
    static_assert(KG == FA_DPAS_NSG && D == 16 * FA_DPAS_NSG, "one key group and 16 dims per sub-group");
    static_assert(RP >= RQ, "P V covers every row of S");

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<fa_dpas_short8, 1> sQ(sycl::range<1>(NRQ * NKS * 16), cgh);  // Q f16 [tile][step][lane]
        sycl::local_accessor<float, 1>          sS(sycl::range<1>(RQ * TK), cgh);         // S [row][key]
        sycl::local_accessor<fa_dpas_int8, 1>   sP(sycl::range<1>(RP * PS / 16), cgh);    // P f16 [row][key]
        sycl::local_accessor<float, 1>          sM(sycl::range<1>(RP), cgh);
        sycl::local_accessor<float, 1>          sL(sycl::range<1>(RP), cgh);
        sycl::local_accessor<float, 1>          sA(sycl::range<1>(RP), cgh);
        const sycl::nd_range<3> ndr(sycl::range<3>((size_t) ne03, nkvh, (size_t) p.nsplit * FA_DPAS_WG),
                                    sycl::range<3>(1, 1, FA_DPAS_WG));
        auto kern = [=](sycl::nd_item<3> it) [[sycl::reqd_sub_group_size(16)]] {
            const int  tid   = it.get_local_id(2);
            const int  split = it.get_group(2);
            const int  kvh   = it.get_group(1);
            const int  seq   = it.get_group(0);
            const int  G     = p.G;
            const int  R     = p.ne01 * G;
            auto       sg    = it.get_sub_group();
            const int  w     = tid / 16;
            const int  lane  = tid % 16;
            const bool fix   = (fa_dpas_sr0() & p.eu_mask) == p.bad_eu;
            short *    sQs   = (short *) &sQ[0];
            short *    sPs   = (short *) &sP[0];

            // Q is rounded to the grid of the s8 Q a quantized-K dot uses (one scale per 32 dims, |q| <= 127, the
            // scale rounded to f16): the CPU reference's Q and TILE's integer path's. A plain f16 Q is closer to the
            // float Q but disagrees with every integer K dot by up to 1/254 of a block's largest element.
            for (int e = tid; e < RQ * (D / 32); e += FA_DPAS_WG) {
                const int r = e / (D / 32);
                const int b = e % (D / 32);
                float     x[32];
                float     amax = 0.0f;
                const float * qr = (const float *) (p.Q + seq * p.nb03 + (int64_t) ((r < R ? r : 0) / G) * p.nb01 +
                                                    (int64_t) (kvh * G + (r < R ? r : 0) % G) * p.nb02) + 32 * b;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    x[i] = r < R ? qr[i] : 0.0f;
                    amax = sycl::fmax(amax, sycl::fabs(x[i]));
                }
                const float d  = amax / 127.0f;
                const float id = d != 0.0f ? 1.0f / d : 0.0f;
                const float dq = (float) sycl::half(d) * p.scale;
#pragma unroll
                for (int i = 0; i < 32; ++i) {
                    const int dd = 32 * b + i;
                    sQs[(((r / 8) * NKS + dd / 16) * 16 + dd % 16) * 8 + r % 8] =
                        (short) sycl::bit_cast<uint16_t>(sycl::half(sycl::round(x[i] * id) * dq));
                }
            }
            for (int e = tid; e < RP * PS; e += FA_DPAS_WG) {
                sPs[e] = 0;
            }
            if (tid < RP) {
                const bool sink = p.sinks && split == 0 && tid < R;
                sM[tid] = sink ? p.sinks[kvh * G + tid % G] : FA_DPAS_MINF;
                sL[tid] = sink ? 1.0f : 0.0f;
                sA[tid] = 1.0f;
            }
            it.barrier(sycl::access::fence_space::local_space);

            const int    k_begin = split * p.chunk;
            const int    k_end   = sycl::min(k_begin + p.chunk, p.k_lim);
            // K and V as 2D surfaces of k_lim rows; this head's segment starts at dword kx / vx of a row
            const char * kb      = p.K + seq * p.nb13;
            const char * vb      = p.V + seq * p.nb23;
            const int    kx      = (int) (kvh * p.nb12 / 4);
            const int    vx      = (int) (kvh * p.nb22 / 4);
            const sycl::half * mrow = p.mask ? (const sycl::half *) (p.mask + (seq % p.ne33) * p.nb33) : nullptr;

            // O^T of this sub-group's 16 dims: lane = row, element m = dim
            fa_dpas_float8 o[NRP][2];
#pragma unroll
            for (int t = 0; t < NRP; ++t) {
                o[t][0] = 0.0f;
                o[t][1] = 0.0f;
            }

            for (int t0 = k_begin; t0 < k_end; t0 += TK) {
                // S = Q K^T for keys t0 + 16 w + lane
                {
                    const int      j  = t0 + w * 16 + lane;
                    const bool     jv = j < k_end;
                    fa_dpas_float8 s[NRQ];
#pragma unroll
                    for (int t = 0; t < NRQ; ++t) {
                        s[t] = 0.0f;
                    }
                    // the lane's whole K row, in 2D block loads issued before the first DPAS
                    constexpr int NKD = KT == FA_R_Q8_0 ? 68 : KT == FA_R_Q5_1 ? 48 : 36;
                    uint32_t      kd[NKD];
#pragma unroll
                    for (int i = 0; i < NKD / 8; ++i) {
                        const fa_dpas_int8 v = fa_r_ld2d<8, fa_dpas_int8>(kb, (int) p.nb11, p.k_lim, kx + 8 * i, t0 + w * 16);
#pragma unroll
                        for (int e = 0; e < 8; ++e) {
                            kd[8 * i + e] = (uint32_t) v[e];
                        }
                    }
                    if constexpr (NKD % 8 != 0) {
                        const fa_dpas_int4 v = fa_r_ld2d<4, fa_dpas_int4>(kb, (int) p.nb11, p.k_lim, kx + NKD - 4, t0 + w * 16);
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            kd[NKD - 4 + e] = (uint32_t) v[e];
                        }
                    }
                    // with several query columns, each column's mask value once, also ahead of the DPAS
                    constexpr bool MK = NRQ > 1;
                    sycl::half     mh[RQ];
                    if constexpr (MK) {
                        const int jm = jv ? j : k_begin;
#pragma unroll
                        for (int r = 0; r < RQ; ++r) {
                            if (r > 0 && r / G == (r - 1) / G) {
                                mh[r] = mh[r - 1];
                            } else {
                                mh[r] = r < R && mrow ? mrow[((int64_t) (r / G) * p.nb31) / (int64_t) sizeof(sycl::half) + jm]
                                                      : sycl::half(0.0f);
                            }
                        }
                    }
                    GGML_UNUSED(mh);
                    // and this sub-group's K and V rows of the next tile into the caches: per call at 98304 keys
                    // that is -3% (one stream) and -4% (two) on the 32-row verify, +13..23% on the 8-row draft
                    if constexpr (NRQ > 1) {
                        fa_r_pf2d(kb, (int) p.nb11, p.k_lim, kx, t0 + TK + w * 16, NKD);
                        fa_r_pf2d(vb, (int) p.nb21, p.k_lim, vx, t0 + TK + w * 16, VT == FA_R_Q8_0 ? 68 : 36);
                    }
#pragma unroll
                    for (int blk = 0; blk < D / 32; ++blk) {
                        sycl::uint4 q[2];
                        uint32_t    qh = 0;
                        float       d, m = 0.0f;
                        const int   qb = KT == FA_R_Q8_0 ? 8 * blk : 4 * blk;
                        q[0] = sycl::uint4(kd[qb], kd[qb + 1], kd[qb + 2], kd[qb + 3]);
                        if constexpr (KT == FA_R_Q8_0) {
                            q[1] = sycl::uint4(kd[qb + 4], kd[qb + 5], kd[qb + 6], kd[qb + 7]);
                            d    = (float) sycl::bit_cast<sycl::half>((uint16_t) (kd[64 + blk / 2] >> (16 * (blk % 2))));
                        } else if constexpr (KT == FA_R_Q5_1) {
                            qh = kd[32 + blk];
                            d  = (float) sycl::bit_cast<sycl::half>((uint16_t) kd[40 + blk]);
                            m  = (float) sycl::bit_cast<sycl::half>((uint16_t) (kd[40 + blk] >> 16));
                        } else {
                            d = (float) sycl::bit_cast<sycl::half>((uint16_t) (kd[32 + blk / 2] >> (16 * (blk % 2))));
                        }
#pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const fa_dpas_int8 b  = fa_r_k_operand<KT>(q, qh, d, m, h);
                            const int          ks = 2 * blk + h;
#pragma unroll
                            for (int t = 0; t < NRQ; ++t) {
                                fa_r_dpas(sQ[(t * NKS + ks) * 16 + lane], b, s[t], fix, p.bad_mask, sg);
                            }
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NRQ; ++t) {
#pragma unroll
                        for (int m = 0; m < 8; ++m) {
                            const int r = t * 8 + m;
                            if (r < R) {
                                const int c  = r / G;
                                bool      ok = jv && (p.n_past < 0 || j <= p.n_past + c);
                                float     mv = 0.0f;
                                if constexpr (MK) {
                                    mv = (float) mh[r];
                                } else if (ok && mrow) {
                                    mv = (float) mrow[((int64_t) c * p.nb31) / (int64_t) sizeof(sycl::half) + j];
                                }
                                sS[r * TK + w * 16 + lane] = ok ? s[t][m] + mv : -INFINITY;
                            }
                        }
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // online softmax per row -> P
                for (int r = w; r < R; r += FA_DPAS_NSG) {
                    float sv[KG];
                    float mt = -INFINITY;
#pragma unroll
                    for (int i = 0; i < KG; ++i) {
                        sv[i] = sS[r * TK + i * 16 + lane];
                        mt    = sycl::fmax(mt, sv[i]);
                    }
                    mt = sycl::reduce_over_group(sg, mt, sycl::maximum<float>());
                    const float m_old = sM[r];
                    const float m_new = sycl::fmax(m_old, mt);
                    float       lt    = 0.0f;
#pragma unroll
                    for (int i = 0; i < KG; ++i) {
                        const float pv = sycl::native::exp(sv[i] - m_new);
                        sPs[r * PS + i * 16 + lane] = (short) sycl::bit_cast<uint16_t>(sycl::half(pv));
                        lt += pv;
                    }
                    lt = sycl::reduce_over_group(sg, lt, sycl::plus<float>());
                    if (lane == 0) {
                        const float a = sycl::native::exp(m_old - m_new);
                        sA[r] = a;
                        sL[r] = sL[r] * a + lt;
                        sM[r] = m_new;
                    }
                }
                it.barrier(sycl::access::fence_space::local_space);

                // O^T = diag(alpha) O^T + V^T P^T over this sub-group's dims 16 w .. 16 w + 15
#pragma unroll
                for (int t = 0; t < NRP; ++t) {
                    const float a = sA[t * 16 + lane];
                    o[t][0] *= a;
                    o[t][1] *= a;
                }
                // V rows t0 + 16 kg + lane: this sub-group's dims in VW dwords, and the dword holding their scale
                constexpr int VW = VT == FA_R_Q8_0 ? 4 : 2;
                constexpr int VS = VT == FA_R_Q8_0 ? 64 : 32;
#pragma unroll 2
                for (int kg = 0; kg < KG; ++kg) {
                    uint32_t  qv[VW];
                    const int y = t0 + kg * 16;
                    if constexpr (VW == 4) {
                        const fa_dpas_int4 v = fa_r_ld2d<4, fa_dpas_int4>(vb, (int) p.nb21, p.k_lim, vx + 4 * w, y);
#pragma unroll
                        for (int e = 0; e < 4; ++e) {
                            qv[e] = (uint32_t) v[e];
                        }
                    } else {
                        const fa_dpas_int2 v = fa_r_ld2d<2, fa_dpas_int2>(vb, (int) p.nb21, p.k_lim, vx + 2 * w, y);
                        qv[0] = (uint32_t) v[0];
                        qv[1] = (uint32_t) v[1];
                    }
                    const uint32_t sc = (uint32_t) fa_r_ld2d<1, int>(vb, (int) p.nb21, p.k_lim, vx + VS + w / 4, y);
                    const float    dv = (float) sycl::bit_cast<sycl::half>((uint16_t) (sc >> (16 * ((w / 2) % 2))));
                    fa_dpas_short8 va[2];
                    if constexpr (VT == FA_R_Q8_0) {
#pragma unroll
                        for (int x = 0; x < 2; ++x) {
#pragma unroll
                            for (int mm = 0; mm < 8; ++mm) {
                                const int e = 8 * x + mm;
                                const int v = (int8_t) (qv[e / 4] >> (8 * (e % 4)));
                                va[x][mm] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) v * dv));
                            }
                        }
                    } else {
#pragma unroll
                        for (int x = 0; x < 2; ++x) {
#pragma unroll
                            for (int mm = 0; mm < 8; ++mm) {
                                const int      e    = 8 * x + mm;
                                const uint32_t byte = (qv[e / 8] >> (8 * ((e / 2) % 4))) & 0xFFu;
                                const int      v    = (int) ((e % 2) ? (byte >> 4) : (byte & 0xFu)) - 8;
                                va[x][mm] = (short) sycl::bit_cast<uint16_t>(sycl::half((float) v * dv));
                            }
                        }
                    }
#pragma unroll
                    for (int t = 0; t < NRP; ++t) {
                        const fa_dpas_int8 pb = sP[((t * 16 + lane) * PS + kg * 16) / 16];
                        fa_r_dpas(va[0], pb, o[t][0], fix, p.bad_mask, sg);
                        fa_r_dpas(va[1], pb, o[t][1], fix, p.bad_mask, sg);
                    }
                }
            }

#pragma unroll
            for (int t = 0; t < NRP; ++t) {
                const int r = t * 16 + lane;
                if (r < R) {
                    const int64_t jdu = ((int64_t) seq * p.ne01 + r / G) * p.ne02 + kvh * G + r % G;
                    const float   l   = sL[r];
#pragma unroll
                    for (int x = 0; x < 2; ++x) {
#pragma unroll
                        for (int mm = 0; mm < 8; ++mm) {
                            const int dd = 16 * w + 8 * x + mm;
                            if (p.nsplit == 1) {
                                p.dst[jdu * D + dd] = l > 0.0f ? o[t][x][mm] / l : 0.0f;
                            } else {
                                p.parts[(jdu * p.nsplit + split) * D + dd] = o[t][x][mm];
                            }
                        }
                    }
                }
            }
            if (p.nsplit > 1 && tid < R) {
                const int64_t jdu = ((int64_t) seq * p.ne01 + tid / G) * p.ne02 + kvh * G + tid % G;
                p.meta[jdu * p.nsplit + split] = sycl::float2(sM[tid], sL[tid]);
            }
        };
        // the large register file: the S and O accumulators of 32 rows spill at 128 registers
        cgh.parallel_for(ndr, sycl::ext::oneapi::experimental::properties{ sycl::ext::intel::experimental::grf_size<256> }, kern);
    });
}

template <int KT, int VT, int NRQ, int NRP>
static void fattn_dpas_r_case(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    const int  ne01 = Q->ne[1];
    const int  ne11 = K->ne[1];
    const int  nkvh = K->ne[2];
    const int  ne03 = Q->ne[3];
    const bool implicit_causal = mask && mask->ne[0] == 1;

    fa_dpas_r_params p;
    p.Q      = (const char *) Q->data;
    p.K      = (const char *) K->data;
    p.V      = (const char *) V->data;
    p.mask   = (mask && !implicit_causal) ? (const char *) mask->data : nullptr;
    p.sinks  = dst->src[4] ? (const float *) dst->src[4]->data : nullptr;
    p.dst    = (float *) dst->data;
    memcpy(&p.scale, (const float *) dst->op_params + 0, sizeof(float));
    p.G      = Q->ne[2] / K->ne[2];
    p.ne01   = ne01;
    p.ne02   = Q->ne[2];
    p.ne33   = mask ? mask->ne[3] : 1;
    p.n_past = implicit_causal ? (int) mask->ne[1] - 1 : -1;
    p.k_lim  = implicit_causal ? std::min(ne11, p.n_past + ne01) : ne11;
    p.nb01 = Q->nb[1];  p.nb02 = Q->nb[2];  p.nb03 = Q->nb[3];
    p.nb11 = K->nb[1];  p.nb12 = K->nb[2];  p.nb13 = K->nb[3];
    p.nb21 = V->nb[1];  p.nb22 = V->nb[2];  p.nb23 = V->nb[3];
    p.nb31 = (mask && !implicit_causal) ? mask->nb[1] : 0;
    p.nb33 = (mask && !implicit_causal) ? mask->nb[3] : 0;
    const ggml_sycl_dpas_health & health = ggml_sycl_dpas_get_health(ctx.device);
    ggml_sycl_dpas_pair_repair(health, p.bad_eu, p.bad_mask);
    p.eu_mask = ggml_sycl_dpas_eu_mask(health.layout);

    // key slices for two work-groups per Xe2 core (8 EUs; nsm = EUs / 16): at 98304 keys one per core costs
    // the one-column q8_0 draft 38% and four per core 9%; the 5-column verify is within 1% of one per core
    const int xe_cores = std::max(1, ggml_sycl_info().devices[ctx.device].nsm * 2);
    const int ntiles   = std::max(1, (p.k_lim + FA_R_TK - 1) / FA_R_TK);
    int nsplit = std::max(1, std::min(xe_cores * 2 / std::max(1, nkvh * ne03), ntiles));
    p.chunk    = ((ntiles + nsplit - 1) / nsplit) * FA_R_TK;
    nsplit     = std::max(1, (p.k_lim + p.chunk - 1) / p.chunk);
    p.nsplit   = nsplit;

    ggml_sycl_pool_alloc<float>        parts(ctx.pool());
    ggml_sycl_pool_alloc<sycl::float2> meta(ctx.pool());
    if (nsplit > 1) {
        parts.alloc((size_t) nsplit * ggml_nelements(dst));
        meta.alloc((size_t) nsplit * ggml_nrows(dst));
    }
    p.parts = parts.get();
    p.meta  = meta.get();

    dpct::queue_ptr stream = ctx.stream();
    fattn_dpas_r_kernel<KT, VT, NRQ, NRP>(p, nkvh, ne03, stream);

    static const int combine_unroll_env = ggml_sycl_get_env("GGML_SYCL_FA_COMBINE_UNROLL", 4);
    const int        combine_unroll     = combine_unroll_env;
    if (nsplit > 1) {
        float *        parts_p = parts.get();
        sycl::float2 * meta_p  = meta.get();
        float *        dst_p   = (float *) dst->data;
        stream->submit([&](sycl::handler & cgh) {
            sycl::local_accessor<uint8_t, 1> lm(sycl::range<1>(nsplit * sizeof(sycl::float2)), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(Q->ne[3], Q->ne[2], (size_t) ne01 * FA_R_D), sycl::range<3>(1, 1, FA_R_D)),
                [=](sycl::nd_item<3>) [[sycl::reqd_sub_group_size(16)]] {
                    flash_attn_combine_results<FA_R_D>(parts_p, meta_p, dst_p, nsplit, combine_unroll,
                                                      lm.get_multi_ptr<sycl::access::decorated::no>().get());
                });
        });
    }
}

template <int KT, int VT> static void fattn_dpas_r_rows(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const int R = (int) Q->ne[1] * (int) (Q->ne[2] / dst->src[1]->ne[2]);
    if (R <= 8) {
        fattn_dpas_r_case<KT, VT, 1, 1>(ctx, dst);
    } else if (R <= 16) {
        fattn_dpas_r_case<KT, VT, 2, 1>(ctx, dst);
    } else {
        fattn_dpas_r_case<KT, VT, 4, 2>(ctx, dst);
    }
}

void ggml_sycl_flash_attn_ext_dpas_r(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    const ggml_type tk = dst->src[1]->type;
    if (tk == GGML_TYPE_Q5_1) {
        fattn_dpas_r_rows<FA_R_Q5_1, FA_R_Q4_0>(ctx, dst);
    } else if (tk == GGML_TYPE_Q4_0) {
        fattn_dpas_r_rows<FA_R_Q4_0, FA_R_Q4_0>(ctx, dst);
    } else {
        fattn_dpas_r_rows<FA_R_Q8_0, FA_R_Q8_0>(ctx, dst);
    }
}

#else

void ggml_sycl_flash_attn_ext_dpas(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn-dpas: not built for this target");
}

void ggml_sycl_flash_attn_ext_dpas_r(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(dst);
    GGML_ABORT("fattn-dpas: not built for this target");
}

#endif // GGML_SYCL_FA_DPAS_BUILD

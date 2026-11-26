#include "dpas-check.hpp"
#include "common.hpp"

#include <algorithm>
#include <mutex>

// GGML_SYCL_SUPPORT_XMX=OFF builds, and spir64_gen builds for sub-group 8 targets or mixed target lists, cannot
// compile the SIMD16 DPAS below
#if !defined(GGML_SYCL_SUPPORT_XMX) || (defined(GGML_SYCL_XMX_AOT_SG) && GGML_SYCL_XMX_AOT_SG != 16)
#    define GGML_SYCL_DPAS_CHECK_BUILD 0
#else
#    define GGML_SYCL_DPAS_CHECK_BUILD 1
#endif

static bool g_ggml_sycl_dpas_faulty[GGML_SYCL_MAX_DEVICES];

bool ggml_sycl_dpas_faulty(int device) {
    return g_ggml_sycl_dpas_faulty[device];
}

static ggml_sycl_dpas_health g_ggml_sycl_dpas_health[GGML_SYCL_MAX_DEVICES];

const ggml_sycl_dpas_health & ggml_sycl_dpas_get_health(int device) {
    return g_ggml_sycl_dpas_health[device];
}

#if GGML_SYCL_DPAS_CHECK_BUILD

typedef short dpas_check_short8 __attribute__((ext_vector_type(8)));
typedef int   dpas_check_int8   __attribute__((ext_vector_type(8)));
typedef float dpas_check_float8 __attribute__((ext_vector_type(8)));

// The DPAS forms of fattn-dpas.cpp: SIMD16, systolic depth 8, 8 rows. a: one 16-bit element of each row per lane
// (two s8 or one f16), b: the 8 K dwords of the lane's output column (four s8 or two f16 each).
static inline void dpas_check_s8(const dpas_check_short8 & a, const dpas_check_int8 & b, dpas_check_int8 & d) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 16) %0.0 %%null.0 DB.0 DA(0,0)\n"
            "}\n"
            : "=rw"(d)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED_VARS(a, b, d);
#endif
}

static inline void dpas_check_s8_acc(const dpas_check_short8 & a, const dpas_check_int8 & b, dpas_check_int8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.s8.s8.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED_VARS(a, b, acc);
#endif
}

static inline void dpas_check_f16_acc(const dpas_check_short8 & a, const dpas_check_int8 & b, dpas_check_float8 & acc) {
#ifdef __SYCL_DEVICE_ONLY__
    __asm__("{\n"
            ".decl DB v_type=G type=ud num_elts=128 align=GRF alias=<%1,0>\n"
            ".decl DA v_type=G type=ud num_elts=64 align=GRF alias=<%2,0>\n"
            "dpas.hf.hf.8.8 (M1, 16) %0.0 %0.0 DB.0 DA(0,0)\n"
            "}\n"
            : "+rw"(acc)
            : "rw"(b), "rw"(a));
#else
    GGML_UNUSED_VARS(a, b, acc);
#endif
}

static inline uint32_t dpas_check_sr0() {
#ifdef __SYCL_DEVICE_ONLY__
    uint32_t v;
    __asm__ volatile("mov (M1_NM, 16) %0(0,0)<1> %%sr0(0,0)<0;1,0>" : "=rw"(v));
    return v;
#else
    return 0;
#endif
}

// a signed zero can differ between the right DPAS result and the host sum
static inline bool dpas_check_same(float x, float y) {
    return sycl::bit_cast<uint32_t>(x) == sycl::bit_cast<uint32_t>(y) || (x == 0.0f && y == 0.0f);
}

// operands and right results of one case, laid out as the lanes of a sub-group read them: [row or K dword][lane]
struct dpas_check_case {
    uint16_t a_s8[8][16];
    int32_t  b_s8[8][16];
    int32_t  d_s8[8][16];
    uint16_t a_f16[8][16];
    int32_t  b_f16[8][16];
    float    c_f16[8][16];
    float    d_f16[8][16];
};

static constexpr int DPAS_CHECK_CASES  = 16;
static constexpr int DPAS_CHECK_ITER   = 4;    // cases per sub-group
static constexpr int DPAS_CHECK_CHAIN  = 16;   // DPAS per dependent chain
static constexpr int DPAS_CHECK_SG_EU  = 128;  // sub-groups per EU and launch
static constexpr int DPAS_CHECK_ROUNDS = 4;    // launches before giving up on reaching every EU

// s8 over the whole range; f16 small integers, so every sum, chains included, is exact in f32
static void dpas_check_make_cases(std::vector<dpas_check_case> & cases) {
    uint32_t seed = 12345;
    auto     rnd  = [&seed](int lo, int hi) {
        seed = seed * 1664525u + 1013904223u;
        return lo + (int) ((seed >> 8) % (uint32_t) (hi - lo + 1));
    };
    for (dpas_check_case & t : cases) {
        int a[8][32];
        int b[32][16];
        for (int r = 0; r < 8; ++r) {
            for (int k = 0; k < 32; ++k) {
                a[r][k] = rnd(-128, 127);
            }
        }
        for (int k = 0; k < 32; ++k) {
            for (int n = 0; n < 16; ++n) {
                b[k][n] = rnd(-128, 127);
            }
        }
        for (int i = 0; i < 8; ++i) {
            for (int l = 0; l < 16; ++l) {
                t.a_s8[i][l] = (uint16_t) ((a[i][2 * l] & 0xff) | (a[i][2 * l + 1] & 0xff) << 8);
                t.b_s8[i][l] = (int32_t) ((b[4 * i][l] & 0xff) | (b[4 * i + 1][l] & 0xff) << 8 |
                                          (b[4 * i + 2][l] & 0xff) << 16 | (uint32_t) (b[4 * i + 3][l] & 0xff) << 24);
                int d = 0;
                for (int k = 0; k < 32; ++k) {
                    d += a[i][k] * b[k][l];
                }
                t.d_s8[i][l] = d;
            }
        }
        for (int r = 0; r < 8; ++r) {
            for (int k = 0; k < 16; ++k) {
                a[r][k] = rnd(-16, 16);
            }
        }
        for (int k = 0; k < 16; ++k) {
            for (int n = 0; n < 16; ++n) {
                b[k][n] = rnd(-16, 16);
            }
        }
        for (int i = 0; i < 8; ++i) {
            for (int l = 0; l < 16; ++l) {
                t.a_f16[i][l] = sycl::bit_cast<uint16_t>(sycl::half((float) a[i][l]));
                t.b_f16[i][l] = (int32_t) (sycl::bit_cast<uint16_t>(sycl::half((float) b[2 * i][l])) |
                                           (uint32_t) sycl::bit_cast<uint16_t>(sycl::half((float) b[2 * i + 1][l])) << 16);
                t.c_f16[i][l] = (float) rnd(-64, 64);
                int d = 0;
                for (int k = 0; k < 16; ++k) {
                    d += a[i][k] * b[k][l];
                }
                t.d_f16[i][l] = (float) d;
            }
        }
    }
}

// each sub-group runs DPAS_CHECK_ITER cases on whatever EU it lands on and records its sr0 and wrong lanes
static void dpas_check_run(sycl::queue & q, const dpas_check_case * cases, ggml_sycl_dpas_record * out, int n_sg) {
    q.parallel_for(sycl::nd_range<1>((size_t) n_sg * 16, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int lane      = it.get_local_id(0) % 16;
        const int sg        = it.get_global_id(0) / 16;
        bool      bad       = false;
        bool      bad_chain = false;
        for (int i = 0; i < DPAS_CHECK_ITER; ++i) {
            const dpas_check_case & t = cases[(sg + i) % DPAS_CHECK_CASES];

            dpas_check_short8 a;
            dpas_check_int8   b;
            for (int r = 0; r < 8; ++r) {
                a[r] = (short) t.a_s8[r][lane];
                b[r] = t.b_s8[r][lane];
            }
            dpas_check_int8 d;
            dpas_check_int8 dc = 0;
            dpas_check_s8(a, b, d);
            for (int c = 0; c < DPAS_CHECK_CHAIN; ++c) {
                dpas_check_s8_acc(a, b, dc);
            }
            for (int r = 0; r < 8; ++r) {
                bad       |= d[r] != t.d_s8[r][lane];
                bad_chain |= dc[r] != DPAS_CHECK_CHAIN * t.d_s8[r][lane];
            }

            for (int r = 0; r < 8; ++r) {
                a[r] = (short) t.a_f16[r][lane];
                b[r] = t.b_f16[r][lane];
            }
            dpas_check_float8 f;
            dpas_check_float8 fc = 0.0f;
            for (int r = 0; r < 8; ++r) {
                f[r] = t.c_f16[r][lane];
            }
            dpas_check_f16_acc(a, b, f);
            for (int c = 0; c < DPAS_CHECK_CHAIN; ++c) {
                dpas_check_f16_acc(a, b, fc);
            }
            for (int r = 0; r < 8; ++r) {
                bad       |= !dpas_check_same(f[r], t.c_f16[r][lane] + t.d_f16[r][lane]);
                bad_chain |= !dpas_check_same(fc[r], DPAS_CHECK_CHAIN * t.d_f16[r][lane]);
            }
        }
        const auto     sgr         = it.get_sub_group();
        const uint32_t lanes       = sycl::reduce_over_group(sgr, bad ? 1u << lane : 0u, sycl::bit_or<uint32_t>());
        const uint32_t lanes_chain = sycl::reduce_over_group(sgr, bad_chain ? 1u << lane : 0u, sycl::bit_or<uint32_t>());
        const uint32_t sr0         = dpas_check_sr0();
        if (lane == 0) {
            out[sg] = { sr0, lanes, lanes_chain };
        }
    }).wait();
}

// GPUs whose DPAS takes the SIMD16 form above, and the sr0 layout of their family
static bool dpas_check_layout(gpu_arch arch, ggml_sycl_dpas_layout & layout) {
    switch (arch) {
        case gpu_arch::intel_gpu_bmg_g21:
        case gpu_arch::intel_gpu_bmg_g31:
        case gpu_arch::intel_gpu_lnl_m:
            layout = GGML_SYCL_DPAS_LAYOUT_XE2;
            return true;
        case gpu_arch::intel_gpu_ptl_h:
        case gpu_arch::intel_gpu_ptl_u:
        case gpu_arch::intel_gpu_wcl:
            layout = GGML_SYCL_DPAS_LAYOUT_XE3;
            return true;
        default:
            return false;
    }
}

static void ggml_sycl_dpas_self_test(int device) try {
    ggml_sycl_dpas_layout layout;
    if (!dpas_check_layout(ggml_sycl_info().devices[device].hw_info.arch, layout)) {
        return;
    }
    const int64_t      t0   = ggml_time_us();
    sycl::queue &      q    = dpct::get_device(device).default_queue();
    const sycl::device dev  = q.get_device();
    const int          n_eu = dev.has(sycl::aspect::ext_intel_gpu_eu_count) ?
                                  (int) dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>() :
                                  (int) dev.get_info<sycl::info::device::max_compute_units>();
    const int          n_sg = std::max(n_eu, 1) * DPAS_CHECK_SG_EU;

    std::vector<dpas_check_case> cases(DPAS_CHECK_CASES);
    dpas_check_make_cases(cases);
    dpas_check_case *       d_cases = sycl::malloc_device<dpas_check_case>(cases.size(), q);
    ggml_sycl_dpas_record * d_out   = sycl::malloc_device<ggml_sycl_dpas_record>(n_sg, q);
    q.memcpy(d_cases, cases.data(), cases.size() * sizeof(dpas_check_case)).wait();

    std::vector<ggml_sycl_dpas_record> records;
    ggml_sycl_dpas_health              h;
    for (int round = 0; round < DPAS_CHECK_ROUNDS && !h.complete(); ++round) {
        dpas_check_run(q, d_cases, d_out, n_sg);
        records.resize(records.size() + n_sg);
        q.memcpy(records.data() + records.size() - n_sg, d_out, n_sg * sizeof(ggml_sycl_dpas_record)).wait();
        h = ggml_sycl_dpas_evaluate(layout, n_eu, records);
    }
    sycl::free(d_cases, q);
    sycl::free(d_out, q);
    g_ggml_sycl_dpas_health[device] = h;

    GGML_SYCL_DEBUG("[SYCL] %s: device %d: %d sub-groups, %.1f ms\n", __func__, device, (int) records.size(),
                    (ggml_time_us() - t0) / 1000.0);

    const std::string what = ggml_sycl_dpas_describe(h);
    if (h.faults.empty()) {
        GGML_LOG_INFO("%s: device %d: %s\n", __func__, device, what.c_str());
        return;
    }
    GGML_LOG_WARN("%s: device %d: %s\n", __func__, device, what.c_str());

    // the kernels with an int32 accumulator, where a wrong lane is off by orders of magnitude
    g_ggml_sycl_dpas_faulty[device] = g_ggml_sycl_dpas_check == 1;

    std::string resp;
    uint32_t    eu;
    int         lane;
    if (g_ggml_sycl_enable_xmx_esimd && ggml_sycl_xmx_supports_arch(ggml_sycl_info().devices[device].hw_info.arch)) {
        if (!ggml_sycl_dpas_lane_repair(h, eu, lane)) {
            resp += "the XMX mat-vec stays off, ";
        } else if (lane >= 0) {
            resp += "the XMX mat-vec recomputes lane " + std::to_string(lane) + " on that EU without DPAS, ";
        }
    }
    if (g_ggml_sycl_dpas_faulty[device]) {
        resp += "DPAS flash attention stays off (GGML_SYCL_DPAS_CHECK=2 keeps it on), ";
    }
    GGML_LOG_WARN("%s: device %d: %sother XMX kernels can still give wrong results\n", __func__, device, resp.c_str());
} catch (const std::exception & e) {
    GGML_LOG_WARN("%s: device %d: the DPAS self-test did not run: %s\n", __func__, device, e.what());
}

#endif // GGML_SYCL_DPAS_CHECK_BUILD

void ggml_sycl_dpas_check(int device) {
#if GGML_SYCL_DPAS_CHECK_BUILD
    static std::once_flag once[GGML_SYCL_MAX_DEVICES];
    std::call_once(once[device], [device] {
        if (g_ggml_sycl_dpas_check) {
            ggml_sycl_dpas_self_test(device);
        }
    });
#else
    GGML_UNUSED(device);
#endif
}

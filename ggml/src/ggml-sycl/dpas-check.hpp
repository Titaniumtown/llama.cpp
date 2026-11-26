#ifndef GGML_SYCL_DPAS_CHECK_HPP
#define GGML_SYCL_DPAS_CHECK_HPP

// Outcome of the DPAS self-test of dpas-check.cpp. No SYCL here: tests/test-sycl-dpas-check.cpp runs it on the CPU.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

// Bit ranges of the sr0.0 fields that name the EU a hardware thread runs on, as IGC lowers intel_get_slice_id,
// intel_get_subslice_id and intel_get_eu_id for the GPU family. The bits below eu_lo name the thread.
struct ggml_sycl_dpas_layout {
    int slice_lo;
    int slice_hi;
    int subslice_lo;
    int subslice_hi;
    int eu_lo;
    int eu_hi;
};

static constexpr ggml_sycl_dpas_layout GGML_SYCL_DPAS_LAYOUT_XE2 = { 11, 15, 8, 9, 4, 6 };
static constexpr ggml_sycl_dpas_layout GGML_SYCL_DPAS_LAYOUT_XE3 = { 14, 17, 8, 11, 4, 6 };

static inline uint32_t ggml_sycl_dpas_field(uint32_t sr0, int lo, int hi) {
    return (sr0 >> lo) & ((1u << (hi - lo + 1)) - 1);
}

static inline uint32_t ggml_sycl_dpas_eu_mask(const ggml_sycl_dpas_layout & l) {
    return (ggml_sycl_dpas_field(~0u, l.slice_lo, l.slice_hi) << l.slice_lo) |
           (ggml_sycl_dpas_field(~0u, l.subslice_lo, l.subslice_hi) << l.subslice_lo) |
           (ggml_sycl_dpas_field(~0u, l.eu_lo, l.eu_hi) << l.eu_lo);
}

// what one sub-group of the test reports: its sr0, and the output lanes (bit n = lane n) that a single DPAS and a
// chain of dependent DPAS (each result fed back as the accumulator) got wrong
struct ggml_sycl_dpas_record {
    uint32_t sr0;
    uint32_t lanes;
    uint32_t lanes_chain;
};

struct ggml_sycl_dpas_fault {
    uint32_t eu;           // sr0 & ggml_sycl_dpas_eu_mask
    uint32_t lanes;
    uint32_t lanes_chain;  // includes lanes
};

struct ggml_sycl_dpas_health {
    ggml_sycl_dpas_layout layout = {};
    int n_eu      = 0;  // EUs of the device
    int n_eu_seen = 0;  // EU ids the test ran on, 0 when it did not run
    std::vector<ggml_sycl_dpas_fault> faults;  // EUs with wrong lanes, by id

    bool complete() const { return n_eu > 0 && n_eu_seen == n_eu; }
};

static inline ggml_sycl_dpas_health ggml_sycl_dpas_evaluate(const ggml_sycl_dpas_layout & layout, int n_eu,
                                                            const std::vector<ggml_sycl_dpas_record> & records) {
    const uint32_t mask = ggml_sycl_dpas_eu_mask(layout);

    std::map<uint32_t, ggml_sycl_dpas_fault> eus;
    for (const ggml_sycl_dpas_record & r : records) {
        ggml_sycl_dpas_fault & f = eus[r.sr0 & mask];
        f.eu           = r.sr0 & mask;
        f.lanes       |= r.lanes;
        f.lanes_chain |= r.lanes | r.lanes_chain;
    }

    ggml_sycl_dpas_health h;
    h.layout    = layout;
    h.n_eu      = n_eu;
    h.n_eu_seen = (int) eus.size();
    for (const auto & it : eus) {
        if (it.second.lanes_chain != 0) {
            h.faults.push_back(it.second);
        }
    }
    return h;
}

// one line for the log: what is wrong where, and how much of the device the test covered
static inline std::string ggml_sycl_dpas_describe(const ggml_sycl_dpas_health & h) {
    const ggml_sycl_dpas_layout & l = h.layout;

    char buf[256];
    if (h.faults.empty()) {
        if (h.complete()) {
            snprintf(buf, sizeof(buf), "DPAS right on all %d EUs", h.n_eu);
        } else {
            snprintf(buf, sizeof(buf), "DPAS right on the %d EU ids the test reached, of %d EUs: coverage not proven",
                     h.n_eu_seen, h.n_eu);
        }
        return buf;
    }

    std::string s;
    if ((int) h.faults.size() == h.n_eu_seen) {
        uint32_t lanes       = 0;
        uint32_t lanes_chain = 0;
        for (const ggml_sycl_dpas_fault & f : h.faults) {
            lanes       |= f.lanes;
            lanes_chain |= f.lanes_chain;
        }
        snprintf(buf, sizeof(buf),
                 "DPAS output lanes 0x%04x wrong, 0x%04x in dependent DPAS, on every EU the test reached (%d): not a "
                 "defective EU, check the GPU driver",
                 lanes, lanes_chain, h.n_eu_seen);
        s = buf;
    } else {
        // more ids than EUs: the layout does not fit the device, so the ids are not decoded
        const bool decode = h.n_eu_seen <= h.n_eu;
        s = "hardware defect:";
        for (size_t i = 0; i < h.faults.size() && i < 4; ++i) {
            const ggml_sycl_dpas_fault & f = h.faults[i];
            if (decode) {
                snprintf(buf, sizeof(buf), "%s EU sr0 0x%04x (slice %u, Xe-core %u, EU %u)", i ? ";" : "", f.eu,
                         ggml_sycl_dpas_field(f.eu, l.slice_lo, l.slice_hi),
                         ggml_sycl_dpas_field(f.eu, l.subslice_lo, l.subslice_hi),
                         ggml_sycl_dpas_field(f.eu, l.eu_lo, l.eu_hi));
            } else {
                snprintf(buf, sizeof(buf), "%s EU id sr0 0x%04x", i ? ";" : "", f.eu);
            }
            s += buf;
            snprintf(buf, sizeof(buf), " gets DPAS output lanes 0x%04x wrong, 0x%04x in dependent DPAS", f.lanes,
                     f.lanes_chain);
            s += buf;
        }
        if (h.faults.size() > 4) {
            snprintf(buf, sizeof(buf), "; %d more EUs", (int) h.faults.size() - 4);
            s += buf;
        }
    }
    if (h.complete()) {
        snprintf(buf, sizeof(buf), "; all %d EUs tested", h.n_eu);
    } else {
        snprintf(buf, sizeof(buf), "; the test reached %d EU ids of %d EUs, coverage not proven", h.n_eu_seen, h.n_eu);
    }
    return s + buf;
}

// The XMX mat-vec issues independent DPAS: it runs when every EU was tested and one EU at most gets one lane of a
// single DPAS wrong, a lane it recomputes on that EU without DPAS (lane -1 when no lane is wrong).
static inline bool ggml_sycl_dpas_lane_repair(const ggml_sycl_dpas_health & h, uint32_t & eu, int & lane) {
    eu   = ~0u;
    lane = -1;
    if (!h.complete()) {
        return false;
    }
    for (const ggml_sycl_dpas_fault & f : h.faults) {
        if (f.lanes == 0) {
            continue;
        }
        if (lane >= 0 || (f.lanes & (f.lanes - 1)) != 0) {
            eu   = ~0u;
            lane = -1;
            return false;
        }
        eu   = f.eu;
        lane = __builtin_ctz(f.lanes);
    }
    return true;
}

// the outcome of the device's self-test, set before ggml_sycl_dpas_check returns
const ggml_sycl_dpas_health & ggml_sycl_dpas_get_health(int device);

#endif // GGML_SYCL_DPAS_CHECK_HPP

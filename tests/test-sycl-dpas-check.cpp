// The outcome of ggml-sycl's DPAS self-test (ggml/src/ggml-sycl/dpas-check.hpp) from synthetic per-thread records:
// EUs are told apart by the sr0 layout of the GPU family, a defective EU is reported with its location, and a test
// that did not reach every EU says so
//
// Build and run without a model:
//   g++ -std=c++17 -o /tmp/t tests/test-sycl-dpas-check.cpp && /tmp/t

#include "../ggml/src/ggml-sycl/dpas-check.hpp"

#include <cstdio>
#include <string>
#include <vector>

static int n_fail = 0;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        n_fail++;
    }
}

static bool has(const std::string & s, const char * part) {
    return s.find(part) != std::string::npos;
}

// sr0 of thread t of an EU, with a bit set above the EU fields as real sr0 values have
static uint32_t make_sr0(const ggml_sycl_dpas_layout & l, uint32_t slice, uint32_t subslice, uint32_t eu, uint32_t t) {
    return (1u << 27) | (slice << l.slice_lo) | (subslice << l.subslice_lo) | (eu << l.eu_lo) | t;
}

// every thread of every EU of a device computes right
static std::vector<ggml_sycl_dpas_record> make_device(const ggml_sycl_dpas_layout & l, int n_slice, int n_subslice,
                                                      int n_eu, int n_thread) {
    std::vector<ggml_sycl_dpas_record> records;
    for (int s = 0; s < n_slice; ++s) {
        for (int c = 0; c < n_subslice; ++c) {
            for (int e = 0; e < n_eu; ++e) {
                for (int t = 0; t < n_thread; ++t) {
                    records.push_back({ make_sr0(l, s, c, e, t), 0, 0 });
                }
            }
        }
    }
    return records;
}

// marks the threads of one EU wrong on the given lanes
static void make_fault(std::vector<ggml_sycl_dpas_record> & records, uint32_t eu, uint32_t mask, uint32_t lanes,
                       uint32_t lanes_chain) {
    for (ggml_sycl_dpas_record & r : records) {
        if ((r.sr0 & mask) == eu) {
            r.lanes       = lanes;
            r.lanes_chain = lanes_chain;
        }
    }
}

int main() {
    const ggml_sycl_dpas_layout xe2  = GGML_SYCL_DPAS_LAYOUT_XE2;
    const uint32_t              mask = ggml_sycl_dpas_eu_mask(xe2);

    // IGC's Xe2 fields: slice 11-15, Xe-core 8-9, EU 4-6
    check(mask == 0xfb70, "Xe2: EU id bits");

    // two threads of the Arc Pro B70's defective EU, as the IGC built-ins decoded them: slice 1, Xe-core 2, EU 3
    for (uint32_t sr0 : { 0x08380a30u, 0x08300a34u }) {
        check(ggml_sycl_dpas_field(sr0, xe2.slice_lo, xe2.slice_hi) == 1, "B70 sr0: slice 1");
        check(ggml_sycl_dpas_field(sr0, xe2.subslice_lo, xe2.subslice_hi) == 2, "B70 sr0: Xe-core 2");
        check(ggml_sycl_dpas_field(sr0, xe2.eu_lo, xe2.eu_hi) == 3, "B70 sr0: EU 3");
        check((sr0 & mask) == 0x0a30, "B70 sr0: EU id 0x0a30, thread bits cleared");
    }

    // the B70: 8 slices x 4 Xe-cores x 8 EUs, 8 threads each
    {
        const auto h = ggml_sycl_dpas_evaluate(xe2, 256, make_device(xe2, 8, 4, 8, 8));
        check(h.complete() && h.n_eu_seen == 256 && h.faults.empty(), "healthy device: complete, no fault");
        check(ggml_sycl_dpas_describe(h) == "DPAS right on all 256 EUs", "healthy device: report");
    }
    {
        auto records = make_device(xe2, 8, 4, 8, 8);
        make_fault(records, 0x0a30, mask, 0x0080, 0x8080);
        const auto h = ggml_sycl_dpas_evaluate(xe2, 256, records);
        check(h.complete() && h.faults.size() == 1, "one defective EU: one fault");
        check(h.faults[0].eu == 0x0a30 && h.faults[0].lanes == 0x0080 && h.faults[0].lanes_chain == 0x8080,
              "one defective EU: id and lanes");
        const std::string s = ggml_sycl_dpas_describe(h);
        check(has(s, "hardware defect: EU sr0 0x0a30 (slice 1, Xe-core 2, EU 3) gets DPAS output lanes 0x0080 wrong, "
                     "0x8080 in dependent DPAS; all 256 EUs tested"),
              "one defective EU: report");
    }
    {
        // a lane wrong on one thread of the EU only, and only in a single DPAS: the chain mask still covers it
        auto records = make_device(xe2, 8, 4, 8, 8);
        records[5].lanes = 0x0004;
        const auto h = ggml_sycl_dpas_evaluate(xe2, 256, records);
        check(h.faults.size() == 1 && h.faults[0].eu == (records[5].sr0 & mask) && h.faults[0].lanes == 0x0004 &&
                  h.faults[0].lanes_chain == 0x0004,
              "one wrong thread: a fault on its EU");
    }
    {
        auto records = make_device(xe2, 8, 4, 8, 8);
        make_fault(records, make_sr0(xe2, 7, 3, 7, 0) & mask, mask, 0, 0x0001);
        make_fault(records, 0x0a30, mask, 0x0080, 0x8080);
        const auto h = ggml_sycl_dpas_evaluate(xe2, 256, records);
        check(h.faults.size() == 2 && h.faults[0].eu == 0x0a30 && h.faults[1].eu == 0x3b70,
              "two defective EUs: both, by id");
        check(h.faults[1].lanes == 0 && h.faults[1].lanes_chain == 0x0001, "two defective EUs: a chain-only fault");
        const std::string s = ggml_sycl_dpas_describe(h);
        check(has(s, "EU sr0 0x0a30 (slice 1, Xe-core 2, EU 3)") && has(s, "; EU sr0 0x3b70 (slice 7, Xe-core 3, EU 7)"),
              "two defective EUs: report");
    }
    {
        // every eighth EU: the report names four and counts the rest
        auto records = make_device(xe2, 8, 4, 8, 8);
        for (size_t i = 0; i < records.size(); i += 64) {
            make_fault(records, records[i].sr0 & mask, mask, 0x0002, 0x0002);
        }
        const auto h = ggml_sycl_dpas_evaluate(xe2, 256, records);
        check(h.complete() && h.faults.size() == 32, "32 defective EUs");
        check(has(ggml_sycl_dpas_describe(h), "; 28 more EUs; all 256 EUs tested"), "32 defective EUs: report");
    }

    // threads that never ran on one EU: right on the rest, but no claim about the device
    {
        std::vector<ggml_sycl_dpas_record> records;
        for (const ggml_sycl_dpas_record & r : make_device(xe2, 8, 4, 8, 8)) {
            if ((r.sr0 & mask) != 0x0a30) {
                records.push_back(r);
            }
        }
        const auto h = ggml_sycl_dpas_evaluate(xe2, 256, records);
        check(!h.complete() && h.n_eu_seen == 255 && h.faults.empty(), "one EU missed: incomplete");
        check(ggml_sycl_dpas_describe(h) == "DPAS right on the 255 EU ids the test reached, of 256 EUs: coverage not proven",
              "one EU missed: report");

        make_fault(records, 0x0040, mask, 0x0080, 0x0080);
        const std::string s = ggml_sycl_dpas_describe(ggml_sycl_dpas_evaluate(xe2, 256, records));
        check(has(s, "EU sr0 0x0040 (slice 0, Xe-core 0, EU 4)") &&
                  has(s, "; the test reached 255 EU ids of 256 EUs, coverage not proven"),
              "one EU missed, one defective: both reported");
    }

    // a layout that takes a thread bit for an EU bit splits each EU in two: no coverage, and the ids stay raw
    {
        const ggml_sycl_dpas_layout wrong = { 11, 15, 8, 9, 2, 6 };
        auto records = make_device(xe2, 8, 4, 8, 8);
        make_fault(records, 0x0a30, mask, 0x0080, 0x8080);
        const auto h = ggml_sycl_dpas_evaluate(wrong, 256, records);
        check(!h.complete() && h.n_eu_seen == 512, "layout too fine: more ids than EUs");
        const std::string s = ggml_sycl_dpas_describe(h);
        check(has(s, "EU id sr0 0x0a30 gets") && !has(s, "slice") && has(s, "reached 512 EU ids of 256 EUs"),
              "layout too fine: raw ids");
    }

    // every EU wrong is not a defective EU
    {
        auto records = make_device(xe2, 8, 4, 8, 8);
        for (ggml_sycl_dpas_record & r : records) {
            r.lanes_chain = 0x00ff;
        }
        const std::string s = ggml_sycl_dpas_describe(ggml_sycl_dpas_evaluate(xe2, 256, records));
        check(has(s, "on every EU the test reached (256): not a defective EU") && !has(s, "hardware defect"),
              "every EU wrong: not reported as hardware");
    }

    // Xe3: slice ids reach above bit 16, and 16 thread slots take bits 0-3
    {
        const ggml_sycl_dpas_layout xe3 = GGML_SYCL_DPAS_LAYOUT_XE3;
        auto records = make_device(xe3, 12, 10, 8, 16);
        const uint32_t eu = make_sr0(xe3, 11, 9, 5, 0) & ggml_sycl_dpas_eu_mask(xe3);
        check(eu == 0x2c950 && ggml_sycl_dpas_eu_mask(xe3) == 0x3cf70, "Xe3: EU id above 16 bits");
        make_fault(records, eu, ggml_sycl_dpas_eu_mask(xe3), 0x0080, 0x8080);
        const auto h = ggml_sycl_dpas_evaluate(xe3, 960, records);
        check(h.complete() && h.faults.size() == 1 && h.faults[0].eu == eu, "Xe3: complete, one fault");
        check(has(ggml_sycl_dpas_describe(h), "EU sr0 0x2c950 (slice 11, Xe-core 9, EU 5)"), "Xe3: report");
    }

    // a test that did not run
    {
        const ggml_sycl_dpas_health h;
        check(!h.complete() && h.n_eu_seen == 0 && h.faults.empty(), "not run: nothing proven");
    }

    // the XMX mat-vec's repair: one lane of a single DPAS on one EU, with every EU tested
    {
        uint32_t eu;
        int      lane;
        auto     records = make_device(xe2, 8, 4, 8, 8);
        check(ggml_sycl_dpas_lane_repair(ggml_sycl_dpas_evaluate(xe2, 256, records), eu, lane) && lane == -1,
              "lane repair: healthy device, nothing to recompute");

        auto chain = records;
        make_fault(chain, 0x0a30, mask, 0, 0x8000);
        check(ggml_sycl_dpas_lane_repair(ggml_sycl_dpas_evaluate(xe2, 256, chain), eu, lane) && lane == -1,
              "lane repair: a fault only in dependent DPAS leaves single DPAS alone");

        make_fault(records, 0x0a30, mask, 0x0080, 0x8080);
        check(ggml_sycl_dpas_lane_repair(ggml_sycl_dpas_evaluate(xe2, 256, records), eu, lane) && eu == 0x0a30 &&
                  lane == 7,
              "lane repair: the B70's lane 7");

        auto two_eu = records;
        make_fault(two_eu, 0x0040, mask, 0x0001, 0x0001);
        check(!ggml_sycl_dpas_lane_repair(ggml_sycl_dpas_evaluate(xe2, 256, two_eu), eu, lane) && lane == -1,
              "lane repair: not two EUs");

        auto two_lanes = records;
        make_fault(two_lanes, 0x0a30, mask, 0x0081, 0x0081);
        check(!ggml_sycl_dpas_lane_repair(ggml_sycl_dpas_evaluate(xe2, 256, two_lanes), eu, lane),
              "lane repair: not two lanes");

        check(!ggml_sycl_dpas_lane_repair(ggml_sycl_dpas_evaluate(xe2, 257, records), eu, lane),
              "lane repair: not without every EU tested");
    }

    // DPAS-R's repair: the dependent-DPAS lanes of one EU, each next to a right lane of its pair
    {
        uint32_t eu;
        uint32_t lanes;
        auto     records = make_device(xe2, 8, 4, 8, 8);
        check(ggml_sycl_dpas_pair_repair(ggml_sycl_dpas_evaluate(xe2, 256, records), eu, lanes) && lanes == 0,
              "pair repair: healthy device, nothing to recompute");

        make_fault(records, 0x0a30, mask, 0x0080, 0x8080);
        check(ggml_sycl_dpas_pair_repair(ggml_sycl_dpas_evaluate(xe2, 256, records), eu, lanes) && eu == 0x0a30 &&
                  lanes == 0x8080,
              "pair repair: the B70's lanes 7 and 15");

        auto pair = records;
        make_fault(pair, 0x0a30, mask, 0x0080, 0x00c0);
        check(!ggml_sycl_dpas_pair_repair(ggml_sycl_dpas_evaluate(xe2, 256, pair), eu, lanes) && lanes == 0,
              "pair repair: not both lanes of a pair");

        auto two_eu = records;
        make_fault(two_eu, 0x0040, mask, 0, 0x0001);
        check(!ggml_sycl_dpas_pair_repair(ggml_sycl_dpas_evaluate(xe2, 256, two_eu), eu, lanes),
              "pair repair: not two EUs");

        check(!ggml_sycl_dpas_pair_repair(ggml_sycl_dpas_evaluate(xe2, 257, records), eu, lanes),
              "pair repair: not without every EU tested");
    }

    if (n_fail) {
        fprintf(stderr, "%d checks failed\n", n_fail);
        return 1;
    }
    printf("OK\n");
    return 0;
}

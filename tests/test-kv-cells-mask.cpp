// llama_kv_cells::seq_mask_row against the per-cell KQ mask rule of llama_kv_cache::set_input_attn_mask
// (no SWA, no ALiBi), over cache states built by appends, rollbacks, shared cells, shifts and restores
//
// Build and run without a model:
//   g++ -std=c++17 -Iinclude -Iggml/include -o /tmp/t tests/test-kv-cells-mask.cpp && /tmp/t

#include "../src/llama-kv-cells.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

static constexpr int n_seq = 3;

template <typename T>
static T ref_cell(const llama_kv_cells & cells, uint32_t j, llama_seq_id seq_id, bool causal, bool is_2d,
        llama_pos p1, llama_pos p1_x, llama_pos p1_y, T v_keep, T v_drop) {
    if (cells.is_empty(j) || !cells.seq_has(j, seq_id)) {
        return v_drop;
    }

    const llama_pos p0 = cells.pos_get(j);

    if (causal) {
        if (p0 > p1) {
            return v_drop;
        }
        if (is_2d && p0 == p1 && cells.ext_get(j).is_2d_gt(p1_x, p1_y)) {
            return v_drop;
        }
    }

    return v_keep;
}

struct checker {
    std::mt19937 rng;
    int n_rows   = 0;
    int failures = 0;

    template <typename T>
    void check_row(const llama_kv_cells & cells, uint32_t n, llama_seq_id seq_id, bool causal, bool is_2d,
            llama_pos p1, llama_pos p1_x, llama_pos p1_y, T v_keep, T v_drop, const char * what) {
        std::vector<T> got(n + 1);
        got[n] = (T) 7; // guard

        cells.seq_mask_row(got.data(), n, seq_id, causal, is_2d, p1, p1_x, p1_y, v_keep, v_drop);

        n_rows++;

        if (got[n] != (T) 7) {
            failures++;
            fprintf(stderr, "%s: wrote past n = %u\n", what, n);
            return;
        }

        for (uint32_t j = 0; j < n; ++j) {
            const T exp = ref_cell(cells, j, seq_id, causal, is_2d, p1, p1_x, p1_y, v_keep, v_drop);
            if (std::memcmp(&exp, &got[j], sizeof(T)) != 0) {
                failures++;
                fprintf(stderr, "%s: seq %d causal %d 2d %d p1 %d (%d,%d) n %u: cell %u differs\n",
                        what, seq_id, causal, is_2d, p1, p1_x, p1_y, n, j);
                return;
            }
        }
    }

    void check_all(const llama_kv_cells & cells, const char * what) {
        const uint32_t size = cells.size();

        llama_pos p_max = 0;
        for (llama_seq_id s = 0; s < n_seq; ++s) {
            p_max = std::max(p_max, cells.seq_pos_max(s));
        }

        for (int q = 0; q < 24; ++q) {
            const llama_seq_id seq_id = rng() % (n_seq + 1); // seq n_seq never has cells

            uint32_t n = size;
            switch (rng() % 3) {
                case 0: n = std::min<uint32_t>(size, cells.used_max_p1() + rng() % 70); break;
                case 1: n = rng() % (size + 1); break;
                default: break;
            }

            llama_pos p1 = (llama_pos) (rng() % (p_max + 4)) - 1;
            if (rng() % 2) {
                p1 = std::max(0, cells.seq_pos_max(seq_id) - (llama_pos) (rng() % 6));
            }
            const llama_pos p1_x = rng() % 4;
            const llama_pos p1_y = rng() % 4;

            for (int causal = 0; causal < 2; ++causal) {
                for (int is_2d = 0; is_2d < 2; ++is_2d) {
                    check_row<uint16_t>(cells, n, seq_id, causal, is_2d, p1, p1_x, p1_y, 0x0000, 0xFC00, what);
                    check_row<float>   (cells, n, seq_id, causal, is_2d, p1, p1_x, p1_y, 0.0f, -INFINITY, what);
                }
            }
        }
    }
};

// place n tokens of seq_id after its last position into the empty cells found from head on, like find_slot
static void append(llama_kv_cells & cells, uint32_t & head, llama_seq_id seq_id, int n, std::mt19937 & rng) {
    llama_pos p = cells.seq_pos_max(seq_id) + 1;
    for (uint32_t k = 0; k < cells.size() && n > 0; ++k) {
        const uint32_t i = (head + k) % cells.size();
        if (!cells.is_empty(i)) {
            continue;
        }
        cells.pos_set(i, p);
        cells.ext_set(i, { (llama_pos) (rng() % 4), (llama_pos) (rng() % 4), (llama_token) p });
        cells.seq_add(i, seq_id);
        head = (i + 1) % cells.size();
        p++;
        n--;
    }
}

// remove the cells of seq_id at positions >= p0 (an MTP rollback of rejected drafts)
static void rollback(llama_kv_cells & cells, uint32_t & head, llama_seq_id seq_id, llama_pos p0) {
    for (uint32_t i = 0; i < cells.size(); ++i) {
        if (!cells.is_empty(i) && cells.seq_has(i, seq_id) && cells.pos_get(i) >= p0) {
            if (cells.seq_rm(i, seq_id)) {
                head = std::min(head, i);
            }
        }
    }
}

int main() {
    checker ck { std::mt19937(1234) };
    auto & rng = ck.rng;

    for (const uint32_t size : { 1u, 63u, 64u, 65u, 300u, 1024u, 1500u }) {
        llama_kv_cells cells;
        cells.resize(size);
        ck.check_all(cells, "empty");

        uint32_t head = 0;

        for (int step = 0; step < 300; ++step) {
            const llama_seq_id seq_id = rng() % n_seq;

            switch (rng() % 9) {
                case 0: case 1: case 2:
                    append(cells, head, seq_id, 1 + rng() % 40, rng);
                    ck.check_all(cells, "append");
                    break;
                case 3: case 4: {
                    // draft tokens appended and partly rejected
                    const llama_pos p0 = cells.seq_pos_max(seq_id) + 1;
                    append(cells, head, seq_id, 5, rng);
                    rollback(cells, head, seq_id, p0 + rng() % 5);
                    ck.check_all(cells, "rollback");
                } break;
                case 5: {
                    // a cell shared by a second sequence (seq_cp)
                    const llama_seq_id dst = (seq_id + 1) % n_seq;
                    for (uint32_t i = rng() % size; i < size; i += 1 + rng() % 7) {
                        if (!cells.is_empty(i) && cells.seq_has(i, seq_id) && !cells.seq_has(i, dst)) {
                            cells.seq_add(i, dst);
                        }
                    }
                    ck.check_all(cells, "seq_cp");
                } break;
                case 6: {
                    // context shift: drop the oldest positions, shift the rest down
                    const llama_pos d = 1 + rng() % 20;
                    for (uint32_t i = 0; i < size; ++i) {
                        if (!cells.is_empty(i) && cells.seq_has(i, seq_id)) {
                            cells.pos_add(i, -d);
                        }
                    }
                    cells.reset_shift();
                    ck.check_all(cells, "shift");
                } break;
                case 7: {
                    // save a range, wipe it, restore it (prompt cache)
                    const uint32_t i0 = rng() % size;
                    const uint32_t n  = rng() % (size - i0 + 1);
                    const llama_kv_cells saved = cells.cp(i0, n);
                    for (uint32_t i = i0; i < i0 + n; ++i) {
                        if (!cells.is_empty(i)) {
                            cells.rm(i);
                        }
                    }
                    ck.check_all(cells, "wiped");
                    cells.set(i0, saved);
                    ck.check_all(cells, "restore");
                } break;
                default:
                    if (rng() % 8 == 0) {
                        cells.reset();
                        head = 0;
                    } else {
                        // a hole: remove one random cell
                        const uint32_t i = rng() % size;
                        if (!cells.is_empty(i)) {
                            cells.rm(i);
                            head = std::min(head, i);
                        }
                    }
                    ck.check_all(cells, "hole");
                    break;
            }
        }
    }

    printf("%s: %d rows checked, %d failures\n", __FILE__, ck.n_rows, ck.failures);

    return ck.failures == 0 ? 0 : 1;
}

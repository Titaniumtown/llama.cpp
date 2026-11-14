#pragma once

#include <string>
#include <cstddef>
#include <algorithm>
#include <cstdint>
#include "server-prompt-cache-value.h"
#include "server-prompt-cache-spread.h"

// a file in the disk tier directory is a loadable entry only if it ends in ".bin";
// anything else -- in particular a ".bin.tmp" left by a writer that died between the
// write and the rename -- is never loaded, never counted, never evicted
static inline bool server_prompt_cache_is_entry_path(const std::string & path) {
    static const char k_suffix[] = ".bin";
    const std::size_t n = sizeof(k_suffix) - 1;

    return path.size() > n + 1 && path[path.size() - n - 1] != '/' && path.compare(path.size() - n, n, k_suffix) == 0;
}

// the first checkpoint the stored entry kept. creation breaks on a strict >, and the two
// tail rungs can push the count over the cap, so k*step is below the rung the restore
// will accept. the floor is that kept n_tokens: below it the resume is 0.
static inline int server_prompt_cache_first_rung(std::size_t entry_tokens, std::size_t min_step, int max_checkpoints,
        std::size_t n_batch = 2048) {
    const std::size_t step = min_step > 0 ? min_step : 1;

    if (entry_tokens < step) {
        return (int) entry_tokens;
    }

    const std::size_t batch = n_batch > 0 ? n_batch : step;
    const auto step_at = [step](std::size_t n) {
        return n >= step * 6 ? step * 4 : step;
    };

    std::vector<int64_t> created;
    std::size_t n = 0;
    while (n < entry_tokens) {
        const std::size_t batch_start = n;
        while (n < entry_tokens && n - batch_start < batch) {
            n++;
            const std::size_t last = created.empty() ? 0 : (std::size_t) created.back();
            if (n > last + step_at(last)) {
                break;
            }
            bool tail = false;
            const std::size_t offsets[] = { 4 + batch, 4 };
            for (const std::size_t offset : offsets) {
                if (entry_tokens == n + std::min(batch, offset)) {
                    tail = true;
                    break;
                }
            }
            if (tail) {
                break;
            }
        }
        if (n == batch_start) {
            break;
        }
        const std::size_t last_ckpt = created.empty() ? 0 : (std::size_t) created.back();
        const bool near_end = entry_tokens < n + batch;
        const bool past = batch_start > last_ckpt + step_at(last_ckpt);
        const bool done = n == entry_tokens;
        if ((done || near_end || past) && batch_start != 0 && (created.empty() || near_end || past)) {
            created.push_back((int64_t) batch_start);
        }
        if (done) {
            break;
        }
    }

    if (created.empty()) {
        return (int) entry_tokens;
    }

    const auto kept = server_prompt_cache_select_checkpoint_positions(created, max_checkpoints, min_step);
    if (kept.empty()) {
        return (int) entry_tokens;
    }
    return (int) kept.front();
}

// a disk fetch is a blocking read on the server thread: both slots stall behind it. it must
// save more prefill than the read costs. measured on the production pool: 0.7-0.9 GB/s of read
// against 500-700 tok/s of prefill, i.e. about one token of reuse per 1.5 MB of entry, doubled
// so the swap clearly wins. the floor is absolute, not just a delta over the RAM tier: with an
// empty RAM tier the delta rule alone let a 5-token head win a 314 MB read. the head also has
// to land on a rung, and what the restore actually reuses is the rung, not the head
static inline bool server_prompt_cache_disk_wins(int lcp_disk, int lcp_ram, std::size_t entry_size, int first_rung) {
    const int fetch_min_gain = std::max(2048, (int) (entry_size / 750'000));

    return lcp_disk >= 0
           && lcp_disk >= first_rung
           && lcp_disk >= fetch_min_gain
           && lcp_disk > (lcp_ram >= 0 ? lcp_ram + fetch_min_gain : -1);
}

// the disk tier's eviction price: the same tokens-per-byte decayed by recency the resident
// tier pays, not file age. the tier ran first-in-first-out on a 96%-full pool: a hot
// conversation's state was the oldest file in the directory, so every write deleted it while
// small stale prefixes survived (measured: a 106k head restored from a 20.5k prefix entry,
// 85.5k tokens re-processed). the states carry size and last_use_us, and the token count is
// supplied by the caller, so the header stays free of the server's types
template <typename States, typename Tokens>
static inline std::size_t server_prompt_cache_disk_victim(const States & states, const Tokens & tokens,
        std::size_t slot_tokens, int64_t now_us) {
    std::size_t worst = 0;

    for (std::size_t i = 1; i < states.size(); i++) {
        if (server_prompt_cache_value((int64_t) tokens(states[i]), (std::size_t) states[i].size, states[i].last_use_us, slot_tokens, now_us)
                < server_prompt_cache_value((int64_t) tokens(states[worst]), (std::size_t) states[worst].size, states[worst].last_use_us, slot_tokens,
                       now_us)) {
            worst = i;
        }
    }

    return worst;
}

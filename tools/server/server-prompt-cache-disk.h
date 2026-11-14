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

// uncontended price: 0.7-0.9 GB/s against 500-700 tok/s, doubled, is one token per 0.75 MB.
// a measured rate tightens that price and never loosens it. bytes_per_token <= 0 means uncontended.
static constexpr int SERVER_PROMPT_CACHE_UNCONTENDED_BPT = 750000;
static constexpr int SERVER_PROMPT_CACHE_PREFILL_TOK_S   = 667;
static constexpr int SERVER_PROMPT_CACHE_FETCH_SAFETY    = 2;
static constexpr std::size_t SERVER_PROMPT_CACHE_WRITE_WINDOW = 128ull << 20;

static inline int server_prompt_cache_bytes_per_token(double mb_s) {
    if (!(mb_s > 0.0)) {
        return SERVER_PROMPT_CACHE_UNCONTENDED_BPT;
    }

    const double bpt = mb_s * 1e6 / (double) (SERVER_PROMPT_CACHE_FETCH_SAFETY * SERVER_PROMPT_CACHE_PREFILL_TOK_S);

    if (bpt < 1.0) {
        return 1;
    }
    if (bpt > (double) SERVER_PROMPT_CACHE_UNCONTENDED_BPT) {
        return SERVER_PROMPT_CACHE_UNCONTENDED_BPT;
    }

    return (int) bpt;
}

static inline double server_prompt_cache_rate_median(const double * samples, int n) {
    if (n <= 0 || samples == nullptr) {
        return 0.0;
    }
    if (n > 8) {
        n = 8;
    }

    double tmp[8];
    for (int i = 0; i < n; i++) {
        tmp[i] = samples[i];
    }
    std::sort(tmp, tmp + n);

    return tmp[n / 2];
}

// the fread is off the server thread. the requester still waits, and the read shares the
// device with writeback, so a head that prefill would finish first must not be fetched.
// the floor is absolute, not just a delta over the RAM tier.
static inline bool server_prompt_cache_disk_wins(int lcp_disk, int lcp_ram, std::size_t entry_size, int first_rung,
        int bytes_per_token = SERVER_PROMPT_CACHE_UNCONTENDED_BPT) {
    if (bytes_per_token < 1) {
        bytes_per_token = SERVER_PROMPT_CACHE_UNCONTENDED_BPT;
    }

    const int fetch_min_gain = (int) std::max<std::size_t>(2048, std::min(entry_size / (std::size_t) bytes_per_token, (std::size_t) 0x7fffffff));

    return lcp_disk >= 0
           && lcp_disk >= first_rung
           && lcp_disk >= fetch_min_gain
           && lcp_disk > (lcp_ram >= 0 ? lcp_ram + fetch_min_gain : -1);
}

// a failed read is not a state. a checksum miss and a short file are the same outcome:
// leaving the index entry makes the next match pay the read and fail again
static inline bool server_prompt_cache_disk_failed_read_unlinks(bool ok, bool corrupt) {
    (void) corrupt;

    return !ok;
}

struct server_prompt_cache_write_behind {
    std::size_t off = 0;
    std::size_t len = 0;
};

// bytes already in the page cache that must be waited and dropped so dirty stays within
// the window. len 0 means the window still holds what has been written
static inline server_prompt_cache_write_behind server_prompt_cache_write_behind_of(
        std::size_t written, std::size_t released, std::size_t window) {
    server_prompt_cache_write_behind behind;
    behind.off = released;

    if (written <= released) {
        return behind;
    }

    const std::size_t dirty = written - released;
    if (dirty <= window) {
        return behind;
    }

    behind.len = dirty - window;

    return behind;
}

enum class server_prompt_cache_queue_act { push, replace, drop_push };

struct server_prompt_cache_queue_decision {
    server_prompt_cache_queue_act act   = server_prompt_cache_queue_act::push;
    std::size_t                   index = 0;
};

// a same-hash state already queued is replaced, not stacked. a full queue drops the least
// valuable entry, not the front: the front is often the conversation that just spilled
template <typename Queue, typename HashOf, typename TokensOf>
static inline server_prompt_cache_queue_decision server_prompt_cache_queue_admit(
        const Queue & queued, const HashOf & hash_of, const TokensOf & tokens_of, uint64_t hash, std::size_t cap,
        std::size_t slot_tokens, int64_t now_us) {
    server_prompt_cache_queue_decision dec;

    for (std::size_t i = 0; i < queued.size(); i++) {
        if (hash_of(queued[i]) == hash) {
            dec.act   = server_prompt_cache_queue_act::replace;
            dec.index = i;

            return dec;
        }
    }

    if (cap == 0 || queued.size() < cap) {
        dec.act = server_prompt_cache_queue_act::push;

        return dec;
    }

    dec.act   = server_prompt_cache_queue_act::drop_push;
    dec.index = server_prompt_cache_disk_victim(queued, tokens_of, slot_tokens, now_us);

    return dec;
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

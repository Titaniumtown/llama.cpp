#pragma once

// The RAM prompt tier's cost/benefit metric, kept free of the server so it can be tested on
// its own. A stored state is worth the tokens a future request takes from it, per byte it
// holds: its length, capped at one slot (a request cannot reuse more than a slot can hold, so
// a 300k state is not worth 300k tokens to anything), decayed by the time since it was last
// used (a disk fetch reads 0.7-0.9 GB/s while the prefill it saves runs at 500-700 tok/s, so
// the state asked for a minute ago is the one likely to be asked for next).
//
// Both the save path (admitting a new state) and the eviction path (choosing a victim) price a
// state with this one function, and the price holds no reuse count. A count would compare a
// resident's accumulated history against an arrival that cannot have one -- a conversation
// being saved for the first time has just started -- and with it every arrival loses to every
// resident that was ever reused, so the tier admits nothing new once it is full. Recency is
// the reuse signal both sides can be priced on: it is a property of the state, not of how many
// turns its conversation has taken.

#include <algorithm>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <cstddef>
#include <cstdint>

static constexpr double SERVER_PROMPT_CACHE_VALUE_HALF_LIFE_S = 600.0;

static inline float server_prompt_cache_value(int64_t tokens, size_t bytes, int64_t last_use_us,
        size_t slot_tokens, int64_t now_us) {
    const size_t n_tok = (size_t) std::min<int64_t>(tokens, (int64_t) slot_tokens);
    const size_t n_byt = std::max<size_t>(1, bytes);

    const double age   = std::max<double>(0.0, (double) (now_us - last_use_us) / 1e6);
    // exp2, so the constant is a half-life: the value halves every
    // SERVER_PROMPT_CACHE_VALUE_HALF_LIFE_S seconds
    const double decay = std::exp2(-age / SERVER_PROMPT_CACHE_VALUE_HALF_LIFE_S);

    return (float) (decay * (double) n_tok / (double) n_byt);
}

// last_use_us is ggml_time_us(). file mtime is not: libstdc++ file_clock counts
// nanoseconds before 2174, so the raw count prices every restarted entry at decay 0.
// two file_clock times cancel that epoch.
static inline int64_t server_prompt_cache_use_from_write_time(
        std::filesystem::file_time_type written, int64_t now_us) {
    const auto age = std::filesystem::file_time_type::clock::now() - written;
    const auto age_us = std::chrono::duration_cast<std::chrono::microseconds>(age).count();
    if (age_us <= 0) {
        return now_us;
    }
    return now_us - age_us;
}

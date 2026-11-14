// The disk tier's eviction price: the same tokens-per-byte decayed by recency the resident
// tier pays, not file age. First-in-first-out on a 96%-full pool deleted a hot
// conversation's state -- the oldest file in the directory -- on every write, while small
// stale prefixes survived (measured: a 106k head restored from a 20.5k prefix entry,
// 85.5k tokens re-processed).
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-disk-prune-value.cpp && /tmp/t
#include "server-prompt-cache-disk.h"

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include <filesystem>
#include <fstream>

struct entry {
    std::size_t tokens;
    std::size_t size;
    int64_t     last_use_us;
};

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

int main() {
    const int64_t     NOW  = 1'000'000'000'000LL;
    const std::size_t SLOT = 8192;
    const int64_t     HALF = 1800'000'000; // 0.5 h: three value half-lives of decay

    const auto tokens = [](const entry & e) { return e.tokens; };

    // a fresh entry in front, a cold one behind: first-in-first-out would evict the fresh
    // one; the price says the cold one is worth a billion times less per byte
    std::vector<entry> states = {
        { 20480, 600'000'000, NOW },
        { 98306, 3'400'000'000, NOW - HALF },
    };
    check(server_prompt_cache_disk_victim(states, tokens, SLOT, NOW) == 1,
          "the cold entry behind a fresh one is the victim, not the front of the list");

    // the reverse order: the price, not the position, decides
    states = {
        { 98306, 3'400'000'000, NOW - HALF },
        { 20480, 600'000'000, NOW },
    };
    check(server_prompt_cache_disk_victim(states, tokens, SLOT, NOW) == 0,
          "the cold entry in front is the victim even though it is larger");

    // equal density, different recency: the older one loses
    states = {
        { 10000, 1'000'000'000, NOW },
        { 10000, 1'000'000'000, NOW - HALF },
    };
    check(server_prompt_cache_disk_victim(states, tokens, SLOT, NOW) == 1,
          "equal density, the older entry loses");

    // tokens are capped at one slot: a 300k entry is not worth 300k tokens to anything
    states = {
        { 300'000, 3'000'000'000, NOW },   // 8192 / 3 GB, capped
        { 8192,    600'000'000, NOW },    // 8192 / 0.6 GB
    };
    check(server_prompt_cache_disk_victim(states, tokens, SLOT, NOW) == 0,
          "the slot cap prices a 300k entry at one slot, below the lean 8k entry");

    // a single entry is its own victim
    states = { { 100, 1000, NOW } };
    check(server_prompt_cache_disk_victim(states, tokens, SLOT, NOW) == 0, "a single entry is its own victim");

    // a restarted entry's mtime is file_clock, not ggml_time_us(). the raw count prices it
    // at decay 0, so the first save after a restart evicts a hot cache for any new state
    {
        const char * path = "/tmp/pcache-mtime-stamp";
        {
            std::ofstream out(path);
            out << 'x';
        }
        const auto fresh = std::filesystem::last_write_time(path);
        const int64_t use_now = server_prompt_cache_use_from_write_time(fresh, NOW);
        check((NOW - use_now) / 1000000 < 5, "a file written now is priced as just used");
        std::filesystem::last_write_time(path, fresh - std::chrono::seconds(600));
        const auto aged = std::filesystem::last_write_time(path);
        const int64_t use_old = server_prompt_cache_use_from_write_time(aged, NOW);
        const int64_t age_s   = (NOW - use_old) / 1000000;
        check(age_s > 590 && age_s < 610, "a file written 600s ago is one half-life old");
        std::filesystem::remove(path);
    }

    printf("PASS\n");
    return 0;
}

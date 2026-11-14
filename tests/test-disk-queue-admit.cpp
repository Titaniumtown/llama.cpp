// The writer queue: a same-hash spill replaces the queued state. A full queue drops the
// least valuable entry, not the front.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-disk-queue-admit.cpp && /tmp/t
#include "server-prompt-cache-disk.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

struct qitem {
    uint64_t hash;
    int64_t  tokens;
    std::size_t size;
    int64_t  last_use_us;
};

static const int64_t NOW  = 1000000000;
static const std::size_t SLOT = 131072;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

static server_prompt_cache_queue_decision admit(const std::vector<qitem> & q, uint64_t hash, std::size_t cap) {
    return server_prompt_cache_queue_admit(q,
            [](const qitem & e) { return e.hash; },
            [](const qitem & e) { return e.tokens; },
            hash, cap, SLOT, NOW);
}

int main() {
    std::vector<qitem> q = {
        { 1, 1000, 1000, NOW },
        { 2, 1000, 1000, NOW },
    };

    const auto pushed = admit(q, 3, 8);
    check(pushed.act == server_prompt_cache_queue_act::push, "a queue under the cap accepts a new hash");

    const auto replaced = admit(q, 2, 8);
    check(replaced.act == server_prompt_cache_queue_act::replace && replaced.index == 1,
            "a same-hash spill replaces the queued state");

    q = {
        { 1, 8000, 8000, NOW },
        { 2, 1000, 1000, NOW - 600000000 },
    };
    const auto dropped = admit(q, 3, 2);
    check(dropped.act == server_prompt_cache_queue_act::drop_push && dropped.index == 1,
            "a full queue drops the cold entry, not the front");

    printf("PASS\n");
    return 0;
}

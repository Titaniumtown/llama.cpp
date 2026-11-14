// The RAM prompt cache's cost metric: which stored state the tier spends its bytes on.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-prompt-cache-value.cpp && /tmp/t
#include "server-prompt-cache-value.h"

#include <cstdio>
#include <cstdlib>

static const size_t  SLOT = 131072;
static const int64_t NOW  = 1000000000;
static const int64_t HALF = (int64_t) (SERVER_PROMPT_CACHE_VALUE_HALF_LIFE_S * 1e6);

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

int main() {
    // the same state, one half-life older, is worth half as much
    const float fresh = server_prompt_cache_value(1000, 1000*24576, NOW, SLOT, NOW);
    const float aged  = server_prompt_cache_value(1000, 1000*24576, NOW - HALF, SLOT, NOW);
    check(aged < fresh && aged > 0.49f*fresh && aged < 0.51f*fresh, "a state half a half-life older is worth half");

    // length is capped at a slot: at equal bytes, ten slots of tokens are worth one slot, not
    // ten (a request cannot reuse more than a slot, so the rest is dead weight in a live tier)
    const float at_cap = server_prompt_cache_value((int64_t) SLOT, SLOT*24576, NOW, SLOT, NOW);
    const float over   = server_prompt_cache_value((int64_t) SLOT*10, SLOT*24576, NOW, SLOT, NOW);
    check(over == at_cap, "at equal bytes, a state longer than a slot is capped at one slot");

    // bytes are the cost side: the same tokens carrying another entry's worth of checkpoints
    // rank below the lean one
    const float lean       = server_prompt_cache_value(1000, 1000*24576, NOW, SLOT, NOW);
    const float ckpt_heavy = server_prompt_cache_value(1000, 1000*24576*4, NOW, SLOT, NOW);
    check(ckpt_heavy < lean, "checkpoint bytes count against the state that carries them");

    // the victim is the minimum over the resident set: the state nothing continued, at the same
    // length and density as one with a use (a 1000-token state a half-life stale is worth less
    // than a fresh one)
    const float stale  = server_prompt_cache_value(1000, 1000*24576, NOW - HALF, SLOT, NOW);
    const float fresh2 = server_prompt_cache_value(1000, 1000*24576, NOW, SLOT, NOW);
    check(stale < fresh2, "the least valuable state is the one nothing continued");

    // a refusal cannot be permanent. an arrival that is checkpoint-heavy (double the bytes for
    // the same tokens) loses against a resident used this instant, and wins against that same
    // resident once it has gone two half-lives unused -- so a full tier keeps making room
    // instead of refusing one conversation's every save, which is what a price carrying the
    // residents' reuse history did: it excluded every arrival the tier had not already seen
    const float arrival = server_prompt_cache_value(1000, 1000*24576*2, NOW, SLOT, NOW);
    const float cold    = server_prompt_cache_value(1000, 1000*24576, NOW - 2*HALF, SLOT, NOW);
    check(arrival < fresh2, "a state carrying double the bytes is refused against a resident used now");
    check(arrival > cold, "the same state is admitted once the resident is two half-lives unused");

    printf("PASS\n");
    return 0;
}

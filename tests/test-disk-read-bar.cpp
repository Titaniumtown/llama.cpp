// The bar a disk read has to clear before a task waits on it: the head the task already has,
// from the RAM tier or in the slot it is handed. Measured in production: a slot pinned with the
// whole 98304-token prompt waited on a 32818-token disk head, restored it over its own state
// and re-processed 65540 tokens.
//
// Links server-context (built by tests/CMakeLists.txt); runs without a model. No reader thread
// is started, so an armed read stays pending and nothing touches the disk.
#include "server-slot-sched.h"

#include <cstdio>
#include <cstdlib>
#include <memory>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

static llama_tokens head(const llama_tokens & toks, size_t n, size_t tail) {
    llama_tokens out(toks.begin(), toks.begin() + n);
    for (size_t i = 0; i < tail; i++) {
        out.push_back(900000 + (llama_token) i);
    }
    return out;
}

using wait_t = server_prompt_cache::disk_wait;

// one fresh tier per case: the read scheduler remembers an armed path
static wait_t arm(const llama_tokens & req, const llama_tokens & entry, size_t entry_bytes,
        const llama_tokens * slot, const llama_tokens * ram = nullptr) {
    auto cache = std::make_unique<server_prompt_cache>(0, 0, 8, 4096);
    cache->disk_dir = "/nonexistent/test-disk-read-bar";

    server_prompt_cache::disk_state e;
    e.tokens = server_tokens(entry, false);
    e.path   = "/nonexistent/test-disk-read-bar/entry.bin";
    e.size   = entry_bytes;
    cache->disk_states.push_back(std::move(e));

    if (ram != nullptr) {
        server_prompt_cache_state st;
        st.prompt.tokens = server_tokens(*ram, false);
        cache->states.push_back(std::move(st));
    }

    server_prompt prompt_slot;
    if (slot != nullptr) {
        prompt_slot.tokens = server_tokens(*slot, false);
    }

    return cache->disk_prepare(1, server_tokens(req, false), slot != nullptr ? &prompt_slot : nullptr);
}

int main() {
    llama_tokens req;
    for (int i = 0; i < 98304; i++) {
        req.push_back(1000 + i);
    }

    // the measured case: a 33023-token entry sharing 32818 tokens, 827 MB of state
    const llama_tokens entry = head(req, 32818, 205);
    const size_t       bytes = 867'000'000;

    llama_tokens slot_full = req;
    slot_full.push_back(7); // the token the warm request generated

    check(arm(req, entry, bytes, &slot_full) == wait_t::skip,
            "a slot holding the whole prompt does not wait on a 32818-token disk head");
    check(arm(req, entry, bytes, nullptr) == wait_t::pending,
            "a task with no slot head waits on the 32818-token disk head");

    const llama_tokens slot_short = head(req, 4096, 50);
    check(arm(req, entry, bytes, &slot_short) == wait_t::pending,
            "a 4096-token slot head does not keep the task off a 32818-token read");

    // the gain over the slot is priced like the gain over a RAM head: max(2048, bytes / bpt)
    const llama_tokens slot_near = head(req, 31000, 10);
    check(arm(req, entry, bytes, &slot_near) == wait_t::skip,
            "1818 tokens over the slot's head do not pay a read");
    check(arm(req, entry, bytes, nullptr, &slot_near) == wait_t::skip,
            "1818 tokens over a RAM head do not pay a read");

    const llama_tokens other = { 1, 2, 3, 4 };
    check(arm(req, entry, bytes, &other) == wait_t::pending,
            "a slot holding another conversation does not keep the task off the read");

    return 0;
}

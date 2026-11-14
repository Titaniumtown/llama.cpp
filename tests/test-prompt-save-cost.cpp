// What a prompt-cache save costs the server thread, the one thread every slot waits on.
// Measured at 98k (Arc Pro B70, 8 checkpoints per entry): a request whose launch
// saved the other, idle slot waited 1.21 s before its prompt started; with the checkpoints
// shared and the replaced entry unmapped off the thread, 0.60 s.
//
// Links server-context (built by tests/CMakeLists.txt); runs without a model. The state
// sizes are passed in, so no context is needed.
#include "server-slot-sched.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

static server_prompt make_prompt(int n_tokens, const std::vector<int> & ckpt_at) {
    llama_tokens toks;
    for (int i = 0; i < n_tokens; i++) {
        toks.push_back(1000 + i);
    }

    server_prompt prompt;
    prompt.tokens = server_tokens(toks, false);
    for (int at : ckpt_at) {
        common_prompt_checkpoint ckpt;
        ckpt.n_tokens = at;
        ckpt.pos_min  = 0;
        ckpt.pos_max  = at - 1;
        ckpt.data_tgt.resize(1024 * 1024);
        prompt.checkpoints.push_back(std::make_shared<const common_prompt_checkpoint>(std::move(ckpt)));
    }

    return prompt;
}

// the reaper frees on its own thread: wait for it, bounded
static bool released_reaches(size_t n) {
    for (int i = 0; i < 500; i++) {
        if (server_state_released() >= n) {
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return false;
}

int main() {
    static const size_t MiB = 1024 * 1024;

    // a stored entry shares the slot's checkpoints: they never change after they are
    // created, and the copy was 1.4 GB of memcpy into fresh pages per save at 98k
    {
        server_prompt_cache cache(0, 0, 8, 4096);
        const server_prompt slot = make_prompt(13000, { 4097, 8193, 12289 });

        auto * entry = cache.alloc(slot, [](size_t & tgt, size_t & dft) { tgt = 4096; dft = 0; });
        check(entry != nullptr, "a new prompt is stored");
        check(entry->prompt.checkpoints.size() == 3, "the entry keeps the slot's three checkpoints");

        bool shared = true;
        auto it = slot.checkpoints.begin();
        for (const auto & ckpt : entry->prompt.checkpoints) {
            shared = shared && ckpt.get() == it->get();
            ++it;
        }
        check(shared, "the entry's checkpoints are the slot's objects, not copies");
        check(entry->size() == 4096 + 3 * MiB, "the shared checkpoints still count against the entry");
    }

    // a prompt the tier already holds is not sized: the size query walks every cell of the
    // sequence, 14 ms at 98k, and an idle slot is offered again on every task launch
    {
        server_prompt_cache cache(0, 0, 8, 4096);
        int n_sized = 0;
        const auto sized = [&](size_t & tgt, size_t & dft) { n_sized++; tgt = 4096; dft = 0; };

        check(cache.alloc(make_prompt(2000, {}), sized) != nullptr && n_sized == 1, "a new prompt is sized once");
        check(cache.alloc(make_prompt(1990, {}), sized) == nullptr && n_sized == 1, "a held prompt is skipped unsized");
        check(cache.alloc(make_prompt(2010, {}), sized) != nullptr && n_sized == 2, "a longer prompt is sized");
        check(cache.states.size() == 1, "the longer prompt replaced the one it contains");
    }

    // a large state buffer is unmapped off the caller's thread; a small one is not handed off
    {
        const size_t n0 = server_state_released();
        {
            server_state_buffer small(1 * MiB);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        check(server_state_released() == n0, "a 1 MiB buffer is freed by its caller");

        {
            server_state_buffer large(64 * MiB);
        }
        check(released_reaches(n0 + 1), "a 64 MiB buffer is freed off the caller's thread");

        // the save path: the entry a longer prompt replaces is released the same way
        server_prompt_cache cache(0, 0, 8, 4096);
        const auto sized = [](size_t & tgt, size_t & dft) { tgt = 64 * MiB; dft = 0; };
        check(cache.alloc(make_prompt(1000, {}), sized) != nullptr, "a 64 MiB state is stored");
        check(cache.alloc(make_prompt(1010, {}), sized) != nullptr, "its continuation is stored");
        check(released_reaches(n0 + 2), "the replaced entry is freed off the server thread");
    }

    printf("PASS\n");
    return 0;
}

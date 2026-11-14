// Which checkpoints a stored entry keeps: the grid a resuming request can land on.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-entry-checkpoint-spread.cpp && /tmp/t
#include "server-prompt-cache-spread.h"

#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

int main() {
    // the production min step is 4096; its grid on a 98k entry is 4097, 8193, 12289,
    // 16385, 20481, 24577, then 16384 steps to the tail. the measured shared-head
    // cluster of one agent's sessions sits at 21-24k, with a secondary at 18k:
    // a 23.5k head must resume at 20481 (loss 3k), an 18k head at 16385 (loss 2k) --
    // a grid with no checkpoint between 8193 and 20481 makes the 18k head pay 10k
    const std::vector<int64_t> grid98k = {4097, 8193, 12289, 16385, 20481, 24577,
                                         40961, 57345, 73729, 90113, 98306};

    const auto kept = server_prompt_cache_select_checkpoint_positions(grid98k, 4, 4096);
    const std::vector<int64_t> want = {8193, 16385, 20481, 98306};
    check(kept == want, "a 23.5k head resumes at 20481 and an 18k head at 16385");

    // the final checkpoint stays: a continuation (lcp = full entry) pays zero
    check(kept.back() == 98306, "the final checkpoint catches a continuation");

    // every kept checkpoint sits within one min step of its target: nothing is kept
    // for its own sake
    const std::size_t targets[] = { 2*4096, 4*4096, 5*4096 };
    for (size_t i = 0; i < 3 && i < kept.size(); i++) {
        const int64_t dist = kept[i] > (int64_t) targets[i] ? kept[i] - (int64_t) targets[i]
                                                           : (int64_t) targets[i] - kept[i];
        check(dist <= 4096, "every kept checkpoint is within one min step of a target");
    }

    // a short prompt keeps its whole grid: the spread only bites past the cap
    const std::vector<int64_t> grid8k = {4097, 8193};
    check(server_prompt_cache_select_checkpoint_positions(grid8k, 4, 4096) == grid8k,
          "a two-checkpoint prompt is kept as-is");

    // the selection never returns a position the grid does not have
    for (const int64_t pos : kept) {
        bool found = false;
        for (const int64_t g : grid98k) {
            if (g == pos) {
                found = true;
                break;
            }
        }
        check(found, "every kept checkpoint exists in the source grid");
    }
    // the production cap is 8: the grid follows the creation cadence into the mid entry, so a
    // head at 45k lands on 40961 (loss 4k), not on 20481 (loss 24k), and a 77k head on 73729
    const auto kept8 = server_prompt_cache_select_checkpoint_positions(grid98k, 8, 4096);
    const std::vector<int64_t> want8 = {8193, 16385, 20481, 24577, 40961, 57345, 73729, 98306};
    check(kept8 == want8, "a 45k head resumes at 40961 and a 77k head at 73729");

    // the final checkpoint stays at the cap of 8 as well: a continuation pays zero
    check(kept8.back() == 98306, "the cap-8 grid still catches a continuation");

    printf("PASS\n");
    return 0;
}

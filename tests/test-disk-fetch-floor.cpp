// The gate for a disk fetch: which head wins the blocking read. The floor is absolute,
// not just a delta over the RAM tier -- with an empty RAM tier the delta rule alone let a
// 5-token head win a 314 MB read. The head also has to land on a checkpoint at all: below
// the entry's first rung the restore resumes at 0 and the read buys nothing.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-disk-fetch-floor.cpp && /tmp/t
#include "server-prompt-cache-disk.h"

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
    const std::size_t small = 314'076'348;  // the 12-token entry on the live tier: 314 MB
    const std::size_t large = 3'424'552'668; // the 98k-token entry on the live tier: 3.4 GB
    const std::size_t step  = 4096;          // the production --checkpoint-min-step
    const int         cap   = 4;             // the production --cache-entry-checkpoints

    // the observed wasted fetches: lcp 5/7/8/40/374 against an empty RAM tier
    check(!server_prompt_cache_disk_wins(5, -1, small, 0), "a 5-token head does not win an empty RAM tier");
    check(!server_prompt_cache_disk_wins(7, -1, small, 0), "a 7-token head does not win an empty RAM tier");
    check(!server_prompt_cache_disk_wins(8, -1, small, 0), "an 8-token head does not win an empty RAM tier");
    check(!server_prompt_cache_disk_wins(40, -1, small, 0), "a 40-token head does not win an empty RAM tier");
    check(!server_prompt_cache_disk_wins(374, -1, small, 0), "a 374-token head does not win an empty RAM tier");

    check(server_prompt_cache_disk_wins(2048, -1, small, 0), "the 2048 floor is inclusive");

    // a real head still wins, priced at the size-scaled gain
    check(server_prompt_cache_disk_wins(45286, -1, large, 8194), "a 45k head wins an empty RAM tier");
    const int large_gain = (int) (large / 750'000); // 4566
    check(!server_prompt_cache_disk_wins(large_gain, 0, large, 0), "gaining exactly the read cost does not win");
    check(server_prompt_cache_disk_wins(large_gain + 1, 0, large, 0), "gaining one token over the read cost wins");

    // against a resident RAM head the delta rule binds
    check(!server_prompt_cache_disk_wins(2048, 3000, 1024, 0), "the floor alone does not beat a 3000-token RAM head");
    check(server_prompt_cache_disk_wins(5100, 3000, 1024, 0), "beating the RAM head by the gain wins");
    check(!server_prompt_cache_disk_wins(-1, -1, 1024, 0), "no head at all does not win");

    // the floor is the first checkpoint the spread keeps, at the production batch of 2048.
    // a k*step estimate sits below that rung, so the fetch restores nothing.
    check(server_prompt_cache_first_rung(48000, step, 8) == 8194, "a 48k entry's spread drops the 4097 rung");
    check(server_prompt_cache_first_rung(57344, step, 8) == 8194, "a 56k entry's first kept rung is 8194");
    check(server_prompt_cache_first_rung(61440, step, 8) == 8194, "a 60k entry's first kept rung is 8194");
    check(server_prompt_cache_first_rung(2048, step, 8) == 2048, "a sub-step entry's first rung is the entry");
    check(server_prompt_cache_first_rung(101228, step, cap) == 8194, "a 98k entry's first kept rung is 8194");
    check(server_prompt_cache_first_rung(17400, step, cap) == 8194, "a 17k entry's first kept rung is 8194");
    check(server_prompt_cache_first_rung(16384, step, cap) == 8194, "a 16k entry's spread drops the 4097 rung");
    check(server_prompt_cache_first_rung(5000, step, cap) == 2952, "a 5k entry's first rung is the tail break");
    check(server_prompt_cache_first_rung(101228, step, 0) == 4097, "with the cap off the first cadence rung is kept");
    check(!server_prompt_cache_disk_wins(8192, -1, large, server_prompt_cache_first_rung(57344, step, 8)), "an LCP on the k*step floor does not win a 56k entry");
    check(server_prompt_cache_disk_wins(8194, -1, large, server_prompt_cache_first_rung(57344, step, 8)), "the kept rung of a 56k entry wins the read");


    // measured on the live tier: heads of 5834/7689/7778 tokens on 48-98k entries passed
    // the size-scaled floor and restored nothing -- the first rung is 8194, so the resume
    // landed on 0 and the whole entry was read for it
    check(!server_prompt_cache_disk_wins(5834, -1, large, 8194), "a 5834-token head does not win a spread entry's read");
    check(!server_prompt_cache_disk_wins(7689, -1, large, 8194), "a 7689-token head does not win a spread entry's read");
    check(!server_prompt_cache_disk_wins(7778, -1, large, 8194), "a 7778-token head does not win a spread entry's read");
    // ... and the head just past the first rung still does
    check(server_prompt_cache_disk_wins(8194, -1, large, 8194), "the first rung itself wins the read");
    // ... and a short entry lends a head its own first rung covers
    check(server_prompt_cache_disk_wins(5000, -1, small, 4096), "a 5k head wins a 5k entry's read");

    printf("PASS\n");
    return 0;
}

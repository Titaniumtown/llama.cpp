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

    // 80 MB/s against 667 tok/s, doubled: 80e6 / 1334 = 59970 bytes per token.
    // a 2.7 GB entry then costs 45022 tokens. the live loser median (13811) must not win;
    // the live winner median (65021) still does. the uncontended price above is unchanged.
    check(server_prompt_cache_bytes_per_token(80.0) == 59970, "80 MB/s prices at 59970 bytes per token");
    check(server_prompt_cache_bytes_per_token(0.0) == 750000, "no sample keeps the uncontended price");
    check(server_prompt_cache_bytes_per_token(2000.0) == 750000, "a fast sample does not loosen the price");
    const std::size_t live = 2'700'000'000ull;
    const int contended = server_prompt_cache_bytes_per_token(80.0);
    check(!server_prompt_cache_disk_wins(13811, -1, live, 8192, contended), "a 13811-token head does not win a slow 2.7 GB read");
    check(server_prompt_cache_disk_wins(65021, -1, live, 8192, contended), "a 65021-token head still wins a slow 2.7 GB read");
    // a quotient past INT_MAX used to wrap negative and collapse to the 2048 floor
    const std::size_t over = 3ull << 30;
    check(!server_prompt_cache_disk_wins(3000, -1, over, 0, 1), "a 3 GiB entry at 1 byte/token does not win on a 3000-token head");
    const double rates[] = { 30.0, 80.0, 1000.0 };
    check(server_prompt_cache_rate_median(rates, 3) == 80.0, "the median of three rates is the middle one");
    check(server_prompt_cache_disk_failed_read_unlinks(false, false), "a truncated read is dropped");
    check(server_prompt_cache_disk_failed_read_unlinks(false, true), "a checksum miss is dropped");
    check(!server_prompt_cache_disk_failed_read_unlinks(true, false), "a good read is kept");
    const std::size_t window = 128;
    const auto behind = server_prompt_cache_write_behind_of(160, 0, window);
    check(behind.off == 0 && behind.len == 32, "160 written into a 128 window drops the first 32");
    const auto held = server_prompt_cache_write_behind_of(100, 0, window);
    check(held.len == 0, "100 written into a 128 window drops nothing");
    const auto drain = server_prompt_cache_write_behind_of(160, 32, 0);
    check(drain.off == 32 && drain.len == 128, "a zero window drains whatever is still dirty");

    printf("PASS\n");
    return 0;
}

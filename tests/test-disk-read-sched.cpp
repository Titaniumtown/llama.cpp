// The off-loop disk read: one fread, joiners wait, a second path does not start a second
// fread, cancel of one waiter does not drop the other, and a finished buffer is taken once
// per waiter.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-disk-read-sched.cpp && /tmp/t
#include "server-prompt-cache-read.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

int main() {
    server_prompt_cache_read_sched sched;

    check(sched.reentry(1) == server_prompt_cache_read_reentry::absent, "an unknown task is absent");

    const auto first = sched.arm("/a.bin", 1);
    check(first.kind == server_prompt_cache_read_kind::pending && first.start, "the first path starts the read");
    check(sched.reentry(1) == server_prompt_cache_read_reentry::pending, "the starter is pending");

    const auto join = sched.arm("/a.bin", 2);
    check(join.kind == server_prompt_cache_read_kind::pending && !join.start, "a second task joins the same path");

    const auto other = sched.arm("/b.bin", 3);
    check(other.kind == server_prompt_cache_read_kind::pending && !other.start, "a second path waits behind the inflight read");

    const auto wake = sched.complete("/a.bin", true);
    check(wake.size() == 2 && wake[0] == 1 && wake[1] == 2, "completing the inflight read wakes both waiters");
    check(sched.reentry(1) == server_prompt_cache_read_reentry::ready, "a woken waiter is ready");

    check(sched.next_path() == "/b.bin", "the queued path starts once the inflight read finishes");
    check(sched.next_path().empty(), "a second next_path does not start another read");

    check(sched.cancel(3), "the queued waiter can be cancelled");
    const auto dropped = sched.complete("/b.bin", true);
    check(dropped.empty(), "a cancelled waiter is not woken");

    server_prompt_cache_read_sched shared;
    shared.arm("/c.bin", 7);
    shared.arm("/c.bin", 8);
    shared.complete("/c.bin", true);

    const auto t7 = shared.take(7);
    check(t7.ok && !t7.last && t7.path == "/c.bin", "the first take keeps the buffer for the other waiter");
    const auto t8 = shared.take(8);
    check(t8.ok && t8.last, "the last take releases the buffer");
    check(!shared.take(7).ok, "a taken waiter cannot take again");

    server_prompt_cache_read_sched bad;
    bad.arm("/d.bin", 9);
    const auto failed = bad.complete("/d.bin", false);
    check(failed.size() == 1 && failed[0] == 9, "a failed read still wakes its waiter");
    check(bad.reentry(9) == server_prompt_cache_read_reentry::failed, "the woken waiter sees the failure");
    check(bad.cancel(9), "a failed waiter is released");
    check(bad.reentry(9) == server_prompt_cache_read_reentry::absent, "a released waiter is gone");

    server_prompt_cache_read_sched alone;
    alone.arm("/e.bin", 4);
    check(alone.cancel(4), "the only inflight waiter can be cancelled");
    check(alone.complete("/e.bin", true).empty(), "a read with no waiters wakes nobody");
    check(alone.reentry(4) == server_prompt_cache_read_reentry::absent, "a cancelled inflight waiter is not re-posted");

    printf("PASS\n");
    return 0;
}

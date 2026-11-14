// What a prompt's evaluation time counts: the iterations that carried the slot's prompt
// tokens, and not the wall time the request sat in the slot. A hold, an injected
// decode-only step and another slot's cache fetch all pass wall time without evaluating
// anything, and a reported "prompt eval time" that includes them reads ~30x slow.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-slot-timer.cpp && /tmp/t
#include "server-slot-timer.h"

#include <cstdio>
#include <cstdlib>

static void check_near(double got, double want, const char * what) {
    if (got < want - 1e-9 || got > want + 1e-9) {
        fprintf(stderr, "FAIL: %s (got %.6f, want %.6f)\n", what, got, want);
        exit(1);
    }
}

int main() {
    // the production shape of the misreport: three 400 ms steps carry the prompt, and the
    // slot then waits 10 s to be scheduled again. eval is the 1.2 s of steps; the 10 s is
    // wait and must not be in it
    {
        slot_timer t;
        t.start(0);
        for (int i = 0; i < 3; ++i) {
            t.step_prompt_tokens(2048);
            t.step_done(400'000);
        }
        t.prompt_done(15'000'000);
        check_near(t.prompt_eval_ms(), 1200.0, "eval time counts only the steps");
        check_near(t.prompt_wall_ms(), 15000.0, "wall time keeps the wait");
    }

    // iterations that carried no prompt tokens are free: a decode-only step given to a
    // competing decoder, a step whose batch was filled by another slot, a held slot
    {
        slot_timer t;
        t.start(0);
        t.step_done(5'000'000);
        t.step_done(1'000'000);
        t.step_prompt_tokens(512);
        t.step_done(100'000);
        t.prompt_done(9'000'000);
        check_near(t.prompt_eval_ms(), 100.0, "steps without prompt tokens are not eval");
    }

    // the pending count is consumed by the iteration it was set in: one step's worth of
    // prompt tokens cannot be billed twice
    {
        slot_timer t;
        t.start(0);
        t.step_prompt_tokens(64);
        t.step_done(10'000);
        t.step_done(990'000);
        t.prompt_done(1'000'000);
        check_near(t.prompt_eval_ms(), 10.0, "prompt tokens are charged to one step only");
    }

    // the cache-hit prompt: the tokens are all in the slot already, but the server still
    // evaluates one token to get the logits, and that step is the eval time
    {
        slot_timer t;
        t.start(0);
        t.step_prompt_tokens(1);
        t.step_done(50'000);
        t.prompt_done(600'000);
        check_near(t.prompt_eval_ms(), 50.0, "a cache-hit prompt costs its logits step");
        check_near(t.prompt_wall_ms(), 600.0, "wall time of a cache-hit prompt");
    }

    // two slots in one iteration are independent: each is charged the step it was in
    {
        slot_timer a, b;
        a.start(0);
        b.start(0);
        a.step_prompt_tokens(1024);
        a.step_done(300'000);
        b.step_done(700'000); // b carried no prompt tokens in this iteration
        check_near(a.prompt_eval_ms(), 300.0, "the slot that carried tokens is charged");
        check_near(b.prompt_eval_ms(), 0.0, "the slot that did not is not");
    }

    // total time is wall: prompt phase plus generation, not the eval sum
    {
        slot_timer t;
        t.start(1'000'000);
        t.step_prompt_tokens(8);
        t.step_done(200'000);
        t.prompt_done(3'000'000);
        check_near(t.total_ms(0), 2000.0, "total time with only the prompt phase done");
        t.gen_done(9'000'000);
        check_near(t.total_ms(0), 8000.0, "total time spans the generation phase too");
        check_near(t.prompt_eval_ms(), 200.0, "eval time is unaffected by generation");
    }

    // nothing has happened yet: every duration is 0, and no division by it
    {
        slot_timer t;
        check_near(t.prompt_eval_ms(), 0.0, "eval time before the prompt phase");
        check_near(t.prompt_wall_ms(), 0.0, "wall time before the prompt phase");
        check_near(t.total_ms(5'000), 0.0, "total time before the prompt phase");
    }

    printf("ok: test-slot-timer\n");
    return 0;
}

// The injection gate of the adaptive step scheduler: how many decode-only steps it owes
// the decoders while a prompt is pending.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-step-sched-inject.cpp && /tmp/t
#include "server-step-sched-inject.h"

#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
}

int main() {
    using namespace step_sched;

    // the threshold is the configured target, whatever the decoder count: a batched decode
    // advances every decoder on every step, so the per-decoder ITL is the decode step
    // time and the count is already priced into it
    check(inject_threshold_ms(400, 1.0, false) == 400, "the threshold is the target");
    check(inject_threshold_ms(400, 3.12, false) == 400, "the target does not move with the emission");
    check(inject_threshold_ms(1, 1.0, false) == 1, "threshold at a 1 ms target");
    check(inject_threshold_ms(0, 3.12, false) == 0, "no target, no threshold");

    // per token: a step that emits k tokens may take k times the target, because the
    // client gets all k at once and its wait between two tokens is the gap over k.
    // production: a mean accepted length of 2.47-3.89 tokens per step over 66 tasks
    check(inject_threshold_ms(400, 3.5, true) == 1400, "a per-token target is k times the target");
    check(inject_threshold_ms(400, 0.0, true) == 400, "an unmeasured emission falls back to the target");
    check(inject_threshold_ms(400, 2.5, true) == 1000, "the unlucky slot's threshold");
    check(inject_threshold_ms(400, 4.0, true) == 1600, "the lucky slot's threshold");

    // the gate is a mean-gap controller: closed at or below the target, open above it
    check(!inject_gate_open(400.0, 400, 3.12, false), "gate open at the target");
    check(!inject_gate_open(375.0, 400, 3.12, false), "gate open below the target");
    check(!inject_gate_open(0.0, 400, 3.12, false), "gate open with no gap");
    check(inject_gate_open(448.0, 400, 3.12, false), "gate closed above the target");
    check(inject_gate_open(1033.0, 400, 3.12, false), "gate closed at a whole-chunk gap");
    check(!inject_gate_open(1033.0, 0, 3.12, false), "gate open with the target off");

    // and the per-token gate is the same controller with the target scaled: a gap the
    // per-step gate would refuse is inside the per-token budget, and one that is not
    check(!inject_gate_open(1200.0, 400, 3.12, true), "a whole-chunk gap is inside the per-token budget");
    check(inject_gate_open(1300.0, 400, 3.12, true), "the per-token gate closes above k times the target");
    check(inject_gate_open(1000.0, 400, 2.47, true), "the unlucky slot's gate closes sooner");

    // the settle point: the smallest k (decode steps per prompt step) whose cycle mean
    // gap is at or below the target. production shape: prompt step 1033 ms, decode step
    // 156 ms (a 10-decoder batch)
    auto settle_k = [](int t_prompt, int t_dec, int itl) {
        for (int k = 0;; ++k) {
            const double mean = (t_prompt + k * t_dec) / (double) (k + 1);
            if (!inject_gate_open(mean, itl, 3.12, false)) {
                return k;
            }
        }
    };
    check(settle_k(1033, 156, 400) == 3, "settle k at the 10-decoder shape"); // k=3: 375 ms, k=2: 448
    check(settle_k(1033, 59, 400) == 2, "settle k at the 1-decoder shape");  // k=2: 384 ms, k=1: 546

    // the settle k is the minimum: one fewer injected step puts the mean gap over the
    // target, so the policy injects exactly what the target requires and no more
    const int k = settle_k(1033, 156, 400);
    check(inject_gate_open((1033 + (k - 1) * 156) / (double) k, 400, 3.12, false), "settle k is not minimal");

    // the gate defends the gap at the depth of the decoders it is looking at, not the
    // run-wide mean: production read 383 ms run-wide while the 64k+ arm read 630 ms over
    // 7096 samples, and the gate closed. the arm in front of the gate wins
    check(gate_gap_ms(630.0, 7096, 383.0, 8) == 630.0, "the depth arm is used once measured");
    check(inject_gate_open(gate_gap_ms(630.0, 7096, 383.0, 8), 400, 3.12, false), "deep decoders keep the gate open");
    check(!inject_gate_open(gate_gap_ms(383.0, 7096, 383.0, 8), 400, 3.12, false), "an on-target depth arm closes the gate");
    check(gate_gap_ms(630.0, 7, 383.0, 8) == 383.0, "an arm short of samples falls back to the mean");
    check(gate_gap_ms(0.0, 7096, 383.0, 8) == 383.0, "an unmeasured arm falls back to the mean");

    // the pacer: the debt is the integral of the excess over the target, without the
    // EWMA's lag. a prompt step of 535 ms against a 400 ms target leaves 135 ms of debt,
    // and the injected step that pays it back (198 ms, 202 ms under the target) clears it
    check(pacer_debt(0.0, 535.0, 400.0) == 135.0, "a long step runs up debt");
    check(pacer_debt(135.0, 198.0, 400.0) == -67.0, "an injected step pays the debt back");
    check(pacer_debt(135.0, 198.0, 400.0) < 0.0, "the gate closes once the debt is paid");
    // the debt is capped at one target: a long stall must not buy a train of injections
    check(pacer_debt(400.0, 10000.0, 400.0) == 400.0, "the debt is capped at the target");
    check(pacer_debt(-400.0, 0.0, 400.0) == -400.0, "the credit is capped too");
    check(pacer_debt(0.0, 535.0, 0.0) == 0.0, "no target, no debt");

    printf("ok: test-step-sched-inject\n");
    return 0;
}

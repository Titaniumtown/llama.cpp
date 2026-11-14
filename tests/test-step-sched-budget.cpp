// The prompt chunk cap of the adaptive step scheduler: how many prompt tokens an iteration
// may carry while a decoder rides in the batch.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-step-sched-budget.cpp && /tmp/t
#include "server-step-sched-budget.h"

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

    // the cap is refused while the capped shape measures slower per prompt token than the whole
    // chunk it displaces -- the production census reads 571 ms / 220 tok against 2357 ms / 1868
    // tok, and the quota alone (400 - 88.5 ms over 1.14 ms/tok = 273 tok, break-even 211) would
    // have taken it. the fitted break-even cannot see this: the small shape is made of the fills
    // the cap produced, so it ratchets down as capping grows.
    check(!cap_pays_measured(571.0 / 220.0, 2357.0 / 1868.0, CAP_PAYS_TOLERANCE), "the production pair refuses the cap");
    check(!cap_pays_measured(2.60, 1.26, CAP_PAYS_TOLERANCE), "twice the whole chunk's rate is refused");
    check(cap_pays_measured(1.30, 1.26, CAP_PAYS_TOLERANCE), "a near-tie keeps the cap");
    check(cap_pays_measured(1.26 * 1.10, 1.26, CAP_PAYS_TOLERANCE), "exactly at the tolerance keeps the cap");
    check(!cap_pays_measured(1.26 * 1.11, 1.26, CAP_PAYS_TOLERANCE), "past the tolerance refuses the cap");
    check(!cap_pays_measured(0.0, 1.26, CAP_PAYS_TOLERANCE), "an unmeasured capped shape refuses");
    check(!cap_pays_measured(1.26, 0.0, CAP_PAYS_TOLERANCE), "an unmeasured whole chunk refuses");
    check(prompt_chunk_cap_min(400, 5 * 17.7, 1.14, 211) > 0, "the quota alone would have capped");

    // production shape: 400 ms target, 5 decoder tokens at 17.7 ms each (a 64k+ decode-only
    // step is 88.4 ms carrying 5 tokens), and a prompt step costing 1.26 ms per token
    // (2592 ms for 2048 tokens). the budget is what is left of the target after the decoders
    const int cap = prompt_chunk_cap(400, 5 * 17.7, 1.26);
    check(cap >= 240 && cap <= 250, "cap is the prompt's share of the target");
    check(cap * 1.26 + 5 * 17.7 <= 400.0, "a step at the cap fits the target");

    // the cap must not depend on what the batch carried last iteration: that is the whole
    // point. a decode-only previous iteration is priced by gen cost, not by the prompt rate
    check(prompt_chunk_cap(400, 5 * 17.7, 1.26) == prompt_chunk_cap(400, 5 * 17.7, 1.26),
          "the cap is a function of the rate it is given");

    // priced at the decode rate instead (what a blended "previous composition" cost does to
    // the prompt tokens) the same target yields no usable cap: this is the state that left
    // the production box on full 2.6 s prompt steps
    check(prompt_chunk_cap(400, 5 * 17.7, 17.7) == 0, "decode-rate pricing disables the cap");

    // below the minimum there is no cap: the prompt takes one batch and the average
    // re-measures at the true marginal rate
    check(prompt_chunk_cap(440, 400.0, 1.26) == 0, "31 tokens is below the minimum");
    check(prompt_chunk_cap(441, 400.0, 1.26) == 32, "32 tokens is the minimum");

    // the decoders alone can exhaust the target: then there is nothing left to cap and the
    // injection gate is the only lever that is left
    check(prompt_chunk_cap(400, 400.0, 1.26) == 0, "the decoders alone spend the whole target");
    check(prompt_chunk_cap(400, 500.0, 1.26) == 0, "decoders over the target");

    // off, and unmeasured: no cap
    check(prompt_chunk_cap(0, 88.0, 1.26) == 0, "target off");
    check(prompt_chunk_cap(400, 88.0, 0.0) == 0, "rate unmeasured");

    // a cheaper prompt rate buys a longer chunk in proportion
    check(prompt_chunk_cap(400, 88.0, 0.63) >= 480, "half the per-token cost, double the chunk");

    // the rate the cap divides by is the prompt's own marginal cost per token. The whole-step
    // average prices the decoders' time in the step, and the step's own fixed cost, as prompt
    // tokens: measured on the production box it reads ~9 ms/token on the fills the cap
    // produces (27 tokens a step once the riding decoders have taken theirs), while the prompt
    // arm's solo rate -- the same prompt with no decoder in the batch, 2088 ms / 2027 tokens
    // at 32-64k -- reads 1.03. The arm is the rate the budget has to see.
    check(cap_rate_ms(1.03, 76, 9.0, 200, 8) == 1.03, "the prompt arm's own rate wins");
    check(cap_rate_ms(1.03, 4, 9.0, 200, 8) == 9.0, "an arm short of samples falls back");
    check(cap_rate_ms(0.0, 76, 9.0, 200, 8) == 9.0, "an unmeasured arm falls back");
    check(cap_rate_ms(1.03, 76, 0.0, 0, 8) == 1.03, "the arm alone is enough");
    check(cap_rate_ms(0.0, 0, 0.0, 0, 8) == -1.0, "nothing measured at all");

    // and at the production shape the divisor is the whole difference: the average leaves a
    // 34-token fill, the arm leaves 302, and both stay inside the target
    const int    polluted_cap = prompt_chunk_cap(400, 5 * 17.7, 9.0);
    const double arm_rate     = cap_rate_ms(2088.0 / 2027.0, 76, 9.0, 200, 8);
    const int    arm_cap      = prompt_chunk_cap(400, 5 * 17.7, arm_rate);
    check(polluted_cap <= 40, "the whole-step average leaves a minimum-sized fill");
    check(arm_cap >= 280, "the prompt's own rate keeps a usable chunk");
    check(arm_cap * arm_rate + 5 * 17.7 <= 400.0, "a step at the arm's cap fits the target");

    // a capped chunk has to amortize a step's fixed cost, or the whole chunk moves the same
    // tokens faster. Production measured the two shapes the census counts: 318.6 ms / 47 prompt
    // tokens beside 1.8 decoder tokens, and 1775.8 ms / 1895 prompt tokens beside 5.0, which is
    // fixed 250 ms and 0.76 ms per prompt token. The cap stands down below the break-even chunk
    const chunk_fit fit = fit_chunk_fixed(318.6, 47.0, 1.8, 1775.8, 1895.0, 5.0, 17.7, 11, 6);
    check(fit.valid, "the two shapes give a fixed/marginal fit");
    check(fit.fixed_ms > 230.0 && fit.fixed_ms < 275.0, "the fixed part is the ~250 ms measured");
    check(fit.ms_per_token > 0.70 && fit.ms_per_token < 0.82, "the marginal rate is the ~0.76 measured");
    check(chunk_break_even(fit.fixed_ms, fit.ms_per_token) >= 300, "break-even is a few hundred tokens");

    const int be = chunk_break_even(250.0, 0.76);
    check(prompt_chunk_cap_min(400, 5 * 17.7, 13.23, be) == 0, "a 23-token quota stands down");
    check(prompt_chunk_cap_min(400, 5 * 17.7, 1.03, be) == 0, "a 302-token quota stands down");
    const int big = prompt_chunk_cap_min(400, 5 * 17.7, 0.45, be);
    check(big >= 690, "a quota past the break-even is kept");
    check(big * 0.45 + 5 * 17.7 <= 400.0, "and a step at it fits the target");

    // guards: an unusable fit leaves the cap on its old floor, an unmeasured one leaves the
    // fill alone, and a shape short of samples is not a fit
    check(!fit_chunk_fixed(100.0, 10.0, 1.0, 120.0, 10.0, 1.0, 17.7, 11, 6).valid, "a flat pair is no fit");
    check(!fit_chunk_fixed(300.0, 47.0, 2.0, 320.0, 60.0, 2.0, 17.7, 11, 6).valid, "a 13-token spread carries no slope");
    check(!fit_chunk_fixed(300.0, 47.0, 2.0, 1700.0, 1900.0, 5.0, 17.7, 3, 6).valid, "a shape short of steps is no fit");
    check(!fit_chunk_fixed(300.0, 47.0, 2.0, 1700.0, 1900.0, 5.0, 17.7, 11, 0).valid, "no whole-chunk steps, no fit");
    check(chunk_break_even(0.0, 0.76) == 0, "no fixed part, no break-even");
    check(prompt_chunk_cap_min(400, 88.0, 9.0, 0) == prompt_chunk_cap(400, 88.0, 9.0), "no break-even keeps the old floor");
    check(prompt_chunk_cap_min(0, 88.0, 1.0, be) == 0, "target off, no cap");
    check(prompt_chunk_cap_min(400, 88.0, 0.0, be) == 0, "rate unmeasured, no cap");

    // the ITL target is a per-token budget: a step that emits k tokens may take k times it.
    // production shape: 5 decoder tokens at 17.7 ms, a prompt at 0.87 ms/token
    check(itl_target_ms(400, 3.5, false) == 400.0, "the shipped reading is the step's gap");
    check(itl_target_ms(400, 3.5, true) == 1400.0, "a per-token target is k times the target");
    check(itl_target_ms(400, 0.0, true) == 400.0, "an unmeasured emission falls back to the target");
    check(itl_target_ms(0, 3.5, true) == 0.0, "no target, no budget");
    check(prompt_chunk_cap_min(itl_target_ms(400, 3.5, false), 5 * 17.7, 0.87, 0) == 358, "the per-step quota");
    check(prompt_chunk_cap_min(itl_target_ms(400, 3.5, true), 5 * 17.7, 0.87, 0) == 1507, "the per-token quota");
    check(prompt_chunk_cap_min(itl_target_ms(400, 2.5, true), 5 * 17.7, 0.87, 0) == 1047, "the unlucky slot's quota");

    // the admission floor is the same budget seen from the other side: the decoders in the
    // batch may spend up to the floor's share of the target before a newcomer has to wait.
    // production: a decoder-bearing step costs 5 * 17.7 = 88.5 ms against a 400 ms target,
    // 22% of it, so a newcomer rides -- and the budget keeps its chunk inside the target
    check(!admit_blocked_by_decoders(400, 5 * 17.7, 50), "22% of the target does not hold a prompt");
    check(!admit_blocked_by_decoders(400, 200.0, 50), "exactly at the floor does not hold");
    check(admit_blocked_by_decoders(400, 210.0, 50), "over the floor holds the prompt");
    check(admit_blocked_by_decoders(400, 300.0, 50), "75% of the target holds the prompt");
    check(!admit_blocked_by_decoders(400, 300.0, 80), "the floor is what it is set to");
    check(!admit_blocked_by_decoders(0, 88.0, 50), "no target, no floor");
    check(!admit_blocked_by_decoders(400, 88.0, 0), "floor off");

    // what an incumbent still owes: the declared count when there is one, the measured turn
    // length when there is not (every turn on this box: no client cap, no --n-predict)
    check(remaining_tokens(120, 300, 700.0) == 120, "a declared remainder is read as declared");
    check(remaining_tokens(-1, 3, 10.0) == 7, "a limitless turn owes what is left of its length");
    check(remaining_tokens(-1, 10, 10.0) == 0, "a limitless turn at its expected length owes nothing");
    check(remaining_tokens(-1, 400, 10.0) == 0, "a limitless turn past its expected length owes nothing");
    check(remaining_tokens(-1, 0, 0.0) == 0, "an unmeasured turn owes nothing");
    check(remaining_tokens(-1, 3, 0.0) == 0, "an unmeasured turn owes nothing even mid-turn");

    // the hold's rule: the decoders' pending tokens stay out of the batch for one step. it
    // costs the prompt no throughput and the decoder one step of latency, so the bound is on
    // the client's wait: the gap plus the make-up step must stay inside the target.
    // production: the make-up step is the decoder's own step, 122.6 ms at 64k+ depth
    check(hold_slack_open(0.0, 122.6, 400.0, 0.0), "a fresh gap has room for the make-up step");
    check(hold_slack_open(200.0, 122.6, 400.0, 0.0), "a gap with room holds");
    check(hold_slack_open(277.4, 122.6, 400.0, 0.0), "the boundary is the step exactly");
    check(!hold_slack_open(300.0, 122.6, 400.0, 0.0), "a gap that cannot absorb the step does not hold");
    check(hold_slack_open(270.0, 122.6, 400.0, 0.0), "the gap fits without the margin");
    check(!hold_slack_open(270.0, 122.6, 400.0, 0.10), "the margin takes the room back");
    check(!hold_slack_open(-1.0, 122.6, 400.0, 0.0), "an unmeasured gap does not hold");
    check(!hold_slack_open(0.0, 0.0, 400.0, 0.0), "no make-up cost, no hold");
    check(!hold_slack_open(0.0, 122.6, 0.0, 0.0), "no target, no hold");
    // the target scales with the emission, so a decoder that emitted k tokens at once has
    // k times the slack of one that emitted one: that is the luck-aware throttle
    check(hold_slack_open(200.0, 122.6, itl_target_ms(118, 4.77, true), 0.0), "a lucky emission leaves the room");
    check(!hold_slack_open(200.0, 122.6, itl_target_ms(118, 2.22, true), 0.0), "an unlucky emission does not");

    printf("ok: test-step-sched-budget\n");
    return 0;
}

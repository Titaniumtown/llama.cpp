#pragma once

// How many prompt tokens one server iteration may carry while a decoder shares the step.
//
// The budget is what is left of the ITL target once the decoder that rides in the batch is
// paid for, divided by the prompt's own marginal cost per token:
//
//     n = (itl_ms - gen_tokens * dec_ms_per_token) / ms_per_prompt_token
//
// The decoders are paid for by the first term, so pricing the prompt's tokens at a blend of
// the prompt rate and the decode rate charges them twice -- and it also disables the cap in
// exactly the regime the cap exists for. After a decode-only iteration (which is what an
// injected step is, and what the iteration before a prompt step usually is) the "previous
// composition" is all decoders, so a blended cost collapses to the decode rate (~18 ms/token
// on this box), n lands under the minimum, the cap is dropped, and the iteration that should
// have been a ~300 ms prompt step is a full 2048-token, 2.6 s one. That gap is 6x the target,
// which opens the injection gate for the next iteration, which makes the iteration after it
// decode-only again: the two mechanisms hold each other in the full-chunk regime.
//
// Measured on the production box in that regime: 692 prompt steps that carried a decoder
// averaged 1983 prompt tokens (a full chunk), and the decoder's own gap arm read 630 ms
// against the 400 ms target. The cap has to see the prompt's marginal rate.
//
// The minimum exists so the prompt does not pay a step's fixed cost in absurdly small
// batches; below it the average re-measures at the batch's true marginal rate.

#include <algorithm>

namespace step_sched {

static constexpr int BUDGET_MIN_CHUNK = 32;

// May the admission guard hold a prompt that has not started, given what the decoders in the
// batch already cost?
//
// It is the same budget seen from the other side. While the decoders' own step cost is under
// the floor's share of the target, the budget can still fit a minimum chunk, so the prompt
// rides and the decoder's gap still meets the target. Over it, every iteration the prompt
// shares is spent over the target and the only lever left is to let the decoders run alone.
//
// This replaces a sampled ratio (t_dec_only / gap_wp, per depth bucket): it was a lagged
// sample of whatever had stalled the decoder before the arrival rather than of the arrival's
// own cost, and needing MIN_SAMPLES per bucket made it inert for the first arrivals after
// every restart -- 4 restarts in 5.3 h on the production box.

// How many tokens an incumbent still owes.
//
// A request that declared a length has the count; a limitless one (no client cap and no
// --n-predict, which is every turn on this box: 897 of 897 released turns carried
// truncated = 0) has none, and pricing it with a sentinel made every remainder look like
// hours -- the guard's "release the held prompt early when the incumbent's remainder is
// cheaper than the newcomer's own prefill" rule could therefore never fire, and a held
// prompt waited out the whole turn. A measured turn length prices it instead: the
// incumbent is past its expected depth or it is not.
static inline int64_t remaining_tokens(int32_t n_remaining, int64_t generated_so_far,
                                       double expected_turn_tokens) {
    if (n_remaining > 0) {
        return n_remaining;
    }
    if (expected_turn_tokens <= 0.0) {
        return 0; // unmeasured: the caller decides whether it may act on that
    }
    const double left = expected_turn_tokens - (double) generated_so_far;
    return left > 0.0 ? (int64_t) left : 0;
}

static inline bool admit_blocked_by_decoders(double itl_ms, double gen_ms_cost, int dec_share_pct) {
    if (itl_ms <= 0 || dec_share_pct <= 0) {
        return false;
    }
    return gen_ms_cost * 100.0 > (double) dec_share_pct * (double) itl_ms;
}

// Which rate the chunk cap divides by: the prompt's own marginal cost per token.
//
// The divisor was ms_per_prompt_tok, the whole step over every token in it, which on a step
// that rode decoders prices their time and the step's fixed cost as prompt tokens: measured
// on the production box it reads ~9 ms/token where the prompt arm's solo step -- the same
// prompt with no decoder in the batch, 2088 ms / 2027 tokens at 32-64k -- reads 1.03. That
// divisor is the 27-token fill the box runs. The arm is per depth, like the rest of the
// budget. A negative return means nothing is measured: the caller leaves the fill uncapped.
static inline double cap_rate_ms(double solo_ms_per_token, int solo_samples,
                                 double fallback_ms_per_token, int fallback_samples,
                                 int min_samples) {
    if (solo_samples >= min_samples && solo_ms_per_token > 0.0) {
        return solo_ms_per_token;
    }
    if (fallback_samples >= min_samples && fallback_ms_per_token > 0.0) {
        return fallback_ms_per_token;
    }
    return -1.0;
}

// The two terms of a prompt-bearing step, from the two shapes the census counts.
struct chunk_fit {
    double fixed_ms     = 0.0; // step time that does not scale with the prompt tokens
    double ms_per_token = 0.0; // the prompt's marginal cost per token
    bool   valid        = false;
};

// how many steps each shape needs before its mean is worth fitting -- the slope is carried by
// the token spread between the shapes, not by the sample count, so a handful is enough -- and
// how far apart their token counts have to be for the slope to mean anything. The production
// census prints both means in one line; they sat 47 and 1895 tokens apart.
static constexpr int    FIT_MIN_STEPS         = 4;
static constexpr double FIT_MIN_TOKEN_SPREAD  = 128.0;

// t = fixed + marginal * prompt_tokens, fitted from a small-chunk and a whole-chunk shape. The
// decoders' share is subtracted first -- each shape reports its own decoder tokens -- so both
// terms are the prompt's. A flat pair, a token spread too small to carry a slope, a shape short
// of steps or a negative fixed part is not a fit, and the caller keeps the divisor it had.
static inline chunk_fit fit_chunk_fixed(double t_small, double tok_small, double gen_small,
                                        double t_whole, double tok_whole, double gen_whole,
                                        double dec_ms_token, int n_small, int n_whole) {
    chunk_fit f;
    if (n_small < FIT_MIN_STEPS || n_whole < FIT_MIN_STEPS) {
        return f;
    }
    const double a  = t_small - gen_small * dec_ms_token;
    const double b  = t_whole - gen_whole * dec_ms_token;
    const double dn = tok_whole - tok_small;
    const double dt = b - a;
    if (dn < FIT_MIN_TOKEN_SPREAD || dt <= 0.0) {
        return f;
    }
    const double c     = dt / dn;
    const double fixed = a - c * tok_small;
    if (c <= 0.0 || fixed < 0.0) {
        return f;
    }
    f.fixed_ms     = fixed;
    f.ms_per_token = c;
    f.valid        = true;
    return f;
}

// The chunk at which a step's fixed cost stops dominating: there the capped step runs at half
// the prompt's marginal rate, and under it a capped step is the slower way to move prompt
// tokens. Production measured fixed 250 ms and 0.76 ms/token: the 302-token quota runs at 522
// prompt tok/s and the 81-token one that fits the target at 195, against 1067 for the
// 1895-token fill the cap leaves when it stands down.
static inline int chunk_break_even(double fixed_ms, double ms_per_token) {
    if (fixed_ms <= 0.0 || ms_per_token <= 0.0) {
        return 0;
    }
    return (int) (fixed_ms / ms_per_token);
}

// Whether the capped shape is measured no slower per prompt token than the whole chunk it
// displaces. Both means come from the same census, so a workload shift moves them together and
// cannot flip this on its own. The fitted break-even can: the small shape is made of the fills
// the cap itself produced, so more capping raises that mean, lowers the break-even and admits
// more capping. Refuse the cap while it is the slower way to move prompt tokens.
// how much slower per prompt token the capped shape may measure before the cap stands down:
// the two census means are medians over different shapes, so a little slack keeps a near-tie
// from flickering
static constexpr double CAP_PAYS_TOLERANCE = 1.10;

static inline bool cap_pays_measured(double part_ms_per_token, double whole_ms_per_token, double tolerance) {
    if (part_ms_per_token <= 0.0 || whole_ms_per_token <= 0.0 || tolerance <= 0.0) {
        return false;
    }
    return part_ms_per_token <= whole_ms_per_token * tolerance;
}

// The ITL target is a per-token budget, and a step that emits k tokens may take k times it.
//
// The client's inter-token latency is not the step's gap: a decoder that emits k tokens in
// one step delivers them together, so the wait between two of its tokens is the gap divided
// by k. Under MTP k is 2.5-3.9 tokens per step on the production box (66 tasks, mean
// accepted length 2.47-3.89), so charging the target against the gap alone makes the
// scheduler k times stricter than the target says: a decoder on a 400 ms/token target with
// k = 3.12 is owed a 1248 ms gap, not 400.
//
// per_token = 0 keeps the shipped reading, where the target is the step's gap.
static inline double itl_target_ms(int itl_ms, double k_emitted, bool per_token) {
    if (itl_ms <= 0) {
        return 0.0;
    }
    if (!per_token || k_emitted <= 0.0) {
        return (double) itl_ms;
    }
    return (double) itl_ms * k_emitted;
}

// The hold's rule: the decoders' pending tokens stay out of the batch for one step.
//
// A hold costs the prompt nothing in throughput -- the make-up step pays the attention the
// prompt's step gave up, so the GPU does the same work either way -- and it costs the
// decoder one step of latency. The only thing at risk is the client's wait, so the rule is
// a bound on that: the gap the client is sitting in plus the make-up step must stay inside
// the target.
//
// The target is per token and scales with the emission, so under MTP a decoder that emitted
// k tokens at once has k times the slack of one that emitted one. That is the whole of the
// luck-aware throttle: a decoder running ahead can be held back for a step without the
// client noticing, because it has tokens in hand.
static inline bool hold_slack_open(double gap_ms, double make_up_ms, double target_ms, double margin) {
    if (gap_ms < 0.0 || make_up_ms <= 0.0 || target_ms <= 0.0) {
        return false;
    }
    return gap_ms + make_up_ms * (1.0 + margin) <= target_ms;
}

// prompt tokens this iteration may add; 0 means "do not cap". `min_chunk` is the floor the
// quota has to clear to be worth taking: at 0 the cap keeps BUDGET_MIN_CHUNK.
static inline int prompt_chunk_cap_min(double itl_ms, double gen_ms_cost, double ms_per_prompt_token, int min_chunk) {
    if (itl_ms <= 0 || ms_per_prompt_token <= 0.0) {
        return 0;
    }
    const double budget = (double) itl_ms - gen_ms_cost;
    if (budget <= 0.0) {
        return 0;
    }
    const int n = (int) (budget / ms_per_prompt_token);
    return n >= std::max(min_chunk, BUDGET_MIN_CHUNK) ? n : 0;
}

static inline int prompt_chunk_cap(double itl_ms, double gen_ms_cost, double ms_per_prompt_token) {
    return prompt_chunk_cap_min(itl_ms, gen_ms_cost, ms_per_prompt_token, 0);
}

} // namespace step_sched

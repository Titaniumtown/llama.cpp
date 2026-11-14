// the injection gate of the adaptive step scheduler: is a decode-only step owed to the
// decoders while a prompt is pending?
//
// the gate compares the mean decoder gap against the configured ITL target as-is: the
// per-decoder ITL of a batched decode equals the decode step time (every decoder
// advances on every step), so the target is invariant to the decoder count -- the
// count is already priced into the measured step time (5 decoders: 95 ms, 10: 156 ms
// against a 400 ms target), and scaling the target down with the count would
// double-count it and put the threshold under the decode step time, where the gate
// can no longer close
//
// the target is a per-TOKEN budget, though, and the gap is per STEP. A decoder that emits
// k tokens in a step delivers them together, so the wait between two of its tokens is the
// gap divided by k. Under MTP k is 2.5-3.9 tokens per step on the production box (66
// tasks, mean accepted length 2.47-3.89), and charging the target against the gap alone
// makes the gate k times stricter than the target says: a decoder on a 400 ms/token target
// with k = 3.12 is owed a 1248 ms gap, not 400.
//
// per_token = 0 keeps the shipped reading, where the target is the step's gap.
#pragma once

#include "server-step-sched-budget.h"

namespace step_sched {

inline double inject_threshold_ms(int itl_ms, double k_emitted, bool per_token) {
    return itl_target_ms(itl_ms, k_emitted, per_token);
}

inline bool inject_gate_open(double mean_gap_ms, int itl_ms, double k_emitted, bool per_token) {
    return itl_ms > 0 && mean_gap_ms > inject_threshold_ms(itl_ms, k_emitted, per_token);
}

// The injection pacer: the debt the decoders have run up against the target.
//
// The gate above reads the MEAN gap, and a mean-gap controller injects in trains: the EWMA
// lags a long prompt step, so the injections arrive as a burst after the mean has crossed
// the target rather than spread over the steps that earned them (production: the client's
// rate swung 6.9-18.4 t/s around a 12.9 mean). The debt is the same integral without the
// lag: every step adds its excess over the target, every injected step pays its own
// shortfall back, and the gate opens while there is debt to pay.
//
// The debt is capped at one target in each direction: a long stall must not buy a train of
// injections, and a quiet spell must not carry a credit that suppresses the next one.
static inline double pacer_debt(double debt, double step_ms, double target_ms) {
    if (target_ms <= 0.0) {
        return 0.0;
    }
    double d = debt + (step_ms - target_ms);
    if (d > target_ms) {
        d = target_ms;
    } else if (d < -target_ms) {
        d = -target_ms;
    }
    return d;
}

// which gap the gate defends: the one the decoders in front of it actually see.
//
// the run-wide mean mixes every depth bucket, and a step's cost is a function of depth, so it
// can read on target while the bucket holding the decoders that are actually batched reads
// well over it. one production dump: decode_gap = 383 ms (the run-wide mean, at the 400 ms
// target) next to 64k+ gap_wp = 630 ms over 7096 samples -- so the gate was closing on
// precisely the deep decoders whose prompt steps cost 2.6 s and who needed it most, and the
// injections they did get came from the arrival rule, not from the gap. use the arm for the
// depth in front of the gate once it has samples; before that the run-wide mean is the only
// measurement there is.
inline double gate_gap_ms(double gap_at_depth_ms, int n_at_depth, double gap_all_ms, int min_samples) {
    return (n_at_depth >= min_samples && gap_at_depth_ms > 0.0) ? gap_at_depth_ms : gap_all_ms;
}

} // namespace step_sched

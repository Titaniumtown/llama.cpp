#pragma once

// What a prompt's evaluation time is, and what it is not.
//
// The slot used to report (t_prompt_last - t_start): the wall time between entering the
// prompt phase and the last prompt token. On this box most of that wall time is not
// evaluation. The slot waits behind an admission hold, its fill is skipped for an iteration
// so a decoder can take a step of its own, and the whole server stalls while another slot's
// cache entry is read from disk -- and all of it was billed as "prompt eval time".
//
// Measured in production, one request: 69.36 s reported for 1541 tokens (45.01 ms per token,
// 22.22 tokens per second) where the prompt-step measurements in the same window put a
// prompt of that size at ~1-1.3 ms per token, i.e. ~97% of the report was time the slot
// evaluated nothing. The same number is what the client sees in the progress line, so a
// scheduling stall is reported to clients as prompt processing being slow.
//
// So count the steps, not the clock. A server iteration is charged to a slot only if that
// iteration actually carried the slot's prompt tokens; holds, injected decode-only steps and
// cache fetches contribute nothing. The difference between the wall time and the eval time
// is then exactly the wait, and both are available to report.

#include <cstdint>

struct slot_timer {
    // absolute timestamps, in us; 0 = not reached yet
    int64_t t_start       = 0; // the slot entered the prompt phase
    int64_t t_prompt_last = 0; // the last prompt token landed
    int64_t t_gen_last    = 0; // the last generated token landed

    // summed durations of the iterations that carried this slot's prompt tokens
    int64_t t_prompt_us = 0;

    // prompt tokens this slot put into the batch being built; consumed by step_done()
    uint64_t n_step_prompt_tokens = 0;

    void start(int64_t now_us)       { t_start       = now_us; }
    void prompt_done(int64_t now_us) { t_prompt_last = now_us; }
    void gen_done(int64_t now_us)    { t_gen_last    = now_us; }

    // the batch builder reports what it added for this slot in the current iteration
    void step_prompt_tokens(uint64_t n) { n_step_prompt_tokens += n; }

    // charge one server iteration to this slot if it carried the slot's prompt tokens, then
    // clear the pending count so a later iteration cannot charge for the same tokens again
    void step_done(int64_t step_us) {
        if (n_step_prompt_tokens > 0 && step_us > 0) {
            t_prompt_us += step_us;
        }
        n_step_prompt_tokens = 0;
    }

    // the prompt's evaluation time
    double prompt_eval_ms() const {
        return t_prompt_us / 1000.0;
    }

    // the prompt phase's wall time: the evaluation plus everything the slot waited through
    double prompt_wall_ms() const {
        return t_prompt_last > 0 ? (double) (t_prompt_last - t_start) / 1000.0 : 0.0;
    }

    // the request's wall time: prompt phase, or prompt plus generation once that has started
    double total_ms(int64_t now_us) const {
        const int64_t end = t_gen_last > 0 ? t_gen_last : (t_prompt_last > 0 ? t_prompt_last : now_us);
        return t_start > 0 ? (double) (end - t_start) / 1000.0 : 0.0;
    }
};

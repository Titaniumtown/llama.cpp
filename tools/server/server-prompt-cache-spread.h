#pragma once

// Which checkpoints a stored entry keeps, kept free of the server so it can be tested on its
// own. The selection is by fixed token position, not list index: an index spread follows the
// list's own skew (user starts pile up in the tail) and leaves the head, where agentic
// sessions of one agent diverge (measured shared-head cluster 21-24k, secondary 18k), with
// one checkpoint. The first targets sit at and just behind that cluster. Creation stops being
// dense past six min steps and falls to four-step spacing, so the remaining targets follow
// that cadence: a head anywhere in a long entry lands within four min steps of a rung instead
// of paying the whole 20k-to-final gap (measured: 20 of 27 lost-tail continuations landed at
// 20481, losses 20-72k). The final checkpoint catches a continuation (lcp = full entry) with
// zero re-processing.
#include <cstddef>
#include <cstdint>
#include <vector>

static inline std::vector<int64_t> server_prompt_cache_select_checkpoint_positions(
        const std::vector<int64_t> & src, int max_checkpoints, std::size_t min_step) {
    if (max_checkpoints <= 0 || src.size() <= (std::size_t) max_checkpoints) {
        return src;
    }

    const std::size_t step = min_step > 0 ? min_step : 1;
    // one slot stays for the final checkpoint; the rest are grid targets, the first four at
    // and just behind the shared-head cluster (creation is dense through six steps), then the
    // creation cadence (10x, 14x, 18x, ...)
    std::vector<std::size_t> targets = { 2*step, 4*step, 5*step, 6*step };
    for (std::size_t t = 10*step; targets.size() < (std::size_t) (max_checkpoints - 1); t += 4*step) {
        targets.push_back(t);
    }

    if (targets.size() > (std::size_t) (max_checkpoints - 1)) {
        targets.resize((std::size_t) (max_checkpoints - 1));
    }
    std::vector<int64_t> out;
    int64_t prev = 0;

    for (const std::size_t target : targets) {
        int64_t best = 0;
        bool have_best = false;

        for (const int64_t cur : src) {
            if (cur <= prev) {
                continue;
            }
            // the grid wobbles by a token (the break rule is a strict >), so take the
            // checkpoint nearest to the target, within one min step of it
            const int64_t dist = cur > (int64_t) target ? cur - (int64_t) target : (int64_t) target - cur;
            if (dist > (int64_t) min_step) {
                continue;
            }
            if (!have_best) {
                best = cur;
                have_best = true;
                continue;
            }
            const int64_t best_dist = best > (int64_t) target ? best - (int64_t) target : (int64_t) target - best;
            if (dist < best_dist || (dist == best_dist && cur < best)) {
                best = cur;
            }
        }

        if (have_best) {
            out.push_back(best);
            prev = best;
        }
    }

    if (!src.empty() && src.back() > prev) {
        out.push_back(src.back());
    }

    while (out.size() > (std::size_t) max_checkpoints) {
        out.erase(out.begin());
    }

    return out;
}

#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"

#include "server-prompt-cache-entry.h"
#include "server-agent-cache-plan.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <thread>
#include <vector>

#ifndef _WIN32
#include <unistd.h>
#endif
#if defined(__linux__)
#include <fcntl.h>
#if !defined(SYNC_FILE_RANGE_WAIT_BEFORE)
extern "C" int sync_file_range(int fd, int64_t offset, int64_t nbytes, unsigned int flags);
#define SYNC_FILE_RANGE_WAIT_BEFORE 1
#endif
#endif

//
// server_prompt
//

server_prompt_reuse server_prompt::get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const {
    server_prompt_reuse res;
    res.n_past   = n_prefix;
    res.pos_next = tokens.pos_next(n_prefix);

    // the first evaluated token attends to the positions > pos_eval - n_swa, so a SWA state
    // starting at pos_eval - n_swa + 1 is enough, while a state without SWA must end before pos_eval
    const llama_pos pos_eval = res.pos_next - (has_new_tokens ? 0 : 1);

    // a state or a checkpoint is useful when its pos_min is below this threshold
    const auto pos_min_thold = std::max(0, n_swa > 0 ? pos_eval - n_swa + 2 : pos_eval);

    if (pos_min < pos_min_thold) {
        return res;
    }

    // search for a context checkpoint
    const auto it = std::find_if(
        checkpoints.rbegin(),
        checkpoints.rend(),
        [&](const auto & ckpt) {
            const auto & cur = *ckpt;
            // guarantee that a checkpoint will result in at least one token being processed [TAG_PROMPT_LOGITS]
            SRV_TRC("checking checkpoint with [%d, %d] against %d...\n", cur.pos_min, cur.pos_max, pos_min_thold);
            // workaround for [TAG_CHECKPOINTS_FIX_POS_MIN]
            if (cur.pos_max > res.pos_next) {
                return false;
            }
            return cur.pos_min < pos_min_thold || cur.pos_min == 0;
        }
    );

    if (it == checkpoints.rend()) {
        res.n_past   = 0;
        res.pos_next = 0;
        return res;
    }

    const auto & cur = **it;

    res.pos_next = std::min(res.pos_next, std::max(cur.pos_min + 1, cur.pos_max));
    res.n_past   = std::min(tokens.size_up_to_pos(res.pos_next), (size_t) cur.n_tokens);
    res.ckpt     = *it;

    return res;
}

void server_state_prefault(void * p, size_t size) {
    // below this size the copy that follows faults the pages in at no extra cost
    static constexpr size_t min_size  = 64ull * 1024 * 1024;
    static constexpr size_t page_size = 4096;

    if (size < min_size) {
        return;
    }

    // 8 threads reach the memory bandwidth of the page zeroing; more do not help
    const size_t n_threads = std::clamp<size_t>(std::thread::hardware_concurrency(), 1, 8);
    const size_t part      = (size + n_threads - 1) / n_threads;

    std::vector<std::thread> threads;
    threads.reserve(n_threads);
    for (size_t t = 0; t < n_threads; ++t) {
        const size_t beg = std::min(size, t * part);
        const size_t end = std::min(size, beg + part);
        try {
            threads.emplace_back([p, beg, end]() {
                // one write per page is enough to fault it in
                for (size_t i = beg; i < end; i += page_size) {
                    ((volatile uint8_t *) p)[i] = 0;
                }
            });
        } catch (const std::exception &) {
            break; // prefaulting is best-effort; the copy below faults the rest
        }
    }
    for (auto & t : threads) {
        t.join();
    }
}

// the large state buffers are freed on a thread of their own: unmapping the 2.2 GB state of a
// 98k entry held the server thread, which every slot waits on, for 0.4-0.5 s (Arc Pro
// B70). created on first use and never joined: a buffer still queued at exit goes with the
// process
struct server_state_reaper {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<std::pair<uint8_t *, size_t>> queue;
    std::atomic<size_t> n_released { 0 };
    bool running = false;

    void run() {
        for (;;) {
            std::pair<uint8_t *, size_t> item;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [this] { return !queue.empty(); });
                item = queue.front();
                queue.pop_front();
            }
            std::allocator<uint8_t>().deallocate(item.first, item.second);
            n_released++;
        }
    }

    static server_state_reaper & get() {
        static server_state_reaper * reaper = [] {
            auto * r = new server_state_reaper();
            try {
                std::thread(&server_state_reaper::run, r).detach();
                r->running = true;
            } catch (const std::exception &) {
                // no thread: every buffer is freed by its caller, as before
            }
            return r;
        }();
        return *reaper;
    }
};

void server_state_release(uint8_t * p, size_t size) {
    // below this size the unmap is cheaper than the hand-off
    static constexpr size_t min_size = 64ull * 1024 * 1024;

    if (size >= min_size) {
        auto & reaper = server_state_reaper::get();
        if (reaper.running) {
            try {
                std::lock_guard<std::mutex> lock(reaper.mutex);
                reaper.queue.emplace_back(p, size);
            } catch (const std::exception &) {
                std::allocator<uint8_t>().deallocate(p, size);
                return;
            }
            reaper.cv.notify_one();
            return;
        }
    }

    std::allocator<uint8_t>().deallocate(p, size);
}

size_t server_state_released() {
    return server_state_reaper::get().n_released.load();
}

//
// server_prompt_cache
//

size_t server_prompt_cache::size() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.size();
    }

    return res;
}

size_t server_prompt_cache::n_tokens() const {
    size_t res = 0;

    for (const auto & state : states) {
        res += state.prompt.n_tokens();
    }

    return res;
}

// what a state is worth keeping: the tokens a future request takes from it, per byte it
// holds, decayed by how long ago it was last used
float server_prompt_cache::value(const server_prompt_cache_state & state) const {
    return server_prompt_cache_value(state.prompt.n_tokens(), state.size(), state.last_use_us,
            slot_tokens, ggml_time_us());
}

std::list<server_prompt_cache_state>::iterator server_prompt_cache::victim() {
    auto worst = states.begin();
    if (states.empty()) {
        return worst;
    }

    for (auto it = states.begin(); it != states.end(); ++it) {
        if (value(*it) < value(*worst)) {
            worst = it;
        }
    }
    // absent a plan, this is the same scan as before: the first strict minimum
    if (!protect) {
        return worst;
    }

    std::vector<float> values;
    std::vector<char> prot;
    values.reserve(states.size());
    bool any = false;
    size_t i = 0;
    size_t min_i = 0;
    for (auto it = states.begin(); it != states.end(); ++it, ++i) {
        values.push_back(value(*it));
        if (i > 0 && values[i] < values[min_i]) {
            min_i = i;
        }
        const bool p = protect(it->agent, it->session);
        prot.push_back(p ? 1 : 0);
        any = any || p;
    }
    if (!any) {
        return worst;
    }

    const size_t vi = agent_plan_victim(values.size(),
            [&](size_t j) { return values[j]; },
            [&](size_t j) { return prot[j] != 0; });
    auto at = [&](size_t index) {
        auto it = states.begin();
        for (size_t j = 0; j < index && it != states.end(); j++) {
            ++it;
        }
        return it;
    };
    auto chosen = at(vi);
    if (vi != min_i) {
        SRV_INF("prompt cache: keeping agent '%s', spilling '%s' instead\n",
                at(min_i)->agent.c_str(), chosen->agent.c_str());
    }
    return chosen;
}

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // keep checkpoints at fixed token positions, not list indices: the selection and its
    // rationale live in server-prompt-cache-spread.h. a checkpoint does not change after it
    // is created, so the entry shares the slot's: a copy is 1.4 GB of memcpy on the server
    // thread at 98k with 8 checkpoints
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;
    {
        const auto & src = prompt.checkpoints;

        std::vector<int64_t> positions;
        positions.reserve(src.size());
        for (const auto & cur : src) {
            positions.push_back(cur->n_tokens);
        }

        const auto kept = server_prompt_cache_select_checkpoint_positions(positions, max_checkpoints, min_step);

        // kept and positions are both ascending: walk them together
        size_t ki = 0;
        for (const auto & cur : src) {
            if (ki < kept.size() && cur->n_tokens == kept[ki]) {
                checkpoints.push_back(cur);
                ki++;
            }
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
    }

    // admission: the tier is full and the new state is worth less than the weakest resident --
    // keep the residents. both sides are priced as the state each holds now, the arrival by
    // being the one just saved and the victim by how long ago it was last used, so what is
    // compared is the tokens per byte each offers. nothing else may enter the price: the
    // arrival is a conversation's first state as often as its twentieth, and a resident that
    // earned a reuse count would refuse every arrival the tier has not already seen
    if (limit_size > 0 && !states.empty() && size() + state_size_new > limit_size) {
        const int64_t now   = ggml_time_us();
        const float   v_new = server_prompt_cache_value(prompt.tokens.size(), state_size_new, now, slot_tokens, now);
        const float   v_min = value(*victim());

        if (v_new < v_min) {
            SRV_WRN(" - not caching this prompt: %d tokens / %.3f MiB is worth %.2f tok/MiB, "
                    "below the weakest of %zu resident entries at %.2f\n",
                    prompt.n_tokens(), state_size_new / (1024.0 * 1024.0),
                    v_new * 1024.0 * 1024.0, states.size(), v_min * 1024.0 * 1024.0);

            return nullptr;
        }
    }

    // remove any cached prompts that are fully contained in the current prompt
    for (auto it = states.begin(); it != states.end();) {
        const int len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (len == (int) it->prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC(" - removing obsolete cached prompt with length %d\n", len);

            it = states.erase(it);
        } else {
            ++it;
        }
    }

    if (limit_size > 0) {
        // make room before allocating the new vectors to avoid breaching the limit. the
        // victim is the least valuable state, not the oldest: a tier that evicts in arrival
        // order drops the session that was asked for a minute ago (its entry is old, its
        // state is the one that has to be there) while keeping a stale 100k one nothing has
        // continued
        while (!states.empty() && size() + state_size_new > limit_size) {
            auto it = victim();

            if (disk_enabled()) {
                SRV_WRN(" - making room: spilling the least valuable entry out of %zu (%d tokens, %.3f MiB, %.2f tok/MiB-minute) to the disk tier\n",
                        states.size(), it->prompt.n_tokens(), it->size() / (1024.0 * 1024.0),
                        value(*it) * 1024.0 * 1024.0);

                disk_spill(*it);
            } else {
                SRV_WRN(" - making room: dropping the least valuable entry out of %zu (%d tokens, %.3f MiB, %.2f tok/MiB-minute)\n",
                        states.size(), it->prompt.n_tokens(), it->size() / (1024.0 * 1024.0),
                        value(*it) * 1024.0 * 1024.0);
            }

            states.erase(it);
        }
    }

    server_state_buffer state_data_tgt;
    server_state_buffer state_data_dft;

    // check if we can allocate enough memory for the new state
    try {
        state_data_tgt.resize(state_size_tgt);
        state_data_dft.resize(state_size_dft);
    } catch (const std::bad_alloc & e) {
        SRV_ERR("failed to allocate memory for prompt cache state: %s\n", e.what());

        limit_size = std::max<size_t>(1, 0.4*size());

        SRV_WRN(" - cache size limit reduced to %.3f MiB\n", limit_size / (1024.0 * 1024.0));

        update();

        return nullptr;
    }

    states.push_back({
        /*.prompt =*/ {
            /*.tokens      =*/ prompt.tokens.clone(),
            /*.checkpoints =*/ std::move(checkpoints),
            /*.lora        =*/ prompt.lora,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
        /*.last_use_us =*/ ggml_time_us(),
    });

    // what a token of this tier costs, measured rather than assumed: the slots-mode byte
    // limit is this times the token budget (see update())
    if (state_size_new > 0) {
        density_max = std::max(density_max, (float) state_size_new / (float) std::max<size_t>(1, prompt.tokens.size()));
    }

    return &states.back();
}

// restores a stored sequence state into the slot's contexts
static bool pcache_restore_state(const server_prompt_data & data, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    if (!data.main.empty()) {
        const size_t size = data.main.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_tgt, data.main.data(), size, id_slot, 0);
        if (n != size) {
            SRV_ERR("failed to restore state with size %zu\n", size);

            return false;
        }
    }

    // the draft only speeds decoding up: a state saved beside a draft this server does not run,
    // or under another draft cache type, still restores the target, and the draft starts cold
    if (!data.drft.empty() && ctx_dft != nullptr) {
        const size_t size = data.drft.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.drft.data(), size, id_slot, 0);
        if (n != size) {
            SRV_WRN("failed to restore draft state with size %zu, the draft starts cold\n", size);
        }
    }

    return true;
}

std::list<server_prompt_cache_state>::const_iterator server_prompt_cache::find_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, int & lcp) const {
    // a prompt made with another lora config is cleared before the new one starts, so it counts as empty
    const bool is_empty = prompt.tokens.empty() || lora_should_clear_cache(prompt.lora, lora_new);

    lcp = is_empty ? 0 : prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = is_empty ? -1.0f : float(lcp) / prompt.tokens.size(); // empty slot: any cache entry wins
    float f_sim_best  = float(lcp) / tokens_new.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.cend();

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.cbegin(); it != states.cend(); ++it) {
        if (lora_should_clear_cache(it->prompt.lora, lora_new)) {
            SRV_TRC("   - prompt with length %7zu, skipping, other lora config\n", it->prompt.tokens.size());
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        // don't trash large prompts
        if (f_keep_cur < 0.25f) {
            continue;
        }

        if (f_keep_best < f_keep_cur && f_sim_best < f_sim_cur) {
            f_keep_best = f_keep_cur;
            f_sim_best  = f_sim_cur;
            lcp         = lcp_cur;

            it_best = it;
        }
    }

    if (it_best != states.cend()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);
    }

    return it_best;
}

bool server_prompt_cache::has_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new) const {
    int lcp = 0;

    return find_better(prompt, tokens_new, lora_new, lcp) != states.cend();
}

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
        server_prompt_cache_decision * decision, int task_id, const std::string & agent, const std::string & session) {
    // a prompt made with another lora config is cleared before the new one starts, so it counts as empty
    const bool is_empty = prompt.tokens.empty() || lora_should_clear_cache(prompt.lora, lora_new);

    // an entry another (agent, session) saved is lent, whatever share of it the head keeps
    const auto owns = [&](const std::string & a, const std::string & s) { return a == agent && s == session; };
    if (task_id >= 0) {
        disk_taken taken = disk_take(task_id);

        // the read was armed against what the slot held then: a slot that holds the head now
        // keeps it, and the entry stays on disk
        if (taken.state && !is_empty && (int) prompt.tokens.get_common_prefix(tokens_new) >= taken.hit.lcp) {
            SRV_INF("prompt cache: the slot already holds the %d-token head, the disk read is dropped\n", taken.hit.lcp);
            taken.state.reset();
        }

        if (taken.state) {
            const int64_t t_restore = ggml_time_us();

            if (!pcache_restore_state(taken.state->data, ctx_tgt, ctx_dft, id_slot)) {
                return false;
            }

            SRV_INF("prompt cache: restored %zu MB from disk in %.2f ms\n",
                    taken.state->data.main.size() / (1024 * 1024), (ggml_time_us() - t_restore) / 1000.0);

            prompt = taken.state->prompt.clone();
            const bool lend = taken.hit.partial || !owns(taken.state->agent, taken.state->session);
            disk_commit(taken.hit, taken.unlink && !lend);

            if (decision != nullptr) {
                server_prompt_cache_decision dec;
                dec.chosen_lcp       = taken.hit.lcp;
                dec.chosen_disk      = true;
                dec.chosen_partial   = lend;
                dec.restored_agent   = taken.state->agent;
                dec.restored_session = taken.state->session;
                dec.restored_id      = true;
                *decision = dec;
            }

            return true;
        }
    }

    const int lcp_best = is_empty ? 0 : prompt.tokens.get_common_prefix(tokens_new);

    float f_keep_best = is_empty ? -1.0f : float(lcp_best) / prompt.tokens.size(); // empty slot: any cache entry wins
    float f_sim_best  = float(lcp_best) / tokens_new.size();

    server_prompt_cache_decision dec;

    dec.slot_lcp  = is_empty ? -1 : lcp_best;
    dec.ram_count = (int) states.size();

    SRV_TRC(" - looking for better prompt, base f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

    auto it_best = states.end();

    int  lcp_ram     = -1;
    bool ram_partial = false; // ... and the entry stays in the tier, its head only borrowed

    // find the most similar cached prompt, that would also preserve the most context
    for (auto it = states.begin(); it != states.end(); ++it) {
        if (lora_should_clear_cache(it->prompt.lora, lora_new)) {
            SRV_TRC("   - prompt with length %7zu, skipping, other lora config\n", it->prompt.tokens.size());
            continue;
        }

        const int lcp_cur = it->prompt.tokens.get_common_prefix(tokens_new);

        const float f_keep_cur = float(lcp_cur) / it->prompt.tokens.size();
        const float f_sim_cur  = float(lcp_cur) / tokens_new.size();

        SRV_TRC("   - prompt with length %7zu, lcp = %7d, f_keep = %.3f, f_sim = %.3f\n", it->prompt.tokens.size(), lcp_cur, f_keep_cur, f_sim_cur);

        dec.ram_seen = std::max(dec.ram_seen, lcp_cur);

        // a head that keeps a quarter of its entry is taken whole and the entry is consumed.
        // a shorter head is the prefix two conversations of one fleet share, and the resident
        // tier can lend it: the state is copied into the slot's context, not moved out of the
        // entry, so the owner's next turn still finds its state where it left it
        const bool keeps      = f_keep_cur >= 0.25f; // don't trash large prompts
        // a borrow under the entry's first rung resumes at 0: the copy and the repack buy
        // nothing, so the floor is that rung, not a token count of its own
        const bool borrowable = lcp_cur >= server_prompt_cache_first_rung(it->prompt.tokens.size(), min_step, (int) max_checkpoints);

        if (!keeps && !borrowable) {
            continue;
        }

        // the longest head wins: requiring a candidate to beat the standing one on both
        // fractions skipped a longer head inside a larger entry
        if (lcp_cur > lcp_ram && lcp_cur > lcp_best) {
            lcp_ram     = lcp_cur;
            ram_partial = !keeps || !owns(it->agent, it->session);
            it_best     = it;
        }
    }

    dec.ram_best = lcp_ram;

    // the disk tier is consulted on every request, not only when the RAM tier has nothing: a
    // RAM entry that keeps the prompt poorly (a short head, or one another conversation
    // shares) used to pre-empt a disk entry that keeps most of it, and the slot resumed from
    // the short head and re-processed the rest
    disk_hit hit;

    if (disk_enabled() && lora_new.empty()) {
        hit = disk_find(prompt, tokens_new, dec);
    }

    // the fetch gate (read cost vs. saved prefill, absolute floor) lives in
    // server-prompt-cache-disk.h, where the test pins it
    const bool disk_wins = server_prompt_cache_disk_wins(hit.lcp, lcp_ram, hit.size, hit.first_rung, fetch_bpt.load());

    if (disk_wins) {
        server_prompt_cache_state fetched;

        if (disk_fetch(hit, fetched)) {
            const int64_t t_restore = ggml_time_us();

            if (!pcache_restore_state(fetched.data, ctx_tgt, ctx_dft, id_slot)) {
                return false;
            }

            const bool lend = hit.partial || !owns(fetched.agent, fetched.session);
            disk_commit(hit, !lend);

            SRV_INF("prompt cache: restored %zu MB from disk in %.2f ms\n",
                    fetched.data.main.size() / (1024 * 1024), (ggml_time_us() - t_restore) / 1000.0);
            dec.restored_agent   = fetched.agent;
            dec.restored_session = fetched.session;
            dec.restored_id      = true;
            prompt = std::move(fetched.prompt);

            dec.chosen_lcp     = hit.lcp;
            dec.chosen_disk    = true;
            dec.chosen_partial = lend;
        }
        // a refused fetch (memory, I/O) falls through to the RAM entry
    }

    if (!dec.chosen_disk && it_best != states.end()) {
        SRV_TRC(" - found better prompt with f_keep = %.3f, f_sim = %.3f\n", f_keep_best, f_sim_best);

        const int64_t t_restore = ggml_time_us();

        if (!pcache_restore_state(it_best->data, ctx_tgt, ctx_dft, id_slot)) {
            return false;
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");
        SRV_INF("prompt cache: restored %zu MB from RAM in %.2f ms\n",
                it_best->data.main.size() / (1024 * 1024), (ggml_time_us() - t_restore) / 1000.0);
        // the slot takes the identity before the entry is erased below
        dec.restored_agent   = it_best->agent;
        dec.restored_session = it_best->session;
        dec.restored_id      = true;


        if (ram_partial) {
            // a borrowed head: the entry is another conversation's state and its owner's next
            // turn wants the rest of it. the restore copied the state into this slot's context,
            // so the entry can stay where it is at the memory it already occupies -- and the
            // slot needs its own copy of the token list to name what its KV now holds. the
            // borrow moves recency and nothing else: a head lent to another conversation says
            // the entry was used, not that its owner came back for it
            it_best->last_use_us = ggml_time_us();

            prompt = it_best->prompt.clone();

            SRV_INF("prompt cache: borrowed the %d-token head of a %zu-token entry\n",
                    lcp_ram, it_best->prompt.tokens.size());
        } else {
            it_best->data.main.clear();
            it_best->data.main.shrink_to_fit();
            it_best->data.drft.clear();
            it_best->data.drft.shrink_to_fit();

            // the whole cached prompt is a prefix of the new one
            if (lcp_best == (int) it_best->prompt.tokens.size()) {
                SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
            }

            prompt = std::move(it_best->prompt);

            states.erase(it_best);
        }

        dec.chosen_lcp = lcp_ram;
    }

    if (decision != nullptr) {
        *decision = dec;
    }

    return true;
}

void server_prompt_cache::update() {
    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // slots mode: the byte limit is the cost of the token budget, at the density this tier has
    // measured. this is the number the cgroup has to hold, so it is derived from the model
    // rather than typed in next to it -- a byte limit below the states the model needs is a
    // server that faults its way through reclaim instead of serving
    if (size_in_slots && limit_tokens > 0 && density_max > 0.0f) {
        limit_size = (size_t) ((double) limit_tokens * (double) density_max);
    }

    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            auto it = victim();

            if (disk_enabled()) {
                SRV_WRN(" - cache size limit reached, spilling the least valuable entry out of %zu (%d tokens, %.3f MiB)\n",
                        states.size(), it->prompt.n_tokens(), it->size() / (1024.0 * 1024.0));

                disk_spill(*it);
            } else {
                SRV_WRN(" - cache size limit reached, dropping the least valuable entry out of %zu (%d tokens, %.3f MiB)\n",
                        states.size(), it->prompt.n_tokens(), it->size() / (1024.0 * 1024.0));
            }

            states.erase(it);
        }
    }

    // dynamically increase the token limit if it can fit in the memory limit. not in slots
    // mode: there the token limit is the budget, and a state the model needs more memory for
    // does not make room for more states
    const size_t limit_tokens_cur = (limit_size > 0 && !size_in_slots)
        ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            auto it = victim();

            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, dropping the least valuable entry out of %zu (%d tokens)\n",
                    limit_tokens, limit_tokens_cur, states.size(), it->prompt.n_tokens());

            states.erase(it);
        }
    }

    SRV_TRC(" - cache state: %zu prompts, %.3f MiB (limits: %.3f MiB, %zu tokens, %zu est)\n",
            states.size(), size() / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0), limit_tokens, limit_tokens_cur);

    for (const auto & state : states) {
        SRV_TRC("   - prompt %p: %7d tokens, checkpoints: %2zu, %9.3f MiB\n",
                (const void *)&state, state.prompt.n_tokens(), state.prompt.checkpoints.size(), state.size() / (1024.0 * 1024.0));
    }
}

//
// server_slot_sched
//

static std::vector<const server_slot_info *> get_free_slot_infos(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) {
    std::vector<const server_slot_info *> free_slots;
    for (const auto & slot : slots) {
        if (!slot.is_processing && slot.id != exclude_id_slot) {
            free_slots.push_back(&slot);
        }
    }

    std::stable_sort(free_slots.begin(), free_slots.end(), [](const server_slot_info * a, const server_slot_info * b) {
        if ((a->prompt->n_tokens() == 0) != (b->prompt->n_tokens() == 0)) {
            return a->prompt->n_tokens() == 0;
        }
        return a->t_last_used < b->t_last_used;
    });

    free_slots.resize(std::min(free_slots.size(), n_slots_needed));

    return free_slots;
}

std::vector<int> server_slot_sched::get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const {
    std::vector<int> res;
    for (const auto * slot : get_free_slot_infos(slots, n_slots_needed, exclude_id_slot)) {
        res.push_back(slot->id);
    }

    return res;
}

server_slot_plan server_slot_sched::select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora, const server_prompt_cache * prompt_cache) const {
    const server_slot_info * ret = nullptr;

    bool update_cache = false;

    // if a specific slot is requested, use it (still goes through cache update logic below)
    if (task.id_slot != -1 && !slots.empty()) {
        // note: allow id_slot to be out of bounds (wrap around)
        const int id_slot = task.id_slot % slots.size();

        for (const auto & slot : slots) {
            if (slot.id == id_slot) {
                ret = &slot;
                break;
            }
        }

        if (ret) {
            // a busy slot is returned untouched, the caller defers the task
            if (ret->is_processing) {
                return { ret->id, false };
            }

            SLT_INF(*ret, "selected slot by id (%d)\n", task.id_slot);
        }
    }

    // find the slot that has at least n% prompt similarity
    if (slot_prompt_similarity != 0.0f) {
        float f_sim_best = 0;

        for (const auto & slot : slots) {
            if (task.id_slot != -1 && slot.id != task.id_slot) {
                continue;
            }

            // skip the slot if it is not available
            if (slot.is_processing) {
                SLT_TRC(slot, " - skipping, is_processing = %d\n", slot.is_processing);
                continue;
            }

            const auto & tokens = slot.prompt->tokens;

            // skip the slot if it does not contains cached tokens
            if (tokens.empty()) {
                SLT_TRC(slot, "%s", " - skipping, slot is empty\n");
                continue;
            }

            // the task would clear this slot, its state cannot be reused with another lora config
            if (lora_should_clear_cache(slot.prompt->lora, lora)) {
                SLT_TRC(slot, "%s", " - skipping, other lora config\n");
                continue;
            }

            // fraction of the Longest Common Prefix length with respect to the input prompt length
            const size_t lcp_len = tokens.get_common_prefix(task.tokens);
            const float f_sim_cur = float(lcp_len) / task.tokens.size();

            SLT_TRC(slot, " - checking sim = %.3f (%zu/%zu) > %.3f\n", f_sim_cur, lcp_len, task.tokens.size(), slot_prompt_similarity);

            // select the current slot if the criteria match
            if (f_sim_cur > f_sim_best && f_sim_cur > slot_prompt_similarity) {
                f_sim_best = f_sim_cur;

                ret = &slot;
            }
        }

        if (ret != nullptr) {
            // empty slot: nothing to keep, avoid 0/0
            const float f_keep = ret->prompt->tokens.size() > 0 ?
                (f_sim_best*task.tokens.size()) / ret->prompt->tokens.size() : 0.0f;

            if (task.id_slot == -1) {
                SLT_INF(*ret, "selected slot by LCP similarity, f_sim_best = %.3f (> %.3f thold), f_keep = %.3f\n",
                        f_sim_best, slot_prompt_similarity, f_keep);
            }

            // if we are about to lose a large portion of the existing context - save it in the prompt cache
            // another agent's pick drops the rest of this one's state: save it as well
            const auto differs = [](const std::string * held, const std::string & wanted) {
                return held ? *held != wanted : !wanted.empty();
            };
            if (f_keep < 0.5f || (f_keep < 1.0f && (differs(ret->agent, task.params.agent) ||
                                                    differs(ret->session, task.params.session)))) {
                update_cache = true;
            }
        }
    }

    // find an empty slot, or the one that has been least recently used
    if (ret == nullptr) {
        const auto free_slots = get_free_slot_infos(slots, 1, -1);

        if (!free_slots.empty()) {
            ret = free_slots[0];

            SLT_INF(*ret, "selected slot by LRU, n_tokens = %d, t_last = %" PRId64 "\n", ret->prompt->n_tokens(), ret->t_last_used);

            update_cache = true;
        }
    }

    if (ret == nullptr) {
        return {};
    }

    // cache prompts only for the tasks that reuse them
    if (prompt_cache == nullptr || task.is_stateless()) {
        return { ret->id, false };
    }

    // the checks above only tell if the outgoing state is worth saving
    // the cache can still hold a better start for this task, so ask it
    update_cache = update_cache || prompt_cache->has_better(*ret->prompt, task.tokens, lora);

    return { ret->id, update_cache };
}

// FNV-1a 64-bit
static uint64_t pcache_hash_bytes(uint64_t h, const void * data, size_t size) {
    const auto * p = (const uint8_t *) data;
    for (size_t i = 0; i < size; i++) {
        h ^= p[i];
        h *= 0x100000001b3ull;
    }
    return h;
}

template <typename T>
static uint64_t pcache_hash_value(uint64_t h, const T & value) {
    return pcache_hash_bytes(h, &value, sizeof(value));
}

static uint64_t pcache_hash_str(uint64_t h, const std::string & s) {
    return pcache_hash_bytes(h, s.data(), s.size());
}

uint64_t server_prompt_cache::context_fingerprint(const llama_model * model, const std::string & model_path,
        const llama_context * ctx, ggml_type type_k, ggml_type type_v) const {
    // the UUID identifies the model; fall back to the model file name when the GGUF
    // does not carry a general.uuid metadata key
    std::string model_id;
    char buf[256] = {0};
    if (model == nullptr || llama_model_meta_val_str(model, "general.uuid", buf, sizeof(buf)) <= 0) {
        model_id = std::filesystem::path(model_path).filename().string();
    } else {
        model_id = buf;
    }

    const int32_t n_ctx     = ctx ? llama_n_ctx(ctx)     : 0;
    const int32_t n_ctx_seq = ctx ? llama_n_ctx_seq(ctx) : 0;

    uint64_t h = 0xcbf29ce484222325ull;
    h = pcache_hash_str (h, model_id);
    h = pcache_hash_value(h, (uint64_t) n_ctx);
    h = pcache_hash_value(h, (uint64_t) n_ctx_seq);
    h = pcache_hash_str (h, ggml_type_name(type_k));
    h = pcache_hash_str (h, ggml_type_name(type_v));

    return h;
}

//
// server_prompt_cache: level 2 on disk (--cache-disk-dir)
//
// one file per entry, so a fetch reads exactly what it needs and an eviction is an unlink:
// the device behind this is ~900 MB/s sequential and ~45 MB/s random, and one entry is
// ~5 GiB at depth, so whole-file I/O is the only shape that works. the states live on
// disk; disk_states is a RAM index (tokens + path + size) that matching runs against, which
// keeps the LCP / f_keep / f_sim selection identical to the RAM tier.
//
// writes happen on eviction and at shutdown and are handed to a writer thread: the request
// that happens to trigger an eviction must not wait ~9 s for another conversation's state
// to reach the disk.
//
// best-effort MemAvailable in bytes; 0 when it cannot be determined (defined below)
static size_t pcache_mem_available_bytes();

// wall-clock budget for the shutdown flush: entries written by then are kept, the rest are
// dropped -- the disk tier is a cache and a half-written cache is still a cache
static constexpr int64_t SERVER_PROMPT_CACHE_FLUSH_DEADLINE_US = 120 * 1000000ll;

// FNV-1a 64-bit over the token ids: names the file for a given state, so writing the same
// state twice replaces instead of accumulating
static uint64_t pcache_tokens_hash(const server_tokens & tokens) {
    uint64_t h = 0xcbf29ce484222325ull;

    for (size_t i = 0; i < tokens.size(); i++) {
        h ^= (uint64_t) tokens[i];
        h *= 0x100000001b3ull;
    }

    return h;
}

server_prompt_cache::~server_prompt_cache() {
    {
        std::lock_guard<std::mutex> lock(disk_mutex);
        disk_stop = true;
    }

    disk_cv.notify_all();
    read_cv.notify_all();

    if (read_thread.joinable()) {
        read_thread.join();
    }
    if (disk_thread.joinable()) {
        disk_thread.join();
    }
}

// a ".bin.agent" sidecar names the entry. the sweep must not delete it: the matching
// .bin reads it. a missing or malformed sidecar leaves the stamp empty and keeps the entry
static bool pcache_is_agent_sidecar(const std::string & path) {
    static const char k_suffix[] = ".agent";
    const size_t n = sizeof(k_suffix) - 1;
    return path.size() > n && path[path.size() - n - 1] != '/' &&
            path.compare(path.size() - n, n, k_suffix) == 0;
}

static void pcache_remove_agent_sidecar(const std::string & path) {
    std::error_code ec;
    std::filesystem::remove(path + ".agent", ec);
}

static void pcache_write_agent_sidecar(const std::string & path, const std::string & agent, const std::string & session) {
    const std::string side = path + ".agent";
    const std::string tmp  = side + ".tmp";
    FILE * file = fopen(tmp.c_str(), "wb");
    if (file == nullptr) {
        SRV_WRN("prompt cache: cannot write agent sidecar '%s': %s\n", side.c_str(), strerror(errno));
        return;
    }
    const std::string body = agent + "\t" + session + "\n";
    const bool ok = fwrite(body.data(), 1, body.size(), file) == body.size() && fflush(file) == 0;
    fclose(file);
    std::error_code ec;
    if (!ok) {
        std::filesystem::remove(tmp, ec);
        SRV_WRN("prompt cache: failed to write agent sidecar '%s'\n", side.c_str());
        return;
    }
    std::filesystem::rename(tmp, side, ec);
    if (ec) {
        std::filesystem::remove(tmp, ec);
        SRV_WRN("prompt cache: cannot rename agent sidecar '%s': %s\n", side.c_str(), ec.message().c_str());
    }
}

static void pcache_read_agent_sidecar(const std::string & path, std::string & agent, std::string & session) {
    agent.clear();
    session.clear();
    FILE * file = fopen((path + ".agent").c_str(), "rb");
    if (file == nullptr) {
        return;
    }
    char buf[80];
    const size_t n = fread(buf, 1, sizeof(buf), file);
    const int extra = fgetc(file);
    fclose(file);
    // 32 + tab + 32 + newline is 66. a full buffer, or another byte, is not that line
    if (n == 0 || n == sizeof(buf) || extra != EOF) {
        return;
    }
    std::string line(buf, n);
    if (!line.empty() && line.back() == '\n') {
        line.pop_back();
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    if (line.find('\n') != std::string::npos) {
        return;
    }
    const auto tab = line.find('\t');
    if (tab == std::string::npos || line.find('\t', tab + 1) != std::string::npos) {
        return;
    }
    const std::string a = line.substr(0, tab);
    const std::string s = line.substr(tab + 1);
    if (agent_plan_sanitize(a) != a || agent_plan_sanitize(s) != s) {
        return;
    }
    agent = a;
    session = s;
}

bool server_prompt_cache::disk_init(const std::string & dir, size_t limit, const std::string & model_path,
        const llama_model * model, const llama_context * ctx, ggml_type type_k, ggml_type type_v, bool fsync) {
    disk_dir   = dir;
    disk_limit = limit;
    disk_used  = 0;
    disk_fsync = fsync;

    disk_fingerprint = context_fingerprint(model, model_path, ctx, type_k, type_v);

    std::error_code ec;
    if (!std::filesystem::is_directory(disk_dir, ec) || ec) {
        SRV_WRN("prompt cache: disk tier '%s' is not a directory (%s), tier disabled\n",
                disk_dir.c_str(), ec ? ec.message().c_str() : "not a directory");
        disk_dir.clear();
        return false;
    }

    size_t n_files = 0;
    size_t n_stale = 0;
    size_t n_swept = 0;

    for (const auto & de : std::filesystem::directory_iterator(disk_dir, ec)) {
        if (ec) {
            break;
        }

        const auto & path = de.path();
        if (!server_prompt_cache_is_entry_path(path.string())) {
            // a sidecar stays: the matching .bin reads it. anything else, including a
            // .bin.tmp the writer died on, is swept
            if (pcache_is_agent_sidecar(path.string())) {
                continue;
            }
            std::filesystem::remove(path, ec);
            n_swept++;
            continue;
        }

        n_files++;

        FILE * file = fopen(path.c_str(), "rb");
        if (file == nullptr) {
            n_stale++;
            continue;
        }

        server_tokens tokens;
        const bool ok = pcache_read_entry_header(file, disk_fingerprint, tokens);
        fclose(file);

        if (!ok) {
            std::filesystem::remove(path, ec);
            pcache_remove_agent_sidecar(path.string());
            n_stale++;
            continue;
        }

        const auto sz = std::filesystem::file_size(path, ec);
        if (ec) {
            n_stale++;
            continue;
        }

        // LRU across restarts: the file's mtime is the last time it was written or fetched
        const auto mt = std::filesystem::last_write_time(path, ec);
        const int64_t use = ec ? 0 : server_prompt_cache_use_from_write_time(mt, ggml_time_us());

        disk_states.push_back({ std::move(tokens), path.string(), (size_t) sz, use });
        pcache_read_agent_sidecar(disk_states.back().path, disk_states.back().agent, disk_states.back().session);
        disk_used += (size_t) sz;
    }

    std::sort(disk_states.begin(), disk_states.end(),
            [](const auto & a, const auto & b) { return a.last_use_us < b.last_use_us; });

    disk_prune();

    disk_thread = std::thread([this] { disk_writer(); });
    read_thread = std::thread([this] { disk_reader(); });

    SRV_INF("prompt cache: disk tier '%s' holds %zu entries (%.2f GiB, limit %.2f GiB), %zu stale removed, %zu orphan tmp swept\n",
            disk_dir.c_str(), disk_states.size(), disk_used / (1024.0 * 1024.0 * 1024.0),
            disk_limit / (1024.0 * 1024.0 * 1024.0), n_stale, n_swept);

    return true;
}

bool server_prompt_cache::disk_enabled() const {
    return !disk_dir.empty();
}

void server_prompt_cache::note_read_rate(double mb_s) {
    if (!(mb_s > 0.0)) {
        return;
    }

    std::lock_guard<std::mutex> lock(read_mu);
    rate_samples[rate_i] = mb_s;
    rate_i = (rate_i + 1) % 8;
    if (rate_n < 8) {
        rate_n++;
    }

    fetch_bpt.store(server_prompt_cache_bytes_per_token(server_prompt_cache_rate_median(rate_samples, rate_n)));
}

void server_prompt_cache::disk_drop_path(const std::string & path) {
    {
        std::lock_guard<std::mutex> lock(disk_mutex);

        auto it = std::find_if(disk_states.begin(), disk_states.end(),
                [&](const disk_state & e) { return e.path == path; });

        if (it != disk_states.end()) {
            disk_used -= it->size;
            disk_states.erase(it);
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);
    pcache_remove_agent_sidecar(path);
}

void server_prompt_cache::disk_commit(const disk_hit & hit, bool unlink_file) {
    {
        std::lock_guard<std::mutex> lock(disk_mutex);

        auto it = disk_states.begin();
        for (; it != disk_states.end(); ++it) {
            if (it->path == hit.path) {
                break;
            }
        }

        if (it != disk_states.end()) {
            if (hit.partial || !unlink_file) {
                it->last_use_us = ggml_time_us();
            } else {
                disk_used -= it->size;
                disk_states.erase(it);
            }
        }
    }

    if (unlink_file && !hit.partial) {
        std::error_code ec;
        std::filesystem::remove(hit.path, ec);
        pcache_remove_agent_sidecar(hit.path);
    }
}

bool server_prompt_cache::disk_read_file(const disk_hit & hit, server_prompt_cache_state & out) {
    if (hit.lcp < 0 || hit.path.empty()) {
        return false;
    }

    const size_t mem_avail = pcache_mem_available_bytes();
    if (mem_avail > 0 && hit.size > mem_avail / 2) {
        SRV_WRN("prompt cache: not fetching %.2f MiB from disk, %.2f GiB of memory available\n",
                hit.size / (1024.0 * 1024.0), mem_avail / (1024.0 * 1024.0 * 1024.0));

        return false;
    }

    SRV_INF("prompt cache: fetching entry from disk (lcp = %d, f_keep = %.3f, f_sim = %.3f%s)\n",
            hit.lcp, hit.keep, hit.sim, hit.partial ? ", keeping it" : "");

    FILE * file = fopen(hit.path.c_str(), "rb");
    if (file == nullptr) {
        SRV_WRN("prompt cache: cannot open '%s': %s\n", hit.path.c_str(), strerror(errno));
        disk_drop_path(hit.path);

        return false;
    }

    out.prompt.clear();
    out.data.main.clear();
    out.data.drft.clear();

    const int64_t t_read = ggml_time_us();
    bool corrupt = false;
    const bool ok = pcache_read_entry(file, disk_fingerprint, out, &corrupt);
    fclose(file);

    const int64_t t_read_us = ggml_time_us() - t_read;
    if (ok) {
        const double mb_s = t_read_us > 0 ? 1e6 * (double) hit.size / (double) t_read_us / (1024.0 * 1024.0) : 0.0;
        SRV_INF("prompt cache: read %s in %.2f ms (%.0f MB/s)\n", hit.path.c_str(), t_read_us / 1000.0, mb_s);
        note_read_rate(mb_s);
        disk_copy_identity(hit.path, out);

        return true;
    }

    if (server_prompt_cache_disk_failed_read_unlinks(ok, corrupt)) {
        SRV_WRN("prompt cache: '%s' %s, dropping it\n", hit.path.c_str(),
                corrupt ? "does not match its checksum" : "is truncated");
        disk_drop_path(hit.path);
    }

    return false;
}

server_prompt_cache::disk_wait server_prompt_cache::disk_prepare(int task_id, const server_tokens & tokens, const server_prompt * prompt_slot) {
    if (!disk_enabled() || task_id < 0 || tokens.size() == 0) {
        return disk_wait::skip;
    }

    {
        std::lock_guard<std::mutex> lock(read_mu);
        const auto again = read_sched.reentry(task_id);

        if (again == server_prompt_cache_read_reentry::ready) {
            return disk_wait::ready;
        }
        if (again == server_prompt_cache_read_reentry::failed) {
            read_sched.cancel(task_id);

            return disk_wait::skip;
        }
        if (again == server_prompt_cache_read_reentry::pending) {
            return disk_wait::pending;
        }
    }

    server_prompt empty;
    const server_prompt & slot = prompt_slot != nullptr ? *prompt_slot : empty;
    server_prompt_cache_decision dec;
    const disk_hit hit = disk_find(slot, tokens, dec);

    // the read has to gain its cost over the longest head the task already has: a RAM entry,
    // or the slot it is handed, which load() does not trade for a shorter one
    int lcp_held = slot.tokens.size() > 0 ? (int) slot.tokens.get_common_prefix(tokens) : -1;
    for (const auto & st : states) {
        lcp_held = std::max(lcp_held, (int) st.prompt.tokens.get_common_prefix(tokens));
    }

    if (!server_prompt_cache_disk_wins(hit.lcp, lcp_held, hit.size, hit.first_rung, fetch_bpt.load())) {
        return disk_wait::skip;
    }

    bool start = false;
    {
        std::lock_guard<std::mutex> lock(read_mu);
        const auto armed = read_sched.arm(hit.path, task_id);

        if (armed.kind == server_prompt_cache_read_kind::failed) {
            read_sched.cancel(task_id);

            return disk_wait::skip;
        }
        if (armed.kind == server_prompt_cache_read_kind::ready) {
            return disk_wait::ready;
        }

        read_hits[hit.path] = hit;
        if (armed.start) {
            read_q.push_back(hit);
            start = true;
        }
    }

    if (start) {
        read_cv.notify_one();
    }

    return disk_wait::pending;
}

void server_prompt_cache::disk_cancel(int task_id) {
    std::lock_guard<std::mutex> lock(read_mu);
    read_sched.cancel(task_id);
}

server_prompt_cache::disk_taken server_prompt_cache::disk_take(int task_id) {
    disk_taken taken;

    std::lock_guard<std::mutex> lock(read_mu);
    const auto got = read_sched.take(task_id);
    if (!got.ok) {
        return taken;
    }

    auto it = read_ready.find(got.path);
    if (it == read_ready.end()) {
        return taken;
    }

    taken.state  = it->second;
    taken.hit    = read_hits[got.path];
    taken.unlink = got.last && !taken.hit.partial;

    if (got.last) {
        read_ready.erase(it);
        read_hits.erase(got.path);
    }

    return taken;
}

void server_prompt_cache::disk_reader() {
    for (;;) {
        disk_hit hit;

        {
            std::unique_lock<std::mutex> lock(read_mu);
            read_cv.wait(lock, [this] { return disk_stop.load() || !read_q.empty(); });

            if (read_q.empty()) {
                return;
            }

            hit = std::move(read_q.front());
            read_q.pop_front();
        }

        server_prompt_cache_state fetched;
        const bool ok = disk_read_file(hit, fetched);

        std::vector<int> wake;
        std::function<void(int)> wake_fn;
        std::string next;

        {
            std::lock_guard<std::mutex> lock(read_mu);
            wake = read_sched.complete(hit.path, ok);

            if (ok && !wake.empty()) {
                read_ready[hit.path] = std::make_shared<server_prompt_cache_state>(std::move(fetched));
                read_hits[hit.path] = hit;
            } else {
                read_ready.erase(hit.path);
                read_hits.erase(hit.path);
            }

            next = read_sched.next_path();
            if (!next.empty()) {
                auto it = read_hits.find(next);
                if (it != read_hits.end()) {
                    read_q.push_back(it->second);
                } else {
                    const auto missed = read_sched.complete(next, false);
                    wake.insert(wake.end(), missed.begin(), missed.end());
                }
            }

            wake_fn = on_read_ready;
        }

        if (!next.empty()) {
            read_cv.notify_one();
        }

        if (wake_fn) {
            for (int id : wake) {
                wake_fn(id);
            }
        }
    }
}

void server_prompt_cache::disk_writer() {
    for (;;) {
        server_prompt_cache_state state;

        {
            std::unique_lock<std::mutex> lock(disk_mutex);
            disk_cv.wait(lock, [this] { return disk_stop.load() || !disk_queue.empty(); });

            if (disk_queue.empty()) {
                return;
            }

            state = std::move(disk_queue.front());
            disk_queue.pop_front();
        }

        disk_write(state);
    }
}

void server_prompt_cache::disk_spill(server_prompt_cache_state & state) {
    // an entry file carries no lora config, so it holds only states made without one
    if (disk_dir.empty() || state.prompt.tokens.size() == 0 || !state.prompt.lora.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(disk_mutex);

    struct qitem {
        uint64_t    hash;
        int64_t     tokens;
        size_t      size;
        int64_t     last_use_us;
    };

    std::vector<qitem> view;
    view.reserve(disk_queue.size());

    for (const auto & cur : disk_queue) {
        view.push_back({ pcache_tokens_hash(cur.prompt.tokens), (int64_t) cur.prompt.tokens.size(), cur.size(), cur.last_use_us });
    }

    const uint64_t hash = pcache_tokens_hash(state.prompt.tokens);
    const auto dec = server_prompt_cache_queue_admit(view,
            [](const qitem & e) { return e.hash; },
            [](const qitem & e) { return e.tokens; },
            hash, 8, slot_tokens, ggml_time_us());

    auto at = [&](size_t index) {
        auto it = disk_queue.begin();
        for (size_t i = 0; i < index && it != disk_queue.end(); i++) {
            ++it;
        }

        return it;
    };

    if (dec.act == server_prompt_cache_queue_act::replace) {
        *at(dec.index) = std::move(state);
        disk_cv.notify_one();

        return;
    }

    if (dec.act == server_prompt_cache_queue_act::drop_push) {
        SRV_WRN("prompt cache: disk tier is behind (%zu queued), dropping the least valuable queued state\n",
                disk_queue.size());
        disk_queue.erase(at(dec.index));
    }

    disk_queue.push_back(std::move(state));
    disk_cv.notify_one();
}

struct pcache_write_pacer {
    int    fd       = -1;
    size_t released = 0;

    void apply(size_t written, size_t window) {
#if defined(__linux__)
        const auto drop = server_prompt_cache_write_behind_of(written, released, window);
        if (drop.len == 0 || fd < 0) {
            return;
        }

        sync_file_range(fd, (int64_t) drop.off, (int64_t) drop.len, SYNC_FILE_RANGE_WAIT_BEFORE);
        posix_fadvise(fd, (off_t) drop.off, (off_t) drop.len, POSIX_FADV_DONTNEED);
        released = drop.off + drop.len;
#else
        (void) written;
        (void) window;
#endif
    }

    static void on_written(void * ctx, size_t written) {
        static_cast<pcache_write_pacer *>(ctx)->apply(written, SERVER_PROMPT_CACHE_WRITE_WINDOW);
    }
};

void server_prompt_cache::disk_write(server_prompt_cache_state & state) {
    if (disk_dir.empty() || state.prompt.tokens.size() == 0) {
        return;
    }

    const std::string name = string_format("%d-%016" PRIx64 ".bin",
            state.prompt.n_tokens(), pcache_tokens_hash(state.prompt.tokens));
    const std::string path = disk_dir + "/" + name;
    const std::string tmp  = path + ".tmp";

    std::vector<char> tokens;

    try {
        tokens = state.prompt.tokens.serialize();
    } catch (const std::exception & e) {
        SRV_WRN("skipping prompt cache entry with %d tokens that cannot be serialized: %s\n",
                state.prompt.n_tokens(), e.what());
        return;
    }

    const uint64_t size_expected = pcache_entry_size(state, tokens.size());

    // the tier already holds this entry: its name is the hash of the tokens and its size
    // is the size of the state those tokens produce, so writing it again would put the
    // same bytes on the device a second time
    bool already = false;
    {
        std::lock_guard<std::mutex> lock(disk_mutex);

        for (auto & e : disk_states) {
            if (e.path == path && e.size == size_expected) {
                e.last_use_us = ggml_time_us();
                e.agent = state.agent;
                e.session = state.session;
                already = true;
                break;
            }
        }
    }
    if (already) {
        // the bin is unchanged, but the owner may not be. the sidecar is the stamp a
        // restart reads, and this write is outside the lock
        pcache_write_agent_sidecar(path, state.agent, state.session);
        SRV_TRC("prompt cache: entry '%s' is already on disk with the same size, not writing it again\n",
                name.c_str());
        return;
    }

    FILE * file = fopen(tmp.c_str(), "wb");
    if (file == nullptr) {
        SRV_WRN("prompt cache: cannot open '%s' for writing: %s\n", tmp.c_str(), strerror(errno));
        return;
    }

    pcache_hasher hasher;

    if (!hasher.init()) {
        SRV_WRN("prompt cache: cannot allocate the checksum state, not writing '%s'\n", tmp.c_str());

        fclose(file);

        std::error_code ec;
        std::filesystem::remove(tmp, ec);

        return;
    }

    const int64_t t_start = ggml_time_us();

    pcache_write_pacer pacer;
    pacer.fd = fileno(file);

    pcache_writer writer;
    writer.file     = file;
    writer.hasher   = &hasher;
    writer.pace     = pcache_write_pacer::on_written;
    writer.pace_ctx = &pacer;

    bool ok = pcache_fwrite_u32(file, SERVER_PROMPT_CACHE_ENTRY_MAGIC) &&
              pcache_fwrite_u32(file, SERVER_PROMPT_CACHE_ENTRY_VERSION) &&
              pcache_fwrite_u32(file, (uint32_t) LLAMA_STATE_SEQ_VERSION) &&
              pcache_fwrite_u64(file, disk_fingerprint) &&
              pcache_write_entry(writer, state, tokens);

    // the trailer closes the entry: the checksum of the payload, written after it, so an
    // entry whose payload the device did not take whole has no valid trailer
    if (ok) {
        ok = pcache_fwrite_u64(file, hasher.digest());
    }

    if (ok && fflush(file) != 0) {
        ok = false;
    }

    if (ok) {
        pacer.apply(writer.written, 0);
    }

#ifndef _WIN32
    // --cache-disk-fsync waits for the inode as well. the pace above already waited the
    // data, so a read is not stuck behind a multi-gigabyte dirty set either way
    if (ok && disk_fsync && fsync(fileno(file)) != 0) {
        ok = false;
    }
#endif

    fclose(file);
    hasher.release();

    std::error_code ec;

    if (!ok) {
        SRV_WRN("prompt cache: failed to write '%s'\n", tmp.c_str());
        std::filesystem::remove(tmp, ec);
        return;
    }

    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        SRV_WRN("prompt cache: cannot rename '%s': %s\n", tmp.c_str(), ec.message().c_str());
        std::filesystem::remove(tmp, ec);
        return;
    }
    // before the tokens are moved into the index. a failed sidecar leaves the bin, and
    // the next load treats a missing sidecar as an unstamped entry
    const std::string agent = state.agent;
    const std::string session = state.session;
    pcache_write_agent_sidecar(path, agent, session);


    const auto sz = std::filesystem::file_size(path, ec);
    if (ec) {
        return;
    }

    const int64_t t_write_us = ggml_time_us() - t_start;

    SRV_INF("prompt cache: wrote entry '%s' (%d tokens, %.2f MiB) in %.2f s (%.0f MB/s%s)\n",
            name.c_str(), state.prompt.n_tokens(), (double) sz / (1024.0 * 1024.0),
            (double) t_write_us / 1e6,
            t_write_us > 0 ? 1e6 * (double) sz / (double) t_write_us / (1024.0 * 1024.0) : 0.0,
            disk_fsync ? ", synced" : "");

    // the state itself goes out of RAM here; only its tokens stay, for matching
    std::lock_guard<std::mutex> lock(disk_mutex);

    const int64_t now = ggml_time_us();

    for (auto & e : disk_states) {
        if (e.path == path) {
            disk_used    += (size_t) sz - e.size;
            e.size        = (size_t) sz;
            e.last_use_us = now;
            e.agent       = agent;
            e.session     = session;
            return;
        }
    }

    disk_states.push_back({ std::move(state.prompt.tokens), path, (size_t) sz, now, agent, session });
    disk_used += (size_t) sz;

    disk_prune();
}

// assumes disk_mutex
void server_prompt_cache::disk_prune() {
    while (disk_limit > 0 && disk_used > disk_limit && !disk_states.empty()) {
        // first-in-first-out deleted a hot conversation's state on every write while small stale
        // prefixes survived; the victim is the least valuable entry, priced like the RAM tier's
        const int64_t now = ggml_time_us();
        const std::size_t vi_value = server_prompt_cache_disk_victim(disk_states, [](const auto & s) { return s.tokens.size(); },
                slot_tokens, now);
        std::size_t vi = vi_value;
        if (protect) {
            std::vector<char> prot(disk_states.size());
            bool any = false;
            for (std::size_t i = 0; i < disk_states.size(); i++) {
                prot[i] = protect(disk_states[i].agent, disk_states[i].session) ? 1 : 0;
                any = any || prot[i];
            }
            if (any) {
                vi = agent_plan_victim(disk_states.size(),
                        [&](std::size_t i) {
                            return server_prompt_cache_value((int64_t) disk_states[i].tokens.size(),
                                    disk_states[i].size, disk_states[i].last_use_us, slot_tokens, now);
                        },
                        [&](std::size_t i) { return prot[i] != 0; });
                if (vi != vi_value) {
                    SRV_INF("prompt cache: keeping agent '%s', spilling '%s' instead\n",
                            disk_states[vi_value].agent.c_str(), disk_states[vi].agent.c_str());
                }
            }
        }
        auto & victim = disk_states[vi];

        SRV_INF("prompt cache: disk tier limit reached, removing '%s' (%zu tokens, %.2f MiB)\n",
                victim.path.c_str(), victim.tokens.size(), victim.size / (1024.0 * 1024.0));

        std::error_code ec;
        std::filesystem::remove(victim.path, ec);
        pcache_remove_agent_sidecar(victim.path);

        disk_used -= victim.size;
        disk_states.erase(disk_states.begin() + vi);
    }
}

std::vector<server_prompt_cache::disk_agent_row> server_prompt_cache::disk_agent_rows() {
    std::lock_guard<std::mutex> lock(disk_mutex);
    std::vector<disk_agent_row> out;
    out.reserve(disk_states.size());
    for (const auto & e : disk_states) {
        if (e.agent.empty()) {
            continue;
        }
        out.push_back({ e.agent, e.session, e.tokens.size() });
    }
    return out;
}

void server_prompt_cache::disk_copy_identity(const std::string & path, server_prompt_cache_state & out) {
    std::lock_guard<std::mutex> lock(disk_mutex);
    for (const auto & e : disk_states) {
        if (e.path == path) {
            out.agent = e.agent;
            out.session = e.session;
            return;
        }
    }
}


server_prompt_cache::disk_hit server_prompt_cache::disk_find(const server_prompt & prompt_slot, const server_tokens & tokens_new, server_prompt_cache_decision & dec) {
    disk_hit hit;

    if (disk_dir.empty()) {
        return hit;
    }

    std::lock_guard<std::mutex> lock(disk_mutex);

    // the slot's own prompt is the bar to clear: a disk entry has to offer a longer head
    const int lcp_slot = prompt_slot.tokens.size() > 0 ? prompt_slot.tokens.get_common_prefix(tokens_new) : 0;

    const disk_state * best = nullptr;
    int lcp_hit = -1;

    dec.disk_count = (int) disk_states.size();

    for (const auto & e : disk_states) {
        const int lcp = e.tokens.get_common_prefix(tokens_new);

        const float f_keep = e.tokens.size() > 0 ? float(lcp) / e.tokens.size() : 0.0f;
        const float f_sim  = float(lcp) / tokens_new.size();

        SRV_TRC("prompt cache: disk entry %zu tokens, lcp = %d, f_keep = %.3f, f_sim = %.3f\n",
                e.tokens.size(), lcp, f_keep, f_sim);

        dec.disk_seen = std::max(dec.disk_seen, lcp);

        if (lcp <= lcp_slot) {
            continue;
        }

        // the RAM tier refuses entries that keep under a quarter of themselves because restoring
        // one consumes it, and the rest of its state goes with it. a disk entry costs the read
        // and, when the head is partial, can stay where it is: what it has to clear is the read
        const int bpt = std::max(1, fetch_bpt.load());
        const int reuse_floor = (int) std::max<std::size_t>(2048, std::min(e.size / (std::size_t) bpt, (std::size_t) 0x7fffffff));

        if (f_keep < 0.25f && lcp < reuse_floor) {
            continue;
        }

        // the longest head wins: that is the number of tokens the request does not re-process
        if (lcp > lcp_hit) {
            best    = &e;
            lcp_hit = lcp;
        }
    }

    if (best == nullptr) {
        return hit;
    }

    dec.disk_best = lcp_hit;

    hit.path = best->path;
    hit.size = best->size;
    hit.lcp  = lcp_hit;
    hit.keep = best->tokens.size() > 0 ? float(lcp_hit) / best->tokens.size() : 0.0f;
    hit.sim  = float(lcp_hit) / tokens_new.size();
    hit.partial = hit.keep < 0.25f;
    hit.first_rung = server_prompt_cache_first_rung(best->tokens.size(), min_step, (int) max_checkpoints);

    SRV_TRC("prompt cache: disk entry '%s' would serve lcp = %d of %zu tokens (f_keep = %.3f, f_sim = %.3f%s)\n",
            hit.path.c_str(), hit.lcp, tokens_new.size(), hit.keep, hit.sim, hit.partial ? ", partial" : "");

    return hit;
}

bool server_prompt_cache::disk_fetch(const disk_hit & hit, server_prompt_cache_state & out) {
    if (hit.lcp < 0) {
        return false;
    }

    // a fetch materializes the whole state at once, next to a loaded model whose ZFS ARC may
    // already hold most of the box: never spend more than half of what is available
    const size_t mem_avail = pcache_mem_available_bytes();
    if (mem_avail > 0 && hit.size > mem_avail / 2) {
        SRV_WRN("prompt cache: not fetching %.2f MiB from disk, %.2f GiB of memory available\n",
                hit.size / (1024.0 * 1024.0), mem_avail / (1024.0 * 1024.0 * 1024.0));
        return false;
    }

    SRV_INF("prompt cache: fetching entry from disk (lcp = %d, f_keep = %.3f, f_sim = %.3f%s)\n",
            hit.lcp, hit.keep, hit.sim, hit.partial ? ", keeping it" : "");

    // the read happens outside the lock: a multi-GiB read is seconds, and the writer thread
    // must not be blocked behind it
    FILE * file = fopen(hit.path.c_str(), "rb");
    if (file == nullptr) {
        SRV_WRN("prompt cache: cannot open '%s': %s\n", hit.path.c_str(), strerror(errno));
        disk_drop_path(hit.path);
        return false;
    }

    out.prompt.clear();
    out.data.main.clear();
    out.data.drft.clear();

    const int64_t t_read = ggml_time_us();

    bool corrupt = false;

    const bool ok = pcache_read_entry(file, disk_fingerprint, out, &corrupt);
    fclose(file);

    const int64_t t_read_us = ggml_time_us() - t_read;

    if (ok) {
        const double mb_s = t_read_us > 0 ? 1e6 * (double) hit.size / (double) t_read_us / (1024.0 * 1024.0) : 0.0;

        SRV_INF("prompt cache: read %s in %.2f ms (%.0f MB/s)\n",
                hit.path.c_str(), t_read_us / 1000.0, mb_s);
        note_read_rate(mb_s);
        disk_copy_identity(hit.path, out);
    }

    if (server_prompt_cache_disk_failed_read_unlinks(ok, corrupt)) {
        SRV_WRN("prompt cache: '%s' %s, dropping it\n", hit.path.c_str(),
                corrupt ? "does not match its checksum" : "is truncated");
        disk_drop_path(hit.path);

        return false;
    }


    return true;
}

void server_prompt_cache::disk_flush() {
    if (disk_dir.empty()) {
        return;
    }

    // stop both workers before the flush writes. the join waits out an in-flight read or
    // write; the deadline below covers only the backlog
    {
        std::lock_guard<std::mutex> lock(read_mu);
        on_read_ready = nullptr;
        disk_stop = true;
    }
    {
        std::lock_guard<std::mutex> lock(disk_mutex);
        disk_stop = true;
    }

    disk_cv.notify_all();
    read_cv.notify_all();

    if (read_thread.joinable()) {
        read_thread.join();
    }
    if (disk_thread.joinable()) {
        disk_thread.join();
    }

    const int64_t t_start = ggml_time_us();

    // whatever the writer still holds: it is not in RAM any more, so nobody else will save it
    for (;;) {
        server_prompt_cache_state state;

        {
            std::lock_guard<std::mutex> lock(disk_mutex);
            if (disk_queue.empty()) {
                break;
            }

            state = std::move(disk_queue.front());
            disk_queue.pop_front();
        }

        disk_write(state);

        if (ggml_time_us() - t_start > SERVER_PROMPT_CACHE_FLUSH_DEADLINE_US) {
            SRV_WRN("%s", "prompt cache: disk flush deadline reached, remaining entries dropped\n");
            break;
        }
    }

    size_t n_saved = 0;

    for (auto & state : states) {
        if (ggml_time_us() - t_start > SERVER_PROMPT_CACHE_FLUSH_DEADLINE_US) {
            SRV_WRN("prompt cache: disk flush deadline reached, saved %zu of %zu entries\n", n_saved, states.size());
            break;
        }

        disk_write(state);
        n_saved++;
    }

    SRV_INF("prompt cache: flushed %zu entries to '%s'\n", n_saved, disk_dir.c_str());
}

// best-effort MemAvailable in bytes; 0 when it cannot be determined
static size_t pcache_mem_available_bytes() {
    FILE * f = fopen("/proc/meminfo", "r");
    if (f == nullptr) {
        return 0;
    }

    char   line[256];
    size_t kib = 0;

    while (fgets(line, sizeof(line), f) != nullptr) {
        if (sscanf(line, "MemAvailable: %zu kB", &kib) == 1) {
            break;
        }
    }

    fclose(f);

    return kib * 1024;
}


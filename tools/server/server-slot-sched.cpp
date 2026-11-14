#include "server-slot-sched.h"

#include "common.h"
#include "llama.h"
#include "log.h"

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

server_prompt_cache_state * server_prompt_cache::alloc(const server_prompt & prompt, size_t state_size_tgt, size_t state_size_dft) {
    // first check if the current state is contained fully in the cache
    for (auto it = states.begin(); it != states.end(); ++it) {
        const int cur_lcp_len = it->prompt.tokens.get_common_prefix(prompt.tokens);

        if (cur_lcp_len == (int) prompt.tokens.size() && are_lora_equal(it->prompt.lora, prompt.lora)) {
            SRV_TRC("%s", " - prompt is already in the cache, skipping\n");
            return nullptr;
        }
    }

    // calculate checkpoints size to see if it will fit with the prompt
    size_t checkpoints_size = 0;
    for (const auto & ckpt : prompt.checkpoints) {
        checkpoints_size += ckpt->size();
    }

    const size_t state_size_new = state_size_tgt + state_size_dft + checkpoints_size;

    // skip over-limit entries to avoid disturbing the cache
    if (limit_size > 0 && state_size_new > limit_size) {
        SRV_WRN(" - prompt state size %.3f MiB exceeds cache size limit %.3f MiB, skipping\n",
                state_size_new / (1024.0 * 1024.0), limit_size / (1024.0 * 1024.0));
        return nullptr;
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
        // make room before allocating the new vectors to avoid breaching the limit
        while (!states.empty() && size() + state_size_new > limit_size) {
            if (disk_enabled()) {
                SRV_WRN(" - making room for prompt cache entry, moving oldest entry (size = %.3f MiB) to the disk tier\n",
                        states.front().size() / (1024.0 * 1024.0));

                disk_spill(states.front());
            } else {
                SRV_WRN(" - making room for prompt cache entry, removing oldest entry (size = %.3f MiB)\n",
                        states.front().size() / (1024.0 * 1024.0));
            }

            states.pop_front();
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
            /*.checkpoints =*/ prompt.checkpoints,
            /*.lora        =*/ prompt.lora,
        },
        /*.data   =*/ {
            /*.main =*/ std::move(state_data_tgt),
            /*.drft =*/ std::move(state_data_dft),
        },
    });

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

    if (!data.drft.empty()) {
        GGML_ASSERT(ctx_dft);

        const size_t size = data.drft.size();
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, data.drft.data(), size, id_slot, 0);
        if (n != size) {
            SRV_WRN("failed to restore state with size %zu\n", size);

            return false;
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

bool server_prompt_cache::load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot) {
    int lcp_best = 0;

    const auto it_found = find_better(prompt, tokens_new, lora_new, lcp_best);

    // an empty erase gives a mutable iterator to the same entry, its prompt is moved out below
    const auto it_best = states.erase(it_found, it_found);

    if (it_best != states.end()) {
        const int64_t t_start = ggml_time_us();

        if (!pcache_restore_state(it_best->data, ctx_tgt, ctx_dft, id_slot)) {
            return false;
        }

        SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_RESTORE__\n");

        // the whole cached prompt is a prefix of the new one
        if (lcp_best == (int) it_best->prompt.tokens.size()) {
            SRV_DBG("%s", "__TEST_TAG_PROMPT_CACHE_FULL_RESTORE__\n");
        }

        prompt = std::move(it_best->prompt);

        states.erase(it_best);

        SRV_TRC("slot %d: prompt cache restore took %.2f ms\n", id_slot, (ggml_time_us() - t_start) / 1000.0);
    } else if (disk_enabled() && lora_new.empty()) {
        // the RAM tier has nothing better; the disk tier holds what the RAM limit evicted
        server_prompt_cache_state fetched;

        if (disk_load(fetched, prompt, tokens_new)) {
            if (!pcache_restore_state(fetched.data, ctx_tgt, ctx_dft, id_slot)) {
                return false;
            }

            prompt = std::move(fetched.prompt);
        }
    }

    return true;
}

void server_prompt_cache::update() {
    if (limit_size > 0) {
        while (!states.empty() && size() > limit_size) {
            if (disk_enabled()) {
                SRV_WRN(" - cache size limit reached, moving oldest entry (size = %.3f MiB) to the disk tier\n",
                        states.front().size() / (1024.0 * 1024.0));

                disk_spill(states.front());
            } else {
                SRV_WRN(" - cache size limit reached, removing oldest entry (size = %.3f MiB)\n",
                        states.front().size() / (1024.0 * 1024.0));
            }

            states.pop_front();
        }
    }

    // average size per token
    const float size_per_token = std::max<float>(1.0f, float(size()) / (std::max<size_t>(1, n_tokens())));

    // dynamically increase the token limit if it can fit in the memory limit
    const size_t limit_tokens_cur = limit_size > 0 ? std::max<size_t>(limit_tokens, limit_size/size_per_token) : limit_tokens;

    if (limit_tokens > 0) {
        while (!states.empty() && n_tokens() > limit_tokens_cur) {
            SRV_WRN(" - cache token limit (%zu, est: %zu) reached, removing oldest entry (size = %.3f MiB)\n",
                    limit_tokens, limit_tokens_cur, states.front().size() / (1024.0 * 1024.0));

            states.pop_front();
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
            if (f_keep < 0.5f) {
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

static bool pcache_fwrite(FILE * file, const void * data, size_t size) {
    return size == 0 || fwrite(data, 1, size, file) == size;
}

static bool pcache_fwrite_u32(FILE * file, uint32_t value) {
    return pcache_fwrite(file, &value, sizeof(value));
}

static bool pcache_fwrite_u64(FILE * file, uint64_t value) {
    return pcache_fwrite(file, &value, sizeof(value));
}

static bool pcache_fread(FILE * file, void * data, size_t size) {
    return size == 0 || fread(data, 1, size, file) == size;
}

static bool pcache_fread_u32(FILE * file, uint32_t & value) {
    return pcache_fread(file, &value, sizeof(value));
}

static bool pcache_fread_u64(FILE * file, uint64_t & value) {
    return pcache_fread(file, &value, sizeof(value));
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
// file format: magic 'LGPC', u32 format version, u32 LLAMA_STATE_SEQ_VERSION, u64 context
// fingerprint, then one entry: u64 tokens blob length + blob (server_tokens serializer),
// u64 main state length + bytes, u64 draft state length + bytes, u32 context checkpoint
// count + per checkpoint: u64 n_tokens, u32 pos_min, u32 pos_max and the target, draft and
// speculative blobs as u64 length + bytes
//
static constexpr uint32_t SERVER_PROMPT_CACHE_ENTRY_MAGIC   = 0x4350474c; // 'LGPC' little-endian
static constexpr uint32_t SERVER_PROMPT_CACHE_ENTRY_VERSION = 3;

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

static bool pcache_write_entry(FILE * file, const server_prompt_cache_state & state) {
    std::vector<char> tokens;

    try {
        tokens = state.prompt.tokens.serialize();
    } catch (const std::exception & e) {
        SRV_WRN("skipping prompt cache entry with %d tokens that cannot be serialized: %s\n",
                state.prompt.n_tokens(), e.what());
        return false;
    }

    if (!pcache_fwrite_u64(file, tokens.size()) || !pcache_fwrite(file, tokens.data(), tokens.size()) ||
        !pcache_fwrite_u64(file, state.data.main.size()) || !pcache_fwrite(file, state.data.main.data(), state.data.main.size()) ||
        !pcache_fwrite_u64(file, state.data.drft.size()) || !pcache_fwrite(file, state.data.drft.data(), state.data.drft.size()) ||
        !pcache_fwrite_u32(file, (uint32_t) state.prompt.checkpoints.size())) {
        return false;
    }

    for (const auto & ckpt : state.prompt.checkpoints) {
        if (!pcache_fwrite_u64(file, (uint64_t) ckpt->n_tokens) ||
            !pcache_fwrite_u32(file, (uint32_t) ckpt->pos_min) ||
            !pcache_fwrite_u32(file, (uint32_t) ckpt->pos_max) ||
            !pcache_fwrite_u64(file, ckpt->data_tgt.size()) || !pcache_fwrite(file, ckpt->data_tgt.data(), ckpt->data_tgt.size()) ||
            !pcache_fwrite_u64(file, ckpt->data_dft.size()) || !pcache_fwrite(file, ckpt->data_dft.data(), ckpt->data_dft.size()) ||
            !pcache_fwrite_u64(file, ckpt->data_spec.size()) || !pcache_fwrite(file, ckpt->data_spec.data(), ckpt->data_spec.size())) {
            return false;
        }
    }

    return true;
}

// header + token blob only: what a scan needs to index a file without reading its state
static bool pcache_read_entry_header(FILE * file, uint64_t fingerprint, server_tokens & tokens) {
    uint32_t magic       = 0;
    uint32_t version     = 0;
    uint32_t seq_version = 0;
    uint64_t fp          = 0;

    if (!pcache_fread_u32(file, magic) || !pcache_fread_u32(file, version) ||
        !pcache_fread_u32(file, seq_version) || !pcache_fread_u64(file, fp)) {
        return false;
    }

    if (magic != SERVER_PROMPT_CACHE_ENTRY_MAGIC ||
        version != SERVER_PROMPT_CACHE_ENTRY_VERSION ||
        seq_version != (uint32_t) LLAMA_STATE_SEQ_VERSION ||
        fp != fingerprint) {
        return false;
    }

    uint64_t sz_tokens = 0;
    if (!pcache_fread_u64(file, sz_tokens) || sz_tokens == 0 || sz_tokens > (uint64_t) 1 << 32) {
        return false;
    }

    std::vector<char> buf(sz_tokens);
    if (!pcache_fread(file, buf.data(), sz_tokens)) {
        return false;
    }

    const llama_tokens packed((const llama_token *) buf.data(), (const llama_token *) (buf.data() + buf.size()));

    tokens = server_tokens::deserialize(packed, false);

    return true;
}

static bool pcache_read_entry(FILE * file, uint64_t fingerprint, server_prompt_cache_state & cur) {
    uint32_t magic       = 0;
    uint32_t version     = 0;
    uint32_t seq_version = 0;
    uint64_t fp          = 0;

    if (!pcache_fread_u32(file, magic) || !pcache_fread_u32(file, version) ||
        !pcache_fread_u32(file, seq_version) || !pcache_fread_u64(file, fp)) {
        return false;
    }

    if (magic != SERVER_PROMPT_CACHE_ENTRY_MAGIC ||
        version != SERVER_PROMPT_CACHE_ENTRY_VERSION ||
        seq_version != (uint32_t) LLAMA_STATE_SEQ_VERSION ||
        fp != fingerprint) {
        return false;
    }

    uint64_t sz_tokens = 0;
    if (!pcache_fread_u64(file, sz_tokens) || sz_tokens == 0 || sz_tokens > (uint64_t) 1 << 32) {
        return false;
    }

    std::vector<char> buf(sz_tokens);
    if (!pcache_fread(file, buf.data(), sz_tokens)) {
        return false;
    }

    const llama_tokens packed((const llama_token *) buf.data(), (const llama_token *) (buf.data() + buf.size()));

    cur.prompt.tokens = server_tokens::deserialize(packed, false);

    uint64_t sz_main = 0;
    if (!pcache_fread_u64(file, sz_main) || sz_main > (uint64_t) 1 << 34) {
        return false;
    }

    cur.data.main.resize(sz_main);
    if (sz_main > 0 && !pcache_fread(file, cur.data.main.data(), sz_main)) {
        return false;
    }

    uint64_t sz_drft = 0;
    if (!pcache_fread_u64(file, sz_drft) || sz_drft > (uint64_t) 1 << 34) {
        return false;
    }

    cur.data.drft.resize(sz_drft);
    if (sz_drft > 0 && !pcache_fread(file, cur.data.drft.data(), sz_drft)) {
        return false;
    }

    uint32_t n_ckpts = 0;
    if (!pcache_fread_u32(file, n_ckpts) || n_ckpts > 4096) {
        return false;
    }

    for (uint32_t i = 0; i < n_ckpts; i++) {
        common_prompt_checkpoint ckpt;

        uint64_t ckpt_tokens  = 0;
        uint32_t ckpt_pos_min = 0;
        uint32_t ckpt_pos_max = 0;
        if (!pcache_fread_u64(file, ckpt_tokens) ||
            !pcache_fread_u32(file, ckpt_pos_min) ||
            !pcache_fread_u32(file, ckpt_pos_max)) {
            return false;
        }

        ckpt.n_tokens = (int64_t) ckpt_tokens;
        ckpt.pos_min  = (llama_pos) ckpt_pos_min;
        ckpt.pos_max  = (llama_pos) ckpt_pos_max;

        auto read_blob = [&](auto & dst) {
            uint64_t sz = 0;
            if (!pcache_fread_u64(file, sz) || sz > (uint64_t) 1 << 34) {
                return false;
            }

            dst.resize(sz);

            return sz == 0 || pcache_fread(file, dst.data(), sz);
        };

        if (!read_blob(ckpt.data_tgt) || !read_blob(ckpt.data_dft) || !read_blob(ckpt.data_spec)) {
            return false;
        }

        cur.prompt.checkpoints.push_back(std::make_shared<const common_prompt_checkpoint>(std::move(ckpt)));
    }

    return true;
}

server_prompt_cache::~server_prompt_cache() {
    {
        std::lock_guard<std::mutex> lock(disk_mutex);
        disk_stop = true;
    }

    disk_cv.notify_all();

    if (disk_thread.joinable()) {
        disk_thread.join();
    }
}

bool server_prompt_cache::disk_init(const std::string & dir, size_t limit, const std::string & model_path,
        const llama_model * model, const llama_context * ctx, ggml_type type_k, ggml_type type_v) {
    disk_dir   = dir;
    disk_limit = limit;
    disk_used  = 0;

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
            // an orphan .bin.tmp: the writer died between the write and the rename; it is
            // never loaded, never counted, never evicted
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
        const int64_t use = ec ? 0 : (int64_t) mt.time_since_epoch().count();

        disk_states.push_back({ std::move(tokens), path.string(), (size_t) sz, use });
        disk_used += (size_t) sz;
    }

    std::sort(disk_states.begin(), disk_states.end(),
            [](const auto & a, const auto & b) { return a.last_use_us < b.last_use_us; });

    disk_prune();

    disk_thread = std::thread([this] { disk_writer(); });

    SRV_INF("prompt cache: disk tier '%s' holds %zu entries (%.2f GiB, limit %.2f GiB), %zu stale removed, %zu orphan tmp swept\n",
            disk_dir.c_str(), disk_states.size(), disk_used / (1024.0 * 1024.0 * 1024.0),
            disk_limit / (1024.0 * 1024.0 * 1024.0), n_stale, n_swept);

    return true;
}

bool server_prompt_cache::disk_enabled() const {
    return !disk_dir.empty();
}

void server_prompt_cache::disk_writer() {
    for (;;) {
        server_prompt_cache_state state;

        {
            std::unique_lock<std::mutex> lock(disk_mutex);
            disk_cv.wait(lock, [this] { return disk_stop || !disk_queue.empty(); });

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

    // the writer is behind: keep the newest few and let the rest go, the disk tier is a cache
    while (disk_queue.size() >= 8) {
        SRV_WRN("prompt cache: disk tier is behind (%zu queued), dropping the oldest queued state\n",
                disk_queue.size());
        disk_queue.pop_front();
    }

    disk_queue.push_back(std::move(state));
    disk_cv.notify_one();
}

void server_prompt_cache::disk_write(server_prompt_cache_state & state) {
    if (disk_dir.empty() || state.prompt.tokens.size() == 0) {
        return;
    }

    const std::string name = string_format("%d-%016" PRIx64 ".bin",
            state.prompt.n_tokens(), pcache_tokens_hash(state.prompt.tokens));
    const std::string path = disk_dir + "/" + name;
    const std::string tmp  = path + ".tmp";

    FILE * file = fopen(tmp.c_str(), "wb");
    if (file == nullptr) {
        SRV_WRN("prompt cache: cannot open '%s' for writing: %s\n", tmp.c_str(), strerror(errno));
        return;
    }

    bool ok = pcache_fwrite_u32(file, SERVER_PROMPT_CACHE_ENTRY_MAGIC) &&
              pcache_fwrite_u32(file, SERVER_PROMPT_CACHE_ENTRY_VERSION) &&
              pcache_fwrite_u32(file, (uint32_t) LLAMA_STATE_SEQ_VERSION) &&
              pcache_fwrite_u64(file, disk_fingerprint) &&
              pcache_write_entry(file, state);

    if (ok && fflush(file) != 0) {
        ok = false;
    }

#ifndef _WIN32
    if (ok && fsync(fileno(file)) != 0) {
        ok = false;
    }
#endif

    fclose(file);

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

    const auto sz = std::filesystem::file_size(path, ec);
    if (ec) {
        return;
    }

    SRV_TRC("prompt cache: wrote entry '%s' (%d tokens, %.2f MiB)\n",
            name.c_str(), state.prompt.n_tokens(), (double) sz / (1024.0 * 1024.0));

    // the state itself goes out of RAM here; only its tokens stay, for matching
    std::lock_guard<std::mutex> lock(disk_mutex);

    const int64_t now = ggml_time_us();

    for (auto & e : disk_states) {
        if (e.path == path) {
            disk_used    += (size_t) sz - e.size;
            e.size        = (size_t) sz;
            e.last_use_us = now;
            return;
        }
    }

    disk_states.push_back({ std::move(state.prompt.tokens), path, (size_t) sz, now });
    disk_used += (size_t) sz;

    disk_prune();
}

// assumes disk_mutex
void server_prompt_cache::disk_prune() {
    while (disk_limit > 0 && disk_used > disk_limit && !disk_states.empty()) {
        auto & victim = disk_states.front();

        SRV_INF("prompt cache: disk tier limit reached, removing '%s' (%zu tokens, %.2f MiB)\n",
                victim.path.c_str(), victim.tokens.size(), victim.size / (1024.0 * 1024.0));

        std::error_code ec;
        std::filesystem::remove(victim.path, ec);

        disk_used -= victim.size;
        disk_states.erase(disk_states.begin());
    }
}

bool server_prompt_cache::disk_load(server_prompt_cache_state & out, const server_prompt & prompt_slot, const server_tokens & tokens_new) {
    if (disk_dir.empty()) {
        return false;
    }

    std::string path;

    {
        std::lock_guard<std::mutex> lock(disk_mutex);

        // the same rule the RAM tier uses: the best entry that keeps at least a quarter of
        // itself, beating both the slot's current prompt and every other candidate
        const int lcp_slot = prompt_slot.tokens.size() > 0 ? prompt_slot.tokens.get_common_prefix(tokens_new) : 0;

        float f_keep_best = prompt_slot.tokens.size() > 0 ? float(lcp_slot) / prompt_slot.tokens.size() : -1.0f;
        float f_sim_best  = float(lcp_slot) / tokens_new.size();

        disk_state * best = nullptr;

        for (auto & e : disk_states) {
            const int lcp = e.tokens.get_common_prefix(tokens_new);

            const float f_keep = e.tokens.size() > 0 ? float(lcp) / e.tokens.size() : 0.0f;
            const float f_sim  = float(lcp) / tokens_new.size();

            SRV_TRC("prompt cache: disk entry %zu tokens, lcp = %d, f_keep = %.3f, f_sim = %.3f\n",
                    e.tokens.size(), lcp, f_keep, f_sim);

            if (f_keep < 0.25f) {
                continue;
            }

            if (f_keep_best < f_keep && f_sim_best < f_sim) {
                f_keep_best = f_keep;
                f_sim_best  = f_sim;
                best        = &e;
            }
        }

        if (best == nullptr) {
            return false;
        }

        // a fetch materializes the whole state at once, next to a loaded model whose ZFS ARC
        // may already hold most of the box: never spend more than half of what is available
        const size_t mem_avail = pcache_mem_available_bytes();
        if (mem_avail > 0 && best->size > mem_avail / 2) {
            SRV_WRN("prompt cache: not fetching %.2f MiB from disk, %.2f GiB of memory available\n",
                    best->size / (1024.0 * 1024.0), mem_avail / (1024.0 * 1024.0 * 1024.0));
            return false;
        }

        SRV_INF("prompt cache: fetching entry from disk (f_keep = %.3f, f_sim = %.3f)\n", f_keep_best, f_sim_best);

        path = best->path;
    }

    // the read happens outside the lock: a multi-GiB read is seconds, and the writer thread
    // must not be blocked behind it
    FILE * file = fopen(path.c_str(), "rb");
    if (file == nullptr) {
        SRV_WRN("prompt cache: cannot open '%s': %s\n", path.c_str(), strerror(errno));
        return false;
    }

    out.prompt.clear();
    out.data.main.clear();
    out.data.drft.clear();

    const bool ok = pcache_read_entry(file, disk_fingerprint, out);
    fclose(file);

    if (!ok) {
        SRV_WRN("prompt cache: '%s' is truncated, dropping it\n", path.c_str());
        return false;
    }

    // the state is about to be in the slot, not in two places: drop the file and its index
    // entry now that the read succeeded
    {
        std::lock_guard<std::mutex> lock(disk_mutex);

        for (auto it = disk_states.begin(); it != disk_states.end(); ++it) {
            if (it->path == path) {
                disk_used -= it->size;
                disk_states.erase(it);
                break;
            }
        }
    }

    std::error_code ec;
    std::filesystem::remove(path, ec);

    return true;
}

void server_prompt_cache::disk_flush() {
    if (disk_dir.empty()) {
        return;
    }

    // this runs on the main thread at shutdown: stop the writer first so the I/O below is
    // single-threaded and the deadline covers all of it
    {
        std::lock_guard<std::mutex> lock(disk_mutex);
        disk_stop = true;
    }

    disk_cv.notify_all();

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


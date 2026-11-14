#pragma once

#include "server-common.h"
#include "server-task.h"
#include "server-prompt-cache-disk.h"
#include "server-prompt-cache-value.h"
#include "server-prompt-cache-spread.h"

#include <condition_variable>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

// which prompt goes to which slot: slot prompt state, prompt cache and slot selection

// what a cached prompt can give to a new prompt
struct server_prompt_reuse {
    int32_t   n_past   = 0; // tokens to keep
    llama_pos pos_next = 0;

    // restore this checkpoint first, nullptr when the live state is used as it is
    std::shared_ptr<const common_prompt_checkpoint> ckpt;
};

struct server_prompt {
    server_tokens tokens;

    // a checkpoint does not change after it is created, so the prompt cache shares it with the slot instead of a copy
    std::list<std::shared_ptr<const common_prompt_checkpoint>> checkpoints;

    // the lora config that the tokens were processed with, it follows the state to the prompt cache and back
    // TODO: slot files do not store it
    std::vector<common_adapter_lora_info> lora;

    void clear() {
        tokens.clear();
        checkpoints.clear();
    }

    int n_tokens() const {
        return tokens.size();
    }

    server_prompt clone() const {
        return server_prompt {
            tokens.clone(),
            checkpoints,
            lora,
        };
    }

    // n_prefix: tokens shared with the new prompt, must be > 0
    // has_new_tokens: the new prompt is longer than n_prefix
    // n_swa: 0 if a state must end right before the next token (no SWA, or recurrent)
    // pos_min: min position of the live state
    // n_past == 0 without a checkpoint means that nothing can be reused
    server_prompt_reuse get_reuse(int32_t n_prefix, bool has_new_tokens, int32_t n_swa, llama_pos pos_min) const;
};

// fault in the pages of a freshly allocated buffer from several threads
void server_state_prefault(void * p, size_t size);

// free a buffer from server_state_allocator; a large one is unmapped on a background thread
void server_state_release(uint8_t * p, size_t size);

// buffers freed off the caller's thread so far
size_t server_state_released();

// allocator for the KV state buffers: they are large (GiBs) and fully overwritten by the
// device->host copy right after allocation, so the zero fill of std::allocator is skipped
// and the pages are faulted in from several threads instead of one by one from the copy
// (first touch of fresh pages: 2.4 GB/s single-threaded, 15 GB/s from 8 threads, measured
// on a 16-thread Zen 3)
template <typename T>
struct server_state_allocator : std::allocator<T> {
    template <typename U> struct rebind { using other = server_state_allocator<U>; };

    server_state_allocator() = default;
    template <typename U> server_state_allocator(const server_state_allocator<U> &) {}

    T * allocate(size_t n) {
        T * p = std::allocator<T>::allocate(n);
        server_state_prefault(p, n * sizeof(T));
        return p;
    }

    void deallocate(T * p, size_t n) {
        static_assert(alignof(T) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__, "server_state_release frees with the default alignment");
        server_state_release((uint8_t *) p, n * sizeof(T));
    }

    template <typename U> void construct(U * p) noexcept(std::is_nothrow_default_constructible<U>::value) {
        ::new ((void *) p) U;
    }

    template <typename U, typename... Args> void construct(U * p, Args &&... args) {
        ::new ((void *) p) U(std::forward<Args>(args)...);
    }
};

using server_state_buffer = std::vector<uint8_t, server_state_allocator<uint8_t>>;

struct server_prompt_data {
    server_state_buffer main;
    server_state_buffer drft;

    size_t size() const {
        return main.size() + drft.size();
    }
};

struct server_prompt_cache_state {
    server_prompt prompt;
    server_prompt_data data;

    // when this state was last used: the tier prices it by how long ago that was, so an entry
    // nothing continues decays out of the tier and one asked for a minute ago keeps its place
    int64_t last_use_us = 0;

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

// what each tier offered one request, and what was taken. the near-cold line prints it, so a
// prompt the cache barely served says why: neither tier had a head for it, or the head was
// there and the keep gate or the RAM-first rule refused it
struct server_prompt_cache_decision {
    int  slot_lcp    = -1;    // tokens the slot's own prompt already shares with the request
    int  ram_count   = 0;     // entries in the RAM tier
    int  ram_best    = -1;    // best lcp among the RAM entries that pass the keep gate
    int  ram_seen    = -1;    // best lcp over every RAM entry, gate included
    int  disk_count  = 0;     // entries in the disk tier
    int  disk_best   = -1;    // best lcp among the disk entries that pass the keep gate
    int  disk_seen   = -1;    // best lcp over every disk entry, gate included
    int  chosen_lcp  = -1;    // lcp of the entry that was restored
    bool chosen_disk = false; // the restored entry came off the disk
    bool chosen_partial = false; // ... and reused under a quarter of it, so it was kept
};

struct server_prompt_cache {
    ~server_prompt_cache();

    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens, int32_t max_checkpoints = 0, int32_t min_step = 8192,
            bool size_in_slots = false, size_t slot_tokens = 0) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
        this->max_checkpoints = max_checkpoints > 0 ? (size_t) max_checkpoints : 0;
        this->min_step = min_step > 0 ? (size_t) min_step : 8192;
        this->size_in_slots = size_in_slots;
        this->slot_tokens = slot_tokens > 0 ? slot_tokens : limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit. in slots mode (--cache-ram-slots) this is derived instead: the
    // cost of the states the tier has actually seen, times the token budget (see update())
    size_t limit_size = 0;

    // the byte limit is the cost of limit_tokens tokens, measured rather than assumed: the
    // model's KV depends on its architecture, the KV types, the checkpoint density and the
    // draft context, so a number fixed at startup is either too small (thrash, which is a
    // frozen server) or too generous (the cgroup was set from it). 0 disables the derivation
    // and the bit limit is the authority
    bool   size_in_slots = false;

    // one slot's full context: a state cannot serve more than this and no entry is worth more
    size_t slot_tokens = 0;

    // bytes per token, the max over the states this tier has saved. the derived byte limit is
    // limit_tokens * this, so it can only ever be generous by how many checkpoints an entry
    // carries -- never by how much memory the model needs for a token
    float  density_max = 0.0f;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    // max number of context checkpoints kept in a stored entry, 0 = keep all
    size_t max_checkpoints = 0;

    // min spacing between context checkpoints, used to place the kept ones
    size_t min_step = 8192;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // the cached state that is a better starting point than `prompt` for `tokens_new`, states.end() if none
    // lcp: the common prefix of that state with `tokens_new`
    // lora_new: the lora config of the new prompt, the states made with another config are skipped
    std::list<server_prompt_cache_state>::const_iterator find_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, int & lcp) const;

    bool has_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new) const;

    // restore the entry with the longest usable head, from either tier, into the slot `id_slot`
    bool load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot,
            server_prompt_cache_decision * decision = nullptr);

    // what a state is worth keeping: the tokens a future request takes from it per byte it
    // holds, decayed by the time since it was last used
    float value(const server_prompt_cache_state & state) const;

    // the state to make room with: the least valuable one, not the oldest
    std::list<server_prompt_cache_state>::iterator victim();


    void update();

    uint64_t context_fingerprint(const llama_model * model, const std::string & model_path, const llama_context * ctx, ggml_type type_k, ggml_type type_v) const;

    // level 2 on disk (--cache-disk-dir): one file per entry, states live there and
    // disk_states is the RAM index (tokens + path) that matching runs against
    struct disk_state {
        server_tokens tokens;
        std::string   path;
        size_t        size        = 0; // bytes on disk
        int64_t       last_use_us = 0;
    };

    bool disk_init(const std::string & dir, size_t limit, const std::string & model_path,
            const llama_model * model, const llama_context * ctx, ggml_type type_k, ggml_type type_v);

    bool disk_enabled() const;

    // write every state the cache still holds (and the writer's backlog) to the disk tier;
    // best effort under a deadline, meant for shutdown
    void disk_flush();

    // hand one state to the writer thread; the caller's copy is gone afterwards
    void disk_spill(server_prompt_cache_state & state);

    // a disk entry that survived the keep gate, before its state is read
    struct disk_hit {
        std::string path;
        size_t      size = 0;
        int         lcp  = -1;     // < 0: the tier held nothing that beats the slot
        float       keep = 0.0f;   // lcp / entry tokens
        float       sim  = 0.0f;   // lcp / request tokens
        bool        partial = false; // the head reuses under a quarter of the entry
        int         first_rung = 0;  // the first checkpoint the entry can land on
    };

    // which entry the disk tier would serve this request with, without materializing it
    disk_hit disk_find(const server_prompt & prompt_slot, const server_tokens & tokens_new, server_prompt_cache_decision & decision);

    // read `hit` off the disk into `out` and remove it: the state then lives in the slot, not
    // in two places. false leaves the tier untouched
    bool disk_fetch(const disk_hit & hit, server_prompt_cache_state & out);

    std::string disk_dir;
    size_t      disk_limit       = 0;
    size_t      disk_used        = 0;
    uint64_t    disk_fingerprint = 0;

    std::vector<disk_state> disk_states;

private:
    void disk_writer();
    void disk_write(server_prompt_cache_state & state);
    void disk_prune();

    std::thread             disk_thread;
    std::mutex              disk_mutex;
    std::condition_variable disk_cv;
    std::list<server_prompt_cache_state> disk_queue;
    bool                    disk_stop = false;
};

// what the scheduler needs to know about a slot
struct server_slot_info {
    int id = -1;
    const server_task * task = nullptr; // only for logging

    bool is_processing = false;
    int64_t t_last_used = -1;

    const server_prompt * prompt = nullptr;
};

struct server_slot_plan {
    // -1 if no slot is available; a busy slot if the task asks for it by id, the caller defers the task
    int id_slot = -1;

    // save the slot state to the prompt cache, then load the best match for the task
    bool update_cache = false;
};

// decides only, the caller applies the plan
struct server_slot_sched {
    float slot_prompt_similarity = 0.0f;

    // idle slots that lose the least by being reused: empty slots first, then the least recently used ones
    std::vector<int> get_free_slots(const std::vector<server_slot_info> & slots, size_t n_slots_needed, int exclude_id_slot) const;

    // lora: the lora config of the task
    // prompt_cache: nullptr if disabled
    server_slot_plan select(const std::vector<server_slot_info> & slots, const server_task & task, const std::vector<common_adapter_lora_info> & lora, const server_prompt_cache * prompt_cache) const;
};

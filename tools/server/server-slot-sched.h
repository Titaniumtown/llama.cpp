#pragma once

#include "server-common.h"
#include "server-task.h"

#include <list>
#include <memory>
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

    size_t size() const {
        size_t res = data.size();

        for (const auto & ckpt : prompt.checkpoints) {
            res += ckpt->size();
        }

        return res;
    }
};

struct server_prompt_cache {
    server_prompt_cache(int32_t limit_size_mib, size_t limit_tokens) {
        this->limit_size   = 1024ull*1024ull*(limit_size_mib < 0 ? 0 : limit_size_mib);
        this->limit_tokens = limit_tokens;
    }

    std::list<server_prompt_cache_state> states;

    // in bytes, 0 = no limit
    size_t limit_size = 0;

    // in tokens, 0 = no limit
    size_t limit_tokens = 0;

    size_t size() const;

    size_t n_tokens() const;

    server_prompt_cache_state * alloc(const server_prompt & prompt, size_t state_size_main, size_t state_size_drft);

    // the cached state that is a better starting point than `prompt` for `tokens_new`, states.end() if none
    // lcp: the common prefix of that state with `tokens_new`
    // lora_new: the lora config of the new prompt, the states made with another config are skipped
    std::list<server_prompt_cache_state>::const_iterator find_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, int & lcp) const;

    bool has_better(const server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new) const;

    // restore the result of find_better into the slot `id_slot`
    bool load(server_prompt & prompt, const server_tokens & tokens_new, const std::vector<common_adapter_lora_info> & lora_new, llama_context * ctx_tgt, llama_context * ctx_dft, int32_t id_slot);

    void update();
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

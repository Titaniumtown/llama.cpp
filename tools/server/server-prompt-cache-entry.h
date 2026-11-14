#pragma once

#include "server-slot-sched.h"

#include "hash/xxhash/xxhash.h"

#include <cstdio>
#include <vector>

//
// the format of an entry in the disk tier
//
// magic 'LGPC', u32 format version, u32 LLAMA_STATE_SEQ_VERSION, u64 context
// fingerprint, then one entry: u64 tokens blob length + blob (server_tokens serializer),
// u64 main state length + bytes, u64 draft state length + bytes, u32 context checkpoint
// count + per checkpoint: u64 n_tokens, u32 pos_min, u32 pos_max and the target, draft and
// speculative blobs as u64 length + bytes. version 4 appends the trailer: the checksum of
// the payload, written after it
//
// the trailer is what makes the write cheap: the entry does not have to be on the device
// before it is published, because a payload the device did not take whole does not match
// its checksum and is refused when it is read back
//
constexpr uint32_t SERVER_PROMPT_CACHE_ENTRY_MAGIC   = 0x4350474c; // 'LGPC' little-endian
constexpr uint32_t SERVER_PROMPT_CACHE_ENTRY_VERSION = 4;

constexpr size_t SERVER_PROMPT_CACHE_ENTRY_HEADER_SIZE  = 20;
constexpr size_t SERVER_PROMPT_CACHE_ENTRY_TRAILER_SIZE = 8;

inline bool pcache_fwrite(FILE * file, const void * data, size_t size) {
    return size == 0 || fwrite(data, 1, size, file) == size;
}

inline bool pcache_fwrite_u32(FILE * file, uint32_t value) {
    return pcache_fwrite(file, &value, sizeof(value));
}

inline bool pcache_fwrite_u64(FILE * file, uint64_t value) {
    return pcache_fwrite(file, &value, sizeof(value));
}

inline bool pcache_fread(FILE * file, void * data, size_t size) {
    return size == 0 || fread(data, 1, size, file) == size;
}

inline bool pcache_fread_u32(FILE * file, uint32_t & value) {
    return pcache_fread(file, &value, sizeof(value));
}

inline bool pcache_fread_u64(FILE * file, uint64_t & value) {
    return pcache_fread(file, &value, sizeof(value));
}

// the checksum of an entry's payload: the write hashes what it writes and the read hashes
// what it reads, so a payload a power cut tore is rejected instead of restored as zeroes
struct pcache_hasher {
    XXH3_state_t * state = nullptr;

    bool init() {
        state = XXH3_createState();
        return state != nullptr && XXH3_64bits_reset(state) == XXH_OK;
    }

    void release() {
        if (state != nullptr) {
            XXH3_freeState(state);
            state = nullptr;
        }
    }

    bool add(const void * data, size_t size) {
        return size == 0 || (state != nullptr && XXH3_64bits_update(state, data, size) == XXH_OK);
    }

    uint64_t digest() const {
        return state != nullptr ? XXH3_64bits_digest(state) : 0;
    }
};

// the writer of an entry: it hashes every byte of the payload as it writes it, so the
// checksum in the trailer is the checksum of the bytes that were handed to the file.
// large puts are chunked so a pace hook can keep the dirty set inside a window.
struct pcache_writer {
    FILE *         file   = nullptr;
    pcache_hasher * hasher = nullptr;
    void (* pace)(void * ctx, size_t written) = nullptr;
    void *         pace_ctx = nullptr;
    size_t         written  = 0;

    bool put(const void * data, size_t size) {
        if (size == 0) {
            return true;
        }

        const auto * p = (const uint8_t *) data;
        size_t off = 0;
        const size_t chunk = 32ull << 20;

        while (off < size) {
            const size_t n = size - off < chunk ? size - off : chunk;

            if (hasher != nullptr && !hasher->add(p + off, n)) {
                return false;
            }
            if (fwrite(p + off, 1, n, file) != n) {
                return false;
            }

            written += n;
            off     += n;

            if (pace != nullptr) {
                pace(pace_ctx, written);
            }
        }

        return true;
    }

    bool put_u32(uint32_t value) { return put(&value, sizeof(value)); }
    bool put_u64(uint64_t value) { return put(&value, sizeof(value)); }
};

// the reader of an entry: it hashes every byte of the payload as it reads it, so the
// checksum in the trailer can be checked against what was actually read
struct pcache_reader {
    FILE *         file   = nullptr;
    pcache_hasher * hasher = nullptr;

    bool get(void * data, size_t size) {
        if (size == 0) {
            return true;
        }
        if (!pcache_fread(file, data, size)) {
            return false;
        }
        return hasher == nullptr || hasher->add(data, size);
    }

    bool get_u32(uint32_t & value) { return get(&value, sizeof(value)); }
    bool get_u64(uint64_t & value) { return get(&value, sizeof(value)); }
};

// the bytes an entry with this state occupies: the header, the payload and the trailer
inline uint64_t pcache_entry_size(const server_prompt_cache_state & state, size_t tokens_size) {
    uint64_t size = SERVER_PROMPT_CACHE_ENTRY_HEADER_SIZE + SERVER_PROMPT_CACHE_ENTRY_TRAILER_SIZE;

    size += 8 + tokens_size;
    size += 8 + state.data.main.size();
    size += 8 + state.data.drft.size();
    size += 4;

    for (const auto & ckpt : state.prompt.checkpoints) {
        size += 8 + 4 + 4;
        size += 8 + ckpt->data_tgt.size();
        size += 8 + ckpt->data_dft.size();
        size += 8 + ckpt->data_spec.size();
    }

    return size;
}

inline bool pcache_write_entry(pcache_writer & writer, const server_prompt_cache_state & state, const std::vector<char> & tokens) {
    if (!writer.put_u64(tokens.size()) || !writer.put(tokens.data(), tokens.size()) ||
        !writer.put_u64(state.data.main.size()) || !writer.put(state.data.main.data(), state.data.main.size()) ||
        !writer.put_u64(state.data.drft.size()) || !writer.put(state.data.drft.data(), state.data.drft.size()) ||
        !writer.put_u32((uint32_t) state.prompt.checkpoints.size())) {
        return false;
    }

    for (const auto & ckpt : state.prompt.checkpoints) {
        if (!writer.put_u64((uint64_t) ckpt->n_tokens) ||
            !writer.put_u32((uint32_t) ckpt->pos_min) ||
            !writer.put_u32((uint32_t) ckpt->pos_max) ||
            !writer.put_u64(ckpt->data_tgt.size()) || !writer.put(ckpt->data_tgt.data(), ckpt->data_tgt.size()) ||
            !writer.put_u64(ckpt->data_dft.size()) || !writer.put(ckpt->data_dft.data(), ckpt->data_dft.size()) ||
            !writer.put_u64(ckpt->data_spec.size()) || !writer.put(ckpt->data_spec.data(), ckpt->data_spec.size())) {
            return false;
        }
    }

    return true;
}

// header + token blob only: what a scan needs to index a file without reading its state
inline bool pcache_read_entry_header(FILE * file, uint64_t fingerprint, server_tokens & tokens) {
    uint32_t magic       = 0;
    uint32_t version     = 0;
    uint32_t seq_version = 0;
    uint64_t fp          = 0;

    if (!pcache_fread_u32(file, magic) || !pcache_fread_u32(file, version) ||
        !pcache_fread_u32(file, seq_version) || !pcache_fread_u64(file, fp)) {
        return false;
    }

    // version 3 entries were written with an fsync and carry no checksum; version 4
    // entries do and are checked when they are read back
    if (magic != SERVER_PROMPT_CACHE_ENTRY_MAGIC ||
        version < 3 || version > SERVER_PROMPT_CACHE_ENTRY_VERSION ||
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

// reads an entry back. corrupt is set when the payload does not match its checksum: the
// entry is then not a state but whatever the device took of one, and it is dropped
inline bool pcache_read_entry(FILE * file, uint64_t fingerprint, server_prompt_cache_state & cur, bool * corrupt = nullptr) {
    uint32_t magic       = 0;
    uint32_t version     = 0;
    uint32_t seq_version = 0;
    uint64_t fp          = 0;

    if (!pcache_fread_u32(file, magic) || !pcache_fread_u32(file, version) ||
        !pcache_fread_u32(file, seq_version) || !pcache_fread_u64(file, fp)) {
        return false;
    }

    if (magic != SERVER_PROMPT_CACHE_ENTRY_MAGIC ||
        version < 3 || version > SERVER_PROMPT_CACHE_ENTRY_VERSION ||
        seq_version != (uint32_t) LLAMA_STATE_SEQ_VERSION ||
        fp != fingerprint) {
        return false;
    }

    // the payload is hashed as it is read; version 4 closes it with the checksum
    pcache_hasher hasher;

    if (version >= 4 && !hasher.init()) {
        return false;
    }

    pcache_reader reader;
    reader.file   = file;
    reader.hasher = version >= 4 ? &hasher : nullptr;

    uint64_t sz_tokens = 0;
    if (!reader.get_u64(sz_tokens) || sz_tokens == 0 || sz_tokens > (uint64_t) 1 << 32) {
        return false;
    }

    std::vector<char> buf(sz_tokens);
    if (!reader.get(buf.data(), sz_tokens)) {
        return false;
    }

    const llama_tokens packed((const llama_token *) buf.data(), (const llama_token *) (buf.data() + buf.size()));

    cur.prompt.tokens = server_tokens::deserialize(packed, false);

    uint64_t sz_main = 0;
    if (!reader.get_u64(sz_main) || sz_main > (uint64_t) 1 << 34) {
        return false;
    }

    cur.data.main.resize(sz_main);
    if (sz_main > 0 && !reader.get(cur.data.main.data(), sz_main)) {
        return false;
    }

    uint64_t sz_drft = 0;
    if (!reader.get_u64(sz_drft) || sz_drft > (uint64_t) 1 << 34) {
        return false;
    }

    cur.data.drft.resize(sz_drft);
    if (sz_drft > 0 && !reader.get(cur.data.drft.data(), sz_drft)) {
        return false;
    }

    uint32_t n_ckpts = 0;
    if (!reader.get_u32(n_ckpts) || n_ckpts > 4096) {
        return false;
    }

    for (uint32_t i = 0; i < n_ckpts; i++) {
        common_prompt_checkpoint ckpt;

        uint64_t ckpt_tokens  = 0;
        uint32_t ckpt_pos_min = 0;
        uint32_t ckpt_pos_max = 0;
        if (!reader.get_u64(ckpt_tokens) ||
            !reader.get_u32(ckpt_pos_min) ||
            !reader.get_u32(ckpt_pos_max)) {
            return false;
        }

        ckpt.n_tokens = (int64_t) ckpt_tokens;
        ckpt.pos_min  = (llama_pos) ckpt_pos_min;
        ckpt.pos_max  = (llama_pos) ckpt_pos_max;

        auto read_blob = [&](auto & dst) {
            uint64_t sz = 0;
            if (!reader.get_u64(sz) || sz > (uint64_t) 1 << 34) {
                return false;
            }

            dst.resize(sz);

            return sz == 0 || reader.get(dst.data(), sz);
        };

        if (!read_blob(ckpt.data_tgt) || !read_blob(ckpt.data_dft) || !read_blob(ckpt.data_spec)) {
            return false;
        }

        cur.prompt.checkpoints.push_back(std::make_shared<const common_prompt_checkpoint>(std::move(ckpt)));
    }

    if (version >= 4) {
        // the trailer is the checksum of the payload: a payload the device did not take
        // whole is refused here instead of being restored as zeroes
        uint64_t checksum = 0;
        if (!pcache_fread_u64(file, checksum) || checksum != hasher.digest()) {
            if (corrupt != nullptr) {
                *corrupt = true;
            }
            return false;
        }
    }

    return true;
}

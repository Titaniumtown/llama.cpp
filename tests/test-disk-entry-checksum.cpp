// A disk tier entry round trip: what the writer wrote, the reader reads back, and an entry
// whose payload the device did not take whole is refused instead of being restored.
//
// The checksum is what lets the write skip the wait for the device: the entry is published
// as soon as the bytes are in the page cache, so a power cut can leave a payload that is
// not the state it claims to be. This is the property that keeps that safe.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -Icommon -Iinclude -Iggml/include -Itools/mtmd \
//       -o /tmp/t tests/test-disk-entry-checksum.cpp && /tmp/t
#include "server-prompt-cache-entry.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

static const uint64_t FINGERPRINT = 0x0123456789abcdefull;

static server_prompt_cache_state make_state() {
    server_prompt_cache_state state;

    llama_tokens toks;
    for (llama_token t = 1; t <= 64; t++) {
        toks.push_back(t);
    }

    state.prompt.tokens = server_tokens(toks, false);

    state.data.main.resize(4096);
    for (size_t i = 0; i < state.data.main.size(); i++) {
        state.data.main[i] = (uint8_t) (i * 31 + 7);
    }

    state.data.drft.resize(256);
    for (size_t i = 0; i < state.data.drft.size(); i++) {
        state.data.drft[i] = (uint8_t) (i ^ 0x5a);
    }

    for (int i = 0; i < 2; i++) {
        common_prompt_checkpoint ckpt;

        ckpt.n_tokens = 16 * (i + 1);
        ckpt.pos_min  = 16 * (i + 1) - 1;
        ckpt.pos_max  = 16 * (i + 1) - 1;

        ckpt.data_tgt.resize(128);
        for (size_t j = 0; j < ckpt.data_tgt.size(); j++) {
            ckpt.data_tgt[j] = (uint8_t) (j + i);
        }

        ckpt.data_dft.resize(32);
        ckpt.data_spec.resize(16);

        state.prompt.checkpoints.push_back(std::make_shared<const common_prompt_checkpoint>(std::move(ckpt)));
    }

    return state;
}

// writes the entry the way the disk tier writes it: the header, the payload, the trailer
static void write_entry(const std::string & path, const server_prompt_cache_state & state, const std::vector<char> & tokens) {
    FILE * file = fopen(path.c_str(), "wb");
    if (file == nullptr) {
        fprintf(stderr, "FAIL: cannot open %s\n", path.c_str());
        exit(1);
    }

    pcache_hasher hasher;
    hasher.init();

    pcache_writer writer;
    writer.file   = file;
    writer.hasher = &hasher;

    pcache_fwrite_u32(file, SERVER_PROMPT_CACHE_ENTRY_MAGIC);
    pcache_fwrite_u32(file, SERVER_PROMPT_CACHE_ENTRY_VERSION);
    pcache_fwrite_u32(file, (uint32_t) LLAMA_STATE_SEQ_VERSION);
    pcache_fwrite_u64(file, FINGERPRINT);

    if (!pcache_write_entry(writer, state, tokens)) {
        fprintf(stderr, "FAIL: cannot write the entry\n");
        exit(1);
    }

    pcache_fwrite_u64(file, hasher.digest());

    fclose(file);
    hasher.release();
}

template <typename A, typename B>
static bool same_blob(const A & a, const B & b) {
    if (a.size() != b.size()) {
        return false;
    }
    return memcmp(a.data(), b.data(), a.size()) == 0;
}

int main() {
    const std::string path = "/tmp/test-disk-entry-checksum.bin";

    server_prompt_cache_state state = make_state();

    const std::vector<char> tokens = state.prompt.tokens.serialize();

    write_entry(path, state, tokens);

    // the file holds the header, the payload and the trailer, and nothing else
    {
        FILE * file = fopen(path.c_str(), "rb");
        fseek(file, 0, SEEK_END);
        const long size = ftell(file);
        fclose(file);

        check((uint64_t) size == pcache_entry_size(state, tokens.size()),
              "the entry is exactly the size the layout says");
    }

    // what was written comes back
    {
        FILE * file = fopen(path.c_str(), "rb");
        server_prompt_cache_state back;
        const bool ok = pcache_read_entry(file, FINGERPRINT, back);
        fclose(file);

        check(ok, "the entry reads back");
        check(back.prompt.tokens.size() == state.prompt.tokens.size(), "the token count comes back");
        check(memcmp(back.prompt.tokens.get_tokens().data(), state.prompt.tokens.get_tokens().data(),
                    state.prompt.tokens.size() * sizeof(llama_token)) == 0, "the tokens come back");
        check(same_blob(state.data.main, back.data.main), "the target state comes back");
        check(same_blob(state.data.drft, back.data.drft), "the draft state comes back");
        check(back.prompt.checkpoints.size() == state.prompt.checkpoints.size(), "the checkpoints come back");
        {
            auto it  = back.prompt.checkpoints.begin();
            auto src = state.prompt.checkpoints.begin();
            bool ok  = back.prompt.checkpoints.size() == 2;
            for (; ok && it != back.prompt.checkpoints.end(); ++it, ++src) {
                ok = (*it)->n_tokens == (*src)->n_tokens &&
                     (*it)->pos_min  == (*src)->pos_min &&
                     (*it)->pos_max  == (*src)->pos_max &&
                     same_blob((*src)->data_tgt, (*it)->data_tgt) &&
                     same_blob((*src)->data_dft, (*it)->data_dft) &&
                     same_blob((*src)->data_spec, (*it)->data_spec);
            }
            check(ok, "the checkpoint contents come back");
        }
    }

    // a payload the device did not take whole is refused, not restored
    {
        FILE * file = fopen(path.c_str(), "r+b");
        fseek(file, 200, SEEK_SET);
        const char byte = 0x00;
        fwrite(&byte, 1, 1, file);
        fclose(file);

        FILE * file2 = fopen(path.c_str(), "rb");
        server_prompt_cache_state back;
        bool corrupt = false;
        const bool ok = pcache_read_entry(file2, FINGERPRINT, back, &corrupt);
        fclose(file2);

        check(!ok, "a payload that does not match its checksum is refused");
        check(corrupt, "the refusal says the entry is corrupt");
    }

    // an entry that was cut short is refused
    {
        FILE * file = fopen(path.c_str(), "r+b");
        fseek(file, 100, SEEK_SET);
        const char zero[16] = {0};
        fwrite(zero, 1, sizeof(zero), file);
        fclose(file);

        FILE * file2 = fopen(path.c_str(), "rb");
        server_prompt_cache_state back;
        const bool ok = pcache_read_entry(file2, FINGERPRINT, back);
        fclose(file2);

        check(!ok, "a payload that was cut short is refused");
    }

    remove(path.c_str());

    printf("PASS\n");
    return 0;
}

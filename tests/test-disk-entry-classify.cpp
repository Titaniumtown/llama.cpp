// Which files in the disk tier directory are loadable entries: the extension
// classification. disk_init unlinks everything else -- in particular the .bin.tmp
// files a writer leaves behind when it dies between the write and the rename.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-disk-entry-classify.cpp && /tmp/t
#include "server-prompt-cache-disk.h"

#include <cstdio>
#include <cstdlib>

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

int main() {
    check(server_prompt_cache_is_entry_path("/mnt/seed-cache/llama-cache/entries/12-2e6a53a7633ab85c.bin"),
          "a .bin path is an entry");
    check(!server_prompt_cache_is_entry_path("/mnt/seed-cache/llama-cache/entries/103775-53027cfa61467478.bin.tmp"),
          "a .bin.tmp path is not an entry");
    check(!server_prompt_cache_is_entry_path("entries/103775-53027cfa61467478.tmp"),
          "a bare .tmp path is not an entry");
    check(!server_prompt_cache_is_entry_path("entries/103775-53027cfa61467478"),
          "a suffix-less path is not an entry");
    check(!server_prompt_cache_is_entry_path("entries/x.bins"),
          "a .bins path is not an entry");
    check(!server_prompt_cache_is_entry_path("entries/.bin"),
          "the suffix alone is not an entry");

    printf("PASS\n");
    return 0;
}

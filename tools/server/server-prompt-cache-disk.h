#pragma once

#include <string>
#include <cstddef>
#include <algorithm>

// a file in the disk tier directory is a loadable entry only if it ends in ".bin";
// anything else -- in particular a ".bin.tmp" left by a writer that died between the
// write and the rename -- is never loaded, never counted, never evicted
static inline bool server_prompt_cache_is_entry_path(const std::string & path) {
    static const char k_suffix[] = ".bin";
    const std::size_t n = sizeof(k_suffix) - 1;

    return path.size() > n + 1 && path[path.size() - n - 1] != '/' && path.compare(path.size() - n, n, k_suffix) == 0;
}

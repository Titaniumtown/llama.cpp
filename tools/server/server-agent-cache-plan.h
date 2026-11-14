#pragma once

// Which local agent runs next, and whether that agent's prompt state stays in RAM.
// omp posts the plan. Eviction spills an unplanned entry before a planned one.
// No llama headers: the tests compile this with g++ -std=c++17 -Itools/server.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// 30 seconds. A plan older than this no longer pins.
static constexpr int64_t AGENT_PLAN_TTL_US = 30000000;

enum class agent_plan_tier : int {
    vram = 0,
    ram  = 1,
    disk = 2,
    cold = 3,
};

struct agent_plan_item {
    std::string agent;
    std::string session;
    int         rank       = 0;
    int64_t     expires_us = 0;
};

static inline const char * agent_plan_tier_name(agent_plan_tier tier) {
    switch (tier) {
        case agent_plan_tier::vram: return "vram";
        case agent_plan_tier::ram:  return "ram";
        case agent_plan_tier::disk: return "disk";
        case agent_plan_tier::cold: return "cold";
    }
    return "cold";
}

// [A-Za-z0-9_.-] and at most 32. Anything else is dropped, not rejected: a header
// must not carry a control byte into a log line or a sidecar.
static inline std::string agent_plan_sanitize(const std::string & raw) {
    std::string out;
    out.reserve(std::min<size_t>(raw.size(), 32));
    for (unsigned char c : raw) {
        if (out.size() >= 32) {
            break;
        }
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                        (c >= '0' && c <= '9') || c == '_' || c == '-' || c == '.';
        if (ok) {
            out.push_back(static_cast<char>(c));
        }
    }
    return out;
}

// Drop expired items. The pin_count lowest ranks are protected (tie: lower rank,
// then agent string). Both agent and session must match. An empty agent is never
// protected, and does not consume a pin. pin_count <= 0 protects nothing.
static inline bool agent_plan_protected(std::vector<agent_plan_item> & items,
        const std::string & agent, const std::string & session, int pin_count, int64_t now_us) {
    items.erase(std::remove_if(items.begin(), items.end(),
            [&](const agent_plan_item & it) { return it.expires_us <= now_us; }),
            items.end());

    if (pin_count <= 0 || agent.empty()) {
        return false;
    }

    std::vector<size_t> order;
    order.reserve(items.size());
    for (size_t i = 0; i < items.size(); i++) {
        if (!items[i].agent.empty()) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (items[a].rank != items[b].rank) {
            return items[a].rank < items[b].rank;
        }
        return items[a].agent < items[b].agent;
    });

    const size_t n = std::min(order.size(), static_cast<size_t>(pin_count));
    for (size_t i = 0; i < n; i++) {
        const agent_plan_item & it = items[order[i]];
        if (it.agent == agent && it.session == session) {
            return true;
        }
    }
    return false;
}

// Index of the minimum value if it is not protected; otherwise the minimum among
// the unprotected; if every entry is protected, the minimum again. Equal values
// keep the lowest index, the same pick as a strict < scan.
template <typename ValueAt, typename ProtectedAt>
static inline size_t agent_plan_victim(size_t n, const ValueAt & value_at, const ProtectedAt & protected_at) {
    if (n == 0) {
        return 0;
    }

    size_t min_i = 0;
    size_t min_open = n;
    for (size_t i = 0; i < n; i++) {
        if (value_at(i) < value_at(min_i)) {
            min_i = i;
        }
        if (!protected_at(i) && (min_open == n || value_at(i) < value_at(min_open))) {
            min_open = i;
        }
    }

    if (!protected_at(min_i)) {
        return min_i;
    }
    if (min_open != n) {
        return min_open;
    }
    return min_i;
}

// Lowest tier, then lowest rank, then lowest index. vram is 0, so it wins.
template <typename TierAt, typename RankAt>
static inline size_t agent_plan_next(size_t n, const TierAt & tier_at, const RankAt & rank_at) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) {
        const int tb = static_cast<int>(tier_at(i));
        const int ta = static_cast<int>(tier_at(best));
        if (tb < ta || (tb == ta && rank_at(i) < rank_at(best))) {
            best = i;
        }
    }
    return best;
}

// Which stored state eviction may spill, and which agent runs next.
//
// Build and run without a model:
//   g++ -std=c++17 -Itools/server -o /tmp/t tests/test-agent-cache-plan.cpp && /tmp/t
#include "server-agent-cache-plan.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static const int64_t NOW = 1000000000;
static const int64_t TTL = AGENT_PLAN_TTL_US;

static void check(bool ok, const char * what) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", what);
        exit(1);
    }
    printf("ok: %s\n", what);
}

static agent_plan_item item(const char * agent, const char * session, int rank, int64_t expires) {
    agent_plan_item it;
    it.agent = agent;
    it.session = session;
    it.rank = rank;
    it.expires_us = expires;
    return it;
}

int main() {
    check(TTL == 30000000, "the plan TTL is 30s");

    // no plan / pin_count 0: the victim is the minimum value, lowest index on a tie
    {
        float vals[] = { 5.f, 1.f, 3.f, 1.f };
        auto val = [&](size_t i) { return vals[i]; };
        std::vector<agent_plan_item> plan = { item("a", "s", 0, NOW + TTL) };
        auto prot = [&](size_t) { return agent_plan_protected(plan, "a", "s", 0, NOW); };
        check(agent_plan_victim(4, val, prot) == 1, "no plan / pin_count 0 keeps the min-value victim");
    }

    // the minimum is protected, so the spill is the cheapest unprotected entry
    {
        float vals[] = { 5.f, 1.f, 3.f };
        auto val = [&](size_t i) { return vals[i]; };
        auto prot = [](size_t i) { return i == 1; };
        check(agent_plan_victim(3, val, prot) == 2, "a protected min-value yields the unprotected one");
    }

    // nothing left to spill except a protected entry: fall back to the minimum
    {
        float vals[] = { 5.f, 1.f, 3.f };
        auto val = [&](size_t i) { return vals[i]; };
        auto prot = [](size_t) { return true; };
        check(agent_plan_victim(3, val, prot) == 1, "all protected falls back to the min-value");
    }

    // an expired plan item is dropped, and the pin it held passes to the next rank
    {
        std::vector<agent_plan_item> plan = {
            item("old", "s", 0, NOW),
            item("live", "s", 5, NOW + TTL),
        };
        check(!agent_plan_protected(plan, "old", "s", 1, NOW), "an expired item is not protected");
        check(plan.size() == 1 && plan[0].agent == "live", "expiry drops the item");
        check(agent_plan_protected(plan, "live", "s", 1, NOW), "the live item takes the freed pin");
    }

    // the pin cap keeps the lowest ranks, and a rank tie keeps the lower agent string
    {
        std::vector<agent_plan_item> plan = {
            item("zeta", "s", 2, NOW + TTL),
            item("mid", "s", 1, NOW + TTL),
            item("alpha", "s", 0, NOW + TTL),
        };
        check(agent_plan_protected(plan, "alpha", "s", 2, NOW), "the lowest rank is inside the pin cap");
        check(agent_plan_protected(plan, "mid", "s", 2, NOW), "the second-lowest rank is inside the pin cap");
        check(!agent_plan_protected(plan, "zeta", "s", 2, NOW), "a rank past the pin cap is not protected");
        check(!agent_plan_protected(plan, "alpha", "other", 2, NOW), "a different session is not protected");
    }
    {
        std::vector<agent_plan_item> plan = {
            item("beta", "s", 0, NOW + TTL),
            item("alpha", "s", 0, NOW + TTL),
        };
        check(agent_plan_protected(plan, "alpha", "s", 1, NOW), "a rank tie keeps the lower agent string");
        check(!agent_plan_protected(plan, "beta", "s", 1, NOW), "a rank tie drops the higher agent string");
    }
    {
        std::vector<agent_plan_item> plan = {
            item("", "s", 0, NOW + TTL),
            item("kept", "s", 1, NOW + TTL),
        };
        check(!agent_plan_protected(plan, "", "s", 1, NOW), "an empty agent is never protected");
        check(agent_plan_protected(plan, "kept", "s", 1, NOW), "an empty agent does not consume the pin");
    }

    // next() prefers vram over ram over disk over cold, then the lower rank, then the lower index
    {
        agent_plan_tier tiers[] = {
            agent_plan_tier::disk, agent_plan_tier::cold, agent_plan_tier::ram, agent_plan_tier::vram,
        };
        int ranks[] = { 0, 0, 0, 9 };
        check(agent_plan_next(4, [&](size_t i) { return tiers[i]; }, [&](size_t i) { return ranks[i]; }) == 3,
                "vram beats ram, disk and cold");
    }
    {
        agent_plan_tier tiers[] = { agent_plan_tier::cold, agent_plan_tier::disk, agent_plan_tier::ram };
        check(agent_plan_next(3, [&](size_t i) { return tiers[i]; }, [&](size_t) { return 0; }) == 2,
                "ram beats disk and disk beats cold");
    }
    {
        agent_plan_tier tiers[] = { agent_plan_tier::ram, agent_plan_tier::ram, agent_plan_tier::disk };
        int ranks[] = { 3, 1, 0 };
        check(agent_plan_next(3, [&](size_t i) { return tiers[i]; }, [&](size_t i) { return ranks[i]; }) == 1,
                "the same tier keeps the lower rank");
    }
    {
        check(agent_plan_next(2, [&](size_t) { return agent_plan_tier::ram; }, [&](size_t) { return 1; }) == 0,
                "the same tier and rank keep the lower index");
    }

    check(agent_plan_sanitize("A;B x") == "ABx", "sanitize strips characters outside the token alphabet");
    check(agent_plan_sanitize(std::string(40, 'n')) == std::string(32, 'n'), "sanitize truncates at 32");
    check(agent_plan_sanitize("ok_name-1.2") == "ok_name-1.2", "sanitize keeps the token alphabet");
    check(agent_plan_sanitize("") == "", "sanitize leaves an empty token empty");

    printf("PASS\n");
    return 0;
}

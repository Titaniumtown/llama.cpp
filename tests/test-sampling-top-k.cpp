#include "sampling.h"

#include "llama.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <vector>

// llama_sampler_top_k on the full candidate array, which common_sampler_top_k_logits must reproduce
static std::vector<llama_token_data> ref_top_k(const std::vector<float> & logits, int32_t k) {
    std::vector<llama_token_data> cur(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        cur[i] = { (llama_token) i, logits[i], 0.0f };
    }

    llama_token_data_array arr = { cur.data(), cur.size(), -1, false };

    llama_sampler * smpl = llama_sampler_init_top_k(k);
    llama_sampler_apply(smpl, &arr);
    llama_sampler_free(smpl);

    return std::vector<llama_token_data>(arr.data, arr.data + arr.size);
}

static bool same_bits(float a, float b) {
    return memcmp(&a, &b, sizeof(float)) == 0;
}

static bool same(const llama_token_data * a, size_t na, const llama_token_data * b, size_t nb) {
    if (na != nb) {
        return false;
    }
    for (size_t i = 0; i < na; ++i) {
        if (a[i].id != b[i].id || !same_bits(a[i].logit, b[i].logit) || !same_bits(a[i].p, b[i].p)) {
            return false;
        }
    }
    return true;
}

// whether the k + 1 largest logits are pairwise distinct: the only case the helper may answer
static bool unique_top(const std::vector<float> & logits, int32_t k) {
    std::vector<float> v(logits);
    std::partial_sort(v.begin(), v.begin() + k + 1, v.end(), std::greater<float>());
    for (int32_t i = 0; i < k; ++i) {
        if (!(v[i] > v[i + 1])) {
            return false;
        }
    }
    return true;
}

static int n_checked = 0;

static void check(const std::vector<float> & logits, int32_t k, bool expect_ok) {
    std::vector<llama_token_data> out;
    const bool ok = common_sampler_top_k_logits(logits.data(), (int32_t) logits.size(), k, out);
    if (ok != expect_ok) {
        fprintf(stderr, "top-k n=%zu k=%d: got %d, want %d\n", logits.size(), k, ok, expect_ok);
        assert(false);
    }
    if (ok) {
        const auto ref = ref_top_k(logits, k);
        assert(same(out.data(), out.size(), ref.data(), ref.size()));
    }
    n_checked++;
}

static void test_top_k_logits() {
    std::mt19937 rng(42);

    const std::vector<int32_t> ks = { 1, 2, 20, 40, 127, 128 };

    for (int32_t n : { 200, 4096, 248320 }) {
        std::normal_distribution<float> nd(0.0f, 3.0f);
        std::vector<float> logits(n);

        for (int rep = 0; rep < 4; ++rep) {
            for (auto & l : logits) {
                l = nd(rng);
            }
            for (int32_t k : ks) {
                if (k < n) {
                    check(logits, k, unique_top(logits, k));
                }
            }
        }

        // ascending ids: every logit enters the list
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = 1e-3f*i;
        }
        check(logits, 20, true);

        // descending ids: no logit after the first k + 1 enters
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = -1e-3f*i;
        }
        check(logits, 20, true);

        // all equal
        std::fill(logits.begin(), logits.end(), 0.5f);
        check(logits, 20, false);

        // all -inf but k + 1 of them
        std::fill(logits.begin(), logits.end(), -INFINITY);
        for (int32_t i = 0; i <= 20; ++i) {
            logits[(i*7919) % n] = (float) i;
        }
        check(logits, 20, true);

        // all -inf but k of them: the k + 1-th largest is -inf, still distinct from the k-th
        logits[0] = -INFINITY; // the 0.0f entry
        check(logits, 20, true);

        // all -inf but k - 1: -inf at the k-th and (k+1)-th
        logits[7919 % n] = -INFINITY;
        check(logits, 20, false);
    }

    // ties at the boundary and inside the top k fall back; ties below it do not matter
    {
        const int32_t n = 4096;
        std::vector<float> logits(n);
        for (int32_t i = 0; i < n; ++i) {
            logits[i] = (float) ((i*35761u) % 100000u); // distinct, scrambled
        }
        const int32_t k = 20;
        assert(unique_top(logits, k));
        check(logits, k, true);

        std::vector<float> v(logits);
        std::sort(v.begin(), v.end(), std::greater<float>());

        auto with_value_at = [&](float value, int32_t rank) {
            std::vector<float> l(logits);
            for (int32_t i = 0; i < n; ++i) {
                if (l[i] == v[rank]) {
                    l[i] = value;
                    break;
                }
            }
            return l;
        };

        check(with_value_at(v[k - 1], k),     k, false); // (k+1)-th == k-th
        check(with_value_at(v[3],     4),     k, false); // tie inside the top k
        check(with_value_at(v[k],     k + 1), k, true);  // (k+2)-th == (k+1)-th

        // +0 and -0 compare equal
        {
            std::vector<float> l(n);
            for (int32_t i = 0; i < n; ++i) {
                l[i] = -1.0f - (float) i;
            }
            l[5] = 0.0f;
            check(l, k, true);
            l[9] = -0.0f;
            check(l, k, false);
        }
    }

    // NaN anywhere, in the first k + 1 or after
    {
        const int32_t n = 4096;
        std::vector<float> logits(n);
        std::mt19937 r(7);
        std::uniform_real_distribution<float> ud(-10.0f, 10.0f);
        for (auto & l : logits) {
            l = ud(r);
        }
        for (int32_t at : { 0, 5, 20, 21, 1000, n - 1 }) {
            std::vector<float> l(logits);
            l[at] = std::numeric_limits<float>::quiet_NaN();
            check(l, 20, false);
        }
    }

    // arguments the helper does not take
    {
        std::vector<float> logits = { 1.0f, 2.0f, 3.0f };
        check(logits, 0, false);
        check(logits, 3, false);
        check(logits, 4, false);
        std::vector<float> big(1000);
        for (int32_t i = 0; i < 1000; ++i) {
            big[i] = (float) i;
        }
        check(big, 128, true);
        check(big, 129, false);
    }

    printf("top-k logits: %d cases OK\n", n_checked);
}

// the gate in common_sampler_init relies on disabled samplers being "?"-named no-ops
static void test_disabled_names() {
    llama_sampler * s[] = {
        llama_sampler_init_penalties(1000, 64, 1.0f, 0.0f, 0.0f),
        llama_sampler_init_penalties(1000, 0, 1.1f, 0.5f, 0.5f),
        llama_sampler_init_dry(nullptr, 0.0f, 1.75f, 2, -1, nullptr, 0),
        llama_sampler_init_top_n_sigma(-1.0f),
    };
    for (auto * smpl : s) {
        assert(llama_sampler_name(smpl)[0] == '?');

        std::vector<llama_token_data> cur = { { 0, 1.0f, 0.0f }, { 1, 3.0f, 0.0f }, { 2, 2.0f, 0.0f } };
        const auto before = cur;
        llama_token_data_array arr = { cur.data(), cur.size(), -1, false };
        llama_sampler_apply(smpl, &arr);
        assert(arr.size == 3 && !arr.sorted && arr.selected == -1);
        assert(same(cur.data(), cur.size(), before.data(), before.size()));

        llama_sampler_free(smpl);
    }

    llama_sampler * on[] = {
        llama_sampler_init_penalties(1000, 64, 1.1f, 0.0f, 0.0f),
        llama_sampler_init_top_n_sigma(1.0f),
        llama_sampler_init_top_k(20),
    };
    for (auto * smpl : on) {
        assert(llama_sampler_name(smpl)[0] != '?');
        llama_sampler_free(smpl);
    }

    printf("disabled sampler names: OK\n");
}

// the server's default chain, as common_sampler_init builds it with no penalties, DRY or top-n-sigma
static llama_sampler * make_chain(int32_t n_vocab, int32_t top_k, float top_p, float min_p, float temp, uint32_t seed) {
    llama_sampler * chain = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(chain, llama_sampler_init_penalties(n_vocab, 64, 1.0f, 0.0f, 0.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_dry(nullptr, 0.0f, 1.75f, 2, -1, nullptr, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_top_n_sigma(-1.0f));
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_typical(1.0f, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(top_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_min_p(min_p, 0));
    llama_sampler_chain_add(chain, llama_sampler_init_xtc(0.0f, 0.1f, 0, seed));
    llama_sampler_chain_add(chain, llama_sampler_init_temp_ext(temp, 0.0f, 1.0f));
    llama_sampler_chain_add(chain, temp == 0.0f ? llama_sampler_init_greedy() : llama_sampler_init_dist(seed));
    return chain;
}

// the chain fed the k prefiltered candidates draws the same tokens from the same RNG stream
static void test_chain_equivalence(int32_t n_vocab, int32_t top_k, float top_p, float min_p, float temp, int n_iter) {
    const uint32_t seed = 1234;

    llama_sampler * full = make_chain(n_vocab, top_k, top_p, min_p, temp, seed);
    llama_sampler * fast = make_chain(n_vocab, top_k, top_p, min_p, temp, seed);

    std::mt19937 rng(99);
    std::normal_distribution<float> nd(0.0f, 2.5f);

    std::vector<float> logits(n_vocab);
    std::vector<llama_token_data> cur_full(n_vocab);
    std::vector<llama_token_data> cur_fast;

    int n_fast = 0;
    for (int it = 0; it < n_iter; ++it) {
        for (auto & l : logits) {
            l = nd(rng);
        }
        // a few peaked rows, where top-p and min-p cut inside the top k
        if (it % 3 == 0) {
            logits[it % n_vocab] += 12.0f;
        }

        for (int32_t i = 0; i < n_vocab; ++i) {
            cur_full[i] = { i, logits[i], 0.0f };
        }
        llama_token_data_array arr_full = { cur_full.data(), cur_full.size(), -1, false };

        llama_token_data_array arr_fast;
        if (common_sampler_top_k_logits(logits.data(), n_vocab, top_k, cur_fast)) {
            arr_fast = { cur_fast.data(), cur_fast.size(), -1, true };
            n_fast++;
        } else {
            cur_fast.resize(n_vocab);
            for (int32_t i = 0; i < n_vocab; ++i) {
                cur_fast[i] = { i, logits[i], 0.0f };
            }
            arr_fast = { cur_fast.data(), cur_fast.size(), -1, false };
        }

        llama_sampler_apply(full, &arr_full);
        llama_sampler_apply(fast, &arr_fast);

        assert(arr_full.selected >= 0 && arr_full.selected == arr_fast.selected);
        assert(arr_full.sorted == arr_fast.sorted);
        assert(same(arr_full.data, arr_full.size, arr_fast.data, arr_fast.size));

        const llama_token id = arr_full.data[arr_full.selected].id;
        llama_sampler_accept(full, id);
        llama_sampler_accept(fast, id);
    }

    // the RNG streams are still in step after the loop
    for (int32_t i = 0; i < n_vocab; ++i) {
        cur_full[i] = { i, logits[i], 0.0f };
    }
    llama_token_data_array a = { cur_full.data(), cur_full.size(), -1, false };
    std::vector<llama_token_data> cur_b(cur_full);
    llama_token_data_array b = { cur_b.data(), cur_b.size(), -1, false };
    llama_sampler_apply(full, &a);
    llama_sampler_apply(fast, &b);
    assert(a.data[a.selected].id == b.data[b.selected].id);

    llama_sampler_free(full);
    llama_sampler_free(fast);

    assert(n_fast > 0);

    printf("chain n_vocab=%d top_k=%d top_p=%.2f min_p=%.2f temp=%.2f: %d/%d prefiltered, same tokens OK\n",
            n_vocab, top_k, top_p, min_p, temp, n_fast, n_iter);
}

int main(void) {
    ggml_time_init();

    test_top_k_logits();
    test_disabled_names();

    test_chain_equivalence(  4096, 20, 0.95f, 0.05f, 1.0f, 2000);
    test_chain_equivalence(  4096, 40, 0.95f, 0.05f, 0.8f, 2000);
    test_chain_equivalence(  4096, 20, 0.80f, 0.00f, 0.6f, 2000);
    test_chain_equivalence(  4096,  1, 1.00f, 0.00f, 0.0f, 500);
    test_chain_equivalence(248320, 20, 0.95f, 0.05f, 1.0f, 20);

    printf("OK\n");

    return 0;
}

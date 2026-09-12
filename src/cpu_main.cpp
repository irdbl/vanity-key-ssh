// CPU vanity searcher. Same pipeline as the CUDA kernel; useful as a
// correctness cross-check, an end-to-end test target, and a (slow) fallback.
//
//   cpu_vanity --suffix ++pham [--table table.bin] [--threads N]
//              [--limit N] [--benchmark]
#include <math.h>

#include <atomic>
#include <chrono>
#include <thread>
#include "vanity.h"

static std::atomic<uint64_t> g_attempts{0};
static std::atomic<bool> g_found{false};

int main(int argc, char **argv) {
    const char *table_path = "table.bin";
    std::vector<std::string> suffixes;
    bool ci = false;
    int threads = (int)std::thread::hardware_concurrency();
    uint64_t limit = 0;
    bool benchmark = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--suffix") && i + 1 < argc) suffixes.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--ci")) ci = true;
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) table_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--benchmark")) benchmark = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (suffixes.empty()) { fprintf(stderr, "--suffix required\n"); return 2; }
    const char *suffix = suffixes[0].c_str();

    uint8_t target[32], mask[32];
    int first = vk_suffix_to_target(suffix, target, mask);
    if (first < 0) { fprintf(stderr, "bad suffix '%s'\n", suffix); return 2; }
    vk_targets tg;
    if (vk_compile_targets(suffixes, ci, tg) < 0) return 2;
    const bool multi = tg.vals.size() > 1;
    int wide = 0;
    auto table = vk_load_table(table_path, &wide);

    uint8_t base[16];
    vk_random_base(base);
    double difficulty = pow(2.0, 6.0 * strlen(suffix)) / (double)tg.vals.size();
    fprintf(stderr, "suffix '%s'%s: %zu target(s), expected attempts %.3g, threads=%d\n",
            suffix, ci ? " (ci)" : "", tg.vals.size(), difficulty, threads);

    // Process keys in batches so one field inversion amortizes over the whole
    // batch (Montgomery's trick), matching the CUDA kernel instead of paying a
    // full fe_invert per key.
    constexpr int K = 16;

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            const uint8_t *tab = table.data();
            ge_p3 pts[K];
            fe prods[K];
            uint8_t seed[32], scalar[32], pub[32];
            uint64_t local = 0;  // flushed to g_attempts in bulk to avoid per-key contention
            bool stop = false;
            for (uint64_t block = 0; !stop && !g_found.load(std::memory_order_relaxed); block++) {
                uint64_t c0 = block * (uint64_t)K;
                if (limit && c0 * (uint64_t)threads >= limit) break;
                for (int k = 0; k < K; k++) {
                    vk_make_seed(seed, base, (uint64_t)t, c0 + k);
                    vk_seed_to_scalar(seed, scalar);
                    pts[k] = wide ? ge_scalarmult_base16(tab, scalar)
                                  : ge_scalarmult_base(tab, scalar);
                    prods[k] = k ? fe_mul(prods[k - 1], pts[k].Z) : pts[k].Z;
                }
                fe u = fe_invert(prods[K - 1]);
                for (int k = K - 1; k >= 0; k--) {
                    fe zinv = k ? fe_mul(u, prods[k - 1]) : u;
                    u = fe_mul(u, pts[k].Z);
                    ge_compress_with_zinv(pub, pts[k], zinv);
                    local++;
                    bool hit;
                    if (multi) {
                        uint64_t w3full = 0;
                        for (int j = 0; j < 8; j++) w3full |= (uint64_t)pub[24 + j] << (8 * j);
                        w3full &= tg.mask_w3full;
                        hit = std::binary_search(tg.vals.begin(), tg.vals.end(), w3full);
                    } else {
                        hit = vk_match(pub, target, mask, first);
                    }
                    if (hit) {
                        if (benchmark) continue;  // benchmarking: keep hashing, ignore matches
                        if (!g_found.exchange(true)) {
                            vk_make_seed(seed, base, (uint64_t)t, c0 + k);
                            printf("FOUND seed=%s pub=%s\n", vk_hex(seed, 32).c_str(), vk_hex(pub, 32).c_str());
                            printf("%s vanity\n", vk_pub_line(pub).c_str());
                            fflush(stdout);
                        }
                        stop = true;
                    }
                }
                if (local >= 1024) { g_attempts.fetch_add(local, std::memory_order_relaxed); local = 0; }
            }
            g_attempts.fetch_add(local, std::memory_order_relaxed);
        });
    }
    // In benchmark mode ignore matches and time a fixed window of real work; with
    // a --limit the workers bound themselves, so just join and time that.
    if (benchmark && limit == 0) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        g_found = true;
    }
    for (auto &th : pool) th.join();
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    uint64_t n = g_attempts.load();
    fprintf(stderr, "%llu attempts in %.1fs = %.0f keys/s\n",
            (unsigned long long)n, dt, n / dt);
    return (benchmark || g_found) ? 0 : 1;
}

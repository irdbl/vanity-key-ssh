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
    const char *suffix = nullptr, *table_path = "table.bin";
    int threads = (int)std::thread::hardware_concurrency();
    uint64_t limit = 0;
    bool benchmark = false;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--suffix") && i + 1 < argc) suffix = argv[++i];
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) table_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = strtoull(argv[++i], nullptr, 10);
        else if (!strcmp(argv[i], "--benchmark")) benchmark = true;
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!suffix) { fprintf(stderr, "--suffix required\n"); return 2; }

    uint8_t target[32], mask[32];
    int first = vk_suffix_to_target(suffix, target, mask);
    if (first < 0) { fprintf(stderr, "bad suffix '%s'\n", suffix); return 2; }
    auto table = vk_load_table(table_path);

    uint8_t base[16];
    vk_random_base(base);
    double difficulty = pow(2.0, 6.0 * strlen(suffix));
    fprintf(stderr, "suffix '%s': expected attempts 2^%zu = %.3g, threads=%d\n",
            suffix, 6 * strlen(suffix), difficulty, threads);

    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> pool;
    for (int t = 0; t < threads; t++) {
        pool.emplace_back([&, t]() {
            const uint8_t *tab = table.data();
            uint8_t seed[32], pub[32];
            for (uint64_t c = 0; !g_found.load(std::memory_order_relaxed); c++) {
                if (limit && c * (uint64_t)threads >= limit) return;
                vk_make_seed(seed, base, (uint64_t)t, c);
                vk_seed_to_pub(tab, seed, pub);
                g_attempts.fetch_add(1, std::memory_order_relaxed);
                if (vk_match(pub, target, mask, first)) {
                    if (!g_found.exchange(true)) {
                        printf("FOUND seed=%s pub=%s\n", vk_hex(seed, 32).c_str(), vk_hex(pub, 32).c_str());
                        printf("%s vanity\n", vk_pub_line(pub).c_str());
                        fflush(stdout);
                    }
                    return;
                }
            }
        });
    }
    if (benchmark) {
        std::this_thread::sleep_for(std::chrono::seconds(5));
        g_found = true;
    }
    for (auto &th : pool) th.join();
    auto dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    uint64_t n = g_attempts.load();
    fprintf(stderr, "%llu attempts in %.1fs = %.0f keys/s\n",
            (unsigned long long)n, dt, n / dt);
    return g_found && !benchmark ? 0 : 1;
}

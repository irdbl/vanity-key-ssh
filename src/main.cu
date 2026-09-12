// GPU vanity searcher for ed25519 SSH keys.
//
//   gpu_vanity --suffix ++pham [--table table.bin] [--device 0]
//              [--keys-per-thread 16] [--blocks-per-sm 8] [--threads 256]
//
// Prints "FOUND seed=<hex> pub=<hex>" plus the authorized_keys line and exits.
// Convert the seed to a private key with: python3 tools/keytool.py privkey <hex>
#include <math.h>

#include <chrono>

#include "vanity.h"

#define CUDA_CHECK(x)                                                              \
    do {                                                                           \
        cudaError_t err_ = (x);                                                    \
        if (err_ != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),  \
                    __FILE__, __LINE__);                                           \
            exit(1);                                                               \
        }                                                                          \
    } while (0)

#define MAX_FOUND 16
#define MAX_K 32

struct Found {
    int count;
    uint8_t seed[MAX_FOUND][32];
    uint8_t pub[MAX_FOUND][32];
};

struct Params {
    uint8_t base[16];
    uint8_t target[32];
    uint8_t mask[32];
    int first_byte;
    int keys_per_thread;
    uint64_t counter_base;
};

__global__ void vanity_kernel(const uint8_t *__restrict__ table, Params p, Found *out) {
    const uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int K = p.keys_per_thread;

    ge_p3 pts[MAX_K];
    fe prods[MAX_K];
    uint8_t seed[32], scalar[32];

    for (int k = 0; k < K; k++) {
        vk_make_seed(seed, p.base, gtid, p.counter_base + k);
        vk_seed_to_scalar(seed, scalar);
        pts[k] = ge_scalarmult_base(table, scalar);
        prods[k] = k ? fe_mul(prods[k - 1], pts[k].Z) : pts[k].Z;
    }

    // Montgomery batch inversion: one fe_invert amortized over K keys
    fe u = fe_invert(prods[K - 1]);
    for (int k = K - 1; k >= 0; k--) {
        fe zinv = k ? fe_mul(u, prods[k - 1]) : u;
        u = fe_mul(u, pts[k].Z);
        uint8_t pub[32];
        ge_compress_with_zinv(pub, pts[k], zinv);
        if (vk_match(pub, p.target, p.mask, p.first_byte)) {
            int slot = atomicAdd(&out->count, 1);
            if (slot < MAX_FOUND) {
                vk_make_seed(out->seed[slot], p.base, gtid, p.counter_base + k);
                memcpy(out->pub[slot], pub, 32);
            }
        }
    }
}

int main(int argc, char **argv) {
    const char *suffix = nullptr, *table_path = "table.bin";
    int device = 0, threads = 256, blocks_per_sm = 8, K = 16;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--suffix") && i + 1 < argc) suffix = argv[++i];
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) table_path = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks-per-sm") && i + 1 < argc) blocks_per_sm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keys-per-thread") && i + 1 < argc) K = atoi(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (!suffix) { fprintf(stderr, "--suffix required\n"); return 2; }
    if (K < 1 || K > MAX_K) { fprintf(stderr, "--keys-per-thread must be 1..%d\n", MAX_K); return 2; }

    Params p = {};
    p.first_byte = vk_suffix_to_target(suffix, p.target, p.mask);
    if (p.first_byte < 0) { fprintf(stderr, "bad suffix '%s'\n", suffix); return 2; }
    p.keys_per_thread = K;
    vk_random_base(p.base);

    auto table = vk_load_table(table_path);

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    uint8_t *d_table;
    Found *d_found;
    CUDA_CHECK(cudaMalloc(&d_table, VK_TABLE_BYTES));
    CUDA_CHECK(cudaMemcpy(d_table, table.data(), VK_TABLE_BYTES, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(Found)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(Found)));

    const int blocks = props.multiProcessorCount * blocks_per_sm;
    const uint64_t keys_per_launch = (uint64_t)blocks * threads * K;
    const double difficulty = pow(2.0, 6.0 * strlen(suffix));
    fprintf(stderr,
            "[gpu %d] %s (%d SMs): suffix '%s', difficulty 2^%zu = %.3g, "
            "%d blocks x %d threads x %d keys = %.2fM keys/launch\n",
            device, props.name, props.multiProcessorCount, suffix, 6 * strlen(suffix),
            difficulty, blocks, threads, K, keys_per_launch / 1e6);

    Found h_found = {};
    uint64_t total = 0, counter = 0, last_total = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto last = t0;

    while (true) {
        vanity_kernel<<<blocks, threads>>>(d_table, p, d_found);
        p.counter_base = (counter += K);
        CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        total += keys_per_launch;

        if (h_found.count > 0) {
            CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(Found), cudaMemcpyDeviceToHost));
            int n = h_found.count < MAX_FOUND ? h_found.count : MAX_FOUND;
            for (int i = 0; i < n; i++) {
                printf("FOUND seed=%s pub=%s\n", vk_hex(h_found.seed[i], 32).c_str(),
                       vk_hex(h_found.pub[i], 32).c_str());
                printf("%s vanity\n", vk_pub_line(h_found.pub[i]).c_str());
            }
            fflush(stdout);
            return 0;
        }

        auto now = std::chrono::steady_clock::now();
        double dt = std::chrono::duration<double>(now - last).count();
        if (dt >= 5.0) {
            double rate = (total - last_total) / dt;
            double elapsed = std::chrono::duration<double>(now - t0).count();
            fprintf(stderr, "[gpu %d] %.1f Mkeys/s | total %.3g | elapsed %.0fs | ETA(mean) %.1fh\n",
                    device, rate / 1e6, (double)total, elapsed, difficulty / rate / 3600.0);
            last = now;
            last_total = total;
        }
    }
}

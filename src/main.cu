// GPU vanity searcher for ed25519 SSH keys.
//
//   gpu_vanity --suffix ++pham [--table table.bin] [--device 0]
//              [--keys-per-thread 16] [--blocks-per-sm 8] [--threads 256]
//              [--benchmark] [--limit N]   # measure keys/s and exit
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
    // word-path fields (AUDIT2 F9/F10/F11)
    uint64_t base_be[2];   // base[0..7], base[8..15] as big-endian words
    uint64_t target_w3;    // target bytes 24..31 as an LE word, sign bit cleared
    uint64_t mask_w3;      // mask bytes 24..31 as an LE word, sign bit cleared
    int use_w3;            // suffix fits in w3 (L <= 10): fast one-word compare
    int wide;              // 16-bit signed comb table (AUDIT3 G1)
    // multi-target matcher (AUDIT F2)
    int n_targets;         // >1 enables the bitmap + binary-search path
    uint64_t mask_w3full;  // like mask_w3 but with the x-sign bit kept
};

// K is a template parameter so the batch arrays are sized exactly (a runtime
// K forced MAX_K-sized frames: 6656 B/thread of local memory, ~1.3 GB reserved
// across a 4090 at full residency) and the loops can fully unroll. T is dead
// after scalarmult, so only X/Y/Z are kept per key (AUDIT2 F5/A1).
__device__ __forceinline__ void record_found(Found *out, volatile int *found_flag,
                                             const Params &p, uint64_t gtid, uint64_t counter,
                                             const uint8_t pub[32]) {
    int slot = atomicAdd(&out->count, 1);
    if (slot < MAX_FOUND) {
        vk_make_seed(out->seed[slot], p.base, gtid, counter);
        memcpy(out->pub[slot], pub, 32);
    }
    __threadfence_system();
    *found_flag = 1;  // mapped host memory: host polls without draining the GPU
}

template <int K>
__global__ void vanity_kernel(const uint8_t *__restrict__ table, Params p, Found *out,
                              volatile int *found_flag,
                              const uint64_t *__restrict__ targets,
                              const uint32_t *__restrict__ bitmap) {
    const uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;

    // Multi-target prefilter (AUDIT F2): 2^16-bit bitmap over the targets'
    // low 16 bits, staged in shared memory. One shared load per key; only
    // bitmap hits (~2^-13 of keys) pay for x-parity + binary search.
    __shared__ uint32_t bm[2048];
    if (p.n_targets > 1) {
        for (int i = threadIdx.x; i < 2048; i += blockDim.x) bm[i] = bitmap[i];
        __syncthreads();
    }

    fe Xs[K], Ys[K], Zs[K], prods[K];

    for (int k = 0; k < K; k++) {
        uint64_t s[4];
        vk_seed_to_scalar_w(p.base_be, gtid, p.counter_base + k, s);
        ge_p3 pt = p.wide ? ge_scalarmult_base16_w(table, s) : ge_scalarmult_base_w(table, s);
        Xs[k] = pt.X;
        Ys[k] = pt.Y;
        Zs[k] = pt.Z;
        prods[k] = k ? fe_mul(prods[k - 1], Zs[k]) : Zs[k];
    }

    // Montgomery batch inversion: one fe_invert amortized over K keys
    fe u = fe_invert(prods[K - 1]);
    for (int k = K - 1; k >= 0; k--) {
        fe zinv = k ? fe_mul(u, prods[k - 1]) : u;
        u = fe_mul(u, Zs[k]);
        ge_p3 pt{Xs[k], Ys[k], Zs[k], fe_zero()};  // T unused by compression

        if (p.n_targets > 1) {
            const uint64_t w3y = ge_compress_y_w3(pt, zinv);
            const uint32_t lo = (uint32_t)(w3y & p.mask_w3) & 0xffff;
            if (bm[lo >> 5] & (1u << (lo & 31))) {
                // rare: full compression gives the sign bit; then exact search
                uint8_t pub[32];
                ge_compress_with_zinv(pub, pt, zinv);
                uint64_t w3full = 0;
                for (int j = 0; j < 8; j++) w3full |= (uint64_t)pub[24 + j] << (8 * j);
                w3full &= p.mask_w3full;
                int lo_i = 0, hi_i = p.n_targets - 1;
                while (lo_i <= hi_i) {
                    const int mid = (lo_i + hi_i) >> 1;
                    const uint64_t t = __ldg(&targets[mid]);
                    if (t == w3full) {
                        record_found(out, found_flag, p, gtid, p.counter_base + k, pub);
                        break;
                    }
                    if (t < w3full) lo_i = mid + 1;
                    else hi_i = mid - 1;
                }
            }
            continue;
        }

        bool candidate;
        if (p.use_w3) {
            // Fast path: test every constrained bit except the x sign against
            // the y encoding alone; x is only computed on the confirm path.
            const uint64_t w3y = ge_compress_y_w3(pt, zinv);
            candidate = ((w3y ^ p.target_w3) & p.mask_w3) == 0;
        } else {
            candidate = true;  // suffix longer than 10 chars: always full path
        }
        if (candidate) {
            uint8_t pub[32];
            ge_compress_with_zinv(pub, pt, zinv);
            if (vk_match(pub, p.target, p.mask, p.first_byte))
                record_found(out, found_flag, p, gtid, p.counter_base + k, pub);
        }
    }
}

// Computes full public keys on-device via the word hot path (one per thread)
// so the host can diff them against the byte-path reference. Validates device
// codegen — host tests cannot catch nvcc/PTX-specific miscompiles.
__global__ void selftest_kernel(const uint8_t *__restrict__ table, Params p, uint8_t *pubs) {
    const uint64_t gtid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t s[4];
    vk_seed_to_scalar_w(p.base_be, gtid, p.counter_base, s);
    ge_p3 pt = p.wide ? ge_scalarmult_base16_w(table, s) : ge_scalarmult_base_w(table, s);
    ge_compress_with_zinv(pubs + gtid * 32, pt, fe_invert(pt.Z));
}

static void launch_vanity(int K, int blocks, int threads, const uint8_t *table, const Params &p,
                          Found *out, volatile int *flag, const uint64_t *targets, const uint32_t *bitmap) {
    switch (K) {
        case 4: vanity_kernel<4><<<blocks, threads>>>(table, p, out, flag, targets, bitmap); break;
        case 8: vanity_kernel<8><<<blocks, threads>>>(table, p, out, flag, targets, bitmap); break;
        case 16: vanity_kernel<16><<<blocks, threads>>>(table, p, out, flag, targets, bitmap); break;
        case 32: vanity_kernel<32><<<blocks, threads>>>(table, p, out, flag, targets, bitmap); break;
    }
}

int main(int argc, char **argv) {
    const char *table_path = "table.bin";
    std::vector<std::string> suffixes;
    bool ci = false;
    int device = 0, threads = 256, blocks_per_sm = 8, K = 32;  // K=32 measured best (206 vs 202 Mkeys/s at K=16)
    bool benchmark = false, selftest = false;
    uint64_t limit = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--suffix") && i + 1 < argc) suffixes.push_back(argv[++i]);
        else if (!strcmp(argv[i], "--ci")) ci = true;
        else if (!strcmp(argv[i], "--table") && i + 1 < argc) table_path = argv[++i];
        else if (!strcmp(argv[i], "--device") && i + 1 < argc) device = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--blocks-per-sm") && i + 1 < argc) blocks_per_sm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--keys-per-thread") && i + 1 < argc) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--benchmark")) benchmark = true;
        else if (!strcmp(argv[i], "--selftest")) selftest = true;
        else if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = strtoull(argv[++i], nullptr, 10);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
    if (suffixes.empty()) { fprintf(stderr, "--suffix required\n"); return 2; }
    const char *suffix = suffixes[0].c_str();
    if (K != 4 && K != 8 && K != 16 && K != 32) { fprintf(stderr, "--keys-per-thread must be 4, 8, 16 or 32\n"); return 2; }

    Params p = {};
    p.first_byte = vk_suffix_to_target(suffix, p.target, p.mask);
    if (p.first_byte < 0) { fprintf(stderr, "bad suffix '%s'\n", suffix); return 2; }
    p.keys_per_thread = K;
    vk_random_base(p.base);
    for (int i = 0; i < 2; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | p.base[8 * i + j];
        p.base_be[i] = v;
    }
    uint64_t tw = 0, mw = 0;
    for (int j = 0; j < 8; j++) {
        tw |= (uint64_t)p.target[24 + j] << (8 * j);
        mw |= (uint64_t)p.mask[24 + j] << (8 * j);
    }
    p.target_w3 = tw & ~(1ULL << 63);
    p.mask_w3 = mw & ~(1ULL << 63);
    p.use_w3 = (p.first_byte >= 24);

    // multi-target compilation (AUDIT F2): several --suffix and/or --ci
    vk_targets tg;
    if (vk_compile_targets(suffixes, ci, tg) < 0) return 2;
    p.n_targets = (int)tg.vals.size();
    p.mask_w3full = tg.mask_w3full;

    auto table = vk_load_table(table_path, &p.wide);

    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    uint8_t *d_table;
    Found *d_found;
    CUDA_CHECK(cudaMalloc(&d_table, table.size()));
    CUDA_CHECK(cudaMemcpy(d_table, table.data(), table.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(Found)));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(Found)));

    uint64_t *d_targets = nullptr;
    uint32_t *d_bitmap = nullptr;
    if (p.n_targets > 1) {
        CUDA_CHECK(cudaMalloc(&d_targets, tg.vals.size() * sizeof(uint64_t)));
        CUDA_CHECK(cudaMemcpy(d_targets, tg.vals.data(), tg.vals.size() * sizeof(uint64_t),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMalloc(&d_bitmap, sizeof(tg.bitmap)));
        CUDA_CHECK(cudaMemcpy(d_bitmap, tg.bitmap, sizeof(tg.bitmap), cudaMemcpyHostToDevice));
    }

    // Mapped host flag (AUDIT2 F13): the hunt loop polls this instead of a
    // synchronous per-launch cudaMemcpy, so a few launches stay queued and the
    // GPU never drains between kernels.
    volatile int *h_flag;
    CUDA_CHECK(cudaHostAlloc((void **)&h_flag, sizeof(int), cudaHostAllocMapped));
    *h_flag = 0;
    volatile int *d_flag;
    CUDA_CHECK(cudaHostGetDevicePointer((void **)&d_flag, (void *)h_flag, 0));

    const int blocks = props.multiProcessorCount * blocks_per_sm;
    const uint64_t keys_per_launch = (uint64_t)blocks * threads * K;
    const double difficulty = pow(2.0, 6.0 * strlen(suffix)) / (p.n_targets > 0 ? p.n_targets : 1);
    fprintf(stderr,
            "[gpu %d] %s (%d SMs): suffix '%s'%s, %d target(s), difficulty %.3g, "
            "%d blocks x %d threads x %d keys = %.2fM keys/launch\n",
            device, props.name, props.multiProcessorCount, suffix, ci ? " (ci)" : "",
            p.n_targets, difficulty, blocks, threads, K, keys_per_launch / 1e6);

    Found h_found = {};
    uint64_t total = 0, counter = 0, last_total = 0;
    auto t0 = std::chrono::steady_clock::now();
    auto last = t0;

    if (selftest) {
        const int n = 512;
        uint8_t *d_pubs;
        CUDA_CHECK(cudaMalloc(&d_pubs, n * 32));
        selftest_kernel<<<n / 256, 256>>>(d_table, p, d_pubs);
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<uint8_t> pubs(n * 32);
        CUDA_CHECK(cudaMemcpy(pubs.data(), d_pubs, pubs.size(), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; i++) {
            uint8_t seed[32], expect[32];
            vk_make_seed(seed, p.base, (uint64_t)i, p.counter_base);
            vk_seed_to_pub(table.data(), p.wide, seed, expect);
            if (memcmp(expect, &pubs[i * 32], 32) != 0) bad++;
        }
        printf("[gpu %d] selftest: %d/%d pubkeys match host reference%s\n",
               device, n - bad, n, bad ? " — FAIL" : "");
        return bad ? 1 : 0;
    }

    // Benchmark mode: measure sustained throughput and exit, without needing a
    // real hunt. Matches are ignored. Runs until --limit keys are hashed, or for
    // ~5s if no limit is given. One warmup launch is excluded from the timing.
    if (benchmark) {
        launch_vanity(K, blocks, threads, d_table, p, d_found, d_flag, d_targets, d_bitmap);
        p.counter_base = (counter += K);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto bstart = std::chrono::steady_clock::now();
        uint64_t bkeys = 0;
        while (true) {
            launch_vanity(K, blocks, threads, d_table, p, d_found, d_flag, d_targets, d_bitmap);
            p.counter_base = (counter += K);
            CUDA_CHECK(cudaDeviceSynchronize());
            bkeys += keys_per_launch;
            double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - bstart).count();
            if ((limit && bkeys >= limit) || (!limit && el >= 5.0)) {
                printf("[gpu %d] benchmark: %.3g keys in %.2fs = %.2f Mkeys/s\n",
                       device, (double)bkeys, el, bkeys / el / 1e6);
                fflush(stdout);
                return 0;
            }
        }
    }

    // Keep up to DEPTH launches in flight; the event ring bounds the queue and
    // the mapped flag says when to stop. No synchronous copies in the loop.
    const int DEPTH = 3;
    cudaEvent_t ev[DEPTH];
    for (int i = 0; i < DEPTH; i++) CUDA_CHECK(cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming));
    uint64_t launched = 0;

    while (true) {
        launch_vanity(K, blocks, threads, d_table, p, d_found, d_flag, d_targets, d_bitmap);
        CUDA_CHECK(cudaEventRecord(ev[launched % DEPTH]));
        p.counter_base = (counter += K);
        launched++;
        if (launched >= DEPTH) {
            CUDA_CHECK(cudaEventSynchronize(ev[(launched - DEPTH) % DEPTH]));
            total = (launched - DEPTH + 1) * keys_per_launch;
        }

        if (*h_flag) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(Found), cudaMemcpyDeviceToHost));
        }
        if (h_found.count > 0) {
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

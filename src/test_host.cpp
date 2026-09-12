// Host-compiled harness exercising the exact code the CUDA kernel runs.
// Driven by tests/test_vanity.py against the pure-Python reference.
//
//   test_host sha512 <seed_hex32>          -> digest hex
//   test_host pub <table> <seed_hex32>     -> pubkey hex (single inversion path)
//   test_host pubbatch <table> <seed_hex32>... -> pubkeys via batch inversion
//   test_host target <suffix>              -> target hex, mask hex, first byte
#include "vanity.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: see source\n");
        return 2;
    }
    if (!strcmp(argv[1], "sha512")) {
        uint8_t seed[32], digest[64];
        if (!vk_unhex(argv[2], seed, 32)) return 2;
        sha512_32(seed, digest);
        printf("%s\n", vk_hex(digest, 64).c_str());
        return 0;
    }
    if (!strcmp(argv[1], "pub")) {
        int wide = 0;
        auto table = vk_load_table(argv[2], &wide);
        uint8_t seed[32], pub[32];
        if (!vk_unhex(argv[3], seed, 32)) return 2;
        vk_seed_to_pub(table.data(), wide, seed, pub);
        printf("%s\n", vk_hex(pub, 32).c_str());
        return 0;
    }
    if (!strcmp(argv[1], "pubbatch")) {
        // mirrors the kernel's Montgomery batch-inversion structure
        int wide = 0;
        auto table = vk_load_table(argv[2], &wide);
        int n = argc - 3;
        std::vector<ge_p3> pts(n);
        std::vector<fe> prods(n);
        for (int i = 0; i < n; i++) {
            uint8_t seed[32], scalar[32];
            if (!vk_unhex(argv[3 + i], seed, 32)) return 2;
            vk_seed_to_scalar(seed, scalar);
            pts[i] = wide ? ge_scalarmult_base16(table.data(), scalar)
                          : ge_scalarmult_base(table.data(), scalar);
            prods[i] = i ? fe_mul(prods[i - 1], pts[i].Z) : pts[i].Z;
        }
        fe u = fe_invert(prods[n - 1]);
        for (int i = n - 1; i >= 0; i--) {
            fe zinv = i ? fe_mul(u, prods[i - 1]) : u;
            u = fe_mul(u, pts[i].Z);
            uint8_t pub[32];
            ge_compress_with_zinv(pub, pts[i], zinv);
            printf("%d %s\n", i, vk_hex(pub, 32).c_str());
        }
        return 0;
    }
    if (!strcmp(argv[1], "expand")) {
        // expand SUFFIX... [ci] -> mask_w3full hex, then sorted w3 targets
        bool ci = false;
        std::vector<std::string> sufs;
        for (int i = 2; i < argc; i++) {
            if (!strcmp(argv[i], "ci")) ci = true;
            else sufs.push_back(argv[i]);
        }
        vk_targets tg;
        if (vk_compile_targets(sufs, ci, tg) < 0) return 1;
        printf("%016llx\n", (unsigned long long)tg.mask_w3full);
        for (uint64_t v : tg.vals) printf("%016llx\n", (unsigned long long)v);
        return 0;
    }
    if (!strcmp(argv[1], "target")) {
        uint8_t target[32], mask[32];
        int first = vk_suffix_to_target(argv[2], target, mask);
        if (first < 0) {
            fprintf(stderr, "bad suffix\n");
            return 1;
        }
        printf("%s\n%s\n%d\n", vk_hex(target, 32).c_str(), vk_hex(mask, 32).c_str(), first);
        return 0;
    }
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
}

// seed -> ed25519 public key pipeline and suffix matching.
#pragma once
#include "ge25519.h"

// seed[32]: 16-byte secret base || 8-byte LE worker id || 8-byte LE counter.
VK_HD void vk_make_seed(uint8_t seed[32], const uint8_t base[16], uint64_t worker, uint64_t counter) {
    for (int i = 0; i < 16; i++) seed[i] = base[i];
    for (int i = 0; i < 8; i++) seed[16 + i] = (uint8_t)(worker >> (8 * i));
    for (int i = 0; i < 8; i++) seed[24 + i] = (uint8_t)(counter >> (8 * i));
}

// RFC 8032: scalar = clamp(SHA512(seed)[0..31])
VK_HD void vk_seed_to_scalar(const uint8_t seed[32], uint8_t scalar[32]) {
    uint8_t digest[64];
    sha512_32(seed, digest);
    for (int i = 0; i < 32; i++) scalar[i] = digest[i];
    scalar[0] &= 248;
    scalar[31] &= 127;
    scalar[31] |= 64;
}

VK_HD uint64_t vk_bswap64(uint64_t x) {
    x = ((x & 0x00ff00ff00ff00ffULL) << 8) | ((x >> 8) & 0x00ff00ff00ff00ffULL);
    x = ((x & 0x0000ffff0000ffffULL) << 16) | ((x >> 16) & 0x0000ffff0000ffffULL);
    return (x << 32) | (x >> 32);
}

// Word-based hot path (AUDIT2 F11): the seed's message words are built
// directly from the pre-swapped base words and the worker/counter, and the
// clamped scalar stays in 4 big-endian words for ge_scalarmult_base_w.
// base_be[i] = big-endian load of base[8i..8i+7]; worker/counter are the same
// little-endian fields vk_make_seed writes, so bswap gives their message words.
VK_HD void vk_seed_to_scalar_w(const uint64_t base_be[2], uint64_t worker, uint64_t counter, uint64_t s[4]) {
    const uint64_t m[4] = {base_be[0], base_be[1], vk_bswap64(worker), vk_bswap64(counter)};
    uint64_t out[8];
    sha512_32_core(m, out);
    s[0] = out[0] & ~(0x07ULL << 56);            // scalar[0] &= 248
    s[1] = out[1];
    s[2] = out[2];
    s[3] = (out[3] & ~0x80ULL) | 0x40ULL;        // scalar[31] &= 127, |= 64
}

// Single-key reference path (CPU searcher, tests). The GPU kernel runs the
// same steps but batches the Z inversions across many keys. `wide` selects
// the 16-bit signed comb table (AUDIT3 G1) over the 8-bit one.
VK_HD void vk_seed_to_pub(const uint8_t *table, int wide, const uint8_t seed[32], uint8_t pub[32]) {
    uint8_t scalar[32];
    vk_seed_to_scalar(seed, scalar);
    ge_p3 p = wide ? ge_scalarmult_base16(table, scalar) : ge_scalarmult_base(table, scalar);
    ge_compress(pub, p);
}

// target/mask are 32-byte arrays aligned with pub[0..31]; see tools/keytool.py.
VK_HD bool vk_match(const uint8_t pub[32], const uint8_t target[32], const uint8_t mask[32], int first_byte) {
    for (int i = first_byte; i < 32; i++)
        if ((pub[i] & mask[i]) != target[i]) return false;
    return true;
}

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

// Single-key reference path (CPU searcher, tests). The GPU kernel runs the
// same steps but batches the Z inversions across many keys.
VK_HD void vk_seed_to_pub(const uint8_t *table, const uint8_t seed[32], uint8_t pub[32]) {
    uint8_t scalar[32];
    vk_seed_to_scalar(seed, scalar);
    ge_p3 p = ge_scalarmult_base(table, scalar);
    ge_compress(pub, p);
}

// target/mask are 32-byte arrays aligned with pub[0..31]; see tools/keytool.py.
VK_HD bool vk_match(const uint8_t pub[32], const uint8_t target[32], const uint8_t mask[32], int first_byte) {
    for (int i = first_byte; i < 32; i++)
        if ((pub[i] & mask[i]) != target[i]) return false;
    return true;
}

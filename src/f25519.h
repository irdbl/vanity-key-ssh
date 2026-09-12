// GF(2^255-19) arithmetic, 5 x 51-bit limbs (curve25519-donna-c64 style).
// Shared between the CUDA kernel and the host CPU searcher/tests.
#pragma once
#include <stdint.h>
#include "sha512.h"  // for VK_HD

typedef unsigned __int128 vk_u128;

struct fe {
    uint64_t v[5];
};

#define VK_M51 0x7ffffffffffffULL

VK_HD fe fe_zero() { return fe{{0, 0, 0, 0, 0}}; }
VK_HD fe fe_one() { return fe{{1, 0, 0, 0, 0}}; }

VK_HD fe fe_add(const fe &a, const fe &b) {
    fe r;
    for (int i = 0; i < 5; i++) r.v[i] = a.v[i] + b.v[i];
    return r;
}

// a - b, adding 8p limbwise so the result stays positive.
// 8p = (2^54-152, 2^54-8, 2^54-8, 2^54-8, 2^54-8) in this radix.
VK_HD fe fe_sub(const fe &a, const fe &b) {
    fe r;
    r.v[0] = a.v[0] + ((1ULL << 54) - 152) - b.v[0];
    r.v[1] = a.v[1] + ((1ULL << 54) - 8) - b.v[1];
    r.v[2] = a.v[2] + ((1ULL << 54) - 8) - b.v[2];
    r.v[3] = a.v[3] + ((1ULL << 54) - 8) - b.v[3];
    r.v[4] = a.v[4] + ((1ULL << 54) - 8) - b.v[4];
    return r;
}

// One carry pass: brings limbs (up to ~2^55 after a doubling) back below
// ~2^51, small enough to serve as a fe_sub subtrahend without underflow.
VK_HD fe fe_reduce(const fe &a) {
    fe h = a;
    uint64_t c;
    c = h.v[0] >> 51; h.v[0] &= VK_M51; h.v[1] += c;
    c = h.v[1] >> 51; h.v[1] &= VK_M51; h.v[2] += c;
    c = h.v[2] >> 51; h.v[2] &= VK_M51; h.v[3] += c;
    c = h.v[3] >> 51; h.v[3] &= VK_M51; h.v[4] += c;
    c = h.v[4] >> 51; h.v[4] &= VK_M51; h.v[0] += c * 19;
    return h;
}

VK_HD fe fe_mul(const fe &f, const fe &g) {
    const uint64_t f0 = f.v[0], f1 = f.v[1], f2 = f.v[2], f3 = f.v[3], f4 = f.v[4];
    const uint64_t g0 = g.v[0], g1 = g.v[1], g2 = g.v[2], g3 = g.v[3], g4 = g.v[4];
#define VK_MM(a, b) ((vk_u128)(a) * (b))
    vk_u128 t0 = VK_MM(f0, g0) + 19 * (VK_MM(f1, g4) + VK_MM(f2, g3) + VK_MM(f3, g2) + VK_MM(f4, g1));
    vk_u128 t1 = VK_MM(f0, g1) + VK_MM(f1, g0) + 19 * (VK_MM(f2, g4) + VK_MM(f3, g3) + VK_MM(f4, g2));
    vk_u128 t2 = VK_MM(f0, g2) + VK_MM(f1, g1) + VK_MM(f2, g0) + 19 * (VK_MM(f3, g4) + VK_MM(f4, g3));
    vk_u128 t3 = VK_MM(f0, g3) + VK_MM(f1, g2) + VK_MM(f2, g1) + VK_MM(f3, g0) + 19 * VK_MM(f4, g4);
    vk_u128 t4 = VK_MM(f0, g4) + VK_MM(f1, g3) + VK_MM(f2, g2) + VK_MM(f3, g1) + VK_MM(f4, g0);
#undef VK_MM
    fe r;
    uint64_t c;
    r.v[0] = (uint64_t)t0 & VK_M51; c = (uint64_t)(t0 >> 51);
    t1 += c; r.v[1] = (uint64_t)t1 & VK_M51; c = (uint64_t)(t1 >> 51);
    t2 += c; r.v[2] = (uint64_t)t2 & VK_M51; c = (uint64_t)(t2 >> 51);
    t3 += c; r.v[3] = (uint64_t)t3 & VK_M51; c = (uint64_t)(t3 >> 51);
    t4 += c; r.v[4] = (uint64_t)t4 & VK_M51; c = (uint64_t)(t4 >> 51);
    r.v[0] += c * 19; c = r.v[0] >> 51; r.v[0] &= VK_M51;
    r.v[1] += c;
    return r;
}

// Dedicated squaring (curve25519-donna-c64 style): symmetric cross terms are
// counted once, so a square costs ~10 wide products instead of the 25 in
// fe_mul. Inputs come from reduced fe outputs (limbs < ~2^52), so the widened
// temporaries d0..d4 stay well within the 128-bit accumulators.
VK_HD fe fe_sq(const fe &f) {
    const uint64_t r0 = f.v[0], r1 = f.v[1], r2 = f.v[2], r3 = f.v[3], r4 = f.v[4];
    const uint64_t d0 = r0 * 2;
    const uint64_t d1 = r1 * 2;
    const uint64_t d2 = r2 * 2 * 19;
    const uint64_t d419 = r4 * 19;
    const uint64_t d4 = d419 * 2;
    vk_u128 t0 = (vk_u128)r0 * r0 + (vk_u128)d4 * r1 + (vk_u128)d2 * r3;
    vk_u128 t1 = (vk_u128)d0 * r1 + (vk_u128)d4 * r2 + (vk_u128)r3 * (r3 * 19);
    vk_u128 t2 = (vk_u128)d0 * r2 + (vk_u128)r1 * r1 + (vk_u128)d4 * r3;
    vk_u128 t3 = (vk_u128)d0 * r3 + (vk_u128)d1 * r2 + (vk_u128)r4 * d419;
    vk_u128 t4 = (vk_u128)d0 * r4 + (vk_u128)d1 * r3 + (vk_u128)r2 * r2;
    fe r;
    uint64_t c;
    r.v[0] = (uint64_t)t0 & VK_M51; c = (uint64_t)(t0 >> 51);
    t1 += c; r.v[1] = (uint64_t)t1 & VK_M51; c = (uint64_t)(t1 >> 51);
    t2 += c; r.v[2] = (uint64_t)t2 & VK_M51; c = (uint64_t)(t2 >> 51);
    t3 += c; r.v[3] = (uint64_t)t3 & VK_M51; c = (uint64_t)(t3 >> 51);
    t4 += c; r.v[4] = (uint64_t)t4 & VK_M51; c = (uint64_t)(t4 >> 51);
    r.v[0] += c * 19; c = r.v[0] >> 51; r.v[0] &= VK_M51;
    r.v[1] += c;
    return r;
}

VK_HD fe fe_sqn(fe f, int n) {
    for (int i = 0; i < n; i++) f = fe_sq(f);
    return f;
}

// z^(p-2) via the standard curve25519 addition chain.
VK_HD fe fe_invert(const fe &z) {
    fe t0 = fe_sq(z);                       // 2
    fe t1 = fe_sqn(t0, 2);                  // 8
    t1 = fe_mul(z, t1);                     // 9
    t0 = fe_mul(t0, t1);                    // 11
    fe t2 = fe_sq(t0);                      // 22
    t1 = fe_mul(t1, t2);                    // 31 = 2^5-1
    t2 = fe_sqn(t1, 5);                     // 2^10-2^5
    t1 = fe_mul(t2, t1);                    // 2^10-1
    t2 = fe_sqn(t1, 10);                    // 2^20-2^10
    t2 = fe_mul(t2, t1);                    // 2^20-1
    fe t3 = fe_sqn(t2, 20);                 // 2^40-2^20
    t2 = fe_mul(t3, t2);                    // 2^40-1
    t2 = fe_sqn(t2, 10);                    // 2^50-2^10
    t1 = fe_mul(t2, t1);                    // 2^50-1
    t2 = fe_sqn(t1, 50);                    // 2^100-2^50
    t2 = fe_mul(t2, t1);                    // 2^100-1
    t3 = fe_sqn(t2, 100);                   // 2^200-2^100
    t2 = fe_mul(t3, t2);                    // 2^200-1
    t2 = fe_sqn(t2, 50);                    // 2^250-2^50
    t1 = fe_mul(t2, t1);                    // 2^250-1
    t1 = fe_sqn(t1, 5);                     // 2^255-2^5
    return fe_mul(t1, t0);                  // 2^255-21 = p-2
}

VK_HD fe fe_from_u64x4(uint64_t w0, uint64_t w1, uint64_t w2, uint64_t w3) {
    fe r;
    r.v[0] = w0 & VK_M51;
    r.v[1] = ((w0 >> 51) | (w1 << 13)) & VK_M51;
    r.v[2] = ((w1 >> 38) | (w2 << 26)) & VK_M51;
    r.v[3] = ((w2 >> 25) | (w3 << 39)) & VK_M51;
    r.v[4] = (w3 >> 12) & VK_M51;
    return r;
}

VK_HD fe fe_frombytes(const uint8_t s[32]) {
    uint64_t w0 = 0, w1 = 0, w2 = 0, w3 = 0;
    for (int i = 7; i >= 0; i--) w0 = (w0 << 8) | s[i];
    for (int i = 7; i >= 0; i--) w1 = (w1 << 8) | s[8 + i];
    for (int i = 7; i >= 0; i--) w2 = (w2 << 8) | s[16 + i];
    for (int i = 7; i >= 0; i--) w3 = (w3 << 8) | s[24 + i];
    fe r;
    r.v[0] = w0 & VK_M51;
    r.v[1] = ((w0 >> 51) | (w1 << 13)) & VK_M51;
    r.v[2] = ((w1 >> 38) | (w2 << 26)) & VK_M51;
    r.v[3] = ((w2 >> 25) | (w3 << 39)) & VK_M51;
    r.v[4] = (w3 >> 12) & VK_M51;
    return r;
}

// Canonical little-endian encoding (fully reduced mod p).
VK_HD void fe_tobytes(uint8_t s[32], const fe &f) {
    fe h = f;
    for (int pass = 0; pass < 2; pass++) {
        uint64_t c;
        c = h.v[0] >> 51; h.v[0] &= VK_M51; h.v[1] += c;
        c = h.v[1] >> 51; h.v[1] &= VK_M51; h.v[2] += c;
        c = h.v[2] >> 51; h.v[2] &= VK_M51; h.v[3] += c;
        c = h.v[3] >> 51; h.v[3] &= VK_M51; h.v[4] += c;
        c = h.v[4] >> 51; h.v[4] &= VK_M51; h.v[0] += c * 19;
    }
    // limbs now < 2^51 + small; assemble into 4 x u64 (value < 2p)
    uint64_t w0 = h.v[0] | (h.v[1] << 51);
    uint64_t w1 = (h.v[1] >> 13) | (h.v[2] << 38);
    uint64_t w2 = (h.v[2] >> 26) | (h.v[3] << 25);
    uint64_t w3 = (h.v[3] >> 39) | (h.v[4] << 12);
    // conditionally subtract p = 2^255-19 (value is < 2p here)
    uint64_t b, t0, t1, t2, t3;
    vk_u128 acc;
    acc = (vk_u128)w0 - 0xffffffffffffffedULL;
    t0 = (uint64_t)acc; b = (uint64_t)(acc >> 64) & 1;
    acc = (vk_u128)w1 - 0xffffffffffffffffULL - b;
    t1 = (uint64_t)acc; b = (uint64_t)(acc >> 64) & 1;
    acc = (vk_u128)w2 - 0xffffffffffffffffULL - b;
    t2 = (uint64_t)acc; b = (uint64_t)(acc >> 64) & 1;
    acc = (vk_u128)w3 - 0x7fffffffffffffffULL - b;
    t3 = (uint64_t)acc; b = (uint64_t)(acc >> 64) & 1;
    if (!b) { w0 = t0; w1 = t1; w2 = t2; w3 = t3; }
    for (int i = 0; i < 8; i++) s[i] = (uint8_t)(w0 >> (8 * i));
    for (int i = 0; i < 8; i++) s[8 + i] = (uint8_t)(w1 >> (8 * i));
    for (int i = 0; i < 8; i++) s[16 + i] = (uint8_t)(w2 >> (8 * i));
    for (int i = 0; i < 8; i++) s[24 + i] = (uint8_t)(w3 >> (8 * i));
}

// Ed25519 group operations over the fixed-base precomputed table.
//
// The table (tools/gen_table.py) holds table[i][j] = j * 2^(8i) * B for
// i in 0..31, j in 0..255, each entry 96 bytes: yplusx || yminusx || xy2d
// (32-byte little-endian field elements). A clamped scalar is consumed one
// byte per window: A = sum_i s[i] * 2^(8i) * B needs only 32 mixed additions,
// no doublings.
#pragma once
#include <string.h>

#include "f25519.h"  // note: u64 table loads assume a little-endian host/device

#define VK_TABLE_ENTRIES (32 * 256)
#define VK_TABLE_BYTES (VK_TABLE_ENTRIES * 96)

// 16-bit signed comb (AUDIT3 G1): 16 windows x 32769 magnitudes, ~48 MB —
// L2-resident on a 4090/5090. Halves the mixed-addition chain per key.
#define VK_TABLE16_MAGS 32769
#define VK_TABLE16_BYTES (16 * VK_TABLE16_MAGS * 96)

struct ge_p3 {
    fe X, Y, Z, T;  // extended coordinates, x = X/Z, y = Y/Z, T = XY/Z
};

VK_HD ge_p3 ge_identity() {
    return ge_p3{fe_zero(), fe_one(), fe_one(), fe_zero()};
}

// r = p + (yplusx, yminusx, xy2d)   [mixed addition via P1P1]
// need_t=false skips r.T when the caller will not read it (last comb window).
VK_HD void ge_madd(ge_p3 &r, const ge_p3 &p, const fe &yplusx, const fe &yminusx, const fe &xy2d, bool need_t = true) {
    fe A = fe_mul(fe_add(p.Y, p.X), yplusx);
    fe B = fe_mul(fe_sub(p.Y, p.X), yminusx);
    fe C = fe_mul(p.T, xy2d);
    fe D = fe_add(p.Z, p.Z);
    fe X3 = fe_sub(A, B);
    fe Y3 = fe_add(A, B);
    fe Z3 = fe_add(D, C);
    fe T3 = fe_sub(D, C);
    r.X = fe_mul(X3, T3);
    r.Y = fe_mul(Y3, Z3);
    r.Z = fe_mul(Z3, T3);
    if (need_t) r.T = fe_mul(X3, Y3);
}

// A = scalar * B using the precomputed table; scalar given as 4 big-endian
// u64 words (byte i of the scalar = byte (i&7) of word i>>3, MSB first), so
// the GPU path never stages the scalar through a byte array (AUDIT2 F11).
VK_HD ge_p3 ge_scalarmult_base_w(const uint8_t *table, const uint64_t s[4]) {
    ge_p3 r;
    for (int i = 0; i < 32; i++) {
        const int sb = (int)((s[i >> 3] >> (56 - 8 * (i & 7))) & 0xff);
        const uint8_t *e = table + ((size_t)(i * 256 + sb)) * 96;
        uint64_t w[12];  // entries are 8-byte aligned; memcpy compiles to wide loads
        memcpy(w, e, 96);
        fe yplusx = fe_from_u64x4(w[0], w[1], w[2], w[3]);
        fe yminusx = fe_from_u64x4(w[4], w[5], w[6], w[7]);
        fe xy2d = fe_from_u64x4(w[8], w[9], w[10], w[11]);
        if (i == 0) {
            // First addition is to the identity (X=0,Y=1,Z=1,T=0), so ge_madd
            // collapses: A=yplusx, B=yminusx, C=0, D=Z3=T3=2. Only r.T needs a
            // multiply; the rest are doublings. 7 fe_mul -> 1.
            fe X3 = fe_sub(yplusx, yminusx);
            fe Y3 = fe_add(yplusx, yminusx);
            // Reduce the doublings: fe_add leaves ~2^55 limbs, but the next
            // window feeds r.X/r.Y through fe_sub, which needs a < ~2^54 input.
            r.X = fe_reduce(fe_add(X3, X3));   // 2*X3 (Z3=2)
            r.Y = fe_reduce(fe_add(Y3, Y3));   // 2*Y3 (T3=2)
            r.Z = fe{{4, 0, 0, 0, 0}};         // Z3*T3 = 4
            r.T = fe_mul(X3, Y3);
        } else {
            // Last window: nothing after the loop reads r.T, so skip it.
            ge_madd(r, r, yplusx, yminusx, xy2d, i != 31);
        }
    }
    return r;
}

// 16-bit signed comb walk (AUDIT3 G1). The clamped scalar is recoded into 16
// signed digits in [-32767, 32768]; the table stores magnitudes only and the
// sign is applied by negating the niels point — in (y+x, y-x, 2dxy) form
// negation is a swap of the first two elements plus a field negation of the
// third, done branchlessly to avoid warp divergence on random digit signs.
// Clamping (bit 254 set, bit 255 clear) caps the top digit at 32768, so no
// 17th window is ever needed.
VK_HD ge_p3 ge_scalarmult_base16_w(const uint8_t *table, const uint64_t s[4]) {
    ge_p3 r;
    int carry = 0;
    for (int i = 0; i < 16; i++) {
        const int lo = (int)((s[(2 * i) >> 3] >> (56 - 8 * ((2 * i) & 7))) & 0xff);
        const int hi = (int)((s[(2 * i + 1) >> 3] >> (56 - 8 * ((2 * i + 1) & 7))) & 0xff);
        const int v = (lo | (hi << 8)) + carry;
        carry = v > 32768;
        const int digit = v - (carry << 16);
        const int mag = digit < 0 ? -digit : digit;
        const uint64_t neg = (uint64_t)0 - (uint64_t)(digit < 0);

        const uint8_t *e = table + ((size_t)i * VK_TABLE16_MAGS + (size_t)mag) * 96;
        uint64_t w[12];
        memcpy(w, e, 96);
        const fe pa = fe_from_u64x4(w[0], w[1], w[2], w[3]);    // y+x
        const fe pb = fe_from_u64x4(w[4], w[5], w[6], w[7]);    // y-x
        const fe pc = fe_from_u64x4(w[8], w[9], w[10], w[11]);  // 2dxy
        const fe pcn = fe_sub(fe_zero(), pc);
        fe yplusx, yminusx, xy2d;
        for (int l = 0; l < 5; l++) {
            yplusx.v[l] = (pa.v[l] & ~neg) | (pb.v[l] & neg);
            yminusx.v[l] = (pb.v[l] & ~neg) | (pa.v[l] & neg);
            xy2d.v[l] = (pc.v[l] & ~neg) | (pcn.v[l] & neg);
        }
        if (i == 0) {
            // identity peel, same algebra as the 8-bit path (F7)
            fe X3 = fe_sub(yplusx, yminusx);
            fe Y3 = fe_add(yplusx, yminusx);
            r.X = fe_reduce(fe_add(X3, X3));
            r.Y = fe_reduce(fe_add(Y3, Y3));
            r.Z = fe{{4, 0, 0, 0, 0}};
            r.T = fe_mul(X3, Y3);
        } else {
            ge_madd(r, r, yplusx, yminusx, xy2d, i != 15);
        }
    }
    return r;
}

VK_HD ge_p3 ge_scalarmult_base16(const uint8_t *table, const uint8_t scalar[32]) {
    uint64_t s[4];
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | scalar[8 * i + j];
        s[i] = v;
    }
    return ge_scalarmult_base16_w(table, s);
}

// Byte-oriented wrapper (tests, CPU tools).
VK_HD ge_p3 ge_scalarmult_base(const uint8_t *table, const uint8_t scalar[32]) {
    uint64_t s[4];
    for (int i = 0; i < 4; i++) {
        uint64_t v = 0;
        for (int j = 0; j < 8; j++) v = (v << 8) | scalar[8 * i + j];
        s[i] = v;
    }
    return ge_scalarmult_base_w(table, s);
}

// y-only compression word: little-endian u64 of encoded bytes 24..31, WITHOUT
// the x sign bit. For suffixes up to 10 chars this is every constrained bit
// except the sign, so the hot loop tests one word and defers the x coordinate
// to the ~never-taken confirm path (AUDIT2 F9/F10).
VK_HD uint64_t ge_compress_y_w3(const ge_p3 &p, const fe &zinv) {
    uint64_t w[4];
    fe_to_u64x4(w, fe_mul(p.Y, zinv));
    return w[3];
}

// Compress with a caller-supplied 1/Z (enables batch inversion).
VK_HD void ge_compress_with_zinv(uint8_t out[32], const ge_p3 &p, const fe &zinv) {
    fe x = fe_mul(p.X, zinv);
    fe y = fe_mul(p.Y, zinv);
    fe_tobytes(out, y);
    uint8_t xb[32];
    fe_tobytes(xb, x);
    out[31] |= (uint8_t)((xb[0] & 1) << 7);
}

VK_HD void ge_compress(uint8_t out[32], const ge_p3 &p) {
    ge_compress_with_zinv(out, p, fe_invert(p.Z));
}

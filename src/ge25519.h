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

// A = scalar * B using the precomputed table; scalar given as 32 clamped bytes.
VK_HD ge_p3 ge_scalarmult_base(const uint8_t *table, const uint8_t scalar[32]) {
    ge_p3 r;
    for (int i = 0; i < 32; i++) {
        const uint8_t *e = table + ((size_t)(i * 256 + scalar[i])) * 96;
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

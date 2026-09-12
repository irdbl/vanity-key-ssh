"""Generate the fixed-base precomputed table used by the miner.

Two formats (AUDIT3 G1):

  8-bit comb (default):  table[i][j] = j * 2^(8*i)  * B, i in 0..31, j in 0..255
                         32 x 256 x 96 B = 768 KB
  16-bit signed comb:    table[i][j] = j * 2^(16*i) * B, i in 0..15, j in 0..32768
                         16 x 32769 x 96 B = ~48 MB   (--wide)

The wide table stores magnitudes only; the miner applies the digit sign by
negating the niels point, which keeps the table inside a 4090's L2. Each entry:

    yplusx (32B LE) || yminusx (32B LE) || xy2d (32B LE)

j = 0 stores the identity (1, 1, 0). The loader tells the formats apart by
file size. Deterministic; regenerate any time.
"""

import sys

from ed25519_ref import B, D, P, point_add, point_mul


def fe_bytes(v: int) -> bytes:
    return int.to_bytes(v % P, 32, "little")


def gen(path: str, wbits: int) -> None:
    windows = 256 // wbits
    mags = 256 if wbits == 8 else 2 ** (wbits - 1) + 1
    out = bytearray()
    for i in range(windows):
        wbase = point_mul(1 << (wbits * i), B)
        # accumulate j*wbase in extended coords; normalize with one batch inversion
        pts = []
        acc = wbase
        for j in range(1, mags):
            if j > 1:
                acc = point_add(acc, wbase)
            pts.append(acc)
        prods = []
        c = 1
        for x, y, z, t in pts:
            c = c * z % P
            prods.append(c)
        u = pow(c, P - 2, P)
        out += fe_bytes(1) + fe_bytes(1) + fe_bytes(0)  # j = 0: identity
        chunks = [b""] * len(pts)
        for k in range(len(pts) - 1, -1, -1):
            zinv = u * prods[k - 1] % P if k else u
            u = u * pts[k][2] % P
            xa = pts[k][0] * zinv % P
            ya = pts[k][1] * zinv % P
            chunks[k] = fe_bytes(ya + xa) + fe_bytes(ya - xa) + fe_bytes(2 * D * xa * ya)
        out += b"".join(chunks)
        print(f"window {i + 1}/{windows}", file=sys.stderr)
    assert len(out) == windows * mags * 96
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {len(out)} bytes to {path} ({wbits}-bit comb)")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--wide"]
    gen(args[0] if args else "table.bin", 16 if "--wide" in sys.argv else 8)

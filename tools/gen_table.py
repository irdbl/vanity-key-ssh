"""Generate the fixed-base precomputed table used by the miner.

table[i][j] = j * 2^(8*i) * B   for i in 0..31, j in 0..255

Each entry is stored in "niels" form for mixed addition:
    yplusx (32B LE) || yminusx (32B LE) || xy2d (32B LE)
j = 0 stores the identity in niels form: (1, 1, 0).

Layout: entry (i, j) at byte offset (i*256 + j) * 96. Total 786,432 bytes.
The file is deterministic; regenerate any time with `python3 tools/gen_table.py out.bin`.
"""

import sys

from ed25519_ref import B, D, IDENTITY, P, point_add, point_mul, point_to_affine


def fe_bytes(v: int) -> bytes:
    return int.to_bytes(v % P, 32, "little")


def niels(point) -> bytes:
    if point == IDENTITY or point[0] % P == 0 and point[1] % P == point[2] % P:
        pass  # identity handled by affine math below just fine unless z==0
    x, y = point_to_affine(point)
    return fe_bytes(y + x) + fe_bytes(y - x) + fe_bytes(2 * D * x * y)


def main(path: str) -> None:
    out = bytearray()
    base = B
    for i in range(32):
        window_base = point_mul(1 << (8 * i), base)
        acc = IDENTITY
        for j in range(256):
            if j == 0:
                out += fe_bytes(1) + fe_bytes(1) + fe_bytes(0)
            else:
                acc = point_add(acc, window_base) if j > 1 else window_base
                out += niels(acc)
    assert len(out) == 32 * 256 * 96
    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {len(out)} bytes to {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "table.bin")

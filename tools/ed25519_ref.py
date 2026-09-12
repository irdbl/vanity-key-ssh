"""Pure-Python ed25519 reference implementation.

Used as ground truth for the CUDA/C++ miner: precomputed-table generation,
test vectors, and independent verification of found keys. Not fast; not
constant-time. Do not use for anything except offline verification.
"""

import hashlib

P = 2**255 - 19
L = 2**252 + 27742317777372353535851937790883648493
D = (-121665 * pow(121666, P - 2, P)) % P


def _inv(x: int) -> int:
    return pow(x, P - 2, P)


def _recover_x(y: int, sign: int) -> int:
    x2 = (y * y - 1) * _inv(D * y * y + 1) % P
    x = pow(x2, (P + 3) // 8, P)
    if (x * x - x2) % P != 0:
        x = x * pow(2, (P - 1) // 4, P) % P
    if (x * x - x2) % P != 0:
        raise ValueError("not a point")
    if x & 1 != sign:
        x = P - x
    return x


# Base point
_By = 4 * _inv(5) % P
_Bx = _recover_x(_By, 0)
B = (_Bx, _By, 1, _Bx * _By % P)  # extended coords (X, Y, Z, T)
IDENTITY = (0, 1, 1, 0)


def point_add(p, q):
    x1, y1, z1, t1 = p
    x2, y2, z2, t2 = q
    a = (y1 - x1) * (y2 - x2) % P
    b = (y1 + x1) * (y2 + x2) % P
    c = 2 * t1 * t2 * D % P
    d = 2 * z1 * z2 % P
    e, f, g, h = b - a, d - c, d + c, b + a
    return (e * f % P, g * h % P, f * g % P, e * h % P)


def point_mul(s: int, p):
    q = IDENTITY
    while s > 0:
        if s & 1:
            q = point_add(q, p)
        p = point_add(p, p)
        s >>= 1
    return q


def point_compress(p) -> bytes:
    x, y, z, _ = p
    zinv = _inv(z)
    x, y = x * zinv % P, y * zinv % P
    return int.to_bytes(y | ((x & 1) << 255), 32, "little")


def point_decompress(b: bytes):
    n = int.from_bytes(b, "little")
    y = n & ((1 << 255) - 1)
    x = _recover_x(y, n >> 255)
    return (x, y, 1, x * y % P)


def clamp(h: bytes) -> int:
    a = bytearray(h[:32])
    a[0] &= 248
    a[31] &= 127
    a[31] |= 64
    return int.from_bytes(a, "little")


def seed_to_public(seed: bytes) -> bytes:
    """32-byte seed -> 32-byte compressed public key (RFC 8032)."""
    assert len(seed) == 32
    h = hashlib.sha512(seed).digest()
    return point_compress(point_mul(clamp(h), B))


def point_to_affine(p):
    x, y, z, _ = p
    zinv = _inv(z)
    return x * zinv % P, y * zinv % P

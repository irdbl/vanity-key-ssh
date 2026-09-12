"""Key tooling for the vanity hunt.

Commands:
  target SUFFIX            print the 32-byte target/mask (hex) a pubkey must match
  privkey SEEDHEX          write an OpenSSH private key for a found seed
  verify SEEDHEX SUFFIX    independently recompute the pubkey and check the suffix
  pub SEEDHEX              print the authorized_keys line for a seed

An ed25519 SSH public key blob is exactly 51 bytes: a 19-byte fixed header
("ssh-ed25519" in SSH wire format) followed by the 32-byte ed25519 public
key. 51 bytes = 408 bits = 68 base64 chars with no padding, so a suffix of
L base64 chars constrains exactly the low 6*L bits of the 32-byte public
key (as a big-endian integer). That lets the GPU compare raw bytes.
"""

import base64
import os
import secrets
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ed25519_ref import seed_to_public  # noqa: E402

B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
HEADER = b"\x00\x00\x00\x0bssh-ed25519\x00\x00\x00 "  # 19 bytes


def suffix_to_target(suffix: str) -> tuple[bytes, bytes]:
    """Return (target, mask) as 32-byte big-endian-aligned arrays.

    A candidate public key `pub` (32 bytes) matches iff
    (pub[i] & mask[i]) == target[i] for all i.
    """
    if not suffix or len(suffix) > 42:
        raise ValueError("suffix must be 1..42 base64 characters")
    val = 0
    for ch in suffix:
        if ch not in B64:
            raise ValueError(f"'{ch}' is not a base64 character (A-Za-z0-9+/)")
        val = (val << 6) | B64.index(ch)
    bits = 6 * len(suffix)
    mask = (1 << bits) - 1
    return val.to_bytes(32, "big"), mask.to_bytes(32, "big")


def pub_blob(pub: bytes) -> bytes:
    return HEADER + pub


def pub_line(pub: bytes, comment: str) -> str:
    b64 = base64.b64encode(pub_blob(pub)).decode()
    return f"ssh-ed25519 {b64} {comment}".rstrip()


def _s(b: bytes) -> bytes:
    return struct.pack(">I", len(b)) + b


def openssh_private_key(seed: bytes, comment: str) -> bytes:
    """Unencrypted openssh-key-v1 private key file contents."""
    pub = seed_to_public(seed)
    check = secrets.token_bytes(4)
    priv = (
        check + check
        + _s(b"ssh-ed25519")
        + _s(pub)
        + _s(seed + pub)
        + _s(comment.encode())
    )
    pad = (-len(priv)) % 8
    priv += bytes(range(1, pad + 1))
    blob = (
        b"openssh-key-v1\x00"
        + _s(b"none") + _s(b"none") + _s(b"")
        + struct.pack(">I", 1)
        + _s(pub_blob(pub))
        + _s(priv)
    )
    b64 = base64.b64encode(blob).decode()
    lines = [b64[i : i + 70] for i in range(0, len(b64), 70)]
    return (
        "-----BEGIN OPENSSH PRIVATE KEY-----\n"
        + "\n".join(lines)
        + "\n-----END OPENSSH PRIVATE KEY-----\n"
    ).encode()


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    cmd = sys.argv[1]

    if cmd == "target":
        target, mask = suffix_to_target(sys.argv[2])
        print(target.hex())
        print(mask.hex())
        return 0

    seed = bytes.fromhex(sys.argv[2])
    if len(seed) != 32:
        print("seed must be 32 bytes of hex", file=sys.stderr)
        return 2

    if cmd == "pub":
        comment = sys.argv[3] if len(sys.argv) > 3 else "vanity"
        print(pub_line(seed_to_public(seed), comment))
        return 0

    if cmd == "verify":
        suffix = sys.argv[3]
        pub = seed_to_public(seed)
        target, mask = suffix_to_target(suffix)
        ok = all((p & m) == t for p, m, t in zip(pub, mask, target))
        line = pub_line(pub, "vanity")
        print(line)
        if not ok or not line.split()[1].endswith(suffix):
            print(f"MISMATCH: key does not end with '{suffix}'", file=sys.stderr)
            return 1
        print(f"OK: ends with '{suffix}'")
        return 0

    if cmd == "privkey":
        comment = sys.argv[4] if len(sys.argv) > 4 else "vanity"
        out = sys.argv[3] if len(sys.argv) > 3 else "id_ed25519_vanity"
        data = openssh_private_key(seed, comment)
        fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        with open(out + ".pub", "w") as f:
            f.write(pub_line(seed_to_public(seed), comment) + "\n")
        print(f"wrote {out} and {out}.pub")
        print(pub_line(seed_to_public(seed), comment))
        return 0

    print(f"unknown command {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())

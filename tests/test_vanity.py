"""Cross-checks: C++ core (same code the CUDA kernel runs) vs pure-Python
reference vs OpenSSH itself. Run via `make test`."""

import os
import secrets
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import ed25519_ref  # noqa: E402
import keytool  # noqa: E402

TEST_HOST = os.path.join(ROOT, "bin", "test_host")
CPU_VANITY = os.path.join(ROOT, "bin", "cpu_vanity")
TABLE = os.path.join(ROOT, "table.bin")

# RFC 8032 test vectors: (seed, public key)
RFC8032 = [
    ("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60",
     "d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
    ("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb",
     "3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
    ("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7",
     "fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"),
    ("f5e5767cf153319517630f226876b86c8160cc583bc013744c6bf255f5cc0ee5",
     "278117fc144c72340f67d0f2316e8386ceffbf2b2428c9c51fef7c597f1d426e"),
]


def run(*args):
    return subprocess.run(args, capture_output=True, text=True, check=True).stdout


def test_python_ref_rfc8032():
    for seed, pub in RFC8032:
        assert ed25519_ref.seed_to_public(bytes.fromhex(seed)).hex() == pub


def test_cpp_pub_rfc8032():
    for seed, pub in RFC8032:
        assert run(TEST_HOST, "pub", TABLE, seed).strip() == pub


def test_cpp_pub_random_seeds_match_python():
    for _ in range(20):
        seed = secrets.token_bytes(32)
        expect = ed25519_ref.seed_to_public(seed).hex()
        assert run(TEST_HOST, "pub", TABLE, seed.hex()).strip() == expect


def test_cpp_batch_inversion_matches_single():
    seeds = [secrets.token_bytes(32) for _ in range(8)]
    out = run(TEST_HOST, "pubbatch", TABLE, *[s.hex() for s in seeds])
    got = dict(line.split() for line in out.strip().splitlines())
    for i, s in enumerate(seeds):
        assert got[str(i)] == ed25519_ref.seed_to_public(s).hex()


def test_cpp_sha512_matches_hashlib():
    import hashlib
    for _ in range(10):
        seed = secrets.token_bytes(32)
        expect = hashlib.sha512(seed).hexdigest()
        assert run(TEST_HOST, "sha512", seed.hex()).strip() == expect


def test_target_mask_c_matches_python():
    for suffix in ["A", "m", "++pham", "pham", "kevin9", "/+/+/+", "z" * 42]:
        t, m = keytool.suffix_to_target(suffix)
        lines = run(TEST_HOST, "target", suffix).strip().splitlines()
        assert lines[0] == t.hex() and lines[1] == m.hex()


def test_target_mask_semantics():
    # a key whose authorized_keys line ends with the suffix must satisfy the mask
    for _ in range(5):
        seed = secrets.token_bytes(32)
        pub = ed25519_ref.seed_to_public(seed)
        line = keytool.pub_line(pub, "x").split()[1]
        suffix = line[-6:]
        t, m = keytool.suffix_to_target(suffix)
        assert all((p & mm) == tt for p, mm, tt in zip(pub, m, t))


def test_end_to_end_cpu_hunt(tmp_path):
    # 2-char suffix: expected 4096 attempts; verify the found key with OpenSSH
    suffix = "Qq"
    out = run(CPU_VANITY, "--suffix", suffix, "--table", TABLE, "--threads", "4")
    found = [l for l in out.splitlines() if l.startswith("FOUND")][0]
    seed = found.split("seed=")[1].split()[0]

    # miner's pub matches the python reference
    pub_hex = found.split("pub=")[1].strip()
    assert ed25519_ref.seed_to_public(bytes.fromhex(seed)).hex() == pub_hex

    # write an OpenSSH private key, ask ssh-keygen for its public half
    priv = tmp_path / "key"
    run(sys.executable, os.path.join(ROOT, "tools", "keytool.py"),
        "privkey", seed, str(priv), "vanity-test")
    sshpub = run("ssh-keygen", "-y", "-f", str(priv)).split()
    assert sshpub[1].endswith(suffix)
    assert sshpub[1] == keytool.pub_line(bytes.fromhex(pub_hex), "").split()[1]


def test_openssh_key_loads_and_fingerprints(tmp_path):
    seed = secrets.token_bytes(32)
    priv = tmp_path / "key"
    run(sys.executable, os.path.join(ROOT, "tools", "keytool.py"),
        "privkey", seed.hex(), str(priv), "fp-test")
    out = run("ssh-keygen", "-l", "-f", str(priv))
    assert "ED25519" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))

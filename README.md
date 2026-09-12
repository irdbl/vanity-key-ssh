# vanity-key-ssh

CUDA-accelerated vanity ed25519 SSH key generator, with vast.ai fleet
scripts for distributed hunts.

Hunts for keys whose `authorized_keys` line **ends** with a chosen base64
suffix, e.g. `ssh-ed25519 AAAA...++pham`.

## Why the suffix, and the math

An ed25519 SSH public key blob is exactly 51 bytes (19-byte fixed header +
32-byte ed25519 point), which base64-encodes to exactly 68 characters with
no padding — every character is fully significant. The first ~25 chars are
fixed by the header, so the end of the string is where vanity goes. A
suffix of **L** base64 characters pins exactly **6·L bits** of the public
key, so a random key matches with probability 64^-L.

Keys/sec (measured / planning estimates):

| Hardware | keys/s |
|---|---|
| Apple M-series laptop, 14 threads (measured) | ~0.9M |
| RTX 4090, 16-bit signed comb (measured 2026-09) | **~334M** |
| RTX 4090, 8-bit comb / small-L2 fallback (measured) | ~208M |

Expected cost at 334M keys/s per GPU and ~$0.35/hr for a 4090 on vast.ai
(cost scales with total attempts, so fleet size changes wall-clock, not $):

| Suffix len | Expected attempts | 1× 4090 (mean) | 10× 4090 (mean) | Expected $ |
|---|---|---|---|---|
| 5 | 2^30 ≈ 1.1e9  | 5 s | — | ~$0.001 |
| 6 | 2^36 ≈ 6.9e10 | 6 min | 35 s | ~$0.03 |
| 7 | 2^42 ≈ 4.4e12 | 6 h | 37 min | ~$2 |
| 8 | 2^48 ≈ 2.8e14 | 9.8 days | 23 h | ~$82 |
| 9 | 2^54 ≈ 1.8e16 | 1.7 years | 2 months | ~$5,200 |

The search is geometric: median = 0.69× the mean, but there's a long tail —
you need 3× the mean for 95% confidence. **Realistic ceiling: 8 characters.**
7 is cheap, 8 is a committed weekend + ~$150–400, 9 is out of reach.

Notes:
- Any base64 chars work (`A-Za-z0-9+/`); all suffixes of equal length are
  equally hard.
- **Case-insensitive and multi-word hunts** (`--ci`, repeatable `--suffix`)
  divide difficulty by the number of accepted variants at ~0.4% measured
  overhead: each letter is worth 1 bit back. Ten case-insensitive letters
  cost 2^50 instead of 2^60 — that is what makes a 10-char suffix real:

  | Hunt | Effective difficulty | 30x 4090 fleet (mean) | Expected $ |
  |---|---|---|---|
  | 9 CI letters | 2^45 | ~1 h | ~$10 |
  | **10 CI letters** | **2^50** | **~31 h** | **~$330 on-demand, ~$215 with BID=1** |

  (334M keys/s per 4090; suffixes up to 10 chars, same length, <= 2^20 variants.)
- **No incremental-addition trick.** `mkp224o` (Tor v3 onion vanity) skips
  the per-candidate scalar mult by walking `A += 8B` and keeping the raw
  scalar — ~15-25× faster. It is deliberately *not* used here: an OpenSSH
  private key stores the 32-byte seed and derives the scalar as
  `clamp(SHA512(seed))`, so a found scalar has no seed preimage and cannot be
  written as a standard key file. Every candidate therefore pays a full
  fixed-base scalar mult (224 field muls, the floor for seed-derived keys).
  Tor gets away with it only because its key format stores the expanded
  scalar directly.

## Quickstart

```bash
python3 tools/gen_table.py table.bin            # 768KB 8-bit table (CPU/tests)
python3 tools/gen_table.py table16.bin --wide   # 48MB 16-bit table (GPUs), ~1 min
make host                              # CPU searcher + test harness
make test                              # cross-checks vs RFC 8032 + ssh-keygen

# CPU hunt (fine up to ~5 chars):
./bin/cpu_vanity --suffix pham --table table.bin

# GPU hunt (needs nvcc; on the CUDA box):
make gpu
./bin/gpu_vanity --suffix ++pham --table table16.bin
./bin/gpu_vanity --suffix KevinPham1 --ci --table table16.bin   # any capitalization
./bin/gpu_vanity --suffix ++pham --benchmark   # measure keys/s and exit
```

On a find:

```
FOUND seed=<64 hex chars> pub=<64 hex chars>
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...++pham vanity
```

Recover a usable key pair from the seed (locally, never on the rented box):

```bash
python3 tools/keytool.py privkey <seed-hex> ~/.ssh/id_ed25519_vanity "you@host"
python3 tools/keytool.py verify  <seed-hex> ++pham   # independent re-check
ssh-keygen -y -f ~/.ssh/id_ed25519_vanity            # OpenSSH agrees
```

## Distributed hunt on vast.ai

```bash
pip install vastai && vastai set api-key <YOUR_KEY>

# push this repo somewhere public first (instances clone + build it):
git remote add origin git@github.com:you/vanity-key-ssh.git && git push -u origin master

SUFFIX='++pham' COUNT=10 MAX_DPH=0.45 ./scripts/vast_launch.sh
AUTO_DESTROY=1 ./scripts/vast_watch.sh     # polls logs, tears fleet down on find
```

`vast_launch.sh` records the instance IDs it creates in `.vast_fleet`, and
`vast_watch.sh` only ever monitors/destroys that list — it never touches
unrelated instances in your account. Delete `.vast_fleet` to start a fresh
hunt record. Set `BID=1` on launch for interruptible (spot) bids: the search
is memoryless, so being outbid costs only the in-flight launch, typically for
30-50% less than on-demand.

No coordination is needed between workers: each instance draws a random
128-bit base for its seed space, and the search is memoryless — the chance
of two workers colliding is negligible, so N workers ≈ N× throughput.
Optional `NTFY_TOPIC=<topic>` sends a content-free ntfy.sh ping on a find
(the seed itself never leaves the instance's logs).

## Security model — read this

- **The rented GPU host can see the private key.** The winning seed is
  computed and logged on machines you don't control. A vanity SSH key found
  in the cloud should be treated accordingly: fine for looks/convenience,
  but if your threat model includes "the GPU landlord kept my key", hunt on
  your own hardware instead (same binary, no vast scripts).
- Seeds are `16 random bytes (CSPRNG) || worker id || counter` — 128 bits of
  entropy, hashed through SHA-512 per RFC 8032 like any normal ed25519 key.
  A vanity suffix reveals nothing an attacker couldn't read off your public
  key anyway; security of the found keys is standard ed25519.
- `keytool.py privkey` writes the key file with mode 0600, unencrypted, and
  refuses to overwrite an existing file (so a found key is never silently
  clobbered or left with a pre-existing laxer mode). Add a passphrase
  afterwards: `ssh-keygen -p -f ~/.ssh/id_ed25519_vanity`.

## How it's fast

- **No base64 in the hot loop**: the suffix compiles to a byte mask over the
  raw 32-byte public key (`tools/keytool.py target`, mirrored in C).
- **16-bit signed comb** (GPU): `table[i][j] = j·2^(16i)·B` magnitudes,
  16×32769 niels entries (~48MB, L2-resident on a 4090). The clamped scalar is
  recoded into 16 signed digits, so a key costs 16 mixed additions — no
  doublings — with digit signs applied by branchless point negation. A 768KB
  8-bit table (32 windows) remains for CPUs and small-L2 GPUs; the loader
  picks by file size. Measured 334 vs 208 Mkeys/s on a 4090.
- **Batch inversion**: each GPU thread computes 16 points, then one Montgomery
  batch inversion amortizes the ~254-squaring field inversion across all 16.
- **5×51-bit field arithmetic** (donna-style) with `unsigned __int128`
  accumulators; identical code compiles for host (tests) and device.
- SHA-512 is a single compression block for 32-byte seeds.

The same headers power `bin/cpu_vanity`, `bin/test_host`, and the CUDA
kernel, so the full pipeline — including the batch-inversion structure — is
validated on any machine against RFC 8032 vectors, a pure-Python reference,
and `ssh-keygen` itself (`make test`).

## Layout

```
src/f25519.h      field arithmetic GF(2^255-19)
src/ge25519.h     point ops + fixed-base table walk
src/sha512.h      single-block SHA-512
src/pipeline.h    seed -> pubkey -> match
src/main.cu       CUDA kernel + host driver
src/cpu_main.cpp  CPU searcher (same pipeline)
src/test_host.cpp harness used by tests/
tools/            pure-Python reference, table generator, key tooling
scripts/          vast.ai fleet launch/watch, container entrypoint
```

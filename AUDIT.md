# Performance audit

Date: 2026-09-12. Scope: `src/*` hot path, `Makefile`, `scripts/vast_*.sh`.
Read-only audit — no code changed. A follow-up audit with reproduced defects
and validation limits appears below. Its qualifications supersede conflicting
claims of certainty or measured performance in the original audit.

The implementation is sound: fixed-base comb, Montgomery batch inversion on the
GPU path, mask compare instead of base64 in the inner loop, and one set of
headers validated against RFC 8032 / `ssh-keygen`. Findings below are
optimizations missed, not defects. A short list of things that were checked and
found **correct** is at the bottom, so future work doesn't re-derive them.

## Cost model used throughout

Counting field multiplications (`fe_mul`; `fe_sq` currently aliases it):

| Operation | `fe_mul` count |
|---|---|
| `ge_madd` | 7 |
| `ge_scalarmult_base` (32 windows) | 224 |
| `fe_invert` (254 sq + 11 mul) | 265 |
| compress (`x·zinv`, `y·zinv`) | 2 |
| batch chain bookkeeping | 2 / key |

- **GPU per key, K=16:** 224 + 2 + 2 + 265/16 ≈ **245**
- **CPU per key:** 224 + 265 + 2 = **491**

Sanity check against the measured CPU number: 0.9 M keys/s over 14 threads =
64 k keys/s/core × 491 = 31 M `fe_mul`/s/core ≈ 110 cycles per `fe_mul` at
~3.5 GHz. That matches a 5×51 limb multiply with `__int128` on arm64, so the
model is calibrated and the percentages below are trustworthy.

## Summary, ranked by payoff

GPU device-code, kernel-launch, and fleet-composition findings (F4, F5, F9–F13,
F18, F19) live in `AUDIT2.md` — they need an `nvcc` build and on-device
profiling to validate. This file covers the CPU, host-build, and safety
findings, plus F1/F2 (structural / product decisions).

| # | Finding | Payoff | Effort | Breaks key format? |
|---|---|---|---|---|
| F1 | Incremental point addition instead of per-key scalarmult | **15–25×** | high | **yes** |
| F2 | Multi-target / case-insensitive matching | **N×** (16× for `++pham`) | low | no |
| F3 | CPU searcher does not batch-invert | **2×** (CPU only) | low | no |
| F6 | Dedicated `fe_sq` | 19% CPU pre-F3, 2.5% after | low | no |
| F7 | Special-case first and last comb iterations | 3.1% | low | no |
| F8 | Host build uses `-O2`, no `-march=native` | unmeasured | trivial | no |
| F14 | No `-Xptxas -v`; spills unknown | diagnostic | trivial | no |
| F15 | `GPU_ARCH` stops at `sm_90`, no consumer Blackwell | startup only | trivial | no |
| F16 | CPU attempt counter is a per-key shared atomic | 1–9% | trivial | no |
| F17 | On-demand instances instead of interruptible bids | **30–50% of $** | low | no |
| F20 | Accept the match at any of the last few positions | free multiplier | low | no |

---

## F1 — Incremental point addition (the structural one)

`src/ge25519.h:41`. Because the scalar is `clamp(SHA512(seed))`, every candidate
needs a full 32-window comb walk: 224 of the 245 `fe_mul` per key. Random
scalars from a hash share no structure, so **224 is the floor for
seed-derived keys** — there is no algorithmic trick below it.

The alternative, as used by `mkp224o` for Tor v3 onion addresses: pick one
random scalar `a`, compute `A = a·B` once, then walk `A += 8B` with a single
precomputed mixed addition per candidate. The scalar for hit *i* is `a + 8i`;
incrementing by 8 preserves the `scalar[0] &= 248` clamping invariant. SHA-512
disappears, and so does the table — the increment point lives in registers,
which also removes ~3 KB/key of L2 traffic.

Per-key cost: **245 → ~10 `fe_mul`.** Call it 15–25× after memory effects.

**Cost:** there is no seed. The OpenSSH private key stores the 32-byte seed and
derives the scalar by hashing it, so expressing a found scalar as a key file
would need a SHA-512 preimage. `tools/keytool.py privkey` cannot work. Usable
only via a custom signer holding the raw scalar (~150 lines; the nonce prefix
can be any random 32 bytes and signatures still verify — only the public key
participates in verification). Tor gets away with it because its
`hs_ed25519_secret_key` stores the expanded scalar directly.

This changes the ceiling the README advertises:

| Suffix | current (1× 4090) | with F1 | with F1, 10× fleet |
|---|---|---|---|
| 8 | 16 days | ~1.5 days | ~4 h |
| 9 | 2.9 years | ~3.5 months | ~10 days (~$900) |

Nine characters moves from "out of reach" to "expensive but real". Treat this
as a product decision about key format, not a patch. If the answer is no, say
so explicitly in the README — it is the first thing anyone familiar with
`mkp224o` will ask about.

## F2 — Multi-target matching

Already flagged at `README.md:64`; it is the best ratio in the repo. `++pham`
has four letters → 16 case variants → **16× for a few ALU ops per key**.
Base64 case changes are not bit flips (`P`=15, `p`=41), so it has to be N
independent compares, but against 245 `fe_mul` a linear scan of N ≤ 64 targets
in `__constant__` memory is invisible. Beyond ~hundreds of targets, prefilter
with a 2^16 bitmap on the low 16 bits.

Same mechanism gives two more multipliers for free: hunt several distinct words
at once, and F20.

**Outcome — IMPLEMENTED, measured 2026-09-12 (RTX 4090).** `--ci` and
repeatable `--suffix` compile at startup to a sorted list of w3 targets plus a
2^16-bit shared-memory bitmap prefilter; only bitmap hits (~2^-13 of keys) pay
for the x-parity and a binary search. Measured overhead with 1024 targets
(10 CI letters): **0.4%** (333.4 vs 334.7 Mkeys/s). C and Python expansions
cross-checked in tests; live GPU hunt for `--suffix kevinp --ci` produced
`...KEvinp` and `...kEVINp` keys, verified via `keytool.py verify --ci`.
Constraints: all suffixes same length, <= 10 chars, <= 2^20 variants.
F20 (position relaxation) remains open.

## F3 — CPU searcher does not batch-invert

`src/cpu_main.cpp:51` calls `vk_seed_to_pub`, which routes through
`ge_compress` → `fe_invert(p.Z)`: one 265-mul inversion **per key**. Only the
CUDA kernel batches.

491 → 245 `fe_mul` per key is a straight **2×**; the measured 0.9 M keys/s
should become ~1.8 M. Lift the K-loop from `src/main.cu:50-71` into the worker
lambda — it is the same code, and `tests/` already validates that structure via
`test_host`.

Note `README.md` currently says the batch-inversion structure is "validated on
any machine", which is true of `bin/test_host` but misleading about
`bin/cpu_vanity`.

## F6 — Dedicated `fe_sq`

`src/f25519.h:58` is `fe_mul(f, f)`. A real squaring saves ~35% of the
multiplies by computing symmetric cross terms once. It only appears in
`fe_invert` (254 squarings), so: **~19% of the un-batched CPU path**, ~2.5%
once F3 lands.

## F7 — Special-case the first and last comb iterations

`src/ge25519.h:41-52`.

**First addition is to the identity** (`X=0, Y=1, Z=1, T=0`), so `ge_madd`
collapses: `A = yplusx`, `B = yminusx`, `C = 0`, `Z3 = T3 = 2`, hence
`r.X = 2·X3`, `r.Y = 2·Y3`, `r.Z = 4`, and only `r.T = X3·Y3` needs a multiply.
**7 `fe_mul` → 1.**

**Last addition does not need `T`** — nothing reads `r.T` after the loop, so
skip `r.T = fe_mul(X3, Y3)` when `i == 31`. **1 `fe_mul` saved.**

7 of 224 = **3.1%** for two peeled iterations.

## F8 — Host build flags

`Makefile:3` is `-O2 -std=c++17 -Wall`. `-O3 -march=native` is free and on x86
hosts lets the compiler use ADX/MULX for exactly these carry chains. Matters
for `cpu_vanity` and for any x86 vast box running the CPU fallback.

## F14 — No spill diagnostics

`Makefile:11` has no `-Xptxas -v`. Register count and spill store/load counts
are the numbers needed to tune `__launch_bounds__` / `-maxrregcount`, and to
confirm F5's effect. Add it.

## F15 — `GPU_ARCH` ends at `sm_90`

`Makefile:5-10` covers Turing through Hopper. The comment says
"Turing..Blackwell", but consumer Blackwell is `sm_120`; an RTX 5090 falls back
to JIT from the `compute_90` PTX. Add `sm_100`/`sm_120`. (Fleet builds use
`-arch=native`, so this only affects prebuilt binaries.)

## F16 — CPU attempt counter

`src/cpu_main.cpp:52` does `g_attempts.fetch_add` per key — a shared cacheline
RMW from every thread. Accumulate in a local and flush every 1024.

## F17 — Use interruptible bids

`scripts/vast_launch.sh:60` creates on-demand instances. A memoryless search
with no checkpoint state is the ideal interruptible workload: losing an instance
costs only its in-flight launch, and `vast_watch.sh` already tolerates
instances disappearing. Typically **30–50% off**, which is the largest single
lever on the dollar column of the README table. `vastai create instance --bid`.

## F20 — Relax the match position

Accepting the suffix at any of the last *m* positions rather than pinned to the
end multiplies the hit rate by about *m*−L+1, for the same per-key cost as F2.
Worth offering as a flag even if the default stays exact.

---

## Checked and correct — do not re-derive

- **Difficulty is exactly 64^L and uniform.** The blob is 19 header + 32 key =
  51 bytes = 68 base64 chars with no padding, so every character is exactly 6
  bits of key material. No alignment slop at the tail, no padding character to
  dilute the search.
- **`vk_suffix_to_target` is right** (`src/vanity.h:16`). The big-endian
  shift-by-6 accumulation and the mask walk both land correctly: at L=1,
  `mask[31] = 0x3f` matches the last base64 char = low 6 bits of `pub[31]`; at
  L=2, `mask[31] = 0xff` and `mask[30] = 0x0f` match chars 67–68 = blob bits
  396–407. Verified against the blob bit layout.
- **The x sign bit is genuinely constrained for L ≥ 2.** It is bit 7 of
  `pub[31]`, blob bit 400, inside base64 char 67. So it cannot be skipped for
  any realistic suffix — F9 defers it, it does not drop it.
- **Targeting the suffix is the right choice.** Chars 1–25 are fixed by the
  header (`AAAAC3NzaC1lZDI1NTE5AAAAI`) and char 26 straddles the final length
  byte, so it is always `A`–`P`. Only the tail is freely searchable.
- **No duplicate candidates.** Seeds are `base(16) || gtid(8) || counter(8)`;
  each thread covers `[counter_base, counter_base + K)` and `counter_base`
  advances by exactly `K` per launch (`src/main.cu:124`). Distinct across both
  threads and launches.
- **No `fe` limb overflow.** Worst case through `ge_madd` is `fe_sub` output
  (<2^54.5) squared and scaled by 19×5, about 2^115 in the `vk_u128`
  accumulators. Comfortable margin.
- **Montgomery batch inversion chain is correct** (`src/main.cu:54-61`), and
  the `pts[k].Z` values cannot be zero for valid points.
- **Table loads are already read-only-cached.** `__restrict__` on the kernel
  parameter plus 16-byte-aligned 96-byte entries lets nvcc emit
  `ld.global.nc.v2.u64`; an explicit `__ldg` would be belt-and-braces only.
- **Multi-GPU scales linearly, with no code changes.** There is no
  inter-GPU communication: the 768 KB table is uploaded once per process, the
  only per-launch host traffic is a 4-byte flag read, and each process seeds
  itself from `/dev/urandom`. No NVLink, no PCIe contention, no sharding. N
  GPUs is exactly N×. See F19.
- **Memory bandwidth is not the bottleneck.** ~3.5 KB/key (3072 B table +
  ~400 B spill) at 200 M keys/s is ~700 GB/s against Ada's multi-TB/s L2, with
  the 768 KB table fully L2-resident. The kernel is ALU-bound, which is why F4
  ranks above F5.

## Suggested order

1. **F3, F6, F8, F16** — CPU path only, no GPU needed to validate, ~2.4×
   combined on `cpu_vanity`.
2. **F2, F7** — kernel changes covered by the existing `make test`
   cross-checks; 16× × ~1.05 for `++pham`.
3. **F14** — enable spill diagnostics (trivial; a prerequisite for the GPU
   tuning tracked in `AUDIT2.md`).
4. **F17** — dollars, independent of all the above.
5. **F1** — only after deciding whether non-OpenSSH key files are acceptable.

The GPU device-code, kernel-launch, and fleet-composition findings
(**F4, F5, F9–F13, F18, F19**) moved to `AUDIT2.md`; they need an `nvcc`
build and on-device profiling to validate and carry their own ordering there.

## Follow-up audit — correctness and performance validation

Date: 2026-09-12. Additional scope: `tools/keytool.py`,
`scripts/entrypoint.sh`, CPU benchmark behavior, and the existing test suite.
No implementation changes were made.

### Validation and confidence

- `make test` could not run with the default Python because pytest was missing.
  `uv run --with pytest python -m pytest tests/ -q` passed all **9 tests**, using
  the existing host binaries. These cover RFC vectors, random reference
  comparisons, batch inversion, SHA-512, suffix matching, and OpenSSH loading.
- No demonstrated error was found in the core seed-to-public-key calculation.
  Passing host tests does not establish CUDA compiler or device correctness.
- This machine has no `nvcc`; CUDA compilation, generated instructions,
  register usage, local-memory allocation, and throughput were not verified.
- The original audit's GPU speedups, bottleneck classification, and README's
  100–300M keys/s estimate are hypotheses, not measurements from this audit.
  In particular, neither an ALU-bound kernel nor a specific local-memory
  penalty is established. A host fallback cannot validate device-specific PTX.

### A2 — Confirmed: CPU benchmark can report dramatically false low speeds

`src/cpu_main.cpp:53-66` stops workers on a match even with `--benchmark`, while
the main thread still sleeps five seconds. Reproduction:

```sh
./bin/cpu_vanity --suffix A --threads 1 --benchmark
```

The observed run found a match after 76 attempts, then reported
`76 attempts in 5.0s = 15 keys/s`. This measures mostly idle time. The precise
attempt count varies with the random seed. An early `--limit` also leaves idle
time in the denominator. Benchmark mode should continue after matches or time
only the actual work, and normal benchmark completion should return success
instead of the current exit status 1.

### A3 — CPU inversion and shared counter costs (confirms F3/F16)

`src/cpu_main.cpp:51` uses a full inversion for every candidate, despite the
batch implementation exercised by the host test harness. Batching is a strong
optimization candidate, but the reduction in field-operation count is not a
measured 2× throughput guarantee. The shared atomic increment at line 52 also
creates avoidable contention; aggregate attempts per worker.

### A4 — Confirmed: private-key output retains unsafe existing permissions

`tools/keytool.py:128` opens the destination with
`O_WRONLY | O_CREAT | O_TRUNC` and mode `0600`. The mode only applies when a
file is created. In a temporary-directory check, overwriting an existing
`0644` file left the new private key at **0644**, readable by other local users
where directory permissions permit access. `O_TRUNC` also silently overwrites
an existing private key.

Prefer exclusive creation with `O_EXCL`, refusing existing destinations, and
handle the public-key destination consistently to avoid unintended overwrites.

### A5 — Serious: automatic cleanup includes unrelated account instances

`scripts/vast_watch.sh:12-17` lists every instance in the account, and lines
34-38 destroy that entire list when `AUTO_DESTROY=1` and any match is observed.
The launcher does not persist a hunt-specific instance list. Unrelated
workloads can therefore be destroyed.

Persist the IDs created for each hunt, and restrict monitoring and destruction
to that list. This finding comes from source inspection; no cloud instances
were created or destroyed during the audit.

### A6 — Cleanup targets tee rather than the GPU search processes

`scripts/entrypoint.sh:17-24` records `$!` for each background pipeline, whose
last process is `tee`. Killing those PIDs does not directly terminate the
miners. They can continue computing until a later pipe write fails, instead
of stopping promptly after another GPU finds a match.

Track miner PIDs explicitly or manage process groups, then terminate and wait
for all workers. This was identified by source inspection, not a GPU run.

### A7 — Wide arithmetic warrants profiling (qualifies F4/F6)

`src/f25519.h:36-58` expresses 25 wide products per field multiplication and
implements squaring through general multiplication. This is a potentially
costly GPU arithmetic design, not a demonstrated correctness defect. Inspect
generated code before claiming a particular lowering or speedup from a new
limb representation or dedicated squaring implementation.

### Follow-up priorities

1. Correct private-key file creation and scope fleet destruction to hunt IDs.
2. Fix CPU benchmark semantics so throughput measurements are meaningful.
3. Obtain CUDA resource diagnostics and measured batch-size sweeps before
   relying on GPU throughput or cost estimates.
4. Evaluate CPU batching and counter aggregation, then profile GPU arithmetic.

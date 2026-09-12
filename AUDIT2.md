# Performance audit — GPU / device & fleet findings (AUDIT2)

Companion to `AUDIT.md`. These findings were split out because they live in
CUDA device code, the kernel launch loop, or vast.ai fleet composition — they
require an `nvcc` build and on-device profiling to validate, which `AUDIT.md`'s
host test suite cannot do. Payoff figures are estimates until measured. See
`AUDIT.md` for the cost model, the CPU/build/safety findings, and the
"Checked and correct" list; cross-references there to F4/F5/F9-F13/F18/F19
point into this file.

## Measured outcomes (2026-09-12, RTX 4090 @ vast.ai, `-arch=native`, nvcc 12.4)

Baseline at session start: **190.8 Mkeys/s** (K=16). After this file plus
AUDIT3 G1: **333.9 Mkeys/s** (K=32, 16-bit signed comb).

| # | Result |
|---|---|
| F18 | **Done.** `--benchmark`/`--limit` verified on-device; also added `--selftest` (512 device pubkeys diffed against the host reference — this is what lets kernel changes land safely). |
| F5/A1 | **Done, footprint only.** Template K + dropping T: frame 6656→2816 B (K=16), 896 B (K=4); throughput unchanged — the kernel was never spill-bound. Frees ~1 GB VRAM reservation, makes 8 GB cards viable. |
| F9/F10/F11 | **Done, ~0% each.** Word-path SHA-512/scalar, one-u64 y-compare, deferred x. Individually unmeasurable under the scalarmult's dependency chain; kept as hygiene (regs 138→130) and as groundwork for multi-target. |
| F4 | **REFUTED on nvcc 12.4.** Two hand PTX variants (mul.lo/hi + add.cc, and fused mad.lo.cc/madc.hi) both measured ~174–176 vs 191 Mkeys/s portable — nvcc's own `__int128` lowering wins; inline-asm blocks defeat its scheduling. Reverted; do not re-attempt without SASS-level evidence. |
| (new) | **maxrregcount=128: +6%** (190.8→201.9). Occupancy was the real ALU-side lever; 112 break-even, ≤96 loses to spills. In the Makefile. |
| F13 | **Done.** Mapped found-flag + 3-deep event ring; the hunt loop no longer drains the GPU per launch. Small (~1%), mostly latency hygiene. |
| F12 | **Skipped, documented.** Pure-ALU trims (F9–F11) measured 0%, so the ~3% unpack saving is not on the critical path; superseded by AUDIT3 G1 halving the window count. |
| F19 | **Done.** vast_launch.sh ranks by $/key (rate table, 4090 measured), accepts multi-GPU offers with cpu_cores ≥ GPUs, COUNT now means total GPUs. |
| K sweep | K=32 default (206.5 vs 201.7 @ K=16 on 8-bit; 333.9 vs 324.8 on 16-bit comb). |

The through-line: the kernel is bound by the *dependent-addition chain* of the
comb walk, not by instruction count, local memory, or the compare path. That
is why only occupancy (+6%) and AUDIT3 G1's halving of the chain (+61%) moved
the number.

## Summary

| # | Finding | Payoff | Effort | Breaks key format? |
|---|---|---|---|---|
| F4 | `unsigned __int128` field arithmetic on device | **1.5–2×** (est.) | med | no |
| F5 | Local-memory blowup in the kernel | unmeasured, likely 10–30% | med | no |
| F9 | Defer the x-coordinate until y matches | ~1% | low | no |
| F10 | Compare one `uint64_t`, skip the byte splat | 1–3% | low | no |
| F11 | Eliminate `seed`/`digest`/`scalar` local arrays | ~6% of local traffic | low | no |
| F12 | Pre-unpacked (5×51) table entries | ~3% ALU, +25% bytes | low | no |
| F13 | Per-launch synchronous `cudaMemcpy` drains the GPU | 1–3% | low | no |
| F18 | `gpu_vanity` has no `--benchmark` / `--limit` | blocks F5, F19 tuning | low | no |
| F19 | Offer selection: single-GPU filter, ranks by $/hr not $/key | 10–30% of $ | low | no |

## Suggested order

1. **F18** — add the benchmark mode; it gates measuring everything else here.
   *(Implemented in `main.cu`; still needs an on-device build to verify.)*
2. **F5** (with A1 below) — size the kernel frame, then re-sweep `K`.
3. **F4** — biggest non-structural GPU win; needs care and a PTX path.
4. **F9, F10, F11, F12, F13** — kernel micro-optimizations; land after F5's
   frame changes so their effect is measurable.
5. **F19** — fleet composition / offer ranking; needs a measured rate table
   from F18.

---

## F4 — `unsigned __int128` on the device

`src/f25519.h:7`. nvcc lowers 128-bit multiply-accumulate chains through
generic helpers: `mul.lo.u64` + `mul.hi.u64` per product (each several 32-bit
IMADs) plus compiler-generated carry propagation over the five accumulators.

Standard fixes, in increasing order of work:
1. `__umul64hi(a,b)` for the high half plus inline PTX `add.cc.u64` / `addc.u64`
   accumulation, replacing the `vk_u128` temporaries.
2. A 10×25.5-bit limb representation over `mul.wide.u32` (donna-c32 style),
   which maps directly onto the 32-bit IMAD pipe.

Typically worth **1.5–2×**; verify before committing. Headroom supports it:
~73 k ALU ops/key against a 4090's ~41 T int-ops/s is ~560 M keys/s
theoretical versus the 100–300 M estimate in the README.

Keep the portable `__int128` path under `#else` so the existing RFC 8032 tests
validate the PTX path against it on any host.

---

## F5 — Local-memory blowup in the kernel

`src/main.cu:46-47`:

```cpp
ge_p3 pts[MAX_K];   // MAX_K = 32, ge_p3 = 4 fe = 160 B  -> 5120 B
fe    prods[MAX_K]; //                         40 B      -> 1280 B
```

Three compounding problems:

1. **`MAX_K` sizes the frame regardless of `K`.** ~6.5 KB/thread always. CUDA
   reserves local memory for maximum residency, so on a 4090 (128 SMs × 1536
   threads) that is ~1.3 GB of VRAM held for arrays half of which are never
   touched at the default K=16.
2. **`K` is a runtime field** (`p.keys_per_thread`), so the loops cannot unroll
   and the dynamically-indexed arrays are guaranteed local memory. Make it a
   template parameter with a small dispatch switch.
3. **`T` is dead after `ge_scalarmult_base`** — `ge_compress_with_zinv` reads
   only X, Y, Z. Storing it is 25% of the spill traffic for nothing.

Template + exact sizing + dropping `T` takes the frame from ~6.5 KB to ~2 KB.

Then re-sweep `K`, because the batch is already deep into diminishing returns:
inversion costs 16.5 `fe_mul`/key at K=16, 8.3 at K=32 (4% better), 33 at K=8
(6% worse). If spill traffic dominates, **K=8 may beat K=16**. This needs F18 to
measure.

---

## F9 — Defer the x-coordinate

`src/ge25519.h:56-63` always computes `fe_mul(p.X, zinv)` and a full
`fe_tobytes(xb, x)` to extract one parity bit. The x sign bit is bit 7 of
`pub[31]`, i.e. exactly **one** of the 6L constrained bits. Test the other
6L−1 bits against y first; compute x only for ~1 in 2^(6L−1) candidates.
With multi-target (F2), mask bit 63 out of the first-pass compare and use the
full target on confirm.

---

## F10 — Compare one `uint64_t`

For L ≤ 10 every constrained byte lies in bytes 24–31, which is exactly the
`w3` word `fe_tobytes` already computes at `src/f25519.h:131` before splatting
it to bytes. Return `w3` instead, mask, compare. That removes both 32-byte
store loops in `ge_compress_with_zinv` and the byte loop in `vk_match`
(`src/pipeline.h:32`). L=10 is far beyond any feasible hunt, so the narrow
path covers every real case; keep the byte version for the tests.

---

## F11 — Local byte arrays in the per-key path

`src/pipeline.h:6-20` and `src/sha512.h:41`. `vk_make_seed` writes `seed[32]`,
`sha512_32` reads it back byte-by-byte to build four words, then splats
**64** digest bytes when only 32 are used, then `vk_seed_to_scalar` copies 32
of them into `scalar[32]`. That is ~200 B/key of local traffic for data that
fits in registers.

Build the four message words directly from `base`/`worker`/`counter`, return
the first four digest words, and index the scalar with
`(w[i>>3] >> (8*(i&7))) & 0xff`.

Minor related note: seeds differ only in bytes 24–31 (the counter), so W[0..2]
and SHA-512 rounds 0–2 are constant per thread and could be hoisted. That is
~0.1% of total — recorded for completeness, not worth doing.

---

## F12 — Pre-unpacked table entries

`src/ge25519.h:47-49` runs `fe_from_u64x4` three times per window, 32 windows
per key — roughly 3% of ALU. Emitting 5×51 limbs from `tools/gen_table.py`
removes it at the cost of 96 → 120 B entries (768 KB → 960 KB, still fully
L2-resident on anything from a 3090 up). Measure; likely a small net win.

---

## F13 — Per-launch synchronous copy

`src/main.cu:125` uses blocking `cudaMemcpy` after every launch, so the GPU
drains completely between kernels and nothing is queued behind the current
wave. Use `cudaHostAlloc(..., cudaHostAllocMapped)` for the found flag and poll
it from the host while two or three launches stay in flight. 1–3%.

---

## F18 — `gpu_vanity` cannot be benchmarked

`bin/cpu_vanity` has `--benchmark` and `--limit`; `bin/gpu_vanity` has neither
(`src/main.cu:77-85`), so the only way to learn a GPU's rate is to start a real
hunt and read the 5-second rate line. Blocks both the F5 `K` sweep and the
F19 offer ranking. Add a fixed-attempt mode that prints keys/s and exits.

---

## F19 — Offer selection and fleet composition

`scripts/vast_launch.sh:37` filters `num_gpus=1` and `-o 'dph'` ranks by price
per hour. Both are wrong for this workload, and the multi-GPU support is
already written.

### Multi-GPU instances are already supported

`scripts/entrypoint.sh:13` reads `nvidia-smi -L` and forks one
`gpu_vanity --device $i` per GPU. Each process calls `vk_random_base`
independently, so N processes on one host get N independent 128-bit bases with
no coordination needed. The single-GPU filter in the offer search is the only
thing blocking it.

Multi-GPU hosts are worth preferring on two counts beyond any per-GPU discount
the host offers:

- **Setup amortizes per instance, not per GPU.** `ONSTART` runs `apt-get
  update`, a clone, `gen_table.py` (~1 min) and `make gpu`. An 8-GPU box pays
  that once instead of eight times — small in dollars, but it is also dead time
  before hashing starts.
- **Fewer instances to track and destroy.** `README.md` already flags runaway
  billing as the main operational risk; two instances is meaningfully safer
  than sixteen.

Two mild gotchas:

- Each `gpu_vanity` process spins a host thread in the synchronous memcpy poll
  loop (F13), so filter on `cpu_cores_effective >= num_gpus`.
- An interruptible multi-GPU box that gets outbid loses every GPU at once.
  Because the search is memoryless this costs only the in-flight launch, so it
  raises wall-clock variance but not expected spend. Not worth diversifying
  hosts to avoid.

### Rank by $/key, not $/hr

Expected cost is proportional to total attempts, so fleet **size** buys
wall-clock only and fleet **composition** matters solely through dollars per
key. The kernel is 64-bit integer multiply-accumulate, so the figure of merit
is INT32 IMAD throughput. Both Ampere GA10x and Ada AD10x SMs expose 128 FP32
lanes but only **64 INT32** lanes — the doubled FP32 datapath contributes
nothing here, which is why naive FP32 TFLOPS rankings mislead.

| GPU | INT32 lanes | ~T IMAD/s | assumed $/hr | keys per $ (4090 = 1.00) |
|---|---|---|---|---|
| RTX 4090 | 8,192 | 20.6 | $0.40 | 1.00 |
| RTX 3090 | 5,248 | 8.9 | $0.20 | 0.87 |
| RTX 3090 (cheap day) | 5,248 | 8.9 | $0.15 | 1.15 |
| RTX 3080 | 4,352 | 7.4 | $0.14 | 1.03 |
| RTX 2080 Ti | 4,352 | 6.7 | $0.12 | 1.08 |
| RTX 4080 | 4,864 | 12.2 | $0.30 | 0.79 |
| RTX 3060 | 1,792 | 3.2 | $0.08 | 0.78 |
| A100 80GB | 6,912 | 9.7 | $1.00 | 0.19 |
| H100 | 8,448 | 14.9 | $2.50 | 0.12 |

**The dollar figures are estimates, not measurements — vast pricing moves.**
Treat the ranking as a hypothesis to be replaced by measured rates (F18). The
structural conclusions do hold regardless of price drift:

- Consumer Ampere, Turing and Ada all sit within about ±20% of each other.
  A fleet of 3090s is roughly break-even with 4090s and wins outright whenever
  3090s are cheap; it takes ~2.3 3090s to match one 4090's wall-clock.
- **Never rent datacenter cards for this.** A100/H100 price in HBM and tensor
  cores, neither of which this kernel touches: 5–8× worse value.
- **Avoid Pascal at any price** (1080 Ti, P106, P100) — 32-bit integer multiply
  lowers to multi-instruction sequences there. `GPU_ARCH` correctly starts at
  `sm_75`.
- VRAM is a non-constraint even with F5's ~1.3 GB local-memory reservation, so
  rank purely on compute per dollar. Fixing F5 drops it to ~400 MB and makes
  8 GB cards eligible.

Keep the relative magnitudes in mind: model choice is a ±20% decision, while
interruptible bidding (F17) is 30–50%. Do F17 first.

### Concrete changes

1. Drop `num_gpus=1`; add `cpu_cores_effective >= num_gpus`.
2. Redefine `COUNT` as **total GPUs** and pack instances until filled. It
   currently means instances, which silently multiplies spend on multi-GPU
   offers.
3. Rank by `dph_total / num_gpus / rate(gpu_name)` against a measured rate
   table, which needs F18.

---

## Follow-up audit

### A1 — Likely significant GPU scratch-memory cost (extends F5)

`src/main.cu:46-47` declares 6,400 bytes per thread for `pts[MAX_K]` and
`prods[MAX_K]`. At K=16, the active entries occupy 3,200 bytes. Dynamic indexing
and the large arrays strongly suggest local-memory traffic and cache pressure,
but actual allocation and impact require compiler diagnostics and profiling.
NVIDIA discusses this behavior in
[Fast Dynamic Indexing of Private Arrays in CUDA](https://developer.nvidia.com/blog/fast-dynamic-indexing-private-arrays-cuda/).

Inspect compiler resource reports and benchmark smaller batch sizes. Consider
fixed-K specializations and retaining only X/Y/Z after scalar multiplication;
T is unused during compression. Specialization alone does not guarantee that
large arrays fit in registers. Do not assign a percentage speedup before
measurement.

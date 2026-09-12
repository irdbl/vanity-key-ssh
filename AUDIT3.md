# Performance audit — closing the gap to 300M keys/s (AUDIT3)

Date: 2026-09-12. Scope: `tools/gen_table.py`, `src/ge25519.h`, the fixed-base
comb. Prompted by a comparison against
[`pcarrier/vanity-keygen`](https://github.com/pcarrier/vanity-keygen/), which
reports **~300M keypairs/s on an RTX 4090**. This repo currently measures
**~190M keys/s** on the same card.

This is not tracked in `AUDIT.md` (CPU/build/safety) or `AUDIT2.md` (GPU
device-code micro-opts). It is a single structural change to the scalar-mult:
**a wider comb window**. It is compatible with the OpenSSH seed→scalar key
format — unlike the incremental-addition trick (`AUDIT.md` F1), which we
rejected — so it is a real, shippable lever, not a product decision.

## The comparison is apples-to-apples

`vanity-keygen` does the same work per key that we do: ed25519 SSH keys,
`SHA-512(seed)` → clamp → fixed-window scalar multiplication against a
precomputed basepoint table, mixed twisted-Edwards additions in
`(y+x, y−x, 2d·xy)` niels form, radix-2⁵¹ 5-limb field arithmetic with 128-bit
products, one Montgomery batch inversion shared across the work item. No
incremental point addition, no alternate key format, no X25519/age shortcut.
So the ~300M is a legitimate target — the difference is purely how cheaply each
key's scalar mult is computed.

## G1 — Use a 16-bit signed comb window (32 additions → 16)

`tools/gen_table.py:32-39`, `src/ge25519.h:44-68`.

The scalar mult dominates cost: ~224 of the ~245 `fe_mul`/key, and each mixed
addition (`ge_madd`) is 7 `fe_mul`. The number of additions equals the number
of comb windows.

| | this repo | `vanity-keygen` |
|---|---|---|
| Window size | 8-bit | **16-bit, signed** |
| Windows = mixed adds / key | **32** | **16** |
| Entries / window | 256 | 32,769 (= 2¹⁵ + 1) |
| Table size (96 B/entry) | 0.77 MB | ~48 MB |
| Batch K | 16 | 32 |

Halving the window count removes ~16 × 7 = **112 `fe_mul`/key**, taking the
per-key budget from ~245 to ~133 — a ~1.8× reduction in the dominant term.
That alone more than spans 190M → 300M; the larger table's extra memory traffic
gives some of it back, which is consistent with the observed ~1.6× rather than a
full 1.8×.

### Why *signed* windows specifically

An unsigned 16-bit window needs 2¹⁶ entries/window ≈ 100 MB total, which blows
past a 4090's ~72 MB L2. Signed-digit recoding stores only magnitudes
`0..2¹⁵` (32,769 entries, the sign applied by negating the looked-up point),
halving the table to ~48 MB so it stays — just barely — L2-resident. Keeping
the table in L2 is what makes the wider window a net win instead of a
memory-bound loss; this is the crux of the design, not an incidental detail.

Point negation in niels form is cheap and branch-light: `−(y+x, y−x, 2d·xy)`
is `(y−x, y+x, −(2d·xy))`, i.e. swap the first two limbs and negate the third.

## Cost model

Counting `fe_mul` per key (K = 32 for the batch column, matching the target):

| Stage | 8-bit comb (now) | 16-bit signed comb |
|---|---|---|
| Scalar mult (`ge_madd` × windows × 7) | 32 × 7 = 224 | 16 × 7 = 112 |
| Batch inversion (265 / K) | 265/16 ≈ 16.5 | 265/32 ≈ 8.3 |
| Compress (x·zinv, y·zinv) | 2 | 2 |
| Chain bookkeeping | ~2 | ~2 |
| **Total** | **~245** | **~124** |

Ratio ~1.97× in field muls. Real throughput gain is less (memory traffic from
the 48 MB table), so budget ~1.5–1.8× and **verify on the card** — see
`AUDIT2.md` F18 (`--benchmark`) and F14 (`-Xptxas -v`).

## Implementation sketch

1. **`tools/gen_table.py`** — emit **16 windows** of `2^(16·i)·B` multiples,
   **32,769 magnitudes** each (`j·2^(16·i)·B` for `j` in `0..2¹⁵`), niels form.
   New table size 16 × 32769 × 96 ≈ 48 MB; update `VK_TABLE_ENTRIES` /
   `VK_TABLE_BYTES` in `src/ge25519.h:13-14`.
2. **Scalar recoding** (`src/ge25519.h`, both `ge_scalarmult_base_w` and the
   byte wrapper) — convert the clamped scalar into 16 signed 16-bit digits:
   read each 16-bit limb, and if a digit exceeds 2¹⁵ subtract 2¹⁶ and carry +1
   into the next window. The top clamp bit (bit 254 set, bit 255 clear) bounds
   the final carry so no 17th window is needed; confirm the top-window digit
   stays in range.
3. **Window loop** — 16 iterations: look up the magnitude entry, conditionally
   negate the niels point by the digit's sign, `ge_madd`. Keep the F7 peels
   (first add to identity → 1 mul; last add skips `T`) — they still apply and
   are now a larger fraction of a shorter loop.
4. **Re-tune K toward 32** and re-sweep (`AUDIT2.md` F5): the shorter scalar
   mult shifts the batch-inversion break-even.

## Tradeoffs and risks

- **Table generation time.** 48 MB of points from the pure-Python reference
  (`tools/ed25519_ref.py`) will be slow to generate (~64× more points than
  now). Consider computing each window's magnitudes by repeated addition (as
  the current generator does) rather than `point_mul` per entry, and/or caching
  `table.bin`.
- **L2 residency is card-dependent.** ~48 MB fits a 4090 (72 MB L2) and a 5090,
  but older/smaller-L2 cards (e.g. 3060, 8 MB) will thrash and may be *slower*
  than the 8-bit comb. This optimization assumes a large-L2 target; keep the
  8-bit path selectable, or pick the window size from the table-fits-in-L2
  budget at build time.
- **VRAM.** 48 MB per process is still trivial against any modern card, and the
  `AUDIT2.md` F5 local-memory work already freed the headroom.
- **Correctness.** The signed-digit recoding is the only subtle part (carry
  propagation and the sign/negation). The existing host suite validates it end
  to end: `test_host pub`/`pubbatch` against the Python reference and
  `ssh-keygen`. GPU codegen is covered by the `--selftest` kernel (`main.cu`).

## Expected outcome

Field-op count says ~1.97×; realistic target ~1.5–1.8× after memory effects,
i.e. **~285–340M keys/s on a 4090**, in line with `vanity-keygen`'s ~300M.
This is the single change most likely to close the gap. Everything in
`AUDIT2.md` (F4 device arithmetic, F5 frame sizing, etc.) stacks on top and is
independent.

## Outcome — IMPLEMENTED, measured 2026-09-12 (RTX 4090, vast.ai)

| build | Mkeys/s |
|---|---|
| session baseline (8-bit comb, K=16) | 190.8 |
| + AUDIT2 F5/F9/F10/F11/F13, maxrregcount=128, K=32 (8-bit comb) | 208.0 |
| + **G1 16-bit signed comb** (K=32) | **333.9** |

**1.75× over baseline, 1.61× from G1 alone** — top of the predicted range,
and past `vanity-keygen`'s ~300M. `--selftest` 512/512 vs the host reference;
host suite validates the signed recoding against RFC 8032 + `ssh-keygen`.
Table generation with batch inversion takes ~1 min in pure Python. K=32 beats
K=16 by ~3% (324.8), as predicted by the inversion-amortization row. The
8-bit table stays selectable (auto-detected by file size) for small-L2 cards.

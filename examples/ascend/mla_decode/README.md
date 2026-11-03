# MLA decode (Ascend NPU)

DeepSeek-style multi-head latent attention (MLA) decode on Ascend 950PR
(`dav-3510`):

    O[b,h,:] = softmax( (Q[b,h,:] KV[b,s,:] + Q_pe[b,h,:] K_pe[b,s,:]) * scale ) * KV[b,s,:]

Two kernels live here:

* `example_mla_decode.py` - the **native decode kernel**: one launch, one KV pass
  per core, no split-KV. It is the reference implementation and the baseline for
  the speedups below.
* `example_mla_decode_paged.py` - the **paged decode kernel**: the KV cache is
  addressed through `block_table` + `cache_seqlens`, the KV blocks are split
  across cores, and a second kernel merges the per-split partials. Phase 1 and
  the merge kernel live in this one module.
* `mla_decode_persistent.py` - the same phase 1 underneath `T.Persistent`: a
  fixed grid of one block per AI core walks the (batch, head tile, split) item
  space, which balances a ragged batch and (measured) beats the paged kernel by
  35% there, at the cost of the cross-block overlap the paged grid gets for free
  on uniform multi-batch work.

Measured on the reference shape (batch=1, heads=128, kv_len=8192, dim=512,
pe_dim=64): native 1549.4 us, paged page 64 **52.6 us** = **29.5x**,
43.4 TFLOPS, relative error 5.4e-4 against the fp32 torch reference.

## Files

```
example_mla_decode.py        # native decode: num_split=1, one launch per call
example_mla_decode_paged.py  # paged decode: split-KV phase 1 + the merge kernel
test_mla_decode.py           # native kernel: 5 shapes against the fp32 reference
test_mla_decode_paged.py     # paged: 21 cases (page geometry, shuffled tables, ragged lengths, empty splits, multi-batch)
mla_decode_persistent.py     # persistent phase 1 (T.Persistent, one block per core) + the shared merge
bench/bench_paged.py         # paged latency + correctness guard, compared with the native kernel
bench/bench_persistent.py    # multi-batch / ragged batch: native vs paged vs persistent
bench/prof_paged.py          # single-kernel msprof entry for the paged phase 1
bench/prof_multi.py          # msprof entry for one kernel of native / paged / persistent
bench/parse_pipes.py         # summarise msprof captures (pipes, waits, L2, bandwidth, MTE counts)
VERIFICATION.md              # self-verification report
```

The correctness suite is 26 cases: 5 native and 21 paged. Everything validates
against an fp32 torch reference with `rtol=atol=1e-2`, and every case is
repeatable across calls.

## Native decode kernel

`mla_decode_fwd` / `compile_mla_decode` is the straightforward Ascend port: one
program per `(batch, head tile)` block, each program walks the whole KV once.

* QK is computed in two halves (pe then content) accumulating into the same L0C,
  so the two halves never need two score tiles in UB.
* softmax and the `o = o*alpha + pv` rescale run on the vector units; PV is
  chunked over output columns, with a transposed V^T chunk in L1 per output
  chunk (an NT GEMM, see the platform constraints).
* Measured 1549.4 us (1.47 TFLOPS) on the reference shape: with one launch and no
  split-KV the 16 AI cores each sweep 8192 tokens' worth of KV, so the kernel is
  memory/vector bound rather than cube bound. It is kept as the reference and
  the baseline, not as the production path.

## Paged decode kernel

The paged kernel is the production path. It splits the KV blocks across the 16
AI cores, runs a first pass that publishes per-split partial attention output
plus log-sum-exp (`O_partial [num_split, batch, heads, dim]`, `ML [num_split,
batch, heads, 8]`), and a second pass that merges the splits with the standard
flash-decoding weights.

Paged addressing:

    page  = block_table[b, (kv_blk * block_N + pc * PAGE_ROWS) // block_size]
    slot  = page * block_size + (kv_blk * block_N + pc * PAGE_ROWS) % block_size

and the live length is per batch (`cache_seqlens[b]`). Four consequences shape
the implementation:

* **one tile may span several pages but never straddles an address range.** A
  `block_N` tile is assembled from `NUM_PC` page chunks of `PAGE_ROWS` tokens
  (both `block_size > block_N` and `block_size <= block_N` work), and each page
  chunk gets its own L1/L0C/UB buffers because a GM->L1 copy with a runtime
  source offset must land on a zero-based full-row region and `T.gemm` sizes K
  from the whole buffer.
* **the rescale still happens once per tile, not once per page**: the row max
  and row sum are combined across the page chunks before `o_ub` is touched.
* **the trip count is runtime**: live KV blocks are split over cores as
  `per = nblk // NS` plus one extra for the first `nblk % NS` splits; a split
  that gets zero blocks publishes `lse = -inf` and zeros, which the merge
  weights to exactly 0. `tiles_per_core` is a compile-time upper bound for the
  scheduler.
* **phase 2 is a shared merge kernel** defined in the same module: it gathers
  the per-split log-sum-exp, computes `exp(l_i - max)`, weights and accumulates
  the partials on the vector units, and writes the fp16 output. It handles batch
  and head tiles independently of phase 1.

Defaults are the measured optimum for the DeepSeek page geometry
(`block_size=64`): `block_N=64` (one page per tile), `chunk_d=128`,
`num_split=16`, `vector_mode="simd_rc"` (the full-SIMD modes are rejected: a
per-page score tile is only 64 columns wide, the explicit MicroAPI path measured
no gain and only adds UB pressure), `partial_dtype="float32"` (fp16 partials
measured slower here), `merge_rows=16`, `merge_stages=4`,
`merge_vector_mode="simd"`.

Phase 1 runs a **two-tile CV-overlap schedule** for the shipped page geometry
(`select_phase1_kernel()` picks it for `block_N=64`, `chunk_d=128`, one page
chunk per tile): K/K_pe of two KV tiles are prefetched, both QK GEMMs are
issued, and only then does the vector unit run the two softmax passes and the
two PV passes, so the cube keeps working while the vector units run the softmax
of the other tile. The two QK results live in two L0C tiles; the UB score buffer
is reused, which keeps the dynamic UBUF budget unchanged. Other geometries
(`block_N=128` with 64-token pages) keep the single-tile loop.

## Performance

Reference shape, `do_bench(backend="event", _n_warmup=10, _n_repeat=50)`, 3
rounds, median; host load 75-132 during the run below (the host is shared, see
the verification report).

| case | e2e | TFLOPS | rel | speedup vs native |
|---|---:|---:|---:|---:|
| native decode (num_split=1) | 1549.4 us | 1.47 | 6.2e-4 | 1.0x |
| **paged, page 64, block_N=64, ns=16 (default, two-tile schedule)** | **52.6 us** | **43.4** | 5.4e-4 | **29.5x** |
| paged, page 128 (two-tile schedule) | 53.0 us | 43.1 | 5.5e-4 | 29.3x |
| paged, block_N=128 (single-tile schedule) | 59.3 us | 38.5 | 5.4e-4 | 26.1x |
| paged, num_split=32 (two-tile schedule) | 70.6 us | 32.3 | 5.4e-4 | 21.9x |

Page 128 and page 64 are equally fast; `block_N=128` falls back to the
single-tile schedule and is slower; extra splits are slower. The two-tile
schedule is worth **-7.6% end-to-end / -8.6% phase 1** against the single-tile
schedule on the reference shape.

## Multi-batch and ragged batches

`bench/bench_persistent.py` sweeps the three kernels over uniform and ragged
batches (`do_bench` event timing, 3 rounds, median; the dense native kernel needs
a uniform length, so it runs on the uniform rows only):

| case | native | paged | persistent | persistent vs paged |
|---|---:|---:|---:|---:|
| B=1, [8192] | 1549.5 us | 53.8 us | 53.7 us | 1.00x |
| B=2, [8192]*2 | 1556.1 us | 102.6 us | 99.6 us | 1.03x |
| B=4, [8192]*4 | 1556.6 us | 156.0 us | 185.8 us | 0.84x |
| B=8, [8192]*8 | 1557.8 us | 260.2 us | 359.5 us | 0.72x |
| B=2, [8192, 1024] | n/a | 100.4 us | **74.5 us** | **1.35x** |
| B=4, [8192, 2048, 512, 4096] | n/a | 106.9 us | 119.6 us | 0.89x |

Why the split result: the paged grid is `batch * splits` blocks, so at batch>=2
two blocks share a core's AIC and AIV ports and the next block's cube starts
while the previous block's vector tail drains -- free cross-block overlap. The
persistent grid is one block per core, so its items serialise (no overlap) but
the work per core is exactly one split of every batch, which is why a ragged
batch is balanced instead of waiting for its longest sequence.

`msprof op` utilisation on the same runs (`bench/prof_multi.py`, parsed by
`bench/parse_pipes.py`; % of the 432 TFLOPS / 4 TB/s peaks):

| kernel (case) | grid | duration | TFLOPS | %cube | GB/s | %GM |
|---|---:|---:|---:|---:|---:|---:|
| native B=4 | 4 | 1556.5 us | 1.5 | 0.3% | 46 | 1.2% |
| paged B=4 | 64 | 128.8 us | 17.7 | 4.1% | 627 | 15.7% |
| persistent B=4 | 16 | 161.3 us | 14.1 | 3.3% | 501 | 12.5% |
| paged ragged B=2 | 32 | 85.9 us | 26.6 | 6.1% | 470 | 11.7% |
| persistent ragged B=2 | 16 | 57.5 us | 39.7 | 9.2% | 470 | 11.7% |

The native kernel launches `batch` blocks (Block Dim 4 at batch=4), so it leaves
12 of the 16 cores idle and is vector-bound (AIV 90% of the duration): that is
where its 1.5 TFLOPS comes from. Use `compile_mla_decode_paged` for uniform
multi-batch serving and `compile_mla_decode_persistent` when the batch is ragged
(the persistent prototype supports the shipped page geometry at fp32 partials
only; other shapes fall back to the paged kernel).

### batch=64, float16 KV-length sweep

`bench/bench_batch64.py` (same KV data for all three kernels: identity page table,
so the dense native view and the paged view share one allocation; event timing,
2 rounds, median, host load 100-120):

| kv_len | tokens | native | paged (ns=16) | persistent (ns=16) | paged / native | paged TFLOPS | paged GB/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 65 536 | 0.63 ms | 0.85 ms | 1.23 ms | 0.74x | 21.4 | 168 |
| 2048 | 131 072 | 1.20 ms | 0.97 ms | 1.24 ms | 1.24x | 37.8 | 295 |
| 4096 | 262 144 | 2.36 ms | 1.27 ms | 1.77 ms | 1.86x | 57.6 | 450 |
| 8192 | 524 288 | 4.67 ms | 1.88 ms | 2.84 ms | 2.48x | 77.5 | 606 |
| 16384 | 1 048 576 | 9.30 ms | 3.19 ms | 4.96 ms | 2.91x | 91.7 | 716 |
| 32768 | 2 097 152 | 18.57 ms | 5.85 ms | 9.21 ms | 3.17x | 99.8 | 780 |

Correctness: relative error of batch element 0 against the fp32 torch reference is
3.9e-4 ... 7.8e-4 for all three kernels and all lengths.

What it shows:

* the native kernel is linear (`8.9 ns/token`) but capped at 29-31 TFLOPS and
  ~245 GB/s for every length: its grid is `batch` blocks, so 64 blocks over 16
  cores run in four waves and each block is vector bound;
* the paged kernel's per-token cost drops from 13.0 ns (kv=1024) to 2.8 ns
  (kv=32768) because the 256 MiB of fp32 partials (16 splits x 64 batch x 128
  heads x 512 dim) is a fixed cost; it only beats the native kernel from
  kv >= 2048. At 32768 it reaches 99.8 TFLOPS (23% of the cube peak) and
  788 GB/s (19.7% of the GM peak);
* the persistent kernel is 0.63-0.83x of paged at batch=64: with 64 items per
  core it loses the cross-block AIC/AIV overlap that the paged grid gets from its
  `batch * splits` blocks (its win is the ragged batch, section 4);
* fp16 partials do not help at this batch size either (kv=1024: 0.88 vs 0.85 ms,
  kv=8192: 1.91 vs 1.88 ms): the halved traffic is offset by the widening pass in
  the merge.

**Split count matters as much as the kernel.** At batch=64 the optimal
`num_split` scales with the KV length (measured, same harness):

| kv_len | best num_split | paged | TFLOPS | GB/s | paged / native |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1 (also 2) | 0.29 ms | 63.5 | 496 | 2.2x |
| 1024 | 4 | 0.35 ms | 51.8 | 404 | 1.8x |
| 1024 | 8 | 0.51 ms | 35.7 | 279 | 1.2x |
| 4096 | 4 | 0.86 ms | 85.4 | 667 | 2.8x |
| 8192 | 8 | 1.60 ms | 91.5 | 715 | 2.9x |
| 16384 | 8 | 2.94 ms | 99.2 | 775 | 3.2x |
| 32768 | 8 | 5.79 ms | 100.9 | 788 | 3.2x |

`compile_mla_decode_paged(num_split=None)` now sizes this automatically with
`pick_num_split(max_pages * block_size, block_N, batch)`: about 16 KV tiles per
split, at least `16 / batch` splits (so a single request still fills the machine)
and at most 8 splits per batch element (beyond that the extra partial traffic buys
nothing). The resulting table is 16/16/16/16/16/16 for batch=1 and
1/2/4/8/8/8 for batch=64, which matches the measured optima, so a serving layer
gets the short-KV win without passing anything: batch=64 / kv=1024 measures
0.29 ms (2.2x native) instead of 0.85 ms with the old fixed 16, and kv=8192
1.65 ms instead of 1.88 ms.

### Where the 90% utilisation target stands

Per KV tile (block_N=64) the pipes need `cube 1.32 us`, `mte2 1.68 us`,
`vec ~1.49 us` per AIV and `fixpipe 0.98 us`. The measured block wall is
2.55 us/tile at batch=64 / kv=32768, so the pipes overlap only partially and even
a perfect overlap would stop at `max(...) = 1.68 us` -- **cube utilisation is
capped near 79% by MTE2, not by the cube**. The cause is the double KV read
(K for QK + transposed V for PV = 136 KB/tile against 72 KB for a single read),
which the NT-only L1 GEMM forces. Cutting MTE2 to one read (0.89 us/tile) puts the
cube at `1.32 / 0.9 = 1.47 us` of wall, the first configuration where 90% cube
utilisation is reachable -- provided the cube/vector overlap is also perfect,
which the auto-scheduler does not deliver today.

Cube and GM cannot both approach 90% on this workload: at 90% cube the kernel runs
~3.0 ms for batch=64 / kv=32768 and moves ~2.4 GB, i.e. ~790 GB/s (20% of the GM
peak); saturating the GM peak would need ~890 TFLOPS, twice the fp16 peak. The
realistic target is cube-bound at 90% with ~40% GM.

Two structural attempts toward that target were measured and rejected (see the
dead-end table): folding the softmax scale/mask pass into the row-max reducer is
16-18% slower (a reducer over a select/multiply expression lowers to a scalar
pass, and the fully fused variant needs +16 KiB of UB), and computing PV as
`P @ K` on L0 operands -- the single-read path -- compiles but faults the AI core
because the K tile is K-major for the QK operand and MN-major for the NN PV
operand, which the L1->L0B copy cannot convert. The single-read path did land on the **PTO
backend**: `compile_mla_decode_paged(pv_l0=True)` computes PV as `P @ K` on L0
operands (the compile entry selects `target="pto"` automatically for it), so the
KV tensor is read once -- 72 KB/tile instead of 136 KB -- and the batch=64 numbers
improve by 5% (kv=8192: 1.65 -> 1.56 ms, kv=32768: 5.86 -> 5.47 ms with
`pair_stages=2`, relative error unchanged). The AscendC backend faults on the same
program, and `msprof` cannot profile PTO kernels here, so `pv_l0` stays opt-in and
the default remains AscendC.

With the single read the per-tile pipe budget becomes cube 1.32 us, mte2 0.89 us,
vec ~1.49 us and fixpipe 0.98 us, i.e. the vector pipe is now the longest and a
perfectly overlapped kernel would reach 1.49 us/tile (cube ceiling ~89%). The
measured wall is 2.64 us/tile, so the remaining gap is the dependency chain the
auto-scheduler does not overlap -- the utilisation target is blocked by
cross-iteration multi-buffering, not by traffic any more. Forcing that buffering
from the kernel is not a workaround: pinning the score/p tiles to two versions
makes the compiler emit a silently wrong kernel (0.2 ms, zero/garbage output,
verified with fresh data), `pair_stages=2` alone moves less than 1%, and three
tile slots, the fused softmax, `vector_mode="simt"` and block_N=128 all measured
slower. The kernel-side work (automatic split sizing, single KV read, the two-tile
schedule) is at its measured optimum on this compiler.

## Profile (msprof)

Captured with

```bash
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache \
    --output=/dev/shm/prof/paged python bench/prof_paged.py --block-size 64
python bench/parse_pipes.py paged=/dev/shm/prof/paged
```

Paged phase 1 - 43.6 us with the two-tile schedule (block dim 16; mix kernel =
16 AIC + 32 AIV):

| pipe | busy | share of duration |
|---|---:|---:|
| AIC cube (MAD) | 10.55 us | 24.2% |
| AIC mte2 (GM->L1) | 13.22 us | 30.3% |
| AIC mte1 (L1->L0) | 7.90 us | 18.1% |
| AIC fixpipe (L0C->UB) | 7.86 us | 18.0% |
| AIC scalar | 1.99 us | 4.6% |
| AIV vec | 29.05 us | 66.7% |
| AIV mte3 (UB->GM) | 2.64 us | 6.1% |

* waits: `aiv_vec_wait` 84.4%, `aiv_mte3_wait` 94.7%, `aic_cube_wait` 94.5%,
  `aic_mte1_wait` 95.3%, **`aic_mte2_wait` 56.8%**
* MTE instruction counts: **mte2 166, mte1 178**
* L2 read hit (AIC) 53.4%; GM->L1 1232 KB/block, pipe bandwidth usage 16.6%
* achieved 52.4 TFLOPS (12.1% of the fp16 cube peak) and 463 GB/s (11.6% of the
  4 TB/s GM peak)

The two-tile schedule raises the vector-unit occupancy from 58.5% to 66.7% of
the duration and cuts the wall clock by 8.6% (single-tile phase 1: 47.7 us).
The ideal cube/vector overlap ceiling is
`max(cube 10.55, mte2 13.22, vec 29.05) = 29.1 us` against 43.6 us measured, so
roughly half of the dependency-chain stall is still there; it needs the PV and
softmax phases of the two tiles to overlap as well, not just QK with softmax.

Bandwidth utilisation across data volume (page 64, batch 1, two-tile schedule
where the geometry allows it): 467 GB/s at 1 MiB of KV cache up to 526 GB/s at
72 MiB, i.e. 11.7% -> 13.2% of the 4 TB/s peak; e2e 23.2 / 52.6 / 153.9 / 287.2
us for kv_len 1024 / 8192 / 32768 / 65536. No tier is bandwidth limited; the
fixed per-launch cost dominates the small tiers and the kernel is linear in the
data volume above ~8 MiB.

## Platform constraints

These are the Ascend 950PR constraints that shaped both kernels; they are not
negotiable in the current tilelang release:

* `block_H == 128`: `T.dual_copy` (L0C->UB) requires M=128, and it is what makes
  one fixpipe instruction write both AIV halves of the tile.
* **L1 GEMM is NT only** (`trans_A=False, trans_B=True`). A GEMM B operand
  therefore has to be stored transposed; PV reads V^T into per-output-chunk L1
  buffers instead of reusing the K tile.
* a GM->L1 copy with a runtime source offset must target a zero-based region
  covering the full row axis, so paged pages cannot fill a row slice of a shared
  tile.
* `T.gemm` derives K from the whole buffer, not from the region, so a score
  accumulator cannot be written column-by-column (hence the per-page buffers).
* UB budget: 221184 B of dynamic UBUF (248 KiB pool minus the 32 KiB SIMT
  reservation); the paged kernel sits at ~200 KiB. L1 is 512 KiB per core.
* `T.copy(ub->l1, transpose=True)` compiles but **silently ignores the
  transpose** (wrong values); a raw UB->L1 copy does not produce the NZ layout a
  GEMM needs; an L1->L1 transpose does not compile (no DMAPath). The KV is
  therefore streamed from GM twice (K and V^T) and there is no way to avoid it
  in this release.
* software pipelines / multi-buffering: `T.Pipelined` is used for the KV and
  split loops, but forcing extra buffer versions on the cross-unit tiles
  (`annotate_buffer_versions`) measured slower in every configuration that fit,
  and `unit_flag_ctrl=3` (pipelined MAD+FIXPIPE) hangs at runtime because it
  requires a matching flag protocol.

## Known limitations and measured dead ends

* paged phase 1 is ~1.6x the MTE2 instruction count of the dense variant
  (page-chunk operands); removing the OOB clamp/fill reduces it to 102 but
  raises `aic_mte2_wait` from 61% to 76% and is slower overall.
* a compile-time trip count for the split loop is slower; an affine page fast
  path (no `block_table` load) recovers exactly the OOB loss, net zero.
* paged with fp16 partials is slower (56.9 -> 59.4 us with 32-column staging,
  68.7 us with a full-width staging buffer), so paged keeps fp32 partials.
* `block_N=32` (to free UB for a hand-written double buffer) costs +43% phase 1.
* page 128 and page 64 are equally fast; extra splits (`num_split=32`) are
  slower.
* restructuring the GEMMs does not pay either: one QK GEMM over a packed
  576-wide `dim+pe_dim` measured +6.7% phase 1, and wider (2*chunk_d) PV GEMMs
  are rejected by the compiler (L0C->UB copy must be zero-origin). The
  scheduler knobs (`num_stages`, merge `rows_per_block`/`num_stages`/`chunk_d`)
  are all at their measured optimum.
* Cube/Vector overlap is only partially solved: the two-tile schedule overlaps
  the QK of one tile with the softmax of the other (phase 1 47.7 -> 43.6 us), but
  the PV phase and the second softmax still serialise. Pushing further is not
  reachable by rearranging this kernel -- three slots, V^T double buffering,
  PV/softmax reordering and a deeper pair loop all measured slower or neutral.
  The next step would be a hand-written `T.MixedKernel` with explicit
  `T.Cube()` / `T.Vector()` regions and manual flags, or compiler support for
  cross-iteration multi-buffering.

## Running

```bash
export TILELANG_DEFAULT_TARGET=ascend

# correctness: 26 cases
python -m pytest test_mla_decode.py test_mla_decode_paged.py -q

# latency + correctness guard (native reference measured first in-process)
python bench/bench_paged.py

# paged phase-1 profile (see the Profile section for the command line)
python bench/prof_paged.py --block-size 64
```

The host is shared: `bench_paged.py` reports the median of three rounds and the
msprof task duration comes from hardware counters. Numbers taken while the load
is high are visible as a larger spread between rounds; re-take them when the
machine is quiet.

## Verification

[`VERIFICATION.md`](VERIFICATION.md) is the self-verification report for this
tree: environment, the 26/26 correctness run, end-to-end numbers, the paged
msprof profile, bandwidth utilisation across data volume, and the measured dead
ends listed above.

# MLA decode (Ascend NPU)

DeepSeek-style multi-head latent attention (MLA) decode on Ascend 950PR
(`dav-3510`):

    O[b,h,:] = softmax( (Q[b,h,:] KV[b,s,:] + Q_pe[b,h,:] K_pe[b,s,:]) * scale ) * KV[b,s,:]

Two kernels live here:

* `example_mla_decode.py` — the **native decode kernel**: one launch, one KV pass
  per core, no split-KV. It is the reference implementation and the baseline for
  the speedups below.
* `example_mla_decode_paged.py` — the **paged decode kernel**: the KV cache is
  addressed through `block_table` + `cache_seqlens`, the KV blocks are split
  across cores, and a second kernel merges the per-split partials. Phase 1 and
  the merge kernel live in this one module.

Measured on the reference shape (batch=1, heads=128, kv_len=8192, dim=512,
pe_dim=64): native 1549.4 us, paged page 64 **56.9-57.3 us** = **27.2x**,
40.0 TFLOPS, relative error 5.4e-4 against the fp32 torch reference.

## Files

```
example_mla_decode.py        # native decode: num_split=1, one launch per call
example_mla_decode_paged.py  # paged decode: split-KV phase 1 + the merge kernel
test_mla_decode.py           # native kernel: 5 shapes against the fp32 reference
test_mla_decode_paged.py     # paged: 21 cases (page geometry, shuffled tables, ragged lengths, empty splits, multi-batch)
bench/bench_paged.py         # paged latency + correctness guard, compared with the native kernel
bench/prof_paged.py          # single-kernel msprof entry for the paged phase 1
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

## Performance

Reference shape, `do_bench(backend="event", _n_warmup=10, _n_repeat=50)`, 3
rounds, median; host load 75-132 during the run below (the host is shared, see
the verification report).

| case | e2e | TFLOPS | rel | speedup vs native |
|---|---:|---:|---:|---:|
| native decode (num_split=1) | 1549.4 us | 1.47 | 6.2e-4 | 1.0x |
| **paged, page 64, block_N=64, ns=16 (default)** | **56.9 us** | **40.1** | 5.4e-4 | **27.2x** |
| paged, page 128 | 57.1 us | 40.0 | 5.5e-4 | 27.1x |
| paged, block_N=128 (previous default) | 60.8 us | 37.5 | 5.4e-4 | 25.5x |
| paged, num_split=32 | 73.8 us | 30.9 | 5.4e-4 | 21.0x |

Page 128 and page 64 are equally fast; `block_N=128` and extra splits are
slower, which is why the defaults are what they are.

## Profile (msprof)

Captured with

```bash
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache \
    --output=/dev/shm/prof/paged python bench/prof_paged.py --block-size 64
python bench/parse_pipes.py paged=/dev/shm/prof/paged
```

Paged phase 1 — 47.7 us (block dim 16; mix kernel = 16 AIC + 32 AIV):

| pipe | busy | share of duration |
|---|---:|---:|
| AIC cube (MAD) | 10.55 us | 22.1% |
| AIC mte2 (GM->L1) | 13.62 us | 28.6% |
| AIC mte1 (L1->L0) | 7.77 us | 16.3% |
| AIC fixpipe (L0C->UB) | 6.75 us | 14.2% |
| AIC scalar | 2.31 us | 4.9% |
| AIV vec | 27.86 us | 58.5% |
| AIV mte3 (UB->GM) | 2.21 us | 4.6% |

* waits: `aiv_vec_wait` 86.1%, `aiv_mte3_wait` 95.3%, `aic_cube_wait` 95.1%,
  `aic_mte1_wait` 96.5%, **`aic_mte2_wait` 61.5%**
* MTE instruction counts: **mte2 166, mte1 178**
* L2 read hit (AIC) 53.1%; GM->L1 1232 KB/block, pipe bandwidth usage 15.2%
* achieved 47.9 TFLOPS (11.1% of the fp16 cube peak) and 424 GB/s (10.6% of the
  4 TB/s GM peak)

The split of the wall clock is the dependency chain, not a single saturated
unit: the ideal cube/vector overlap ceiling is
`max(cube 10.55, mte2 13.62, vec 27.86) = 27.9 us` against 47.7 us measured.
The paged kernel carries ~1.6x the MTE2 instructions of the dense variant
because every page-chunk operand needs its own copy (166 vs 102), and that is
what keeps `aic_mte2_wait` at 61%.

Bandwidth utilisation across data volume (measured 2026-10-30 with the then
shipped sweep driver; the paged kernel is unchanged since): paged throughput
49.3 GB/s at 0.56 MiB of KV cache up to 438.9 GB/s at 72 MiB, i.e. 1.2% -> 11.0%
of the 4 TB/s peak. No tier is bandwidth limited; the fixed per-launch cost
(~12-15 us) dominates the small tiers and the kernel is linear in the data
volume above ~8 MiB.

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
* the remaining large win is Cube/Vector overlap: the auto-scheduler cannot
  overlap the softmax of tile i with the QK of tile i+1 across the ping-pong
  dependency. Reaching the 27.9 us ceiling needs a hand-written
  `T.MixedKernel` with explicit `T.Cube()` / `T.Vector()` regions and manual
  flags, or compiler support for cross-iteration multi-buffering.

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

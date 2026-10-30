# MLA decode (Ascend NPU)

Multi-head Latent Attention (MLA) decode forward, as used by DeepSeek-V2/V3, for
both the dense and the paged KV cache layouts on Ascend 950. The KV cache holds
the compressed latent (`dim` wide) plus the decoupled RoPE key (`pe_dim` wide),
so one score needs two GEMMs and one attention output needs the latent part of
the cache only:

```
O[b,h,:] = sum_s softmax( (Q[b,h,:] @ KV[b,s,:] + Q_pe[b,h,:] @ K_pe[b,s,:]) * scale ) @ KV[b,s,:]
scale    = (dim + pe_dim) ** -0.5
```

`Q`/`KV` are fp16, every softmax statistic and accumulator is fp32, and `O` is
fp16. Decode has one query row per head, so the arithmetic intensity is low: the
whole problem for the reference shape (`batch=1, heads=128, kv_len=8192, dim=512,
pe_dim=64`) is 2.2817 GFLOP against 8 MiB of latent KV (9 MiB including the RoPE
key).

## Files

```
mla_decode_api.py                # layout dispatch + measured-best defaults, one entry point
example_mla_decode.py            # baseline: num_split=1, one launch, one KV pass per core
example_mla_decode_splitk.py     # dense split-KV phase 1, the shared merge kernel, pick_tiling
example_mla_decode_paged.py      # paged split-KV phase 1 (imports the merge kernel)
test_mla_decode.py               # baseline kernel, 5 shapes, either backend
test_mla_decode_splitk.py        # split-KV regression + cross-check against num_split=1
test_mla_decode_splitk_matrix.py # shape/edge matrix for the dense split-KV kernel
test_mla_decode_splitk_tiles.py  # block_N / chunk_d / num_stages sweep
test_mla_decode_paged.py         # paged semantics: shuffled pages, variable lengths, both geometries, multi-batch, empty splits
test_mla_decode_coverage.py      # dense gaps: batch=16, dim/pe combinations, long tail, extra tiles, call hygiene, soak
test_mla_decode_api.py           # dispatch/defaulting policy, no device required
bench/bench_default.py           # dense baseline vs shipped defaults (correctness + latency)
bench/bench_paged.py             # paged page sizes vs dense, with a correctness guard
bench/bench_partial_dtype.py     # A/B of O_partial fp32 vs fp16 (measured: no gain)
bench/prof_*.py, bench/parse_pipes.py  # single-kernel msprof captures + CSV summary; see bench/README.md
```

The suite is 101 cases: 5 baseline, 7 dense split-KV regression, 39 dense shape
matrix, 12 dense tile sweep, 21 paged (11 semantics + 10 multi-batch/empty-split),
11 dense coverage gaps (`batch=16`, `dim`/`pe_dim` combinations, a `kv_len=8193`
long tail, extra tile combinations, call hygiene and a 100-call soak) and 6
unified-API dispatch/equivalence cases. Everything validates against an fp32 torch
reference with `rtol=atol=1e-2`, and every case is repeatable bitwise across
calls.

## Unified entry point

`mla_decode_api.compile_mla_decode()` selects the kernel and the measured-best
tiling from the shape; the caller only changes the input tensors between the two
layouts. For the dense reference shape it resolves to `block_N=64`, `chunk_d=128`,
`num_split=16`, `vector_mode="simd"` and `ml_width=8`. `resolve_config()` returns
the same decision as a dict without touching the device, and the returned callable
exposes its decision (`paged`, `num_split`, `block_N`, `chunk_d`,
`partial_dtype`, `vector_mode`, `ml_width`).

```python
import math

from mla_decode_api import compile_mla_decode

# dense KV cache: KV is [batch, kv_len, dim]
run = compile_mla_decode(batch=1, heads=128, kv_len=8192, dim=512, pe_dim=64)
out = run(q, q_pe, kv, k_pe)

# paged KV cache: KV is [num_slots, dim], addressing through block_table
run = compile_mla_decode(
    batch=1, heads=128, dim=512, pe_dim=64,
    num_slots=kv.shape[0], block_size=64, max_pages=math.ceil(max_seqlen / 64),
)
out = run(q, q_pe, kv, k_pe, block_table, cache_seqlens)
```

`example_mla_decode.py` is the unmodified `num_split=1` port: the kernel is
partitioned as `grid = batch * heads / block_H`, so the reference shape runs on a
single AI core. It is kept as the baseline that the numbers below are relative to
and as the correctness cross-check for the split kernels.

## Why split-KV

`block_H` is fixed at 128 (see the constraints below), so the grid is
`batch * heads / 128` and the reference shape gets a grid of 1: a single block
runs all 8192 KV positions while the remaining cube cores idle. Latency is flat
from `batch=1` to `batch=16` (1.606 ms to 1.612 ms), which confirms the
parallelism comes only from `batch`.

The split-KV variant (flash-decoding) moves the parallelism to the KV axis and
splits the launch in two:

```
phase 1  grid = batch * heads / block_H * num_split        (Cube + Vector cores)
         per split: O_p[s] = sum_{kv in range} exp(x - m_s) * V,  lse_s = m_s + log(l_s)
phase 2  grid = batch * heads / block_H * dim / chunk_d * rows_per_block   (Vector cores)
         w_s = exp(lse_s - max_s lse_s)
         O   = (sum_s w_s * O_p[s]) / (sum_s w_s)
```

`O_p` is unnormalised, so each phase-1 core normalises by its own `l` before
writing; the merge only needs the split weights. Two launches instead of one
cross-core barrier: a barrier has to be reached by every core, which serialises
the combine behind the slowest split and still round-trips the partials through
GM.

Phase 1 writes `O_partial`, a `[num_split, batch, heads, dim]` fp32 tensor. The
fp32 round trip is deliberate - the partials are unnormalised, so fp16 would lose
accuracy before the merge - and it costs 64 MiB in the worst measured shape
(`batch=16, num_split=16`). Next to it, `ML` is a `[num_split, batch, heads,
ml_width]` fp32 tensor whose column 0 holds `lse`. `ml_width=8` is one 32 B DMA
granule: `T.dual_copy` moves the two AIV halves of the M axis in one DMA, and a
1-wide region is not a useful transfer shape. The width is a parameter because it
only buys UB headroom with nothing in return.

`effective_split()` clamps the split count to the KV block count and divides it
down until it divides that count, so a short sequence never pays for empty
splits.

The merge took the same treatment as phase 1. One block per `dim/chunk_d` column
chunk is only 4 blocks for the reference shape, so the split loop was latency
bound; `rows_per_block=16` splits the 128-row head block across 8x more blocks
and `num_stages=4` pipelines the split loop. Measured: 85.2 us (rows=128,
stages=1) -> 37.0 -> 21.1 -> **14.3 us** (rows=16, stages=4).

## Vector programming model

Almost all of the phase-1 elementwise work is a handful of per-KV-block passes
over `ROWS x block_N` or `ROWS x dim` fp32 tiles. `vector_mode` selects how they
are expressed. `None` (the default) resolves per layout: the dense kernel picks
`"simd"` at `block_N <= 64` and `"simd_rc"` for the 128-token tile, while the
paged kernel always resolves to `"simd_rc"`. An explicit value always wins.

| `vector_mode` | Passes moved to `T.SimdVF` | Fits when |
| --- | --- | --- |
| `"simt"` | none - the original `T.SimtVF` thread-parallel code | always (A/B reference) |
| `"simd_rc"` | PV rescale-accumulate: one `vmul` + one `vadd` per 64 fp32 lanes instead of one scalar op per thread | always |
| `"simd_exp"` | exp + fp16 cast (`T.simd.vexpdif` is a fused `exp(a-b)`) | dense, `block_N <= 64` |
| `"simd"` | both of the above - **the default for the dense `block_N <= 64`** | dense, `block_N <= 64` |

Both kernels fall back to `T.SimtVF` automatically when the tiled width is not a
whole number of 64-lane registers (`block_N % 64` or `chunk_d % 64`), so narrow
tilings keep working.

Two steps were measured. First, `"simd_rc"` at the old 128-token tile took the
reference shape from 78.9 to 70.8 us (-10.2%), relative error unchanged at 3.4e-4.
Second, once the tile was retuned to 64 tokens (next section) the full-SIMD mode
fits the UB budget and takes the same shape to **54.2 us**. What blocks `simd_exp`
at 128 tokens is exactly the score-tile double buffering:
`dyn_ubuf_bytes = 222208 > 221184`.

The paged kernel accepts only `"simt"` and `"simd_rc"`, and passing `"simd"`
explicitly fails its assert with a clear message rather than being silently
downgraded. The reason is that a per-page score tile is only 64 columns wide, so
the explicit MicroAPI path cannot amortise its setup: it measured no gain
(phase 1 49.2 vs 49.1 us) and only adds UB pressure.

## Why `block_N=64` (tiling re-tune)

The inherited tuning picked `block_N=128`: one rescale/accumulate sweep per 128 KV
columns halves the vector traffic per element. That was measured **before** the
SimdVF passes and no longer holds. With the rescale on the SIMD unit the 128-token
score tile is pure UB cost, and it is the one tiling where the full-SIMD mode does
not fit (`dyn_ubuf_bytes = 222208 > 221184`).

The 64-token tile wins on all three counts - half the score buffers leave UB
headroom, it halves the L1 footprint (448 -> 296 KiB, which is what lets the
compiler's automatic pipeline overlap stages), and only `block_N=64` lets
`"simd"` fit at `kv_len=8192`:

| Change | e2e (us) |
| --- | ---: |
| `block_N=128`, `"simd_rc"` (old default) | ~70.6 |
| `block_N=64`, `"simd"` (**new default**) | **54.2** |

Around the new point: `chunk_d=128` is ~16% faster than `chunk_d=64` (fewer, wider
PV chunks), `num_split=16` stays optimal (`ns=8` is +41%, `ns=32` is +39%), and
`num_stages=2` stays optimal (`3` is +2.2%, `4` is +7.4%). `kv_len < 64` falls
back to `block_N=32`, where every SIMD pass drops back to `T.SimtVF`.

## Paged KV cache

The paged variant keeps the same operator and the same phase-1/merge pair; only
the addressing of `s` changes. `block_table[b, p]` names the physical page that
holds logical positions `[p*block_size, (p+1)*block_size)`, and
`cache_seqlens[b]` is the live length:

```
page  = block_table[b, (kv_blk * block_N + pc * PAGE_ROWS) // block_size]
slot  = page  * block_size  + (kv_blk * block_N + pc * PAGE_ROWS) % block_size
```

A `block_N` tile is assembled from `NUM_PC` page chunks of `PAGE_ROWS` tokens,
which covers both page geometries: `block_size > block_N` (several tiles inside
one page, non-zero in-page offset) and `block_size <= block_N` (one tile spans
whole pages). Out-of-range columns are masked to `-inf`, and `block_table` reads
are clamped to `max_pages-1`.

The runtime trip count is split over cores exactly like the CUDA kernel:
`per = nblk // NS` plus one extra block for the first `nblk % NS` splits. A split
that receives zero blocks publishes `lse = -inf` and zeros, which the merge
weights to exactly 0 - so `num_split` may exceed the number of live KV blocks.

`example_mla_decode_paged.py` **imports** `mla_decode_merge` instead of copying
it; the two extra parameters that make one kernel serve both layouts are
`ns_explicit` (paged lengths are runtime values, so the split count is fixed by
the caller rather than derived from `kv_len`) and `ml_width` (the `ML` staging-row
width, 8 floats in both kernels).

## Platform constraints

The kernels look the way they do because of these; none is a style choice.

- **`block_H` is fixed at 128.** `T.dual_copy`'s L0C-to-UB path requires a
  128-row source (`PTO L0C->UB MTE source stride must be at least n_size`), so
  each core owns 64 rows (`ROWS = block_H // 2`) and `heads` must be a multiple
  of 128. Making `heads` arbitrary needs per-AIV row limits, which the
  `T.Kernel(N)` form cannot express.
- **L1 GEMM is NT-only.** `T.gemm` supports `trans_A=False, trans_B=True` with L1
  operands only, so the PV product needs `V` transposed: `T.copy(V, v_l1,
  transpose=True)` into one independent L1 buffer per output-column chunk. A
  dynamic row slice of a shared L1 buffer is not usable as a GEMM operand (it
  compiles and returns wrong results), and `T.gemm` sizes K from the whole
  buffer, ignoring the region - hence the per-chunk and per-page buffer lists.
- **A GM-to-L1 copy with a runtime source offset must land on a zero-based
  region covering the full row axis.** A page cannot fill a row slice of a shared
  tile, so the paged kernel carries `k_l1`/`k_pe_l1`/`p_l1`/`v_l1` as lists, one
  entry per page chunk. Accumulating several pages' PV into one *complete*
  `o_l0c` (`clear_accum=False`) is allowed and is what keeps the L0C-to-UB
  traffic from doubling.
- **`dual_copy` destinations are whole buffers.** Writes go through dedicated
  contiguous buffers, and a buffer written inside a pipelined loop cannot be
  reused as a `dual_copy` destination after the loop (an `aicore timeout`).
- **`range` inside a `prim_func` is a TIR loop.** `for i in range(n)` becomes
  `T.serial` and `i` a `Var`, which cannot index a Python list. Every compile-time
  unroll uses `enumerate()` or a list comprehension instead.
- **UB and L0C are the binding budgets.** The dav-3510 UB pool is 248 KiB with
  32 KiB reserved for SimtVF, so the dynamic UBUF limit is 221184 B. At the new
  `block_N=64` tiling the dense kernel leaves about 40 KiB of margin (one extra
  buffer version fits - `s_ub` or `o_tmp`, not both) and the paged kernel about
  50 KiB; the 128-token tile is where UB runs out, since `"simd"` needs 222208 B
  there. L0B is 64 KiB and L0C 128 KiB. `o_ub` alone is `64 * dim * 4 B` = 128 KiB
  at `dim=512`, which caps `dim` at 512. The same budget is why fewer, wider
  `chunk_d=128` PV chunks pay off at `block_N=64`, why `"simd_exp"`/`"simd"` only
  fit the dense kernel, and why larger tiles do not: `block_N=256` needs a 128 KiB
  `s_l0c` on top of `o_l0c`, and `block_H=256` would need a 256 KiB `o_ub`.
- **Buffers on the UB cannot be DMA base addresses.** The paged kernel therefore
  reads `block_table` entries straight from GM; preloading the row into UB is not
  schedulable in this release.
- **L1 is 512 KiB per core, and it is the other budget.** At `block_N=128` the
  phase-1 set already takes `q_l1` 128 + `q_pe` 16 + `k_l1` 128 + `k_pe` 16 +
  `p_l1` 32 + `v_l1` 128 = 448 KiB, so no KV buffer can be hand double-buffered;
  at `block_N=64` it is 296 KiB, which is what gives the automatic pipeline room
  to overlap stages.

`pick_tiling()` encodes the measured-best shape policy: `block_N=64, chunk_d=128`,
with `block_N=32` for `kv_len < 64`; the merge picks its column width
independently (`merge_chunk_d=128`) so that phase-1 tiling changes do not fragment
the merge DMAs.

## Performance

Environment: Ascend 950PR (`ASCEND_RT_VISIBLE_DEVICES=6`), CANN 9.3.0, tilelang
`0.1.15+cpu.git876a59e6`, torch 2.10.0+cpu / torch_npu 2.10.0, Python 3.11.
Shape `batch=1, heads=128, kv_len=8192, dim=512, pe_dim=64`, FLOPs
`2*b*h*s*(dim+pe_dim) + 2*b*h*s*dim = 2.2817 GFLOP`. Timings are wall clock with
`do_bench(backend="event")`, median of 3-5 rounds, spread <= 0.3 us. The round 1-4
history was collected at load below 60, the SimdVF/fp16-partial A/B at load
90-120, and the final `block_N=64` numbers at load 27-45 (dense) / 22.9 (paged).

| Step | e2e (us) | phase 1 | merge | Speedup | TFLOPS |
|---|---:|---:|---:|---:|---:|
| Baseline `num_split=1` (unchanged) | 1549.3 | - | - | 1.00x | 1.47 |
| Split-KV (flash-decoding) | 188.8 | 111.4 | 84.3 | 8.21x | 12.08 |
| + single-L0C QK, fused rescale, merge rework | 93.1 | 80.5 | 14.5 | 16.64x | 24.50 |
| + `block_N=128` / `chunk_d=64`, decoupled merge, `threads=1024` | 78.9 | 66.2 | 14.4 | 19.64x | 28.92 |
| + merged exp/fp16-cast pass | 78.4 | 63.5 | 14.4 | 19.76x | 29.09 |
| + SimdVF rescale/accumulate (`block_N=128`) | 70.8 | 58.4 | 14.4 | 21.9x | 32.3 |
| + 64-token score tile + full-SIMD vector mode (**current default**) | **54.2** | **41.9** | **14.3** | **28.6x** | **42.1** |

The current default was measured three times independently at 54.09, 54.22 and
54.47 us (load 27-45); the previous `block_N=128` default measured 70.4-70.9 us.
Phase 1 and merge are independent single-kernel measurements; the shared merge
kernel contributes 14.3 us, so phase 1 alone is 41.9 us (dense) and 49.1 us
(paged, page 64).

Paged, same shape and the new `block_N=64` / `chunk_d=128` default, measured in
the same process as the dense number above:

| Page size | e2e (us) | phase 1 (us) | TFLOPS | vs dense |
|---|---:|---:|---:|---:|
| **64 (recommended)** | **61.4** | **49.1** | **37.2** | **1.13x slower** |
| 16 / 32 / 128 | - | - | - | not re-swept |

Page sizes 16/32/128 were measured under the previous `block_N=128` default
(99.0 / 77.4 / 75.0 us) and were not re-swept after the tiling change.

`page_size=64` is both the DeepSeek cache geometry and the measured optimum: it
matches the 64-token tile, and two 64-wide score tiles keep the QK product as
independent GEMM chains. Against the `num_split=1` baseline, paged at page 64 is
25.2x. The tiling re-tune and the full-SIMD mode helped the dense kernel more than
paged - the paged kernel caps at `"simd_rc"` and a per-page score tile is only 64
columns wide - so the ordering flipped: paged was 5% faster than dense at
`block_N=128` and is now 13% slower, while still gaining 6% over its own previous
default.

The first paged version was 93.9 us (19% slower than dense). The fix was to
accumulate both pages' PV into the same `o_l0c` (`clear_accum=False`): the GEMM
count stays at 16 per tile but `dual_copy` and the UB accumulate drop from 16 to
8, and the profiler's doubled `fixpipe` share disappeared exactly as predicted.

## Pipeline

msprof `op` with the 8-CSV set, one kernel per capture (`--warm-up=10
--launch-count=5`), Ascend 950PR at 1650/1650 MHz.
`ratio` is busy cycles over that core's own cycles; AIC and AIV run concurrently,
so the two groups do not add up, and pipes within a group overlap.

| Metric | dense | paged, page 64 | paged, page 128 | merge |
|---|---:|---:|---:|---:|
| Task duration (us) | 63.5 | **56.8** | 64.0 | 12.9 |
| AIC cube (MAD) | 16.6% | **18.6%** | 16.5% | - |
| AIC mte2 (GM->L1) | 20.2% | 23.2% | 23.4% | - |
| AIC fixpipe (L0C->UB) | 6.6% | 7.7% | 6.5% | - |
| AIV vec | 77.2% | **72.2%** | 77.4% | **91.0%** |
| AIV mte2 (GM->UB) | 0% | 0% | 0% | **69.9%** |
| Achieved compute | 36.0 TFLOPS (8.3%) | 40.2 TFLOPS (9.3%) | 35.7 TFLOPS (8.3%) | - |
| Aggregate GM bandwidth | 318 GB/s (8.0%) | 356 GB/s (8.9%) | 316 GB/s (7.9%) | - |
| `aiv_vec_wait` | 88.9% | 87.4% | 89.6% | 89.6% |
| `aic_cube_wait` | 95.8% | 88.0% | 96.7% | - |

The captures predate both SimdVF passes and the tiling re-tune (dense phase 1 is
63.5 us here against 41.9 us now, paged page 64 56.8 against 49.1), so the shares
are used for attribution rather than as current latencies.

Neither variant is compute bound nor bandwidth bound: the cube is busy 16-19% of
the duration and the memory system is at 8-9% of its peak. The vector pipeline
owns 72-91% of the duration and waits 87-90% of its cycles. The root cause is the
**six-stage serial dependency per KV tile**: `QK -> dual_copy -> softmax ->
dual_copy -> PV -> dual_copy`. A tile takes about 15.9 us of which the cube has
about 2.6 us of real work, so deepening the pipeline cannot close the gap
(`num_stages=3` measured 14% slower at the old tile, `num_split=8` 21% slower than
16); the lever that did work was shrinking the tile, which lowers the UB pressure
that the scheduler is fighting (see the `block_N=64` section).

Ceilings already checked, so they are not worth retrying: `block_N=256` exceeds
L0C, `block_H=256` exceeds UB, larger `num_split` leaves too few tiles per core
to fill the pipeline, and the two-pass max-then-accumulate softmax removes only
12% of the vector work while adding a second KV read and a second QK pass.

The profile pointed at three levers; two have since been taken and one is still
open:

1. **SimtVF -> SimdVF (done, in two steps).** `aiv_vec_simt_vf_ratio = 0.78`
   against 0.005 for pure SIMD meant 78% of vector time ran in the SIMT thread
   model. Moving the PV rescale-accumulate to `T.SimdVF` (`"simd_rc"`) took the
   reference shape 78.9 -> 70.8 us (-10.2%); retuning the tile to 64 tokens then
   let the exp/cast pass move too (`"simd"`), taking it to 54.2 us. The exp pass
   is UB-blocked only at `block_N=128`.
2. **fp16 `O_partial` (tried, no gain).** Halving the merge read from 4 MiB to
   2 MiB left the merge time unchanged (-0.23 / +0.04 us across two A/B rounds)
   and end-to-end 0.4-0.7 us slower: the merge is latency bound, not byte bound
   (`AIV vec` is 91.0% of its duration against `mte2` 69.9%), and the extra
   fp16-to-fp32 cast in the accumulate cancels the narrower read.
   `partial_dtype="float16"` stays as an opt-in knob (verified to rel <= 5.3e-4);
   `"float32"` is the default.
3. **Read the KV once (open).** `read_main_memory_datas` measures 1232 KiB per
   tile, which is the KV data twice (once as K for QK, once transposed as V for
   PV). Loading it into UB once and splitting into `k_l1`/`v_l1T` would halve the
   GM read and lift the 53% L2 read hit rate.

The dependency chain itself is untouched - `aiv_vec_wait` was ~87% in the
pre-change capture - so the remaining large win is structural: fewer stages per
tile, or overlapping the softmax of one block with the PV of the previous one.
Tiling itself is exhausted, and cross-block software pipelining is blocked by the
scheduler (next section).

## Pipeline / multi-buffer experiments (negative results)

Manual software pipelining across KV blocks is the obvious next step once the
profile showed the six-stage dependency, so it was attempted from several angles
and failed. The experimental switches live only in the experiment trees - nothing
from this section was merged into the deliverable, which is why the file list does
not change.

**The hard limit is L1 capacity.** A core has 512 KiB of L1, and at `block_N=128`
the phase-1 buffers already take

```
q_l1 128 + q_pe 16 + k_l1 128 + k_pe 16 + p_l1 32 + v_l1 128 = 448 KiB
```

so a second version of even one `k_l1`/`v_l1` (+128 KiB) cannot fit: hand-written
double buffering is physically impossible, not merely unprofitable. At
`block_N=64` the same set is 296 KiB, which is why the compiler's automatic
pipelining finally has room to overlap stages. The "six-stage serial dependency"
diagnosis was correct; the fix is to shrink the tile, not to hand-write the
pipeline.

Everything below was tried on top of that and lost. The other half of the limit is
that the current tilelang/Z3 scheduler sizes buffer multiversioning against UB
capacity up front, so a version count that does not fit fails at compile time
instead of being traded off:

| Experiment | Result |
| --- | --- |
| `T.annotate_buffer_versions({s_ub: (2, "counter")})` | +10.8% (worse) |
| `(2, "auto")` / `(2, "iteration")` | same order of magnitude, no gain |
| `{s_ub: 2, o_tmp: 2}` | Z3 compile-time failure: `capacity is negative (-8704)` |
| paged `score_versions=2` | `scope "shared" capacity 40448 insufficient for minimum buffer usage 41344` (896 B short) |
| `annotate_unlimit_memory` | Z3 goes unsat and degrades to a topological sort: UB 238592 over the limit, time unchanged |
| `num_stages = 3 / 4` | at `block_N=64` no gain (paged 61.4-61.8 us, inside the noise) |
| merge `rows_per_block` / `num_stages` / `threads` | fully swept; `rows=16, stages=4, threads=256` best at 14.34 us |
| merge 2 / 4 partial accumulators | +1.6% / +3.8% (worse) |

Cross-KV-block software pipelining is therefore infeasible on this compiler (the
Z3 in-loop UB budget model) rather than merely untuned; the leverage is the tile
size, which lowers the UB pressure (see the `block_N=64` section).

## Running

From the repository root. CANN 9.3.0 or newer is required - 9.1.0 lacks the SIMD
builtins the Ascend backend emits. The default target resolves to `ascend`; set
`TILELANG_DEFAULT_TARGET=pto` to run the same tests through the PTO backend
(first compile is much slower).

```bash
# correctness: 101 passed
python -m pytest examples/ascend/mla_decode -q
TILELANG_DEFAULT_TARGET=pto python -m pytest examples/ascend/mla_decode/test_mla_decode.py -q

# bench: baseline vs dense default, paged page sizes vs dense, O_partial dtype A/B
python examples/ascend/mla_decode/bench/bench_default.py
python examples/ascend/mla_decode/bench/bench_paged.py
python examples/ascend/mla_decode/bench/bench_partial_dtype.py

# unified entry point
python -c "import sys; sys.path.insert(0, 'examples/ascend/mla_decode'); import mla_decode_api"

# pipeline attribution (one kernel per capture; output must not land on /)
msprof op --warm-up=10 --launch-count=5 \
    --output=/dev/shm/opp_dense python examples/ascend/mla_decode/bench/prof_phase1.py --which phase1
msprof op --warm-up=10 --launch-count=5 --output=/dev/shm/opp_pg64 \
    python examples/ascend/mla_decode/bench/prof_paged.py --block-size 64
msprof op --warm-up=10 --launch-count=5 \
    --output=/dev/shm/opp_merge python examples/ascend/mla_decode/bench/prof_merge.py
python examples/ascend/mla_decode/bench/parse_pipes.py dense=/dev/shm/opp_dense paged_p64=/dev/shm/opp_pg64
```

On a shared machine, point the tilelang cache and temporary files off a
potentially full root partition (`TILELANG_CACHE_DIR`, `TMPDIR`) and write msprof
captures to `/dev/shm`; a full `/` fails compilation and capture alike.

## Measurement methodology

- **Wall clock only through `do_bench(backend="event")`,** `_n_warmup=10`,
  `_n_repeat=50`, 3-5 rounds, median. The reported numbers have spreads of
  0.02-0.3 us; if the spread exceeds 0.3 us the machine is too loaded and the data
  is not used.
- **msprof is for attribution, never for end-to-end timing.** When one call
  launches two kernels, `do_bench(msprof_detail)` attributes only one of them (the
  two `prim_func`s used to share the name `main`), which once inflated a speedup
  by 1.7x. The merge kernel is now named `merge_main` and each `prof_*.py` entry
  runs exactly one kernel.
- **The machine is shared.** Load moves between 15 and 650 with 150+ users, and
  the first measurement point in a process is inflated under load (86.9 us vs
  78.8 us later in the same process). Head-to-head comparisons therefore stay
  inside one process, after the baseline has been warmed.
- The round 1-4 history was collected on 2026-10-28 at load below 60; the
  SimdVF/fp16-partial numbers on 2026-10-29 at load 90-120, and the final
  `block_N=64` numbers at load 27-45 (dense) / 22.9 (paged). Spreads stay <= 0.3 us
  and relative A/B comparisons stay inside one process.

## Known limitations

- **fp16 only.** A bf16 path needs a `dtype` parameter on the fwd/merge/compile
  entry points (~7 lines) and is measured feasible on one shape (maxdiff 2.5e-3),
  but it is not shipped: it would need a full accuracy and performance re-sweep.
- **`block_H=128` and `heads % 128 == 0`.** Both follow from the `dual_copy`
  128-row requirement.
- **`dim <= 512`.** `o_ub` is `64 * dim * fp32` = 128 KiB at `dim=512` against the
  221184 B dynamic UBUF limit; larger `dim` needs the output column-chunked *and*
  the accumulator restructured. `dim % chunk_d == 0` is asserted.
- **`kv_len >= block_N`** for the dense layout. `effective_split()` clamps
  `num_split` to the KV block count and divides it down until it divides that
  count, so short sequences keep at least one KV block per split.
- **Paged multi-batch is covered:** batch 4/8 with shuffled page tables, per-batch
  variable length, page 64/128 x `block_N` 64/128, empty splits and `seq_len=1`.
  Not covered: page size 256, `max_pages > 8`, concurrent multi-stream use, and
  page sizes 16/32/128 were not re-swept after the tiling change.
- **The tile sweep covers `dim` 256/512, `pe_dim` 128, `kv_len` up to 8193 and
  `num_stages` 1/2/3** (see `test_mla_decode_splitk_tiles.py` and
  `test_mla_decode_coverage.py`), and the measured-best point was re-swept after
  the `block_N=64` retune (`chunk_d` 64/128, `num_split` 8/16/32, `num_stages`
  1-4); it is still not a full Cartesian product.
- **`partial_dtype="float16"` halves the merge's read traffic but measured no
  gain** (the merge is latency bound: `AIV vec` 91% against `mte2` 69.9%), so
  `"float32"` is the default.
- **No causal mask, dropout, attention bias, padding mask, or `q_len > 1`.**
- **Non-contiguous inputs are rejected** by the runtime packed-ABI stride check
  (`strides[2] violates packed ABI constraint`); call `.contiguous()` first - it
  fails loudly, never silently. Inputs are never modified in place and repeated
  calls are bitwise identical (100-call soak), but multi-process, multi-stream and
  hour-long soak runs are untested.
- Accuracy: `rel <= 6.2e-4` across the dense matrix and `<= 5.4e-4` for paged
  (`rtol=atol=1e-2`); with the SIMD vector modes the reference shape measures
  3.4e-4, so splitting, merging and the SIMD passes add no measurable error over
  the `num_split=1` kernel.

---

Validated against tilelang `0.1.15+cpu.git876a59e6` on Ascend 950PR with CANN
9.3.0; the kernels only use public `tilelang.ascend.language` APIs.

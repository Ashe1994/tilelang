# Self-verification report - paged MLA decode (Ascend 950PR)

Verification of the `examples/ascend/mla_decode` tree (native decode reference +
paged decode kernel + its tests/benchmarks) on the PR development host,
2026-11-02. The paged kernel gained a two-tile CV-overlap phase-1 schedule in the
same session; this report covers the final state.

## 1. Environment

| item | value |
|---|---|
| NPU | Ascend 950PR (`dav-3510`), 8 devices on the host, one process on one device |
| core model | 1 cube (AIC) + 2 vector (AIV) per core; the kernel launches as a mix kernel (`Mix Block Dim = 32` = 16 AIC + 32 AIV) |
| clock / peaks | 1650 MHz; 4 TB/s GM and 432 TFLOPS fp16 cube used for utilisation |
| CANN | 9.3.0 (toolkit + 950 ops) |
| tilelang | `0.1.15+cpu.git876a59e6` |
| PyTorch | `2.10.0+cpu` + `torch_npu 2.10.0`, Python 3.11.16 |
| host load | correctness 37-60; latency runs 37-132 (shared host, see section 9) |

Reference shape: `batch=1, heads=128, kv_len=8192, dim=512, pe_dim=64`.

## 2. Correctness - 26/26 passed

```
python -m pytest test_mla_decode.py test_mla_decode_paged.py -q
26 passed, 15 warnings in 9.09s
```

| file | cases | coverage |
|---|---:|---|
| `test_mla_decode.py` | 5 | native decode kernel vs the fp32 torch reference |
| `test_mla_decode_paged.py` | 21 | page geometries (64/128/256), shuffled block tables, ragged `cache_seqlens`, empty splits, multi-batch, dense equivalence |

Relative error: native 6.2e-4, paged 5.4e-4 (tolerance 1e-2). The paged suite also
exercises the two-tile schedule's masked-slot path (a split with an odd number of
tiles runs one fully masked slot, which must contribute exactly zero).

## 3. Performance

`do_bench(backend="event", _n_warmup=10, _n_repeat=50)`, 3 rounds, median; the
native reference is measured first in the same process so the ratio is
meaningful.

| case | e2e | TFLOPS | rel | speedup vs native |
|---|---:|---:|---:|---:|
| native decode (num_split=1) | 1549.4 us | 1.47 | 6.2e-4 | 1.0x |
| **paged, page 64, block_N=64, ns=16 (default, two-tile schedule)** | **52.6 us** | **43.4** | 5.4e-4 | **29.5x** |
| paged, page 128 (two-tile schedule) | 53.0 us | 43.1 | 5.5e-4 | 29.3x |
| paged, block_N=128 (single-tile schedule) | 59.3 us | 38.5 | 5.4e-4 | 26.1x |
| paged, num_split=32 (two-tile schedule) | 70.6 us | 32.3 | 5.4e-4 | 21.9x |

The two-tile CV-overlap schedule is worth -7.6% end-to-end against the
single-tile schedule (56.9 -> 52.6 us; phase 1 47.7 -> 43.6 us, -8.6%).

## 4. Multi-batch scaling (native vs paged vs persistent)

`bench/bench_persistent.py`, event timing, 3 rounds, median, host load 21-46.
`mla_decode_persistent.py` runs the same two-tile phase 1 but walks the
(batch, head tile, split) space with `T.Persistent` on one block per core.

| case | native | paged | persistent | persistent / paged |
|---|---:|---:|---:|---:|
| B=1, [8192] | 1549.5 us | 53.8 us | 53.7 us | 1.00x |
| B=2, [8192]*2 | 1556.1 us | 102.6 us | 99.6 us | 1.03x |
| B=4, [8192]*4 | 1556.6 us | 156.0 us | 185.8 us | 0.84x |
| B=8, [8192]*8 | 1557.8 us | 260.2 us | 359.5 us | 0.72x |
| B=2, [8192, 1024] | n/a (dense needs one length) | 100.4 us | **74.5 us** | **1.35x** |
| B=4, [8192, 2048, 512, 4096] | n/a | 106.9 us | 119.6 us | 0.89x |

`msprof op --warm-up=10 --launch-count=5 --aic-metrics=PipeUtilization,Memory,
MemoryL0,ResourceConflictRatio,L2Cache` on one launch per variant:

| kernel (case) | grid | duration | TFLOPS | %cube peak | GB/s | %GM peak | AIC busy | AIV busy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native B=4 | 4 | 1556.5 us | 1.5 | 0.3% | 46 | 1.2% | 45.3% | 90.7% |
| paged B=4 | 64 | 128.8 us | 17.7 | 4.1% | 627 | 15.7% | 32.2% | 24.5% |
| persistent B=4 | 16 | 161.3 us | 14.1 | 3.3% | 501 | 12.5% | 100.3% | 77.1% |
| paged ragged B=2 | 32 | 85.9 us | 26.6 | 6.1% | 470 | 11.7% | 49.3% | 37.6% |
| persistent ragged B=2 | 16 | 57.5 us | 39.7 | 9.2% | 470 | 11.7% | 91.5% | 77.4% |

What the numbers say:

* the **native** kernel launches `batch` blocks (Block Dim 4 for batch=4) and its
  AIV is busy 90% of the duration, so it leaves 12 of the 16 cores idle and is
  vector bound -- 1.5 TFLOPS and 46 GB/s at any batch size;
* the **paged** grid (`batch * splits` blocks) turns that into free cube/vector
  overlap: a core starts the next block's cube while the previous block's vector
  tail drains, so B=4 phase 1 is 128.8 us instead of 4 x 43.6 us;
* the **persistent** grid is one block per core, so its items serialise (no
  cross-block overlap): B=4/B=8 phase 1 loses 25-30% against the paged grid, and
  `num_stages>1` on `T.Persistent` is not usable because the scheduler versions
  the phase-1 buffers and the dynamic UBUF grows to 339 KB (limit 221184 B);
* for a **ragged** batch the persistent assignment (one split of every batch per
  core) is a straight win: batch=2 [8192, 1024] goes 100.4 -> 74.5 us because the
  paged grid otherwise waits for the sequence that owns 8 tiles per split while
  the short one owns one. With four unequal sequences the per-item serialisation
  eats the balance again (0.89x), so the win is batch-size dependent.

Supported envelope of the prototype: `block_N=64`, one page per tile,
`chunk_d=128`, fp32 partials (the wider geometries and the fp16 publish staging
exceed the UB budget and are rejected with an explicit assertion; use
`compile_mla_decode_paged` for them).

The obvious follow-up is a **packed, length-proportional split plan** on the
paged grid: keep the `batch * splits` blocks (and therefore the cross-block
overlap) but choose each block's tile range from a global plan so every block
owns the same number of KV tiles. That should keep the uniform-batch numbers and
recover the ragged-batch win without the persistent item serialisation.

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

The measured optimum matches `num_split = clamp(kv_tiles / 16, 1, 16)` (with
`kv_tiles = ceil(kv_len / 64)`), i.e. keep about 16 tiles per split: 1/2/4/8/16/16
against the measured 1/2/4/8/8/8 best points (within 8% at the two largest
lengths). Callers pass `num_split` to `compile_mla_decode_paged`, so a serving
layer should size it from the max live length instead of the fixed default 16.

### Automatic split sizing and the 90% utilisation target

`compile_mla_decode_paged(num_split=None)` now derives the split count with
`pick_num_split`: about 16 KV tiles per split, at least `16 / batch` splits and at
most 8 per batch element. That reproduces the measured optima (batch=1: 16
everywhere; batch=64: 1/2/4/8/8/8 for kv=1024..32768) and is covered by
`test_pick_num_split` plus `test_paged_auto_split` (28/28 tests pass). The batch=64
sweep with the automatic count: 0.29 / 0.46 / 0.85 / 1.65 / 3.03 / 5.86 ms for
kv=1024..32768 against 0.85 / 0.97 / 1.27 / 1.88 / 3.19 / 5.85 ms with the old
fixed 16 -- a 2.9x win at kv=1024 and no regression at the top end.

`msprof` at batch=64 (`bench/prof_multi.py --variant paged`) gives the per-tile
budget that bounds the utilisation target:

| pipe | per tile | note |
|---|---:|---|
| AIC cube | 1.32 us | 10.55 us per 8-tile block at batch=1 |
| AIC mte2 | 1.68 us | 136 KB/tile: K (72 KB) + transposed V (64 KB) |
| AIV vec | ~1.49 us per AIV | softmax 4 passes + PV rescale |
| AIC fixpipe | 0.98 us | |

The measured block wall is 2.55 us/tile (kv=32768: 163 us per 64-tile block,
512 blocks), so the pipes overlap partially and the perfect-overlap floor is
`max = 1.68 us`. Consequences:

* **cube busy cannot exceed ~79%** with the current traffic, because MTE2 is the
  longest pipe; raising it needs the KV single read (0.89 us/tile), which the
  NT-only L1 GEMM blocks (section 8);
* **cube and GM cannot both be 90%**: at 90% cube the kernel would move ~2.4 GB in
  ~3.0 ms (~790 GB/s, 20% of peak); saturating 3.6 TB/s would need ~890 TFLOPS,
  twice the fp16 peak. The realistic target is cube-bound at 90% with ~40% GM;
* the concrete order of work is therefore (1) KV single read (compiler support for
  a transposed L1 operand), (2) vector-pass reduction so the vector pipe stops
  being the second-longest, (3) more tiles in flight per core.

The fused-softmax attempt is 16-18% slower because a reducer over a
select/multiply expression lowers to a scalar pass (and the fully fused variant
adds 16 KiB of UB), so vector-pass fusion is closed.

The single-read `P @ K` L0 GEMM is the one that worked, but only on the **PTO
backend** (`pv_l0=True`, `target="pto"`): the AscendC L0 path faults on the
K-major/MN-major layout mismatch, while PTO supports the L0 operand shapes we
need. Measured at batch=64 with `num_split=8` (event timing, 2 rounds):

| case | staged V^T (ascend) | single read (pto + pv_l0) | delta |
|---|---:|---:|---:|
| kv=8192 | 1.65 ms | **1.57 ms** | -4.8% |
| kv=32768 | 5.86 ms | **5.57 ms** | -5.0% |

MTE2 traffic per tile drops from 136 KB (K + transposed V) to 72 KB (K only) by
construction, and the relative error is unchanged (6.6e-4 / 6.3e-4). Three tile
slots on top of the single read were measured too and are slower (5.81 ms), so
the two-slot schedule stays. `msprof op` cannot profile PTO kernels in this
environment ("profiling results are empty"), so the pipe-level split of the PTO
build is not captured yet; the PTO-only path is exposed as an opt-in flag and the
default stays on the AscendC backend.

Reaching 90% cube utilisation still needs cross-iteration multi-buffering (or a
manually scheduled mixed kernel), and the single-read path needs PTO as the
deployment backend.

## 5. Two-tile CV-overlap schedule (phase 1)

Design: two KV tiles are kept in flight. K/K_pe of both tiles are prefetched into
double-buffered L1 tiles, both QK GEMMs are issued (two L0C score tiles), and only
then does the vector unit run the two softmax passes and the two PV passes. The
cube therefore keeps working while the vector units run the other tile's softmax,
instead of ping-ponging one tile at a time. The UB score buffer is reused between
the two slots, so the dynamic UBUF footprint is unchanged; V^T is single-buffered
and loaded inside each slot's PV phase to stay inside the L1 budget.

Gating: `select_phase1_kernel()` enables it only for the shipped geometry
(`block_N=64`, `chunk_d=128`, one page chunk per tile) *and* when the compile-time
live length gives each core at least two KV tiles
(`max_pages*block_size/block_N >= 2*num_split`). Other geometries (`block_N=128`
with 64-token pages, tiny sequences) keep the single-tile loop, which is faster
there because the masked second slot would be pure overhead.

The buffer pressure needs two scheduler escapes:
`T.annotate_unlimit_memory("shared", "shared.l1")` lets Z3 version the shared/L1
pools beyond its default partition. The physical budgets still hold: L1 ~400 KiB
of 512 KiB, dynamic UBUF about the same as the single-tile kernel (the launch
succeeds).

## 6. Profile - paged phase 1, 43.6 us (two-tile schedule)

`msprof op --warm-up=10 --launch-count=5 --aic-metrics=PipeUtilization,Memory,
MemoryL0,ResourceConflictRatio,L2Cache`, one kernel per capture, summarised by
`bench/parse_pipes.py`.

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
  `aic_mte1_wait` 95.3%, `aic_mte2_wait` 56.8%
* MTE instruction counts: mte2 **166**, mte1 178
* L2 read hit (AIC) 53.4%; GM->L1 1232 KB/block, pipe bandwidth usage 16.6%
* achieved 52.4 TFLOPS (12.1% of the fp16 cube peak) and 463 GB/s (11.6% of the
  4 TB/s GM peak)

Single-tile reference: 47.7 us, vec 27.86 us (58.5%), `aic_mte2_wait` 61.5%,
47.9 TFLOPS, 424 GB/s. The two-tile schedule lifts vector occupancy from 58.5%
to 66.7% of the duration and cuts the wall clock by 8.6%. The ideal cube/vector
overlap ceiling is `max(cube 10.55, mte2 13.22, vec 29.05) = 29.1 us`, so about
half of the dependency chain stall is gone; the PV phase and the second softmax
still serialise.

## 7. Bandwidth utilisation over data volume (two-tile schedule)

Page 64, batch 1, 3 rounds; "moved" is the KV read (K + V^T + k_pe) plus the
fp32 partials write/read:

| kv_len | KV cache | e2e | TFLOPS | GB/s | util (of 4 TB/s) |
|---:|---:|---:|---:|---:|---:|
| 1024 (single-tile: one tile per core) | 1.12 MiB | 23.2 us | 12.3 | 467 | 11.67% |
| 8192 | 9.0 MiB | 53.1 us | 43.0 | 498 | 12.44% |
| 32768 | 36 MiB | 153.9 us | 59.3 | 519 | 12.98% |
| 65536 | 72 MiB | 287.2 us | 63.6 | 526 | 13.16% |

Utilisation rises with data volume because the fixed per-launch cost dominates
the small tiers; from ~8 MiB upward the kernel is linear in the data volume and
no tier is bandwidth limited.

## 8. Measured dead ends

| direction | measurement | verdict |
|---|---|---|
| remove OOB clamp/fill from the runtime-offset copies | MTE2 instructions 166 -> 102 but `aic_mte2_wait` 61.2% -> 75.9%, phase 1 slower | closed |
| compile-time split-loop trip count | phase 1 49.2 -> 51.8 us | closed |
| affine page fast path (no `block_table` load) | recovers exactly the OOB loss, net zero against the baseline | closed |
| fp16 `O_partial` | 56.9 -> 59.4 us (32-column staging) / 68.7 us (full-width staging) | closed |
| `block_N=32` | phase 1 56.3 us vs 39.4 us dense reference (+43% per tile) | closed |
| forced multi-buffering of the cross-unit tiles (`annotate_buffer_versions`) | score tile x2 +17.5%, L0C x2 +8.7%, p_l1 x2 +6.6%, p_l1+k_l1 x2 +1.2%; o_tmp x2 does not fit UB | closed |
| `unit_flag_ctrl=3` (pipelined MAD+FIXPIPE) on gemm / dual_copy | compiles, hangs at run time (needs a matching manual flag protocol) | closed |
| single KV read (reuse the K tile for PV) | L1 GEMM is NT-only (`trans_A=False, trans_B=True`); using one tile for both gemms hits a layout conflict; UB->L1 `transpose=True` compiles but silently ignores the transpose (wrong values); L1->L1 transpose does not compile | closed |
| QK as one GEMM over a packed `dim+pe_dim` (K=576) | correct but phase 1 49.1 -> 52.4 us (+6.7%), e2e 56.9 -> 60.5 us | closed |
| PV GEMMs `2*chunk_d` (256) wide | rejected by the compiler: `L0C->UB compact source requires zero-origin trailing matrix dimensions`; a full-width UB accumulator would exceed the 221184 B budget | closed |
| phase-1 `num_stages=3` | 49.19 vs 49.22 us | closed |
| merge `rows_per_block=32` / `num_stages=6` | 13.20 / 12.37 us vs 12.26 us | closed |
| merge `chunk_d` 256 / 512 | e2e 57.7 / 60.8 us vs 57.0 us | closed |
| `threads` 512 / 2048 | e2e 57.2 / 57.8 us vs 57.0 us | closed |
| merge `tmp` x4 versions on the paged (fp32-partial) merge | e2e 56.85 -> 56.92 us, phase 1 49.11 -> 49.05 us: no effect | closed |
| page 128 | identical to page 64 in both schedules | closed |
| fused softmax (scale/mask folded into the row-max reducer, score tile read-only) | semi-fused: e2e 6.77 vs 5.82 ms at batch=64/kv=32768 (+16%) and 1.88 vs 1.59 ms at kv=8192; a reducer over a select/multiply expression lowers to a scalar pass. The fully-fused variant additionally needs +16 KiB UB (224512 > 221184) | closed |
| PV as `P @ K` on L0 operands (single KV read, `transpose_B=False`) | **AscendC backend**: compiles but the AI core faults (device error 507015) -- the K tile is K-major for QK and MN-major for the NN PV operand and the L1->L0B copy cannot convert majors. **PTO backend** (`target="pto"`): works and is the current best single-read path (see below) | open (PTO-only) |
| three KV tiles in flight (`TILE_SLOTS=3`) | phase 1 50.9 vs 43.6 us, e2e 58.7 vs 52.6 us; L1 at ~496 KiB leaves the scheduler no room | closed |
| V^T double-buffered and prefetched in the copy phase | phase 1 45.1 vs 43.6 us, e2e 52.9 vs 52.6 us (same run) | closed |
| reorder: slot-0 PV placed between the two softmax passes | e2e 56.8 vs 53.6 us (same run); the scheduler prefers the batched phases | closed |
| two-tile pair loop `num_stages` 2 / 3 | 53.65 / 53.85 us vs 53.60 us e2e | closed |

### Manual mixed-kernel probe (exploratory, 2026-11-02)

Tried the hand-written `T.Cube()` / `T.Vector()` path (auto-scheduling off) as
the next step for CV overlap. What the probe established:

* manual mode is entered with
  `pass_configs={PassConfigKey.TL_ENABLE_AUTO_SCHEDULE: False}` **and** a
  `tvm.target.Target("ascend")` context, otherwise layout inference raises
  `Target context required`;
* `T.MixedKernel` is rejected in manual mode: software `dual_copy` must sit in an
  explicit `T.Vector()` block, so the kernel has to be `T.Kernel` + explicit
  `Cube()` / `Vector()` regions (the sub-block id comes from `T.Vector()`);
* write-time semantic checks: `T.Parallel` must be inside a VF block, DMA copies
  cannot cast dtypes, `dual_copy` needs an exact 2:1 split ratio;
* cross-core and intra-core flags do lower (`asc_sync_inter_arrive/wait`,
  `asc_sync_notify/wait`, `asc_threadfence`), but `T.ascend_pipe_barrier` is
  silently dropped and **no `InsertSync` runs in manual mode**, so every
  MTE2->MTE1, MAD->FIX and V->MTE3 ordering is the user's responsibility;
* a one-tile manual kernel (QK + softmax + PV + publish) compiles and launches
  (`__mix__(1, 2)`, fixpipe `dual_copy`, `asc_copy_ub2l1`, VF calls all present)
  but computes garbage (rel ~1 or nan), and an isolated `p @ W` probe fails as
  well: the generated `asc_copy_ub2l1` destination uses a `sid * 1024` byte
  offset for a per-AIV L1 tile that is 8 KiB, i.e. the manual per-AIV L1
  addressing convention is undocumented in this build.

Verdict: the hand-written mixed kernel needs a dedicated compiler-side effort
(documented per-AIV L1 offsets, an emitting barrier API, sync protocol for the
2-AIV flag counting) rather than a session prototype. The shipped two-tile
auto-scheduled schedule remains the best verified kernel.

## 9. Trust conditions on a shared host

The host is shared with other users' benchmarking jobs; load moved between 37 and
900 during this session. Correctness is load-independent (only slower), and the
latency table above was taken at load 37-132 with the median of three rounds;
`msprof` task durations come from hardware counters and matched the event timings
within 5%. Runs taken at higher load are visible in the raw logs and were not
used for the numbers above.

## 10. Reproduction

```bash
export TILELANG_DEFAULT_TARGET=ascend
cd examples/ascend/mla_decode
python -m pytest test_mla_decode.py test_mla_decode_paged.py -q
python bench/bench_paged.py
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache \
    --output=/dev/shm/prof/paged python bench/prof_paged.py --block-size 64
python bench/parse_pipes.py paged=/dev/shm/prof/paged
```

`bench/prof_paged.py` selects the same phase-1 kernel as the compile entry
(it prints `dual_schedule=True/False`), so the capture always profiles the kernel
the benchmark would run.

## 11. Open items

1. **Cube/Vector overlap, part 2** - the two-tile schedule overlaps QK with the
   other tile's softmax. Pushing further is *not* reachable by rearranging this
   kernel: three slots, V^T double buffering, PV/softmax reordering and a deeper
   pair loop all measured slower or neutral (section 8), so the batched-phase
   two-tile schedule is a local optimum for this auto-scheduler. The next step
   would be a hand-written `T.MixedKernel` with explicit `T.Cube()` /
   `T.Vector()` regions and manual flags, or compiler support for cross-iteration
   multi-buffering.
2. **split-count sizing for serving** - the batch=64 sweep (section 4) shows the
   optimal `num_split` follows the KV length (`clamp(kv_tiles / 16, 1, 16)`), and
   a fixed 16 costs 2.9x at kv=1024 (0.85 -> 0.29 ms). A serving layer should pick
   it from the max live length; the same plan is what the packed split idea below
   generalises to uneven batches.
4. **packed length-proportional split plan** - keep the paged grid (and its
   cross-block overlap) but balance the KV tiles per block from a global plan;
   this is the follow-up to the persistent probe (section 4), where balance won
   1.35x on a ragged batch but the one-block-per-core grid lost 0.72-0.89x on
   uniform multi-batch work.
3. **paged full-SIMD vector modes** (`simd` / `simd_exp`) are not implemented in
   this kernel; the earlier fp32-partial build measured `simd_exp` at +56%.
3. **KV single read** remains blocked by the NT-only L1 GEMM and the layout
   inference conflict (see section 8).

## 12. History

The directory was slimmed to the native reference + the paged kernel (with its
tests, two benchmark entries, the profile summariser, this report and the
README). The paged phase-1 kernel then gained the two-tile CV-overlap schedule
(-7.6% end-to-end); everything else is unchanged from the earlier full-tree
verification.

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
| host load | correctness 37-60; latency runs 37-132 (shared host, see section 8) |

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

## 4. Two-tile CV-overlap schedule (phase 1)

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

## 5. Profile - paged phase 1, 43.6 us (two-tile schedule)

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

## 6. Bandwidth utilisation over data volume (two-tile schedule)

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

## 7. Measured dead ends

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

## 8. Trust conditions on a shared host

The host is shared with other users' benchmarking jobs; load moved between 37 and
900 during this session. Correctness is load-independent (only slower), and the
latency table above was taken at load 37-132 with the median of three rounds;
`msprof` task durations come from hardware counters and matched the event timings
within 5%. Runs taken at higher load are visible in the raw logs and were not
used for the numbers above.

## 9. Reproduction

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

## 10. Open items

1. **Cube/Vector overlap, part 2** - the two-tile schedule overlaps QK with the
   other tile's softmax. Overlapping the PV phase with the other tile's softmax
   (or keeping three tiles in flight) would push phase 1 below 40 us; the UB
   score buffer reuse is the current blocker (a second UB score tile needs
   ~16 KiB that the 221184 B budget does not have).
2. **paged full-SIMD vector modes** (`simd` / `simd_exp`) are not implemented in
   this kernel; the earlier fp32-partial build measured `simd_exp` at +56%.
3. **KV single read** remains blocked by the NT-only L1 GEMM and the layout
   inference conflict (see section 7).

## 11. History

The directory was slimmed to the native reference + the paged kernel (with its
tests, two benchmark entries, the profile summariser, this report and the
README). The paged phase-1 kernel then gained the two-tile CV-overlap schedule
(-7.6% end-to-end); everything else is unchanged from the earlier full-tree
verification.

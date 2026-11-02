# Self-verification report — paged MLA decode (Ascend 950PR)

Verification of the slimmed `examples/ascend/mla_decode` tree (native decode
reference + paged decode kernel + its tests/benchmarks) on the PR development
host, 2026-11-02. The paged kernel code is unchanged from the earlier full-tree
verification (commit `24bd4dac`); this round re-runs the slim tree end to end.

## 1. Environment

| item | value |
|---|---|
| NPU | Ascend 950PR (`dav-3510`), 8 devices on the host, one process on one device |
| core model | 1 cube (AIC) + 2 vector (AIV) per core; the kernel launches as a mix kernel (`Mix Block Dim = 32` = 16 AIC + 32 AIV) |
| clock / peaks | 1650 MHz; 4 TB/s GM and 432 TFLOPS fp16 cube used for utilisation |
| CANN | 9.3.0 (toolkit + 950 ops) |
| tilelang | `0.1.15+cpu.git876a59e6` |
| PyTorch | `2.10.0+cpu` + `torch_npu 2.10.0`, Python 3.11.16 |
| host load | correctness 540 -> 176 (cache-warm run); latency runs 75-132 |

Reference shape: `batch=1, heads=128, kv_len=8192, dim=512, pe_dim=64`.

## 2. Correctness — 26/26 passed

```
python -m pytest test_mla_decode.py test_mla_decode_paged.py -q
26 passed, 15 warnings in 31.10s
```

| file | cases | coverage |
|---|---:|---|
| `test_mla_decode.py` | 5 | native decode kernel vs the fp32 torch reference |
| `test_mla_decode_paged.py` | 21 | page geometries (64/128/256), shuffled block tables, ragged `cache_seqlens`, empty splits, multi-batch, dense equivalence |

Relative error: native 6.2e-4, paged 5.4e-4 (tolerance 1e-2).

## 3. Performance

`do_bench(backend="event", _n_warmup=10, _n_repeat=50)`, 3 rounds, median; host
load 75-132; the native reference is measured first in the same process so the
ratio is meaningful.

| case | e2e | TFLOPS | rel | speedup vs native |
|---|---:|---:|---:|---:|
| native decode (num_split=1) | 1549.4 us | 1.47 | 6.2e-4 | 1.0x |
| **paged, page 64, block_N=64, ns=16 (default)** | **56.9 us** | **40.1** | 5.4e-4 | **27.2x** |
| paged, page 128 | 57.1 us | 40.0 | 5.5e-4 | 27.1x |
| paged, block_N=128 (previous default) | 60.8 us | 37.5 | 5.4e-4 | 25.5x |
| paged, num_split=32 | 73.8 us | 30.9 | 5.4e-4 | 21.0x |

The default configuration was re-measured in the 12:02 slim run (load 540) at
57.3 us / 39.84 TFLOPS, i.e. the slimming did not change kernel behaviour.

## 4. Profile — paged phase 1, 47.7 us

`msprof op --warm-up=10 --launch-count=5 --aic-metrics=PipeUtilization,Memory,
MemoryL0,ResourceConflictRatio,L2Cache`, one kernel per capture, summarised by
`bench/parse_pipes.py`.

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
  `aic_mte1_wait` 96.5%, `aic_mte2_wait` 61.5%
* MTE instruction counts: mte2 **166**, mte1 178
* L2 read hit (AIC) 53.1%; GM->L1 1232 KB/block, pipe bandwidth usage 15.2%
* achieved 47.9 TFLOPS (11.1% of the fp16 cube peak) and **424 GB/s (10.6% of
  the 4 TB/s GM peak)**

Ideal cube/vector overlap ceiling: `max(cube 10.55, mte2 13.62, vec 27.86) =
27.9 us`, versus 47.7 us measured — the gap is the per-tile ping-pong dependency
chain, not one saturated pipe.

## 5. Bandwidth utilisation over data volume

Paged throughput measured 2026-10-30 with the then-shipped tier sweep (the paged
kernel is unchanged since; page 64, batch 1):

| kv_len | KV cache | moved | e2e | GB/s | util (of 4 TB/s) |
|---:|---:|---:|---:|---:|---:|
| 1024 | 1.12 MiB | 2.12 MiB | 23.8 us | 93.7 | 2.34% |
| 8192 | 9.0 MiB | 17.0 MiB | 57.2 us | 311.8 | 7.80% |
| 32768 | 36 MiB | 68 MiB | 172.0 us | 414.6 | 10.37% |
| 65536 | 72 MiB | 136 MiB | 324.9 us | 438.9 | 10.97% |

Utilisation rises with data volume because the fixed per-launch cost (~12-15 us
for the phase-1 + merge pair) dominates the small tiers; from ~8 MiB upward the
kernel is linear in the data volume and no tier is bandwidth limited.

## 6. Measured dead ends (paged phase 1)

| direction | measurement | verdict |
|---|---|---|
| remove OOB clamp/fill from the runtime-offset copies | MTE2 instructions 166 -> 102 but `aic_mte2_wait` 61.2% -> 75.9%, phase 1 slower | closed |
| compile-time split-loop trip count | phase 1 49.2 -> 51.8 us | closed |
| affine page fast path (no `block_table` load) | recovers exactly the OOB loss, net zero against the baseline; the idea has no headroom | closed |
| fp16 `O_partial` | 56.9 -> 59.4 us (32-column staging) / 68.7 us (full-width staging); the page-chunk publish is the bottleneck | closed |
| `block_N=32` (to free UB for a hand-written double buffer) | phase 1 56.3 us vs 39.4 us dense reference (+43% per tile) | closed |
| forced multi-buffering of the cross-unit tiles (`annotate_buffer_versions`) | score tile x2 +17.5%, L0C x2 +8.7%, p_l1 x2 +6.6%, p_l1+k_l1 x2 +1.2%; o_tmp x2 does not fit UB | closed |
| `unit_flag_ctrl=3` (pipelined MAD+FIXPIPE) on gemm / dual_copy | compiles, hangs at run time (needs a matching manual flag protocol) | closed |
| single KV read (reuse the K tile for PV) | L1 GEMM is NT-only (`trans_A=False, trans_B=True`); using one tile for both gemms hits a layout conflict; UB->L1 `transpose=True` compiles but silently ignores the transpose (wrong values); L1->L1 transpose does not compile | closed |
| page 128 | 57.1 us vs 56.9 us for page 64 — no win | closed |

## 7. Trust conditions on a shared host

The host is shared with other users' benchmarking jobs; load moved between 75
and 900 during this session. Correctness is load-independent (only slower), and
the latency table above was taken at load 75-132 with the median of three
rounds. `msprof` task durations come from hardware counters and matched the
event timings within 5%. Early rounds taken at load >400 are visible in the raw
logs (paged default 57.3 us, phase 1 47.7 us — same as the quiet window).

## 8. Reproduction

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

## 9. Open items

1. **merge split-loop `tmp` x4 multi-version** — merge-only A/B measured
   9.28 -> 8.79 us (-5.3%, identical results) but under host load 300-900, so
   only the medians were stable; expected e2e effect ~-0.5 us. Needs a quiet
   window before it is adopted.
2. **paged full-SIMD vector modes** (`simd` / `simd_exp`) are not implemented in
   this kernel; the earlier fp32-partial build measured `simd_exp` at +56%, so it
   stays unported.
3. **Cube/Vector overlap** — all auto-scheduler knobs are exhausted (see §6).
   Reaching the 27.9 us ceiling needs a hand-written `T.MixedKernel` with
   explicit `T.Cube()` / `T.Vector()` regions and manual ping-pong flags, or
   compiler support for cross-iteration multi-buffering.

## 10. History

The directory previously also carried the dense split-KV kernel, a unified
entry point, 101 tests, the tier sweep and the profiling helper scripts. Per the
delivery decision it was slimmed to the native reference + the paged kernel (plus
their tests, two benchmark entries, the profile summariser, this report and the
README). The paged kernel and merge kernel are byte-identical to the full-tree
version, and their measured numbers are unchanged (26/26 tests, 56.9-57.3 us
e2e, 47.7 us phase 1).

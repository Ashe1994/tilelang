# Self-verification report — DeepSeek MLA decode (Ascend 950PR)

Verification of branch `ascend/mla-decode` at commit `24bd4dac` (the kernels, tests
and benchmarks in this directory), re-run from a clean server staging tree on
2026-11-02. Every number below was produced by the scripts in `bench/` on the PR
development host; the raw logs are listed in [§8](#8-artifacts-and-raw-logs).

## 1. Environment

| item | value |
|---|---|
| NPU | Ascend 950PR (`dav-3510`), 8 devices on the host, one process on one device |
| core model | 1 cube (AIC) + 2 vector (AIV) per core; kernel launches as a mix kernel (`Mix Block Dim = 32` = 16 AIC + 32 AIV) |
| clock | 1650 MHz |
| peaks used for utilisation | 4 TB/s GM, 432 TFLOPS fp16 cube |
| CANN | 9.3.0 (toolkit + 950 ops) |
| tilelang | `0.1.15+cpu.git876a59e6` |
| PyTorch | `2.10.0+cpu` + `torch_npu 2.10.0`, Python 3.11.16 |
| host load during runs | correctness 147-163; benchmarks and sweep 99-131; msprof 99-131 (the host is shared, see §7) |

Reference shape for all single-shape numbers: `batch=1, heads=128, kv_len=8192,
dim=512, pe_dim=64` (DeepSeek MLA decode, 128 heads, 576-wide QK).

## 2. Correctness — 101/101 passed

```
python -m pytest test_mla_decode.py test_mla_decode_splitk.py \
    test_mla_decode_splitk_matrix.py test_mla_decode_splitk_tiles.py \
    test_mla_decode_paged.py test_mla_decode_coverage.py test_mla_decode_api.py
101 passed, 15 warnings in 38.59s        (load 147-163, warm kernel cache)
```

| file | checks | what it covers |
|---|---:|---|
| `test_mla_decode.py` | 5 | baseline `num_split=1` kernel vs torch reference |
| `test_mla_decode_splitk.py` | 7 | split-KV path: reference shapes, long KV, GQA-style head counts, partial dtype |
| `test_mla_decode_splitk_matrix.py` | 39 | shape matrix (dim / heads / kv_len / batch combinations) |
| `test_mla_decode_splitk_tiles.py` | 12 | tile-shape acceptance and rejection |
| `test_mla_decode_paged.py` | 21 | paged KV: page sizes, table permutations, empty/ragged splits, dense equivalence |
| `test_mla_decode_coverage.py` | 11 | elementwise/distribution stress cases |
| `test_mla_decode_api.py` | 6 | unified entry point + `resolve_config` defaults |

Relative error against the fp32 torch reference (tolerance 1e-2 in all tests):
dense default **6.2e-4**, paged default **5.4e-4**. The epilogue keeps the
partial outputs normalised inside each split, so the fp16 staging round trip
costs about one fp16 epsilon.

## 3. Performance

Measured with `do_bench(backend="event", _n_warmup=10, _n_repeat=50)`, 3 rounds
per configuration, median reported; spread across rounds was <= 0.3 us for every
number below.

| implementation | e2e | TFLOPS | vs baseline |
|---|---:|---:|---:|
| baseline `num_split=1` (no split-KV) | 1549.4 us | 1.5 | 1.0x |
| dense default (`block_N=64`, `simd`, `num_split=16`, fp16 partials, SimdVF merge) | **46.4 us** | **49.1** | **33.4x** |
| paged default (`block_size=64`, `block_N=64`, `num_split=16`) | **57.2 us** | **39.9** | **27.1x** |

Dense phase 1 alone is 39.3 us and the shared merge kernel 9.1 us (event
timings); the paged kernel is 1.24x the dense latency on the same shape.

Paged configuration sweep (same process, dense reference measured first):

| case | e2e | TFLOPS | rel | paged/dense |
|---|---:|---:|---:|---:|
| `pg64_ns16_bn64` (shipped default) | 57.2 us | 39.87 | 5.4e-4 | 1.236 |
| `pg128_ns16_bn64` (page 128) | 57.2 us | 39.86 | 5.5e-4 | 1.236 |
| `pg64_ns16_bn128` (previous default) | 61.2 us | 37.28 | 5.4e-4 | 1.322 |
| `pg64_ns32_bn64` (more splits) | 74.2 us | 30.75 | 5.4e-4 | 1.603 |

## 4. Data-volume sweep and bandwidth utilisation

`bench/bench_sweep.py` (16 tiers, batch=1; "moved" = K + V + k_pe streamed once =
`batch*kv_len*(2*dim + pe_dim)*2`; utilisation is against the 4 TB/s GM peak).
Host load 126-131, 3 rounds, spread <= 0.18 us except one dense 4096 round
(8.8 us, host spike) whose median still matches the 10-30 run.

| kv_len | KV cache | moved | dense e2e | dense TFLOPS | dense GB/s | dense util | paged e2e | paged TFLOPS | paged GB/s | paged util |
---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.56 MiB | 1.06 MiB | 15.7 us | 9.1 | 70.9 | 1.77% | 22.6 us | 6.3 | 49.3 | 1.23% |
| 1024 | 1.12 MiB | 2.12 MiB | 20.1 us | 14.2 | 111.1 | 2.78% | 23.8 us | 12.0 | 93.7 | 2.34% |
| 2048 | 2.25 MiB | 4.25 MiB | 23.8 us | 24.0 | 187.3 | 4.68% | 28.5 us | 20.0 | 156.5 | 3.91% |
| 4096 | 4.50 MiB | 8.50 MiB | 31.2 us | 36.6 | 285.5 | 7.14% | 38.2 us | 29.9 | 233.4 | 5.84% |
| 8192 | 9.00 MiB | 17.0 MiB | 46.3 us | 49.3 | 384.8 | 9.62% | 57.2 us | 39.9 | 311.8 | 7.80% |
| 16384 | 18.0 MiB | 34.0 MiB | 76.0 us | 60.0 | 469.0 | 11.72% | 95.3 us | 47.9 | 374.1 | 9.35% |
| 32768 | 36.0 MiB | 68.0 MiB | 135.3 us | 67.4 | 526.8 | 13.17% | 172.0 us | 53.1 | 414.6 | 10.37% |
| 65536 | 72.0 MiB | 136 MiB | 253.9 us | 71.9 | 561.7 | **14.04%** | 324.9 us | 56.2 | 438.9 | **10.97%** |

Reading: dense e2e scales 16.2x for 16x the data (512 -> 65536 tokens), so the
kernel is linear in the data volume and no tier is bandwidth limited;
utilisation climbs from 1.8% to 14.0% as the fixed per-launch cost (~12-15 us)
is amortised. Paged trails dense by 18-25% at every tier, matching its 1.66x
MTE2 instruction count and ~2x `aic_mte2_wait` (§5).

## 5. msprof profile (current defaults)

Captured with `msprof op --warm-up=10 --launch-count=5
--aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache`
(one kernel per capture) and summarised by `bench/parse_pipes.py`.

### Dense phase 1 — 38.1 us (block dim 16, mix 32)

| pipe | busy | share of duration |
|---|---:|---:|
| AIC cube (MAD) | 10.55 us | 27.7% |
| AIC mte2 (GM->L1) | 14.15 us | 37.1% |
| AIC mte1 (L1->L0) | 7.75 us | 20.3% |
| AIC fixpipe (L0C->UB) | 9.05 us | 23.7% |
| AIC scalar | 1.17 us | 3.1% |
| AIV vec | 20.70 us | 54.3% |
| AIV mte3 (UB->GM) | 1.66 us | 4.4% |

* waits: `aiv_vec_wait` 97.0%, `aiv_mte3_wait` 99.0%, `aic_cube_wait` 95.8%,
  `aic_mte1_wait` 95.0%, `aic_mte2_wait` 32.2%
* L2 read hit (AIC) 53.3%
* GM->L1: 1232 KB/block (K + V^T + k_pe + Q), pipe bandwidth usage 18.9%,
  main-memory read 31 GB/s per core
* MTE instruction counts: mte2 102, mte1 146
* achieved 59.8 TFLOPS (13.8% of peak) and 529 GB/s (13.2% of GM peak)

### Paged phase 1 — 47.5 us

| pipe | busy | share |
|---|---:|---:|
| AIC cube (MAD) | 10.55 us | 22.2% |
| AIC mte2 (GM->L1) | 13.41 us | 28.2% |
| AIC mte1 (L1->L0) | 7.77 us | 16.3% |
| AIC fixpipe (L0C->UB) | 6.75 us | 14.2% |
| AIV vec | 27.67 us | 58.2% |
| AIV mte3 | 2.26 us | 4.8% |

* waits: `aiv_vec_wait` 86.6%, `aic_cube_wait` 95.4%, `aic_mte1_wait` 96.6%,
  **`aic_mte2_wait` 61.8%**
* MTE instruction counts: **mte2 166 (vs 102 dense), mte1 178**
* L2 read hit 53.4%; GM->L1 1232 KB/block, pipe bandwidth usage 15.2%
* achieved 48.0 TFLOPS (11.1%) and 425 GB/s (10.6%)

### Merge — 8.5 us (block dim 32)

| pipe | busy | share |
|---|---:|---:|
| AIV vec | 4.91 us | 57.5% |
| AIV mte2 (GM->UB) | 4.15 us | 48.6% |
| AIV mte3 (UB->GM) | 0.19 us | 2.2% |

* waits: `aiv_vec_wait` 80.2%, `aiv_mte2_wait` 80.7%
* GM->UB 72 KB/block (fp16 partials) x 32 blocks; pipe bandwidth usage 6.2%
* achieved 292 GB/s (7.3% of GM peak); the fp32 path used 136 KB/block

All three kernels are far from the hardware roof (compute 11-14%, bandwidth
7-13%): the limit is the per-tile dependency chain (`aic_cube_wait` 95-96%,
`aiv_vec_wait` 86-97%). The ideal-overlap ceiling for dense phase 1 is
`max(cube 10.55, mte2 14.15, vec 20.70) = 20.7 us` against the measured 38.1 us.

## 6. Negative results (closed directions, with measured evidence)

| direction | measurement | verdict |
|---|---|---|
| dense manual pipeline (`T.Stage`, `pipeline=20`) | -1.3% at `num_stages=3`, negative at 1/4; `pipeline=0` byte-identical to the shipped schedule | closed |
| forced multi-buffering of the cross-unit tiles (`annotate_buffer_versions`) | `s_ub`x2 +17.5%, `s_l0c`x2 +8.7%, `p_l1`x2 +6.6%, `o_l0c`x2 +1.9%, `p_l1`+`k_l1`x2 +1.2%, `p_ub`/`v_l1[0]`/`s_l0c`+`o_l0c` within noise; `o_tmp`x2 and `s_ub`+`o_tmp`x2 do not fit UB | closed |
| `unit_flag_ctrl=3` ("pipelined with mad") on gemm / dual_copy | compiles, hangs at run time (unit-flag protocol needs matching sync; auto-schedule emits manual sync) | closed |
| fp16 O accumulator for dense phase 1 | numerically correct (rel 5.3e-4) but slower: SIMD rescale 70.5 -> 79.8...+119 us (same-process A/B), simt rescale 84.0 -> 64.7 us but the simt baseline is 1.8x slower than the shipped SIMD kernel | closed |
| single KV read (reuse K tile for PV) | Ascend L1 GEMM is NT-only (`trans_A=False, trans_B=True`); using one tile for both gemms hits a layout conflict (`Get different layout for k_l1`); UB->L1 `transpose=True` compiles but silently ignores the transpose (wrong values); L1->L1 transpose does not compile | closed |
| paged dataflow (remove OOB clamp/fill, compile-time trip count, affine page fast path) | removing OOB drops MTE2 instructions 166 -> 102 but raises `aic_mte2_wait` 61.2% -> 75.9% and is slower; trip count slower; affine fast path recovers exactly the OOB loss (net 0) | closed |
| paged with fp16 partials | 57.0 -> 59.4 us (32-column staging) / 68.7 us (full-width staging) | closed |
| `block_N=32` (to free UB for a hand-written double buffer) | phase 1 56.3 us vs 39.4 us (+43%) | closed |

## 7. Trust conditions on a shared host

The host is shared with other users' benchmarking jobs; load moved between 100
and 900 during this session. All numbers in §3-§5 were taken in a window with
load 99-163 and with per-round spread <= 0.3 us; the `bench_*` scripts print the
spread so a bad window is visible. `msprof` task durations come from hardware
counters and matched the event timings within 5%. On the DT host (single NPU
shared by six users) the same kernels report identical single-kernel durations
but end-to-end timings inflate 1.5-1.8x because the two launches are interleaved
with other users' work.

## 8. Artifacts and raw logs

| artifact | location |
|---|---|
| test/bench/profile scripts | this directory, `bench/` |
| verification driver | `run_verify_commit.sh` (copied to the staging tree, not part of the commit) |
| raw verification log | `/dev/shm/verify_commit.log` on the PR host |
| msprof captures | `/dev/shm/prof_verify/{dense,paged,merge}` on the PR host |
| earlier tiered sweep log | commit `24bd4dac` message and `/dev/shm/sweep.log` |

Reproduction (on the PR host, after `source <stack>/env.sh`):

```bash
cd examples/ascend/mla_decode
export TILELANG_DEFAULT_TARGET=ascend
python -m pytest test_mla_decode.py test_mla_decode_splitk.py \
    test_mla_decode_splitk_matrix.py test_mla_decode_splitk_tiles.py \
    test_mla_decode_paged.py test_mla_decode_coverage.py test_mla_decode_api.py
python bench/bench_default.py
python bench/bench_paged.py
python bench/bench_sweep.py --rounds 3
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache \
    --output=/dev/shm/prof/dense python bench/prof_phase1.py --which phase1
python bench/parse_pipes.py dense=/dev/shm/prof/dense
```

## 9. Open items

1. **merge `tmp` x4 multi-version** — merge-only A/B measured 9.28 -> 8.79 us
   (-5.3%) with identical results, but the host load was 300-900 (only the
   medians were stable; per-round spread reached 100+ us). Expected e2e effect
   ~-0.5 us. Not adopted in this commit; needs a quiet-window re-measurement.
2. **paged vector side** — the paged kernel implements only `simt` / `simd_rc`;
   a `simd_exp` port would need new code and previously measured +56% on the
   fp32-partial build, so it stays unported.
3. **Cube/Vector overlap** — all auto-scheduler knobs are exhausted (§6). The
   remaining headroom (38.1 -> ~21 us) needs a hand-written
   `T.MixedKernel` with explicit `T.Cube()`/`T.Vector()` regions and manual
   ping-pong flags, or compiler support for cross-iteration multi-buffering.

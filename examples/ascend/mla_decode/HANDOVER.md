# Handover - MLA decode on Ascend 950PR

Status as of 2026-11-25: branch `ascend/mla-decode`, HEAD `f5bad79f`, pushed to
`git@github.com:Ashe1994/tilelang.git` (private) with upstream
`git@github.com:tile-ai/tilelang.git` still configured as `origin`. Suite is
green (28 cases) and the measured numbers below are the ones to keep.

This document is written so that someone can pick the work up on a new machine
without the previous session's context.

## 1. What the operator is

    O[b,h,:] = softmax( (Q[b,h,:] . KV[b,s,:] + Q_pe[b,h,:] . K_pe[b,s,:]) * scale ) . KV[b,s,:]

DeepSeek-style MLA decode: `heads=128`, `dim=512`, `pe_dim=64`, `kv_len` up to
32768, fp16 in/out, fp32 accumulation. The reference shape used everywhere is
`batch=1, kv_len=8192` (2.28 GFLOP).

Three kernels live in `examples/ascend/mla_decode/`:

* `example_mla_decode.py` - native decode (num_split=1), the reference/baseline;
* `example_mla_decode_paged.py` - **production path**: page-table addressed KV,
  split-KV phase 1 (two-tile CV-overlap schedule) + a merge kernel;
* `mla_decode_persistent.py` - the same phase 1 under `T.Persistent`
  (one block per core); only wins on ragged batches, kept as an experiment.

## 2. Current state (verified)

| item | value |
|---|---|
| tests | `28 passed` (`5 native + 23 paged`), each against an fp32 torch reference (rtol=atol=1e-2) |
| B=1, kv=8192 | native 1549.4 us -> paged **52.6 us** = **29.5x**, 43.4 TFLOPS, rel 5.4e-4 |
| B=64, kv=8192 (auto split 4) | **1.49-1.61 ms**, ~98-102 TFLOPS |
| B=64, kv=32768 (auto split 8) | **5.31-5.96 ms**, ~100-112 TFLOPS |
| B=64 sweep 1024..32768 | 0.277 / 0.440 / 0.768 / 1.496 / 2.815 / 5.312 ms (DT host) |
| cube utilisation | ~47-52% of the per-block wall, ~22-24% of the 432 TFLOPS peak |
| CANN FIA reference, same data | 1.5-1.6x faster than this kernel (151 TFLOPS, 35% of peak) |

`README.md` has the per-case tables; `VERIFICATION.md` has the environment,
profile captures and the dead-end log with the numbers.

## 3. Environment

Two shared hosts, same IP `123.60.114.238`, different ports (see
`D:\Work\AI_WorkSpace\.ssh\config`: aliases `pr` = port 10118, `dt` = port 10236,
user root, key auth):

| item | PR host (`pr`) | DT host (`dt`) |
|---|---|---|
| purpose | main verification, CANN release build | fallback verification, weekly CANN |
| env script | `/data/h00662772/mla_opt/scripts/env.sh` | `/data/h00662772/env.sh` |
| python | 3.11 venv `/data/h00662772/envs/tilelang-ascend` | 3.12 venv `/data/h00662772/venv312` |
| CANN | 9.3.0 (`/home/h00517926/cann-930/cann-9.3.0`) | weekly 9.3.0 (`/home/yangqiang/Ascend-weekly-20261021/cann-9.3.0`) |
| tilelang | `0.1.15+cpu.git876a59e6` | `0.1.15+cpu.gitf2b8866e` (wheel) |
| staging tree | `/data/h00662772/mla_opt_exp/val` | `/data/h00662772/mla_opt_exp/dt_v1` (240 KB, matches HEAD) |
| bench lock | `flock /dev/shm/mla_bench.lock` | `flock /dev/shm/mla_dt.lock` |

Rules that cost time when forgotten:

* **`source` the env script first, then `cd`** - `env.sh` changes the working
  directory; never `cd` before sourcing.
* wrap every benchmark/profiling run in the `flock` above: the NPU is shared and
  two concurrent runs corrupt both the numbers and the captures.
* DT is slower than PR at small shapes (B=1: 0.112 vs 0.053 ms) but matches at
  batch=64; use DT for A/B, PR for absolute numbers.
* both hosts are heavily shared (load moved 30-900); re-take numbers when quiet,
  quote the median of 2-3 `do_bench` rounds.

## 4. New machine setup

1. Install the Ascend CANN toolkit (9.3.0) + driver for the 950PR, then
   `source $ASCEND_HOME_PATH/set_env.sh`.
2. Install Python 3.11/3.12 with `torch 2.10.0+cpu` and `torch_npu 2.10.0`
   (the Ascend wheels are on the DT host under `/home/z00946738/`; the DT setup
   script that builds the venv is `mla-opt-exp/tools/dt_setup_env.sh`).
3. `git clone git@github.com:Ashe1994/tilelang.git && cd tilelang && git
   checkout ascend/mla-decode`; install with `pip install -e .` (or the matching
   `tilelang-0.1.15+cpu.git*.whl` if the build toolchain is not present).
4. `export TILELANG_DEFAULT_TARGET=ascend`; add
   `ASCEND_HOME_PATH`/`LD_LIBRARY_PATH` from CANN's `set_env.sh`.
5. Verify: `python -c "import torch_npu, tilelang; print(tilelang.__version__)"`,
   then run the suite (section 5). If it passes, the machine is ready.
6. Copy the ssh config for `pr`/`dt` (or recreate the two Host blocks), and set
   `GIT_SSH=C:\Windows\System32\OpenSSH\ssh.exe` on Windows before `git push`
   (the private remote needs the OpenSSH client explicitly).
7. Optional: copy `D:\Work\AI_WorkSpace\mla-opt-exp\` (tools + profiling
   exports, see section 8) - 225 MB of msprof evidence; everything else in it
   is reproducible.

## 5. Runbook

```bash
export TILELANG_DEFAULT_TARGET=ascend
cd examples/ascend/mla_decode

# correctness (28 cases, ~10-30 s)
python -m pytest test_mla_decode.py test_mla_decode_paged.py -q

# latency + correctness guard, B=1 reference shape
python bench/bench_paged.py

# multi-batch / ragged comparison (native vs paged vs persistent)
python bench/bench_persistent.py

# batch=64 KV-length sweep (needs ~2 GB per case at kv=32768)
python bench/bench_batch64.py --lens 8192 --batch 64

# msprof capture of phase 1 + parse
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,ResourceConflictRatio,L2Cache \
    --output=/dev/shm/prof/paged python bench/prof_paged.py --block-size 64
python bench/parse_pipes.py paged=/dev/shm/prof/paged

# MindStudio Insight (Timeline needs PipeTimeline!)
msopprof --output=/dev/shm/insight/paged --aic-metrics=Default,Roofline,Occupancy,BasicInfo,MemoryDetail,PipeUtilization,PipeTimeline \
    python bench/prof_multi.py --variant paged --batch 64 --lens 8192
# import /dev/shm/insight/paged/OPPROF_*/visualize_data.bin, keep dump/ next to it
```

Remote helper (Windows): `mla-opt-exp/tools/remote.ps1 -Action push|pull
-HostAlias pr|dt -Local <file> -Remote <path>` (base64 over ssh, no scp needed).
`capture_msprof.sh` / `capture_insight.sh` reproduce the whole capture sets.

## 6. Design in one paragraph

Phase 1 is a grid of `batch * num_split` blocks; each block owns one split of
one sequence and walks it in 64-token tiles, keeping **two tiles in flight**
(`TILE_SLOTS=2`): K/K_pe are double-buffered in L1, the two QK results land in
two L0C score tiles, so slot 1's QK runs while slot 0's softmax is on the AIVs.
QK is two GEMMs (pe half + content half) accumulating into one L0C, PV is NT
with per-output-chunk transposed V^T tiles in L1, and the online softmax keeps
`m/l/o` in UB. Blocks publish fp32 partials + log-sum-exp to `O_partial`/`ML`,
and a small merge kernel combines `NS` splits per (batch, head).
`pick_num_split` sizes `NS` (~32 KV tiles per split, floor 2, cap 8, floor 8 for
batch 8-16) and matches the measured optima.

## 7. Measured dead ends - do not re-run these

| attempted | result |
|---|---|
| PTO backend rebuild / `P @ K` L0 path | excluded by decision; on AscendC it faults the AI core (507015) |
| single KV read by reusing the K tile (`transpose_B=False`) | rejected: **Ascend L1 GEMM is NT only** |
| natural V instead of V^T | same NT restriction |
| `block_H=64` (frees 72 KiB L1) | `T.dual_copy` requires M=128 |
| `chunk_d=256` / one 64 KiB V^T tile | compile (Z3 UB) or launch (`rtsLaunchKernelWithHostArgs`) failure |
| `TILE_SLOTS` 3/4 | 3 slots -20%, 4 slots exceeds L1 |
| V^T double buffering | within noise |
| single-tile schedule with `num_stages=2..4` | 13% slower than the two-tile schedule |
| fused softmax, `simd_exp`, `block_N=128`, `num_split=32` | all slower (+8..+56%) |
| packed length-proportional split plan | correct but 3-10x slower; `if` around the body breaks ThreadSync (races) |
| L2 cache hints, page-table locality, fp16 partials | neutral (0-2%) |
| persistent phase 1 on uniform multi-batch | 0.72-0.89x of paged (wins 1.35x only when ragged) |
| manual `T.MixedKernel` | rejected by the compiler in this release |

Full numbers and compiler messages: `VERIFICATION.md` section 8.

## 8. Artifacts outside the repo

`D:\Work\AI_WorkSpace\mla-opt-exp\` (local only, copy it if you want the history):

| path | what |
|---|---|
| `tools/` | `remote.ps1`, `capture_msprof.sh`, `capture_insight.sh`, experiment generators (`build_*.py`), DT setup scripts |
| `opt/` | ~40 one-off probes/benches used for the dead ends (keep for reference) |
| `msprof/` | **225 MB** of exported captures: `msprof_export/` (8 raw msprof runs), `insight/` (4 msopprof bins), `tl_export/` (2 timeline captures with `trace.json` + `dump/timeline.bin`) |
| `intake/` | commit messages used on the branch |

The staging trees on both hosts hold only the current tree (240 KB DT); kernel
caches (`tl_cache`) and anything under `/dev/shm` are disposable.

## 9. Open items / where to go next

1. **Dispatch to CANN FIA** for the decode attention (measured **1.5-1.6x** at
   151 TFLOPS): wrap `npu_fused_infer_attention_score` (MLA mode: `query_rope` /
   `key_rope`, `input_layout="BNSD"`, cache `[num_blocks, N=1, block_size, D]`,
   fp16) behind one entry point and keep this kernel as the fallback. Highest
   value per hour of work.
2. **Serving level**: continuous batching (per-token cost is 2.1x better at
   batch=64 than batch=1), length-bucketed compilation so `num_split` matches the
   live lengths, pre-allocated partials.
3. **Hand-written AscendC kernel** if the tilelang kernel must reach FIA:
   the same two-tile algorithm with manual multi-buffering, explicit flags and
   the L0 NN path (halves the KV read). Everything cheaper was measured and lost.
4. **Compiler asks** (with evidence in section 7): NN GEMM on the AscendC L1
   path, `dual_copy` with M<128, and a safe per-block branch (no ThreadSync
   hoisting).
5. batch 16/32 split table; merge-kernel fold-in; FIA `PipeTimeline` capture for
   an operator-to-operator comparison in Insight.

## 10. Conventions

* Branch `ascend/mla-decode`, commits prefixed `[Ascend] MLA decode: ...`, each
  commit body carries the measured numbers and the compiler messages.
* Repo docs stay ASCII; keep `README.md` (user-facing) and `VERIFICATION.md`
  (self-verification) in sync with any measurement.
* Every experiment gets a number and a verdict in `VERIFICATION.md` section 8,
  even when it fails - the list is the most valuable part of this handover.
* Never leave a knob the measurements did not earn; the removed ones
  (`pv_l0`, `l2_hints`, `mla_decode_packed.py`) are recorded as dead ends.

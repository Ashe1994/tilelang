# MLA decode benchmarks and profiling

Every script is standalone: run it from any directory (`python
examples/ascend/mla_decode/bench/bench_default.py`), no environment variables are
needed beyond the Ascend runtime (see the [parent README](../README.md)).

| Script | What it does | Output |
| --- | --- | --- |
| [bench_default.py](bench_default.py) | Baseline `num_split=1` kernel vs the shipped dense defaults; checks correctness of the optimized path first | `DEFAULTRUN rel=... baseline_us=... e2e_us=... speedup=... tflops=...` |
| [bench_paged.py](bench_paged.py) | Paged kernel over page size, tile width and split count, including the previous `block_N=128` default as an A/B; the dense default kernel is measured first in the same process as the reference | `PGDENSE dense_optimized_us=...` then one `PG <case> ... e2e_us=... rel=... paged_over_dense=...` line per case |
| [bench_partial_dtype.py](bench_partial_dtype.py) | A/B of the merge input dtype: compiles phase 1 + merge for fp32 and fp16 `O_partial` in one process, checks both against the reference (plus two softmax-distribution stresses) and times merge-only, phase 1 and end-to-end interleaved. The shipped defaults differ per layout (dense fp16, paged fp32 - see the parent README), so this is the tool to re-check the choice on a new shape | `REL ...`, then `TIME` / `AB` / `DECOMP` / `E2E_TFLOPS` / `VERDICT` lines |
| [prof_phase1.py](prof_phase1.py) | msprof entry that builds **only** the dense phase-1 kernel (or the full call with `--which e2e`) | prints the partial-output magnitude |
| [prof_paged.py](prof_paged.py) | msprof entry that builds only the paged phase-1 kernel (`--block-size`, `--block-n`, `--num-split`) | prints the partial-output magnitude |
| [prof_merge.py](prof_merge.py) | msprof entry that builds only the merge kernel (`--num-split`, `--ml-width`, `--partial-dtype` default `float16`) | prints the output magnitude |
| [bench_sweep.py](bench_sweep.py) | Complete data-volume tier table: KV sizes (and an optional batch axis) for dense and paged, each verified against the reference, with e2e latency, TFLOPS, achieved KV read bandwidth and its utilisation against the 4 TB/s GM peak | `SWEEP ... kv_len=... moved_mb=... e2e_us=... tflops=... gbps=... util_pct=...` lines, then `SWEEP_BEST` per layout |
| [prof_sweep.py](prof_sweep.py) | msprof entry for one tier of the sweep (`--layout dense|paged --kv-len N --batch B`), building only the phase-1 kernel so the capture has a single kernel | prints the partial-output magnitude |
| [parse_pipes.py](parse_pipes.py) | Summarises a capture's CSVs: per-pipe busy time and share, wait ratios, the L2 read hit rate, GM->L1 / GM->UB traffic and bandwidth usage, L0 bandwidth, the MTE instruction counts, group-busy totals, and the achieved TFLOPS / GB/s | one block per capture |

`bench_default.py` reports the baseline above the optimized path on purpose: the
first measurement point in a process is inflated by the warm-up, so the ratio is
only meaningful when both numbers come from the same process. `bench_paged.py`
prints `dense_optimized_us` before any paged case for the same reason.

Field meanings: `e2e_us` is one call (phase 1 + merge) in microseconds,
`tflops` is `2*b*h*s*(dim+pe_dim) + 2*b*h*s*dim` divided by the latency, `rel` is
`max|out - ref| / max|ref|` against the fp32 torch reference, and
`paged_over_dense` is the paged latency divided by the dense one (below 1.0 means
paged is faster).

`bench_partial_dtype.py` compiles four kernels (a phase-1/merge pair per dtype), so
it is the slowest entry. The shipped choice per layout is recorded in the parent
README, and the script prints a `VERDICT` line combining its acceptance checks with
the load/spread trust flag, so it can re-validate that choice on a new shape.

`bench_sweep.py` is the data-volume sweep: it prints one line per tier plus a
`SWEEP_BEST` line per layout, and it is the tool behind the tier table in the
parent README. `prof_sweep.py` is its profiling half - run it under `msprof op`
for a chosen tier and feed the output directory to `parse_pipes.py`. Tiers are
built with `--kv-lens`, the optional batch axis with `--batches`; a tier is
roughly `2x` cheaper than the previous one in compile time, so a full 8-tier run
takes a few minutes on a warm cache.

## Timing methodology

Latency comes from `do_bench(backend="event", _n_warmup=10, _n_repeat=50)`,
3-5 rounds per configuration, median reported. Data is only used when the spread
across rounds is at most 0.3 us; the reported numbers have 0.02-0.3 us.
The machine is shared - if the spread grows or the first round is far above the
others, the load is too high and the numbers should be re-taken later.

## msprof captures

One kernel per capture. All three `prof_*.py` entries build exactly one
`prim_func`, so the capture contains a single kernel name. Do not profile a
combined call: when a single call launches phase 1 and merge, the profiler
helper attributes only one of them (which once inflated an end-to-end speedup by
1.7x). Combined timings always use `backend="event"`.

```bash
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,L2Cache,ResourceConflictRatio \
    --output=/dev/shm/opp_dense python .../bench/prof_phase1.py --which phase1
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,L2Cache,ResourceConflictRatio \
    --output=/dev/shm/opp_pg64 python .../bench/prof_paged.py --block-size 64
msprof op --warm-up=10 --launch-count=5 \
    --aic-metrics=PipeUtilization,Memory,MemoryL0,L2Cache,ResourceConflictRatio \
    --output=/dev/shm/opp_merge python .../bench/prof_merge.py --partial-dtype float16
python .../bench/parse_pipes.py \
    dense=/dev/shm/opp_dense paged_p64=/dev/shm/opp_pg64 merge=/dev/shm/opp_merge
```

- **Pass all five `--aic-metrics` groups.** `parse_pipes.py` reads
  `PipeUtilization`, `Memory`, `MemoryL0`, `L2Cache` and `ResourceConflictRatio`
  (duration and frequency come from `OpBasicInfo`); omitting one silently drops
  the corresponding lines (L2 hit rate, traffic, L0 bandwidth, MTE instruction
  counts).
- **Write the capture to `/dev/shm`** (or a data partition). A full root
  partition fails the capture, and it also fails the kernel compilation because
  the tilelang cache lives under the home directory by default. On a shared
  machine point `TILELANG_CACHE_DIR` and `TMPDIR` at a partition with room too.
- **Adding `--kernel-name` changes the directory layout**: without it the CSVs
  are under `OPPROF_*/<kernel>/0/`, with it they are directly under
  `OPPROF_*/*.csv`. `parse_pipes.py` searches recursively for
  `OpBasicInfo*.csv`, so both work. Omitting the flag keeps the command
  independent of the kernel's internal name, which is why the commands above do.
- **`--warm-up=10` removes the first-launch outlier.** A capture that includes
  JIT preparation shows a task duration orders of magnitude above the steady
  state (1.85 s instead of ~64 us); discard such a record and re-capture.
- The merge kernel's `prim_func` is named `merge_main` (not `main`) so that a
  filtered capture distinguishes it from phase 1.

`parse_pipes.py` prints, per capture: task duration, block dim and frequency; AIC
busy time and share for cube / mte1 / mte2 / fixpipe / scalar; AIV busy time and
share for vec / mte2 / mte3; the dominant wait ratios; the L2 read hit rate
(close/far split); GM->L1 and GM->UB traffic with the per-block bandwidth and pipe
busy share; L0 read/write bandwidth; the MTE2/MTE1 instruction counts; the group
busy totals; and the achieved compute and aggregate bandwidth as a share of the
platform peak. AIC and AIV are separate cores running concurrently, so the two
groups do not add up to 100%, and pipes within a group can overlap.

# MLA decode benchmarks and profiling

Every script is standalone: run it from any directory (`python
examples/ascend/mla_decode/bench/bench_default.py`), no environment variables are
needed beyond the Ascend runtime (see the [parent README](../README.md)).

| Script | What it does | Output |
| --- | --- | --- |
| [bench_default.py](bench_default.py) | Baseline `num_split=1` kernel vs the shipped dense defaults; checks correctness of the optimized path first | `DEFAULTRUN rel=... baseline_us=... e2e_us=... speedup=... tflops=...` |
| [bench_paged.py](bench_paged.py) | Paged kernel at several page sizes and split counts, with the dense optimized kernel as the in-process reference | `PGDENSE dense_optimized_us=...` then one `PG <case> ... e2e_us=... rel=... paged_over_dense=...` line per case |
| [bench_partial_dtype.py](bench_partial_dtype.py) | A/B of the merge input dtype: compiles phase 1 + merge for fp32 and fp16 `O_partial` in one process, checks both against the reference (plus two softmax-distribution stresses) and times merge-only, phase 1 and end-to-end interleaved. **Measured conclusion: no gain** - the merge is latency bound, so fp32 stays the default | `REL ...`, then `TIME` / `AB` / `DECOMP` / `E2E_TFLOPS` / `VERDICT` lines |
| [prof_phase1.py](prof_phase1.py) | msprof entry that builds **only** the dense phase-1 kernel (or the full call with `--which e2e`) | prints the partial-output magnitude |
| [prof_paged.py](prof_paged.py) | msprof entry that builds only the paged phase-1 kernel (`--block-size`, `--block-n`, `--num-split`) | prints the partial-output magnitude |
| [prof_merge.py](prof_merge.py) | msprof entry that builds only the merge kernel (`--num-split`, `--ml-width`) | prints the output magnitude |
| [parse_pipes.py](parse_pipes.py) | Summarises a capture's 8 CSVs: per-pipe busy time and share, wait ratios, achieved TFLOPS and GB/s | one block per capture |

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
it is the slowest entry; its conclusion is already recorded in the parent README's
known limitations, and it prints a `VERDICT` line combining its acceptance checks
with the load/spread trust flag.

## Timing methodology

Latency comes from `do_bench(backend="event", _n_warmup=10, _n_repeat=50)`,
three rounds per configuration, median reported. Data is only used when the
spread across rounds is at most 0.3 us; the reported numbers have 0.02-0.11 us.
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
    --output=/dev/shm/opp_dense python .../bench/prof_phase1.py --which phase1
msprof op --warm-up=10 --launch-count=5 \
    --output=/dev/shm/opp_pg64 python .../bench/prof_paged.py --block-size 64
msprof op --warm-up=10 --launch-count=5 \
    --output=/dev/shm/opp_merge python .../bench/prof_merge.py
python .../bench/parse_pipes.py dense=/dev/shm/opp_dense paged_p64=/dev/shm/opp_pg64
```

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
share for vec / mte2 / mte3; the dominant wait ratios; and the achieved compute
and aggregate bandwidth as a share of the platform peak. AIC and AIV are separate
cores running concurrently, so the two groups do not add up to 100%, and pipes
within a group can overlap.

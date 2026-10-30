"""A/B benchmark: MLA decode O_partial in fp32 vs fp16 (phase 1 -> merge).

Evidence this tests (from the handover profiling): the shared merge kernel costs
12.9 us and 69.9% of its cycles are ``mte2`` -- it is a pure streaming reduction
that reads exactly ``num_split * batch * heads * dim`` elements of O_partial.
Storing those partials as fp16 halves that read and should cut the merge time by
roughly half, at the price of one fp16 rounding of an already-normalised partial
(see ``example_mla_decode_splitk`` module docstring).

What it does, in ONE process so the two variants see the same load / clocks:
  1. compile phase 1 + merge twice, ``partial_dtype="float32"`` and ``"float16"``
     (the same kernels ``compile_mla_decode_splitk(partial_dtype=...)`` builds);
  2. check both against a float32 torch reference (``rel = max|out-exp|/max|exp|``,
     the same metric as ``bench_default.py``), on the reference data and on the
     two distribution stresses (``d_peaked``: q*8, ``d_flat``: q*0.05);
  3. time merge-only (the compiled merge callable, pre-computed O_partial) and
     e2e (phase 1 + merge), ``backend="event"``, 3 interleaved rounds per
     variant -> median + spread.

Reference shape: B=1, H=128, S=8192, D=512, PE=64, num_split=16.

Measured on 2026-10-30 with the vector-unit merge: fp16 partials are the dense
default (phase 1 41.8 -> 39.3 us, merge reads 136 -> 72 KB/block), while the
paged kernel keeps fp32 because its column-sliced fp16 publish measured slower
(57.0 -> 59.4 us; a full-width staging buffer starves the auto-scheduler and
lands at 68.7 us). Run this script before changing either default.

    flock -w 5400 /dev/shm/mla_bench.lock -c 'python bench_partial_dtype.py'

Only ``backend="event"`` is used: msprof_detail undercounts a two-kernel e2e
(handover 04-B1).
"""

import os
import statistics
import sys
from pathlib import Path

import torch
import tilelang
from tilelang.profiler import do_bench

try:  # a CUDA-only build has no torch_npu; the benchmark itself needs the NPU
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

# Importable from any working directory: kernels and tests live one level up.
PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode_splitk import (
    mla_decode_merge,
    mla_decode_splitk_fwd,
    pick_tiling,
)

B, H, S, D, PE = 1, 128, 8192, 512, 64
NS = 16

DTYPES = ("float32", "float16")
# acceptance thresholds from the handover optimisation notes (P2)
MERGE_TARGET_US = 9.0
REL_TARGET = 1e-2


def wall(fn):
    """Wall clock of one call, microseconds, event backend (04-B1)."""
    return do_bench(fn, backend="event", _n_warmup=10, _n_repeat=50) * 1000.0


def make_inputs(seed=0, q_scale=1.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn((B, H, D), generator=g) * q_scale).half().npu()
    q_pe = (torch.randn((B, H, PE), generator=g) * q_scale).half().npu()
    kv = torch.randn((B, S, D), generator=g).half().npu()
    k_pe = torch.randn((B, S, PE), generator=g).half().npu()
    return q, q_pe, kv, k_pe


def ref(q, q_pe, kv, k_pe, scale):
    """float32 reference, same wording as bench_default.py."""
    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bsd->bhd", attn, kv.float())


def rel_err(out, expected):
    return (out.float() - expected).abs().max().item() / expected.abs().max().item()


def build_pair(partial_dtype, scale):
    """Compile the phase-1/merge pair exactly as compile_mla_decode_splitk does."""
    block_N, chunk_d, merge_chunk_d = pick_tiling(S, D)
    partial = tilelang.compile(
        mla_decode_splitk_fwd(
            batch=B, heads=H, kv_len=S, dim=D, pe_dim=PE,
            block_N=block_N, chunk_d=chunk_d, softmax_scale=scale,
            num_split=NS, partial_dtype=partial_dtype,
        ),
        out_idx=[4, 5],
    )
    merge = tilelang.compile(
        mla_decode_merge(
            batch=B, heads=H, kv_len=S, dim=D, pe_dim=PE,
            block_N=block_N, chunk_d=merge_chunk_d, num_split=NS,
            rows_per_block=16, num_stages=4, partial_dtype=partial_dtype,
        ),
        out_idx=[2],
    )
    return partial, merge


def main():
    scale = (D + PE) ** -0.5
    flops = 2 * B * H * S * (D + PE) + 2 * B * H * S * D
    load_begin = os.getloadavg()
    print(f"CONFIG B={B} H={H} S={S} D={D} PE={PE} num_split={NS}", flush=True)
    print(f"LOAD_BEGIN {load_begin}", flush=True)

    # ---------------- compile both variants ----------------
    pairs = {}
    for dt in DTYPES:
        pairs[dt] = build_pair(dt, scale)
        print(f"COMPILED partial_dtype={dt}", flush=True)

    # ---------------- correctness: reference + two stresses ----------------
    refs = {}
    rels = {}
    for label, q_scale in (("d_ref", 1.0), ("d_peaked", 8.0), ("d_flat", 0.05)):
        q, q_pe, kv, k_pe = make_inputs(q_scale=q_scale)
        refs[label] = ref(q, q_pe, kv, k_pe, scale)
        for dt in DTYPES:
            partial, merge = pairs[dt]
            out = merge(*partial(q, q_pe, kv, k_pe))
            torch.npu.synchronize()
            r = rel_err(out, refs[label])
            rels[(label, dt)] = r
            finite = bool(torch.isfinite(out.float()).all().item())
            print(
                f"REL {label} partial={dt} rel={r:.6f} finite={finite} "
                f"{'PASS' if r < REL_TARGET and finite else 'FAIL'}",
                flush=True,
            )

    # ---------------- timing, interleaved rounds ----------------
    q, q_pe, kv, k_pe = make_inputs()
    # pre-compute one O_partial per variant for the merge-only probe
    held = {}
    for dt in DTYPES:
        partial, merge = pairs[dt]
        op, ml = partial(q, q_pe, kv, k_pe)
        torch.npu.synchronize()
        held[dt] = (op, ml)

    samples = {("e2e", dt): [] for dt in DTYPES}
    samples.update({("merge", dt): [] for dt in DTYPES})
    samples.update({("phase1", dt): [] for dt in DTYPES})
    for rnd in range(3):
        for dt in DTYPES:
            partial, merge = pairs[dt]
            # bind through default args: a bare closure over the loop variable is
            # fine here because do_bench runs it inside this iteration, but the
            # explicit binding makes the intent (and a later refactor) safe.
            samples[("e2e", dt)].append(
                wall(lambda p=partial, m=merge: m(*p(q, q_pe, kv, k_pe)))
            )
        for dt in DTYPES:
            partial = pairs[dt][0]
            # phase 1 alone: isolates the publish/store side of the A/B from the
            # merge side (e2e = phase1 + merge, both launches serialised).
            samples[("phase1", dt)].append(
                wall(lambda p=partial: p(q, q_pe, kv, k_pe))
            )
        for dt in DTYPES:
            merge = pairs[dt][1]
            op, ml = held[dt]
            samples[("merge", dt)].append(wall(lambda m=merge, o=op, l=ml: m(o, l)))
        print(f"ROUND {rnd}", flush=True)

    load_end = os.getloadavg()
    stats = {}
    for (what, dt), runs in samples.items():
        stats[(what, dt)] = statistics.median(runs)
        print(
            f"TIME {what} partial={dt} runs_us={[round(x, 3) for x in runs]} "
            f"median_us={stats[(what, dt)]:.2f} spread_us={max(runs) - min(runs):.3f}",
            flush=True,
        )

    # ---------------- summary ----------------
    o_partial_mib = {dt: NS * B * H * D * (4 if dt == "float32" else 2) / 2**20 for dt in DTYPES}
    print()
    print(f"O_PARTIAL_SIZE fp32={o_partial_mib['float32']:.2f}MiB fp16={o_partial_mib['float16']:.2f}MiB", flush=True)
    for what in ("merge", "phase1", "e2e"):
        a, b = stats[(what, "float32")], stats[(what, "float16")]
        print(
            f"AB {what}_us fp32={a:.2f} fp16={b:.2f} delta_us={b - a:+.2f} "
            f"ratio={b / a:.3f} delta_pct={(b / a - 1) * 100:+.1f}%",
            flush=True,
        )
    # e2e should equal phase1 + merge up to launch/serialisation overhead
    for dt in DTYPES:
        total = stats[("phase1", dt)] + stats[("merge", dt)]
        print(
            f"DECOMP {dt} phase1+merge={total:.2f} e2e={stats[('e2e', dt)]:.2f} "
            f"gap={stats[('e2e', dt)] - total:+.2f}",
            flush=True,
        )
    print(
        f"E2E_TFLOPS fp32={flops / stats[('e2e', 'float32')] / 1e6:.2f} "
        f"fp16={flops / stats[('e2e', 'float16')] / 1e6:.2f}",
        flush=True,
    )

    merge_fp16 = stats[("merge", "float16")]
    merge_ok = merge_fp16 <= MERGE_TARGET_US
    e2e_ok = stats[("e2e", "float16")] <= stats[("e2e", "float32")]
    rel_ok = all(rels[(label, "float16")] < REL_TARGET for label in ("d_ref", "d_peaked", "d_flat"))
    spread_all = max(max(v) - min(v) for v in samples.values())
    loaded = load_end[0] > 60 or load_begin[0] > 60
    trust = "TRUSTED" if (spread_all <= 0.3 and not loaded) else "UNTRUSTED"
    print(
        f"VERDICT merge_ok={merge_ok} (target<={MERGE_TARGET_US}) e2e_ok={e2e_ok} "
        f"rel_ok={rel_ok} max_spread_us={spread_all:.3f} "
        f"load={load_begin[0]:.0f}->{load_end[0]:.0f} trust={trust}",
        flush=True,
    )
    print(f"LOAD_END {load_end}", flush=True)


if __name__ == "__main__":
    main()

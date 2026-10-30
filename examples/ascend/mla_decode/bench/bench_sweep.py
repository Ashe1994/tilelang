"""Tiered data-volume sweep for MLA decode, with bandwidth utilisation.

The sweep builds a complete case table over KV-cache sizes (the "data volume"
axis) and, optionally, over batch, and reports for every tier:

  * correctness (rel error against the fp32 torch reference, tolerance 1e-2),
  * e2e latency (``do_bench(backend="event")``, median of ``--rounds``),
  * achieved compute (TFLOPS) and read bandwidth (GB/s),
  * bandwidth utilisation against the 4 TB/s GM peak of Ascend 950PR.

"moved" bytes are the bytes the kernel actually has to stream from GM for the
attention itself: ``batch * kv_len * (2*dim + pe_dim) * 2`` -- K and V are each
read once (V transposed) and k_pe once.  This is ~2x the KV-cache footprint
(``batch * kv_len * (dim + pe_dim) * 2``) because K and V are separate streams;
both numbers are printed.

Reading the table: small tiers are launch-latency bound (low utilisation),
large tiers approach the bandwidth roof.  Compare a tier across layouts
(dense vs paged) and across code changes -- not across different tiers.

    python bench_sweep.py                       # default tier table
    python bench_sweep.py --kv-lens 4096,8192   # custom tiers
    python bench_sweep.py --batches 1,2,4 --kv-lens 8192   # batch axis
"""

import argparse
import statistics
import sys
from pathlib import Path

import torch
from tilelang.profiler import do_bench

try:  # a CUDA-only build has no torch_npu; the benchmark itself needs the NPU
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

# Importable from any working directory: kernels and tests live one level up.
PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode_paged import compile_mla_decode_paged
from example_mla_decode_splitk import compile_mla_decode_splitk
from test_mla_decode_paged import make_paged, ref_paged

D, PE = 512, 64
PEAK_GBPS = 4000.0  # Ascend 950PR GM peak, same constant as bench/parse_pipes.py
DEFAULT_TIERS = "512,1024,2048,4096,8192,16384,32768,65536"


def wall_us(fn):
    return do_bench(fn, backend="event", _n_warmup=10, _n_repeat=50) * 1000.0


def moved_mb(batch, kv_len):
    return batch * kv_len * (2 * D + PE) * 2 / (1024.0 * 1024.0)


def cache_mb(batch, kv_len):
    return batch * kv_len * (D + PE) * 2 / (1024.0 * 1024.0)


def flops(batch, kv_len):
    return 2 * batch * 128 * kv_len * (D + PE) + 2 * batch * 128 * kv_len * D


def report(layout, batch, kv_len, out, exp, times):
    med = statistics.median(times)
    spread = max(times) - min(times)
    rel = (out.float() - exp).float().abs().max().item() / exp.float().abs().max().item()
    gbps = moved_mb(batch, kv_len) * 1024 * 1024 / (med * 1e-6) / 1e9
    print(
        f"SWEEP layout={layout} batch={batch} kv_len={kv_len} "
        f"cache_mb={cache_mb(batch, kv_len):.2f} moved_mb={moved_mb(batch, kv_len):.2f} "
        f"e2e_us={med:.2f} spread_us={spread:.2f} "
        f"tflops={flops(batch, kv_len) / med / 1e6:.2f} "
        f"gbps={gbps:.1f} util_pct={100.0 * gbps / PEAK_GBPS:.2f} "
        f"rel={rel:.5f} {'PASS' if rel < 1e-2 else 'FAIL'}",
        flush=True,
    )
    return med, gbps, rel


def run_dense(batch, kv_len, scale, rounds, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((batch, 128, D), generator=g).half().npu()
    q_pe = torch.randn((batch, 128, PE), generator=g).half().npu()
    kv = torch.randn((batch, kv_len, D), generator=g).half().npu()
    k_pe = torch.randn((batch, kv_len, PE), generator=g).half().npu()

    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    exp = torch.einsum("bhs,bsd->bhd", attn, kv.float()).to(torch.float16)

    run = compile_mla_decode_splitk(
        batch=batch, heads=128, kv_len=kv_len, dim=D, pe_dim=PE,
        softmax_scale=scale, num_split=16,
    )
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    times = [wall_us(lambda: run(q, q_pe, kv, k_pe)) for _ in range(rounds)]
    del kv, k_pe, key, scores, attn, query
    return report("dense", batch, kv_len, out, exp, times)


def run_paged(batch, kv_len, scale, rounds, block_size=64, seed=5):
    q, q_pe, kv, k_pe, table, seqlens, mp = make_paged(batch, [kv_len] * batch, block_size, seed=seed)
    exp = ref_paged(q, q_pe, kv, k_pe, table, seqlens, block_size, scale).float()
    run = compile_mla_decode_paged(
        batch=batch, heads=128, num_slots=kv.shape[0], dim=D, pe_dim=PE,
        block_size=block_size, max_pages=mp, block_N=64, num_split=16,
        softmax_scale=scale,
    )
    out = run(q, q_pe, kv, k_pe, table, seqlens)
    torch.npu.synchronize()
    times = [wall_us(lambda: run(q, q_pe, kv, k_pe, table, seqlens)) for _ in range(rounds)]
    return report("paged", batch, kv_len, out, exp, times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-lens", default=DEFAULT_TIERS,
                    help="comma separated kv_len tiers (data-volume axis)")
    ap.add_argument("--batches", default="",
                    help="optional comma separated batch tiers at a fixed kv_len")
    ap.add_argument("--batch-kv-len", type=int, default=8192)
    ap.add_argument("--layouts", default="dense,paged")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--block-size", type=int, default=64)
    args = ap.parse_args()

    scale = (D + PE) ** -0.5
    layouts = [x.strip() for x in args.layouts.split(",") if x.strip()]
    tiers = [int(x) for x in args.kv_lens.split(",") if x.strip()]

    print(f"SWEEP_CFG layouts={layouts} kv_lens={tiers} rounds={args.rounds} "
          f"peak_gbps={PEAK_GBPS} block_size={args.block_size}", flush=True)
    print(f"SWEEP_LOAD host_load1={open('/proc/loadavg').read().split()[0]}", flush=True)

    best = {}
    for kv_len in tiers:
        for layout in layouts:
            try:
                if layout == "dense":
                    med, gbps, _ = run_dense(1, kv_len, scale, args.rounds)
                else:
                    med, gbps, _ = run_paged(1, kv_len, scale, args.rounds, args.block_size)
            except Exception as exc:  # keep the rest of the table usable
                print(f"SWEEP layout={layout} kv_len={kv_len} FAILED "
                      f"{str(exc).splitlines()[0][:160]}", flush=True)
                continue
            if gbps > best.get(layout, (0, 0))[0]:
                best[layout] = (gbps, kv_len, med)

    if args.batches:
        for batch in [int(x) for x in args.batches.split(",") if x.strip()]:
            for layout in layouts:
                try:
                    if layout == "dense":
                        run_dense(batch, args.batch_kv_len, scale, args.rounds)
                    else:
                        run_paged(batch, args.batch_kv_len, scale, args.rounds, args.block_size)
                except Exception as exc:
                    print(f"SWEEP layout={layout} batch={batch} FAILED "
                          f"{str(exc).splitlines()[0][:160]}", flush=True)

    for layout, (gbps, kv_len, med) in best.items():
        print(f"SWEEP_BEST layout={layout} kv_len={kv_len} gbps={gbps:.1f} "
              f"util_pct={100.0 * gbps / PEAK_GBPS:.2f} e2e_us={med:.2f}", flush=True)
    print(f"SWEEP_LOAD_END host_load1={open('/proc/loadavg').read().split()[0]}", flush=True)


if __name__ == "__main__":
    main()

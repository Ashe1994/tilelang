"""batch=64, float16: MLA decode latency over the KV-length sweep.

For kv_len in {1024, 2048, 4096, 8192, 16384, 32768} and batch=64 this runs the
three kernels on the same KV data (identity page table, so the dense native view
and the paged view share one allocation):

  native      example_mla_decode.py        one block per (batch, head tile), no split-KV
  paged       example_mla_decode_paged.py  split-KV phase 1 + merge (default fp32 partials)
  persistent  mla_decode_persistent.py     same phase 1 under T.Persistent (one block per core)

Correctness is checked on batch element 0 against a float32 torch reference (the
other 63 elements reuse the same kernel; the paged suite covers them).
"""

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch
import tilelang
from tilelang.profiler import do_bench

try:  # a CUDA-only build has no torch_npu
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode import compile_mla_decode
from example_mla_decode_paged import compile_mla_decode_paged
from mla_decode_persistent import compile_mla_decode_persistent

D, PE, H = 512, 64, 128
PEAK_TFLOPS, PEAK_GBPS = 432.0, 4000.0
SEQ_LENS = (1024, 2048, 4096, 8192, 16384, 32768)


def wall(fn, rounds=2):
    return statistics.median(
        [do_bench(fn, backend="event", _n_warmup=10, _n_repeat=50) * 1000.0 for _ in range(rounds)]
    )


def build(batch, kv_len, block_size, seed=3):
    """Identity page table over one shared KV allocation (dense view + paged view)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_pages = kv_len // block_size
    kv_all = torch.randn((batch * kv_len, D), generator=g).half().npu()
    kpe_all = torch.randn((batch * kv_len, PE), generator=g).half().npu()
    q = torch.randn((batch, H, D), generator=g).half().npu()
    q_pe = torch.randn((batch, H, PE), generator=g).half().npu()
    table = (
        torch.arange(batch * max_pages, dtype=torch.int32).view(batch, max_pages).contiguous().npu()
    )
    seqlens = torch.tensor([kv_len] * batch, dtype=torch.int32).npu()
    dense_kv = kv_all.view(batch, kv_len, D)
    dense_kpe = kpe_all.view(batch, kv_len, PE)
    return q, q_pe, kv_all, kpe_all, dense_kv, dense_kpe, table, seqlens, max_pages


def reference_b0(q, q_pe, kv_all, kpe_all, kv_len, scale):
    q0 = torch.cat([q[0].float(), q_pe[0].float()], dim=-1)
    key = torch.cat([kv_all[0:kv_len].float(), kpe_all[0:kv_len].float()], dim=-1)
    scores = (q0 @ key.T) * scale
    probs = torch.softmax(scores, dim=-1)
    return (probs @ kv_all[0:kv_len].float()).half()


def rel(out, ref):
    return (out.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", default=",".join(str(v) for v in SEQ_LENS))
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--partial-dtype", default="float32", choices=("float32", "float16"))
    ap.add_argument("--num-split", type=int, default=16)
    args = ap.parse_args()
    lens = [int(v) for v in args.lens.split(",")]
    scale = (D + PE) ** -0.5
    batch = args.batch

    for kv_len in lens:
        q, q_pe, kv_all, kpe_all, dense_kv, dense_kpe, table, seqlens, max_pages = build(
            batch, kv_len, 64
        )
        tokens = batch * kv_len
        flops = 2 * H * tokens * ((D + PE) + D)
        moved = tokens * (2 * D + PE) * 2  # KV read (K and V^T) in bytes
        ref0 = reference_b0(q, q_pe, kv_all, kpe_all, kv_len, scale)

        nat = compile_mla_decode(
            batch=batch, heads=H, kv_len=kv_len, dim=D, pe_dim=PE, softmax_scale=scale
        )
        pg = compile_mla_decode_paged(
            batch=batch, heads=H, num_slots=kv_all.shape[0], dim=D, pe_dim=PE,
            block_size=64, max_pages=max_pages, block_N=64, num_split=args.num_split,
            softmax_scale=scale, partial_dtype=args.partial_dtype,
        )
        ps = compile_mla_decode_persistent(
            batch=batch, heads=H, num_slots=kv_all.shape[0], dim=D, pe_dim=PE,
            block_size=64, max_pages=max_pages, block_N=64, num_split=args.num_split,
            softmax_scale=scale, num_cores=16,
        )

        rows = []
        for tag, run, call in (
            ("native", nat, lambda: nat(q, q_pe, dense_kv, dense_kpe)),
            ("paged", pg, lambda: pg(q, q_pe, kv_all, kpe_all, table, seqlens)),
            ("persist", ps, lambda: ps(q, q_pe, kv_all, kpe_all, table, seqlens)),
        ):
            out = call()
            torch.npu.synchronize()
            r0 = rel(out[0], ref0)
            us = wall(call, args.rounds)
            rows.append((tag, us, r0))

        line = [f"B64 kv={kv_len} batch={batch} tokens={tokens}"]
        for tag, us, r0 in rows:
            line.append(
                f"{tag}={us / 1000.0:.2f}ms({flops / us / 1e3:.1f}T,"
                f"{moved / (us * 1e-6) / 1e9:.0f}GB/s,rel0={r0:.5f})"
            )
        line.append(f"pg_vs_nat={rows[0][1] / rows[1][1]:.1f}x")
        line.append(f"ps_vs_pg={rows[1][1] / rows[2][1]:.3f}x")
        print(" ".join(line), flush=True)
        del q, q_pe, kv_all, kpe_all, dense_kv, dense_kpe, table, seqlens, ref0, nat, pg, ps
        torch.npu.empty_cache()


if __name__ == "__main__":
    main()

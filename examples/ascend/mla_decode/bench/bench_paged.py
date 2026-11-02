"""Paged MLA decode: correctness guard + wall-clock benchmark vs the native decode kernel.

The native kernel (``example_mla_decode.py``, one launch, one KV pass per core) is
the reference implementation; the paged kernel is the optimized split-KV path.
"""

import math
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

from example_mla_decode import compile_mla_decode
from example_mla_decode_paged import compile_mla_decode_paged
from test_mla_decode_paged import make_paged, ref_paged

D, PE = 512, 64
CASES = [
    # name, batch, seq_len, block_size, block_N, num_split
    ("pg64_ns16_bn64", 1, 8192, 64, 64, 16),   # shipped default
    ("pg128_ns16_bn64", 1, 8192, 128, 64, 16),
    ("pg64_ns16_bn128", 1, 8192, 64, 128, 16),  # previous default (A/B)
    ("pg64_ns32_bn64", 1, 8192, 64, 64, 32),
]


def wall(fn):
    return do_bench(fn, backend="event", _n_warmup=10, _n_repeat=50) * 1000.0


def main():
    scale = (D + PE) ** -0.5
    flops = 2 * 1 * 128 * 8192 * (D + PE) + 2 * 1 * 128 * 8192 * D

    dense = compile_mla_decode(batch=1, heads=128, kv_len=8192, dim=D, pe_dim=PE,
                               softmax_scale=scale)
    g = torch.Generator(device="cpu").manual_seed(0)
    qd = torch.randn((1, 128, D), generator=g).half().npu()
    qpd = torch.randn((1, 128, PE), generator=g).half().npu()
    kvd = torch.randn((1, 8192, D), generator=g).half().npu()
    kpd = torch.randn((1, 8192, PE), generator=g).half().npu()
    dense(qd, qpd, kvd, kpd)
    torch.npu.synchronize()
    d_us = statistics.median([wall(lambda: dense(qd, qpd, kvd, kpd)) for _ in range(3)])
    print(f"NATIVEDENSE native_us={d_us:.1f} tflops={flops / d_us / 1e6:.2f}", flush=True)

    for name, batch, seq_len, bs, bn, ns in CASES:
        q, qp, kv, kp, table, seqlens, mp = make_paged(batch, [seq_len], bs, seed=5)
        run = compile_mla_decode_paged(
            batch=batch, heads=128, num_slots=kv.shape[0], dim=D, pe_dim=PE,
            block_size=bs, max_pages=mp, block_N=bn, num_split=ns, softmax_scale=scale,
        )
        out = run(q, qp, kv, kp, table, seqlens)
        torch.npu.synchronize()
        exp = ref_paged(q, qp, kv, kp, table, seqlens, bs, scale).float()
        rel = (out.float() - exp).abs().max().item() / exp.abs().max().item()
        us = statistics.median([wall(lambda: run(q, qp, kv, kp, table, seqlens)) for _ in range(3)])
        print(
            f"PG {name} bs={bs} bn={bn} ns={ns} e2e_us={us:.1f} "
            f"tflops={flops / us / 1e6:.2f} rel={rel:.5f} "
            f"{'PASS' if rel < 1e-2 else 'FAIL'} speedup_vs_native={d_us / us:.2f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Multi-batch / ragged-batch benchmark: native decode vs paged vs persistent.

For every case the script reports wall clock, achieved TFLOPS and the KV-bandwidth
utilisation for all kernels that can run the shape (the dense native kernel needs
a uniform sequence length, so it is skipped for ragged batches).
"""

import argparse
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

PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode import compile_mla_decode
from example_mla_decode_paged import compile_mla_decode_paged
from mla_decode_persistent import compile_mla_decode_persistent
from test_mla_decode_paged import make_paged, ref_paged

D, PE, H = 512, 64, 128
PEAK_TFLOPS, PEAK_GBPS = 432.0, 4000.0

CASES = [
    ("u_b1", [8192]),
    ("u_b2", [8192, 8192]),
    ("u_b4", [8192] * 4),
    ("u_b8", [8192] * 8),
    ("r_b2", [8192, 1024]),
    ("r_b4", [8192, 2048, 512, 4096]),
]


def wall(fn):
    return do_bench(fn, backend="event", _n_warmup=10, _n_repeat=50) * 1000.0


def med(fn, rounds=3):
    return statistics.median([wall(fn) for _ in range(rounds)])


def traffic(tokens, batch, ns=16):
    """KV read + fp32 partial write/read + ML traffic, in bytes."""
    kv = tokens * (2 * D + PE) * 2
    partial = ns * batch * H * D * 4 * 2 + ns * batch * H * 8 * 4 * 2
    return kv + partial


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(name for name, _ in CASES))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--skip-native", action="store_true")
    ap.add_argument("--persistent-stages", type=int, default=0)
    args = ap.parse_args()
    wanted = set(args.cases.split(","))
    scale = (D + PE) ** -0.5

    native_cache = {}

    def native_for(batch, kv_len):
        key = (batch, kv_len)
        if key not in native_cache:
            native_cache[key] = compile_mla_decode(
                batch=batch, heads=H, kv_len=kv_len, dim=D, pe_dim=PE, softmax_scale=scale
            )
        g = torch.Generator(device="cpu").manual_seed(11)
        q = torch.randn((batch, H, D), generator=g).half().npu()
        qp = torch.randn((batch, H, PE), generator=g).half().npu()
        kv = torch.randn((batch, kv_len, D), generator=g).half().npu()
        kp = torch.randn((batch, kv_len, PE), generator=g).half().npu()
        return native_cache[key], q, qp, kv, kp

    for name, lens in CASES:
        if name not in wanted:
            continue
        batch = len(lens)
        tokens = sum(lens)
        flops = 2 * H * tokens * ((D + PE) + D)
        moved = traffic(tokens, batch)

        q, qp, kv, kp, table, seqlens, mp = make_paged(batch, lens, 64, seed=5)
        exp = ref_paged(q, qp, kv, kp, table, seqlens, 64, scale).float()

        run_paged = compile_mla_decode_paged(
            batch=batch, heads=H, num_slots=kv.shape[0], dim=D, pe_dim=PE,
            block_size=64, max_pages=mp, block_N=64, num_split=16, softmax_scale=scale,
        )
        run_pers = compile_mla_decode_persistent(
            batch=batch, heads=H, num_slots=kv.shape[0], dim=D, pe_dim=PE,
            block_size=64, max_pages=mp, block_N=64, num_split=16, softmax_scale=scale,
            num_cores=16, persistent_stages=args.persistent_stages,
        )

        rows = []
        for tag, run in (("paged", run_paged), ("persist", run_pers)):
            out = run(q, qp, kv, kp, table, seqlens)
            torch.npu.synchronize()
            rel = (out.float() - exp).abs().max().item() / exp.abs().max().item()
            us = med(lambda: run(q, qp, kv, kp, table, seqlens), args.rounds)
            rows.append((tag, us, rel))

        native_us = None
        if not args.skip_native and len(set(lens)) == 1:
            kern, qd, qpd, kvd, kpd = native_for(batch, lens[0])
            kern(qd, qpd, kvd, kpd)
            torch.npu.synchronize()
            native_us = med(lambda: kern(qd, qpd, kvd, kpd), args.rounds)

        line = [f"MULTI {name} B={batch} lens={lens} tokens={tokens}"]
        if native_us:
            line.append(f"native={native_us:.1f}us({flops / native_us / 1e6:.1f}T)")
        base = None
        for tag, us, rel in rows:
            base = base or us
            line.append(
                f"{tag}={us:.1f}us({flops / us / 1e6:.1f}T,"
                f"{moved / (us * 1e-6) / 1e9:.0f}GB/s,"
                f"{100 * moved / (us * 1e-6) / 1e9 / PEAK_GBPS:.1f}%bw,"
                f"{100 * flops / us / 1e6 / PEAK_TFLOPS:.1f}%tflops,"
                f"rel={rel:.5f})"
            )
        if len(rows) == 2:
            line.append(f"persist_vs_paged={rows[0][1] / rows[1][1]:.3f}x")
        print(" ".join(line), flush=True)


if __name__ == "__main__":
    main()

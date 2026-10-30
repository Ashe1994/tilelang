"""One-tier profiling entry for the data-volume sweep (dense or paged phase 1).

    msprof op --warm-up=10 --launch-count=5 --aic-metrics=... \
        python prof_sweep.py --layout dense --kv-len 32768
    msprof op --warm-up=10 --launch-count=5 --aic-metrics=... \
        python prof_sweep.py --layout paged --kv-len 1024 --batch 1

Only phase 1 is compiled so the capture contains a single kernel; feed the
output directory to ``bench/parse_pipes.py`` for the pipe / bandwidth table.
"""

import argparse
import sys
from pathlib import Path

import torch
import tilelang

try:  # a CUDA-only build has no torch_npu; profiling itself needs the NPU
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

# Importable from any working directory: kernels and tests live one level up.
PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode_paged import mla_decode_paged_fwd
from example_mla_decode_splitk import mla_decode_splitk_fwd
from test_mla_decode_paged import make_paged

D, PE = 512, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="dense", choices=["dense", "paged"])
    ap.add_argument("--kv-len", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--num-split", type=int, default=16)
    args = ap.parse_args()
    scale = (D + PE) ** -0.5

    if args.layout == "dense":
        g = torch.Generator(device="cpu").manual_seed(0)
        q = torch.randn((args.batch, 128, D), generator=g).half().npu()
        q_pe = torch.randn((args.batch, 128, PE), generator=g).half().npu()
        kv = torch.randn((args.batch, args.kv_len, D), generator=g).half().npu()
        k_pe = torch.randn((args.batch, args.kv_len, PE), generator=g).half().npu()
        kernel = tilelang.compile(
            mla_decode_splitk_fwd(
                batch=args.batch, heads=128, kv_len=args.kv_len, dim=D, pe_dim=PE,
                softmax_scale=scale, num_split=args.num_split, block_N=64, chunk_d=128,
            ),
            out_idx=[4, 5],
        )
        op, ml = kernel(q, q_pe, kv, k_pe)
        torch.npu.synchronize()
        print(f"PROF_SWEEP layout=dense batch={args.batch} kv_len={args.kv_len} "
              f"o_partial_absmax={op.abs().max().item():.3f}", flush=True)
    else:
        q, q_pe, kv, k_pe, table, seqlens, mp = make_paged(
            args.batch, [args.kv_len], args.block_size, seed=5
        )
        kernel = tilelang.compile(
            mla_decode_paged_fwd(
                batch=args.batch, heads=128, num_slots=kv.shape[0], dim=D, pe_dim=PE,
                block_size=args.block_size, max_pages=mp, block_N=64, chunk_d=128,
                softmax_scale=scale, num_split=args.num_split,
            ),
            out_idx=[6, 7],
        )
        op, ml = kernel(q, q_pe, kv, k_pe, table, seqlens)
        torch.npu.synchronize()
        print(f"PROF_SWEEP layout=paged batch={args.batch} kv_len={args.kv_len} "
              f"o_partial_absmax={op.abs().max().item():.3f}", flush=True)


if __name__ == "__main__":
    main()

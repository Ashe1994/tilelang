"""Profiling entry: run ONE kernel so msprof can attribute it cleanly.

    msprof op --warm-up=10 --launch-count=5 --output=DIR python prof_phase1.py --which phase1
    msprof op --warm-up=10 --launch-count=1 --replay-mode=kernel \
        --aic-metrics=PipeTimeline --output=DIR python prof_phase1.py --which phase1

--which phase1 builds only the split kernel, so the merge never runs and the
capture contains a single kernel name.
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

from example_mla_decode_splitk import compile_mla_decode_splitk, mla_decode_splitk_fwd

B, H, S, D, PE = 1, 128, 8192, 512, 64
NS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="phase1", choices=["phase1", "e2e"])
    ap.add_argument("--block-n", type=int, default=128)
    ap.add_argument("--chunk-d", type=int, default=64)
    args = ap.parse_args()

    scale = (D + PE) ** -0.5
    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn((B, H, D), generator=g).half().npu()
    q_pe = torch.randn((B, H, PE), generator=g).half().npu()
    kv = torch.randn((B, S, D), generator=g).half().npu()
    k_pe = torch.randn((B, S, PE), generator=g).half().npu()

    if args.which == "phase1":
        kernel = tilelang.compile(
            mla_decode_splitk_fwd(
                batch=B, heads=H, kv_len=S, dim=D, pe_dim=PE, softmax_scale=scale,
                num_split=NS, block_N=args.block_n, chunk_d=args.chunk_d,
            ),
            out_idx=[4, 5],
        )
        op, ml = kernel(q, q_pe, kv, k_pe)
        torch.npu.synchronize()
        print(f"PROF phase1 block_N={args.block_n} chunk_d={args.chunk_d} "
              f"o_partial_absmax={op.abs().max().item():.3f}")
    else:
        run = compile_mla_decode_splitk(
            batch=B, heads=H, kv_len=S, dim=D, pe_dim=PE, softmax_scale=scale, num_split=NS
        )
        out = run(q, q_pe, kv, k_pe)
        torch.npu.synchronize()
        print(f"PROF e2e out_absmax={out.abs().max().item():.3f}")


if __name__ == "__main__":
    main()

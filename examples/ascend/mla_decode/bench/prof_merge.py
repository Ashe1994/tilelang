"""Profiling entry for the shared merge kernel (phase 2 of MLA decode)."""

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

from example_mla_decode_splitk import mla_decode_merge

B, H, D, PE = 1, 128, 512, 64


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-split", type=int, default=16)
    ap.add_argument("--ml-width", type=int, default=8)
    ap.add_argument("--partial-dtype", default="float16", choices=["float16", "float32"])
    args = ap.parse_args()

    ns = args.num_split
    kernel = tilelang.compile(
        mla_decode_merge(
            batch=B, heads=H, kv_len=ns * 64, dim=D, pe_dim=PE, block_H=128,
            block_N=64, chunk_d=128, num_split=ns, ns_explicit=ns,
            rows_per_block=16, num_stages=4, ml_width=args.ml_width,
            partial_dtype=args.partial_dtype,
        ),
        out_idx=[2],
    )
    g = torch.Generator(device="cpu").manual_seed(0)
    op = torch.randn((ns, B, H, D), generator=g).to(
        torch.float16 if args.partial_dtype == "float16" else torch.float32
    ).npu()
    ml = torch.randn((ns, B, H, args.ml_width), generator=g).npu()
    out = kernel(op, ml)
    torch.npu.synchronize()
    print(f"PROF merge ns={ns} ml_width={args.ml_width} "
          f"partial_dtype={args.partial_dtype} out_absmax={out.abs().max().item():.3f}")


if __name__ == "__main__":
    main()

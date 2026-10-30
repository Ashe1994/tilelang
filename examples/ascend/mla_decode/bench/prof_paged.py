"""Profiling entry: run ONE paged phase-1 kernel so msprof can attribute it.

    msprof op --warm-up=10 --launch-count=5 --output=DIR \
        python prof_paged.py --block-size 64
"""

import argparse
import math
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

B, H, S, D, PE = 1, 128, 8192, 512, 64
NS = 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--block-n", type=int, default=64)
    ap.add_argument("--num-split", type=int, default=NS)
    args = ap.parse_args()

    bs = args.block_size
    max_pages = math.ceil(S / bs)
    spare = 8
    num_pages = B * max_pages + spare
    num_slots = num_pages * bs
    scale = (D + PE) ** -0.5

    g = torch.Generator(device="cpu").manual_seed(0)
    q = torch.randn((B, H, D), generator=g).half().npu()
    q_pe = torch.randn((B, H, PE), generator=g).half().npu()
    kv = torch.randn((num_slots, D), generator=g).half().npu()
    k_pe = torch.randn((num_slots, PE), generator=g).half().npu()
    perm = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = perm[: B * max_pages].view(B, max_pages).contiguous().npu()
    seqlens = torch.tensor([S] * B, dtype=torch.int32).npu()

    kernel = tilelang.compile(
        mla_decode_paged_fwd(
            batch=B, heads=H, num_slots=num_slots, dim=D, pe_dim=PE,
            block_size=bs, max_pages=max_pages, block_N=args.block_n,
            num_split=args.num_split, softmax_scale=scale,
        ),
        out_idx=[6, 7],
    )
    op, ml = kernel(q, q_pe, kv, k_pe, table, seqlens)
    torch.npu.synchronize()
    print(f"PROF paged block_size={bs} block_N={args.block_n} ns={args.num_split} "
          f"o_partial_absmax={op.abs().max().item():.3f}")


if __name__ == "__main__":
    main()

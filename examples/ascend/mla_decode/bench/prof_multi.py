"""Profiling entry for one kernel of a chosen variant, so msprof can attribute it.

    msprof op --warm-up=10 --launch-count=5 --output=DIR \
        python bench/prof_multi.py --variant persistent --batch 4

Variants:
  native      one launch of the dense native decode kernel
  paged       phase 1 only (the split-KV kernel), batch/max_pages from --lens
  persistent  phase 1 only of the persistent variant
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

PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode import compile_mla_decode
from example_mla_decode_paged import pick_num_split, select_phase1_kernel
from mla_decode_persistent import mla_decode_persistent_fwd
from test_mla_decode_paged import make_paged

D, PE, H = 512, 64, 128
BS, BN, NS = 64, 64, 16


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=("native", "paged", "persistent"), required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lens", default=None, help="comma separated lengths, default uniform 8192")
    ap.add_argument("--cores", type=int, default=16)
    ap.add_argument("--block-n", type=int, default=BN)
    ap.add_argument("--num-split", type=int, default=0, help="0 = auto from kv_len and batch")
    args = ap.parse_args()
    scale = (D + PE) ** -0.5

    if args.variant == "native":
        kv_len = 8192
        kernel = compile_mla_decode(
            batch=args.batch, heads=H, kv_len=kv_len, dim=D, pe_dim=PE, softmax_scale=scale
        )
        g = torch.Generator(device="cpu").manual_seed(0)
        q = torch.randn((args.batch, H, D), generator=g).half().npu()
        qp = torch.randn((args.batch, H, PE), generator=g).half().npu()
        kv = torch.randn((args.batch, kv_len, D), generator=g).half().npu()
        kp = torch.randn((args.batch, kv_len, PE), generator=g).half().npu()
        kernel(q, qp, kv, kp)
        torch.npu.synchronize()
        print(f"PROF native batch={args.batch} kv_len={kv_len}")
        return

    lens = (
        [int(v) for v in args.lens.split(",")]
        if args.lens
        else [8192] * args.batch
    )
    if len(lens) == 1:
        lens = lens * args.batch
    assert len(lens) == args.batch
    ns = args.num_split or pick_num_split(max(lens), args.block_n, args.batch)
    q, qp, kv, kp, table, seqlens, mp = make_paged(args.batch, lens, BS, seed=5)
    if args.variant == "paged":
        fwd, dual = select_phase1_kernel(
            args.block_n, BS, 64 if args.block_n >= 128 else 128,
            kv_hint=mp * BS, num_split=ns,
        )
    else:
        fwd, dual = mla_decode_persistent_fwd, True
    kwargs = dict(
        batch=args.batch, heads=H, num_slots=kv.shape[0], dim=D, pe_dim=PE,
        block_size=BS, max_pages=mp, block_N=args.block_n, num_split=ns, softmax_scale=scale,
    )
    if args.variant == "persistent":
        kwargs.update(num_cores=args.cores, persistent_stages=0)
    kernel = tilelang.compile(fwd(**kwargs), out_idx=[6, 7])
    op, ml = kernel(q, qp, kv, kp, table, seqlens)
    torch.npu.synchronize()
    print(f"PROF {args.variant} batch={args.batch} lens={lens} dual={dual} "
          f"o_partial_absmax={op.abs().max().item():.4f}")


if __name__ == "__main__":
    main()

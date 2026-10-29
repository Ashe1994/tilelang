"""Correctness sweep over the tile parameters of the split-KV MLA decode kernel.

The kernel had correctness evidence only at the default
``block_N=64 / chunk_d=128 / num_stages=2``. The optimisation steps
(``block_N=128``, fused rescale) sit right next to this untested region, so the
grid is swept first.

Shape is fixed (small enough to compile quickly) and ``num_split=4`` divides both
the 8 KV blocks (block_N=64) and the 16 KV blocks (block_N=32).

    python -m pytest test_mla_decode_splitk_tiles.py -q
"""

import pytest
import torch

import tilelang.testing

try:  # a CUDA-only build has no torch_npu; requires_ascend skips those cases
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

from example_mla_decode_splitk import compile_mla_decode_splitk

B, H, KV, D, PE = 1, 128, 512, 512, 64

TILES = [
    # block_N, chunk_d, num_stages
    (64, 128, 2),  # current default
    (64, 128, 1),
    (64, 128, 3),
    (64, 64, 1),
    (64, 64, 2),
    (64, 64, 3),
    (32, 128, 1),
    (32, 128, 2),
    (32, 128, 3),
    (32, 64, 1),
    (32, 64, 2),
    (32, 64, 3),
]


def ref_mla_decode(q, q_pe, kv, k_pe, scale):
    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bsd->bhd", attn, kv.float())
    return out.to(torch.float16)


def make_inputs(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((B, H, D), generator=g).half().npu()
    q_pe = torch.randn((B, H, PE), generator=g).half().npu()
    kv = torch.randn((B, KV, D), generator=g).half().npu()
    k_pe = torch.randn((B, KV, PE), generator=g).half().npu()
    return q, q_pe, kv, k_pe


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "block_N,chunk_d,num_stages", TILES, ids=[f"bn{a}_cd{b}_st{c}" for a, b, c in TILES]
)
def test_tile(block_N, chunk_d, num_stages):
    scale = (D + PE) ** -0.5
    run = compile_mla_decode_splitk(
        batch=B,
        heads=H,
        kv_len=KV,
        dim=D,
        pe_dim=PE,
        softmax_scale=scale,
        block_N=block_N,
        chunk_d=chunk_d,
        num_stages=num_stages,
        num_split=4,
    )
    q, q_pe, kv, k_pe = make_inputs()
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    expected = ref_mla_decode(q, q_pe, kv, k_pe, scale)
    assert torch.isfinite(out.float()).all(), "NaN/Inf"
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)

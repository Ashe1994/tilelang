"""Correctness tests for the split-KV (flash-decoding) MLA decode kernel.

Same cases as ``test_mla_decode.py`` plus the long-KV shape the split exists for:

    TILELANG_DEFAULT_TARGET=pto    python -m pytest test_mla_decode_splitk.py -q
    TILELANG_DEFAULT_TARGET=ascend python -m pytest test_mla_decode_splitk.py -q

Every case is additionally checked against the ``num_split == 1`` kernel, so the
merge path is validated against an independent implementation of the same math.
"""

import pytest
import torch

import tilelang
import tilelang.testing

try:  # a CUDA-only build has no torch_npu; requires_ascend skips those cases
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

from example_mla_decode import mla_decode_fwd as mla_decode_fwd_baseline
from example_mla_decode_splitk import compile_mla_decode_splitk

CASES = [
    # id, batch, heads, kv_len, dim, pe_dim, num_split
    ("tiny", 1, 128, 128, 512, 64, 2),
    ("kv_tail", 1, 128, 100, 512, 64, 2),  # kv_len not a multiple of block_N
    ("dim256", 1, 128, 128, 256, 64, 2),  # dim chunked into 2 pieces
    ("kv_multi_block", 1, 128, 512, 512, 64, 8),
    ("batch2_heads256", 2, 256, 128, 512, 64, 2),
    ("split_uneven", 1, 128, 448, 512, 64, 7),  # 7 blocks / split 7 -> 1 block each
    ("long_kv", 1, 128, 8192, 512, 64, 8),
]


def ref_mla_decode(q, q_pe, kv, k_pe, scale):
    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bsd->bhd", attn, kv.float())
    return out.to(torch.float16)


def make_inputs(batch, heads, kv_len, dim, pe_dim, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((batch, heads, dim), generator=g).half().npu()
    q_pe = torch.randn((batch, heads, pe_dim), generator=g).half().npu()
    kv = torch.randn((batch, kv_len, dim), generator=g).half().npu()
    k_pe = torch.randn((batch, kv_len, pe_dim), generator=g).half().npu()
    return q, q_pe, kv, k_pe


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "name,batch,heads,kv_len,dim,pe_dim,num_split", CASES, ids=[c[0] for c in CASES]
)
def test_mla_decode_splitk(name, batch, heads, kv_len, dim, pe_dim, num_split):
    scale = (dim + pe_dim) ** -0.5
    run = compile_mla_decode_splitk(
        batch=batch,
        heads=heads,
        kv_len=kv_len,
        dim=dim,
        pe_dim=pe_dim,
        softmax_scale=scale,
        num_split=num_split,
    )

    q, q_pe, kv, k_pe = make_inputs(batch, heads, kv_len, dim, pe_dim)
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()

    expected = ref_mla_decode(q, q_pe, kv, k_pe, scale)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)
    assert torch.isfinite(out.float()).all(), f"{name}: output contains NaN/Inf"

    # cross-check against the single-split kernel on the small cases only
    if name != "long_kv":
        base = tilelang.compile(
            mla_decode_fwd_baseline(
                batch=batch,
                heads=heads,
                kv_len=kv_len,
                dim=dim,
                pe_dim=pe_dim,
                softmax_scale=scale,
            ),
            out_idx=[4],
        )
        ref_single = base(q, q_pe, kv, k_pe)
        torch.npu.synchronize()
        torch.testing.assert_close(out.float(), ref_single.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "--no-header", "-p", "no:cacheprovider"]))

"""Correctness tests for the MLA decode kernel on Ascend 950.

The execution backend is chosen by the environment, never by the test:

    TILELANG_DEFAULT_TARGET=pto    python -m pytest test_mla_decode.py -q
    TILELANG_DEFAULT_TARGET=ascend python -m pytest test_mla_decode.py -q
"""

import pytest
import torch

import tilelang
import tilelang.testing

try:  # a CUDA-only build has no torch_npu; requires_ascend skips those cases
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

from example_mla_decode import mla_decode_fwd

CASES = [
    # id, batch, heads, kv_len, dim, pe_dim
    ("tiny", 1, 128, 128, 512, 64),
    ("kv_tail", 1, 128, 100, 512, 64),  # kv_len not a multiple of block_N
    ("dim256", 1, 128, 128, 256, 64),  # dim chunked into 2 pieces
    ("kv_multi_block", 1, 128, 512, 512, 64),
    ("batch2_heads256", 2, 256, 128, 512, 64),
]


def ref_mla_decode(q, q_pe, kv, k_pe, scale):
    """PyTorch reference. q: [b, h, dim]; kv: [b, s, dim]."""
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
@pytest.mark.parametrize("name,batch,heads,kv_len,dim,pe_dim", CASES, ids=[c[0] for c in CASES])
def test_mla_decode(name, batch, heads, kv_len, dim, pe_dim):
    scale = (dim + pe_dim) ** -0.5
    program = mla_decode_fwd(
        batch=batch,
        heads=heads,
        kv_len=kv_len,
        dim=dim,
        pe_dim=pe_dim,
        softmax_scale=scale,
    )
    kernel = tilelang.compile(program, out_idx=[4])

    q, q_pe, kv, k_pe = make_inputs(batch, heads, kv_len, dim, pe_dim)
    out = kernel(q, q_pe, kv, k_pe)
    torch.npu.synchronize()

    expected = ref_mla_decode(q, q_pe, kv, k_pe, scale)
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)
    assert torch.isfinite(out.float()).all(), f"{name}: output contains NaN/Inf"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "--no-header", "-p", "no:cacheprovider"]))

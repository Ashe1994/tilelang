"""Dispatch, defaulting and cross-layout equivalence tests for ``mla_decode_api``.

The point of the unified entry point is that a caller only chooses a *layout*
(dense KV or paged KV) and the module picks the measured-best tiling.  These
tests pin the policy down (on CPU, no device needed) and check that the two
layouts agree numerically on the same data.
"""

import pytest
import torch

import tilelang
import tilelang.testing

try:  # a CUDA-only build has no torch_npu; ``requires_ascend`` skips those cases
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

from mla_decode_api import compile_mla_decode, resolve_config

D, PE = 512, 64
SCALE = (D + PE) ** -0.5


# --------------------------------------------------------------------------
# policy: no device involved
# --------------------------------------------------------------------------
def test_resolve_dense_defaults():
    """Long KV picks the 128-wide tile and 16 splits; short KV falls back."""
    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, kv_len=8192)
    assert cfg["paged"] is False
    assert cfg["block_N"] == 64
    assert cfg["chunk_d"] == 128
    assert cfg["num_split"] == 16
    assert cfg["merge_chunk_d"] == 128
    assert cfg["vector_mode"] == "simd"
    assert cfg["ml_width"] == 8
    assert cfg["partial_dtype"] == "float16"  # dense uses fp16 partials

    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, kv_len=64)
    assert cfg["block_N"] == 64
    assert cfg["chunk_d"] == 128
    assert cfg["num_split"] == 1

    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, kv_len=128)
    assert cfg["block_N"] == 64
    assert cfg["num_split"] == 2


def test_resolve_paged_defaults():
    """Paged geometry alone selects the paged kernel; splits shrink with the cache."""
    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, num_slots=8192, block_size=64, max_pages=128)
    assert cfg["paged"] is True
    assert cfg["block_N"] == 64
    assert cfg["chunk_d"] == 128
    assert cfg["num_split"] == 16
    assert cfg["vector_mode"] == "simd_rc"
    assert cfg["partial_dtype"] == "float32"  # paged fp16 publish measured slower

    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, num_slots=256, block_size=64, max_pages=4)
    assert cfg["paged"] is True
    assert cfg["num_split"] == 4  # 4 tiles of 64 tokens -> 4 splits


def test_resolve_rejects_mixed_geometry():
    with pytest.raises(ValueError, match="needs num_slots"):
        resolve_config(1, 128, dim=D, pe_dim=PE, block_size=64, paged=True)
    with pytest.raises(ValueError, match="do not mix"):
        resolve_config(1, 128, dim=D, pe_dim=PE, kv_len=512, num_slots=512, block_size=64)
    with pytest.raises(ValueError, match="needs kv_len"):
        resolve_config(1, 128, dim=D, pe_dim=PE)


def test_explicit_split_is_honoured():
    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, kv_len=8192, num_split=8)
    assert cfg["num_split"] == 8
    cfg = resolve_config(1, 128, dim=D, pe_dim=PE, kv_len=8192, num_split=32)
    assert cfg["num_split"] == 32


# --------------------------------------------------------------------------
# numerics: device
# --------------------------------------------------------------------------
def ref_mla_decode(q, q_pe, kv, k_pe, scale):
    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bsd->bhd", attn, kv.float()).to(torch.float16)


def make_dense(batch, heads, kv_len, dim, pe_dim, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((batch, heads, dim), generator=g).half().npu()
    q_pe = torch.randn((batch, heads, pe_dim), generator=g).half().npu()
    kv = torch.randn((batch, kv_len, dim), generator=g).half().npu()
    k_pe = torch.randn((batch, kv_len, pe_dim), generator=g).half().npu()
    return q, q_pe, kv, k_pe


@tilelang.testing.requires_ascend
def test_dense_via_api_matches_reference():
    batch, heads, kv_len = 1, 128, 512
    run = compile_mla_decode(
        batch, heads, D, PE, kv_len=kv_len, softmax_scale=SCALE
    )
    assert run.paged is False
    assert run.vector_mode == "simd"
    assert run.partial_dtype == "float16"
    assert run.ml_width == 8
    assert run.merge_vector_mode == "simd"
    q, q_pe, kv, k_pe = make_dense(batch, heads, kv_len, D, PE)
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    expected = ref_mla_decode(q, q_pe, kv, k_pe, SCALE)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_ascend
def test_paged_identity_table_matches_dense():
    """Identity page map == dense layout, so both entry points must agree."""
    batch, heads, kv_len = 1, 128, 512
    block_size = 128
    max_pages = kv_len // block_size
    q, q_pe, kv, k_pe = make_dense(batch, heads, kv_len, D, PE)
    block_table = torch.arange(max_pages, dtype=torch.int32).view(1, max_pages).npu()
    seqlens = torch.tensor([kv_len], dtype=torch.int32).npu()

    dense = compile_mla_decode(batch, heads, D, PE, kv_len=kv_len, softmax_scale=SCALE)
    paged = compile_mla_decode(
        batch,
        heads,
        D,
        PE,
        num_slots=kv_len,
        block_size=block_size,
        max_pages=max_pages,
        softmax_scale=SCALE,
    )
    assert paged.paged is True
    assert paged.vector_mode == "simd_rc"
    assert paged.block_N == 64
    out_dense = dense(q, q_pe, kv, k_pe)
    out_paged = paged(q, q_pe, kv.reshape(kv_len, D), k_pe.reshape(kv_len, PE), block_table, seqlens)
    torch.npu.synchronize()
    assert torch.isfinite(out_paged.float()).all()
    torch.testing.assert_close(out_paged.float(), out_dense.float(), rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q", "--no-header", "-p", "no:cacheprovider"]))

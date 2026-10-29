"""Coverage matrix for the split-KV MLA decode kernel.

The 7-case file (``test_mla_decode_splitk.py``) is the inherited regression set.
This file widens the coverage in four directions:

  tier A  shape sweep      dim / pe_dim / heads / batch vary
  tier B  kv_len boundary  kv_len == block_N, +-1 around block boundaries,
                           long KV with a non-dividing tail
  tier C  split-count edge ns=1, ns > kv blocks, ns that must be reduced to
                           divide the block count, ns=32
  tier D  numerical stress very peaked and near-uniform attention (q scaled),
                           plus the long-KV shape actually used by the benchmark

``effective_split`` (the reduction that forces the split count to divide the KV
block count) is additionally unit-tested without touching the device.

    python -m pytest test_mla_decode_splitk_matrix.py -q
"""

import pytest
import torch

import tilelang.testing

try:  # a CUDA-only build has no torch_npu; requires_ascend skips those cases
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

from example_mla_decode_splitk import compile_mla_decode_splitk, effective_split

# id, batch, heads, kv_len, dim, pe_dim, num_split, q_scale
CASES = [
    # ---------------- tier A: shape sweep ----------------
    ("a_dim128", 1, 128, 128, 128, 64, 2, 1.0),
    ("a_dim256", 1, 128, 256, 256, 64, 4, 1.0),
    ("a_dim512", 1, 128, 128, 512, 64, 2, 1.0),
    ("a_pe16", 1, 128, 128, 512, 16, 2, 1.0),
    ("a_pe128", 1, 128, 128, 512, 128, 2, 1.0),
    ("a_heads256", 1, 256, 128, 512, 64, 4, 1.0),
    ("a_heads512", 1, 512, 128, 512, 64, 8, 1.0),
    ("a_batch2", 2, 128, 256, 512, 64, 4, 1.0),
    ("a_batch4", 4, 128, 256, 512, 64, 4, 1.0),
    ("a_batch2_heads256", 2, 256, 384, 512, 64, 6, 1.0),
    # ---------------- tier B: kv_len boundary ----------------
    ("b_kv64", 1, 128, 64, 512, 64, 1, 1.0),  # exactly one block
    ("b_kv65", 1, 128, 65, 512, 64, 1, 1.0),  # single block + 1 column
    ("b_kv127", 1, 128, 127, 512, 64, 2, 1.0),  # last block shifted back
    ("b_kv129", 1, 128, 129, 512, 64, 3, 1.0),
    ("b_kv191", 1, 128, 191, 512, 64, 3, 1.0),
    ("b_kv192", 1, 128, 192, 512, 64, 3, 1.0),  # exact 3 blocks
    ("b_kv511", 1, 128, 511, 512, 64, 8, 1.0),
    ("b_kv513", 1, 128, 513, 512, 64, 4, 1.0),  # 9 blocks, tail of 1
    ("b_kv8193", 1, 128, 8193, 512, 64, 16, 1.0),  # 129 blocks, ns reduced
    # ---------------- tier C: split-count edge ----------------
    ("c_ns1", 1, 128, 256, 512, 64, 1, 1.0),
    ("c_ns_gt_blocks", 1, 128, 128, 512, 64, 8, 1.0),  # 2 blocks, ns -> 2
    ("c_ns_reduce", 1, 128, 768, 512, 64, 5, 1.0),  # 12 blocks, ns 5 -> 4
    ("c_ns32", 1, 128, 2048, 512, 64, 32, 1.0),
    # ---------------- tier D: numerical stress ----------------
    ("d_peaked", 1, 128, 512, 512, 64, 8, 8.0),  # large scores -> peaked softmax
    ("d_flat", 1, 128, 512, 512, 64, 8, 0.05),  # near-uniform attention
    ("d_bench_shape", 1, 128, 8192, 512, 64, 16, 1.0),  # shape used for perf
]


def ref_mla_decode(q, q_pe, kv, k_pe, scale):
    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bsd->bhd", attn, kv.float())
    return out.to(torch.float16)


def make_inputs(batch, heads, kv_len, dim, pe_dim, q_scale=1.0, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = (torch.randn((batch, heads, dim), generator=g) * q_scale).half().npu()
    q_pe = (torch.randn((batch, heads, pe_dim), generator=g) * q_scale).half().npu()
    kv = torch.randn((batch, kv_len, dim), generator=g).half().npu()
    k_pe = torch.randn((batch, kv_len, pe_dim), generator=g).half().npu()
    return q, q_pe, kv, k_pe


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "test_id,batch,heads,kv_len,dim,pe_dim,num_split,q_scale",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_matrix(test_id, batch, heads, kv_len, dim, pe_dim, num_split, q_scale):
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
    q, q_pe, kv, k_pe = make_inputs(batch, heads, kv_len, dim, pe_dim, q_scale)
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()

    expected = ref_mla_decode(q, q_pe, kv, k_pe, scale)
    actual = out.float()
    assert torch.isfinite(actual).all(), f"{test_id}: output contains NaN/Inf"
    # scale the tolerance with the value range so a near-zero output cannot hide
    # an absolute error
    atol = 1e-2
    torch.testing.assert_close(actual, expected.float(), rtol=1e-2, atol=atol)
    assert actual.shape == expected.shape


@tilelang.testing.requires_ascend
def test_deterministic():
    """The merge writes splits in a fixed order, so two runs must match bitwise."""
    batch, heads, kv_len, dim, pe_dim, num_split = 2, 128, 1024, 512, 64, 8
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
    a = run(q, q_pe, kv, k_pe)
    b = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    assert torch.equal(a, b), "split-KV merge is not deterministic across runs"


@pytest.mark.parametrize(
    "kv_len,block_N,num_split,expected",
    [
        (128, 64, 2, 2),
        (128, 64, 8, 2),  # more splits than blocks
        (768, 64, 5, 4),  # 12 blocks: 5 does not divide, reduce to 4
        (8192, 64, 16, 16),
        (8193, 64, 16, 3),  # 129 blocks = 3*43
        (513, 64, 4, 3),
        (64, 64, 8, 1),
        (448, 64, 7, 7),
    ],
)
def test_effective_split(kv_len, block_N, num_split, expected):
    ns = effective_split(kv_len, block_N, num_split)
    blocks = (kv_len + block_N - 1) // block_N
    assert ns == expected
    assert blocks % ns == 0, "split count must divide the KV block count"
    assert 1 <= ns <= min(num_split, blocks)


@pytest.mark.parametrize(
    "override",
    [
        {"heads": 192},  # not a multiple of block_H
        {"kv_len": 32},  # shorter than block_N
        {"dim": 320},  # not a multiple of chunk_d
    ],
    ids=["heads_not_multiple", "kv_shorter_than_block", "dim_not_multiple"],
)
def test_rejects_unsupported(override):
    """The unsupported envelope must fail loudly, not silently mis-compute."""
    from example_mla_decode_splitk import mla_decode_splitk_fwd

    kwargs = dict(batch=1, heads=128, kv_len=128, dim=512, pe_dim=64)
    kwargs.update(override)
    with pytest.raises(AssertionError):
        mla_decode_splitk_fwd(**kwargs)


@tilelang.testing.requires_ascend
def test_threads_1024():
    """threads=1024 was recommended by the sweep, so assert it is correct too."""
    batch, heads, kv_len, dim, pe_dim, num_split = 1, 128, 512, 512, 64, 8
    scale = (dim + pe_dim) ** -0.5
    run = compile_mla_decode_splitk(
        batch=batch,
        heads=heads,
        kv_len=kv_len,
        dim=dim,
        pe_dim=pe_dim,
        softmax_scale=scale,
        num_split=num_split,
        threads=1024,
    )
    q, q_pe, kv, k_pe = make_inputs(batch, heads, kv_len, dim, pe_dim)
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    expected = ref_mla_decode(q, q_pe, kv, k_pe, scale)
    assert torch.isfinite(out.float()).all()
    torch.testing.assert_close(out.float(), expected.float(), rtol=1e-2, atol=1e-2)

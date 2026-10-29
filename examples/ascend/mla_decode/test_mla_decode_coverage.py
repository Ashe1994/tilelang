"""P5 coverage: dense MLA decode gaps left open by the inherited 58 cases.

The handover notes (section 5) and their optimisation TODO (P5) list the
unverified areas.
The paged ones live in ``test_mla_decode_paged.py`` (see the
``MULTIBATCH_CASES`` block there); this file covers the dense ones:

  * ``batch = 16`` correctness -- only performance data existed, although the
    partial buffer is already 16 MiB at ``b=16 / ns=4`` (64 MiB at ns=16).
  * ``dim = 256`` together with ``pe_dim = 128``.
  * ``kv_len = 8193`` -- the long tail leaves 65 KV blocks, so the split count is
    reduced to 13 (65 = 5 x 13); a non-dividing-tail path only checked before at
    ``block_N=64``.
  * ``block_N`` / ``chunk_d`` / ``num_stages`` at shapes other than
    ``kv_len=512, dim=512`` (the inherited tile sweep fixed that one shape).
  * call hygiene: the callable is reusable, the inputs are not mutated in place,
    and a 100-launch soak stays bitwise stable.

Tolerance and reference are the house standard: float32 torch reference,
``rtol=atol=1e-2``, plus an explicit ``isfinite`` assertion. ``make_inputs`` and
``ref_mla_decode`` are imported from the inherited matrix test rather than copied
so the three files cannot drift apart.

Two limitations found while writing this file are recorded in the README
known-limitations section rather than asserted here: ``bf16`` is rejected because
the kernel ``dtype`` is hardcoded to fp16, and non-contiguous inputs are rejected
by the runtime's packed-ABI stride check (they fail loudly, never silently).

    pytest test_mla_decode_coverage.py -q
"""

import os
import sys

import pytest
import torch

import tilelang.testing

try:  # the kernels only compile on an Ascend host; keep this importable elsewhere
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

# The example kernels live next to this test file, so a same-directory import is
# enough. pytest already prepends the test directory; the explicit insert keeps
# `python -m pytest <file>` working from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from example_mla_decode_splitk import compile_mla_decode_splitk
from test_mla_decode_splitk_matrix import make_inputs, ref_mla_decode

D, PE, H = 512, 64, 128

# Compiling phase 1 + merge dominates the runtime, and several tests below reuse
# the same shape (soak / call hygiene / determinism), so the compiled callables
# are cached for the module instead of recompiled per test.
_RUNS = {}


def get_run(batch, kv_len, dim=D, pe_dim=PE, num_split=8, heads=H, **tile):
    key = (batch, heads, kv_len, dim, pe_dim, num_split, tuple(sorted(tile.items())))
    if key not in _RUNS:
        _RUNS[key] = compile_mla_decode_splitk(
            batch=batch,
            heads=heads,
            kv_len=kv_len,
            dim=dim,
            pe_dim=pe_dim,
            softmax_scale=(dim + pe_dim) ** -0.5,
            num_split=num_split,
            **tile,
        )
    return _RUNS[key]


def run_and_check(run, batch, kv_len, dim=D, pe_dim=PE, seed=0, test_id=""):
    q, q_pe, kv, k_pe = make_inputs(batch, H, kv_len, dim, pe_dim, seed=seed)
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    expected = ref_mla_decode(q, q_pe, kv, k_pe, (dim + pe_dim) ** -0.5)
    actual = out.float()
    assert torch.isfinite(actual).all(), f"{test_id}: output contains NaN/Inf"
    assert actual.shape == expected.shape, f"{test_id}: shape {actual.shape} != {expected.shape}"
    torch.testing.assert_close(actual, expected.float(), rtol=1e-2, atol=1e-2)
    return actual


# id, batch, kv_len, dim, pe_dim, num_split
SHAPES = [
    ("b16_kv128", 16, 128, 512, 64, 8),   # 1 KV block -> effective ns=1
    ("b16_kv512", 16, 512, 512, 64, 8),   # effective ns=4, O_partial 16 MiB
    ("b2_dim256_pe128", 2, 512, 256, 128, 4),
    ("kv8193_tail", 1, 8193, 512, 64, 16),  # 65 blocks -> effective ns=13
]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "test_id,batch,kv_len,dim,pe_dim,num_split",
    SHAPES,
    ids=[c[0] for c in SHAPES],
)
def test_shape_coverage(test_id, batch, kv_len, dim, pe_dim, num_split):
    run = get_run(batch, kv_len, dim=dim, pe_dim=pe_dim, num_split=num_split)
    run_and_check(run, batch, kv_len, dim=dim, pe_dim=pe_dim, seed=1, test_id=test_id)


# id, block_N, chunk_d, num_stages, kv_len, dim, pe_dim, num_split, batch
TILES = [
    # num_stages=1 with block_N=64 at a KV length other than 512
    ("t_bn64_cd128_ns1_kv768", 64, 128, 1, 768, 512, 64, 6, 1),
    # chunk_d=128 together with block_N=128, at dim=256 (old sweep: dim=512 only)
    ("t_bn128_cd128_dim256", 128, 128, 2, 512, 256, 64, 4, 1),
    # default tiling family but num_stages=1 on a longer KV axis
    ("t_bn128_cd64_ns1_kv1024", 128, 64, 1, 1024, 512, 64, 4, 1),
]


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "test_id,block_N,chunk_d,num_stages,kv_len,dim,pe_dim,num_split,batch",
    TILES,
    ids=[c[0] for c in TILES],
)
def test_tile_coverage(
    test_id, block_N, chunk_d, num_stages, kv_len, dim, pe_dim, num_split, batch
):
    run = get_run(
        batch, kv_len, dim=dim, pe_dim=pe_dim, num_split=num_split,
        block_N=block_N, chunk_d=chunk_d, num_stages=num_stages,
    )
    run_and_check(run, batch, kv_len, dim=dim, pe_dim=pe_dim, seed=2, test_id=test_id)


@tilelang.testing.requires_ascend
def test_call_hygiene_b16():
    """batch=16 callable: inputs untouched, repeatable, and actually re-reads inputs."""
    batch, kv_len = 16, 512
    run = get_run(batch, kv_len)
    scale = (D + PE) ** -0.5
    q, q_pe, kv, k_pe = make_inputs(batch, H, kv_len, D, PE, seed=5)
    snapshot = [t.clone() for t in (q, q_pe, kv, k_pe)]

    a = run(q, q_pe, kv, k_pe)
    b = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()

    for name, now, before in zip(("Q", "Q_pe", "KV", "K_pe"), (q, q_pe, kv, k_pe), snapshot):
        assert torch.equal(now, before), f"{name} was mutated in place by the kernel"

    assert torch.equal(a, b), "repeated calls on the same buffers are not bitwise equal"

    # A second call on the same buffers with new data must reflect the new data:
    # this catches a callable that captures its inputs or never re-runs phase 1.
    q2, q_pe2, kv2, k_pe2 = make_inputs(batch, H, kv_len, D, PE, seed=6)
    c = run(q2, q_pe2, kv2, k_pe2)
    torch.npu.synchronize()
    assert not torch.allclose(c.float(), a.float(), rtol=1e-2, atol=1e-2), (
        "different inputs produced the same output: the callable is not stateless"
    )
    expected = ref_mla_decode(q2, q_pe2, kv2, k_pe2, scale)
    assert torch.isfinite(c.float()).all(), "second-call output contains NaN/Inf"
    torch.testing.assert_close(c.float(), expected.float(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_ascend
def test_soak_stability():
    """Short soak: 100 back-to-back launches must stay finite and bitwise stable."""
    batch, kv_len = 16, 512
    run = get_run(batch, kv_len)
    q, q_pe, kv, k_pe = make_inputs(batch, H, kv_len, D, PE, seed=7)

    first = run(q, q_pe, kv, k_pe).clone()
    torch.npu.synchronize()
    mismatches = 0
    for _ in range(100):
        out = run(q, q_pe, kv, k_pe)
        if not torch.equal(out, first):
            mismatches += 1
    torch.npu.synchronize()
    assert torch.isfinite(first.float()).all(), "soak baseline contains NaN/Inf"
    assert mismatches == 0, f"{mismatches}/100 soak iterations differ from the first run"


@tilelang.testing.requires_ascend
def test_deterministic_across_compiles():
    """Two independent compiles of the same shape must agree bitwise on fixed data.

    (The inherited matrix test re-runs one callable; this checks that the compiled
    artefact does not embed any run-to-run state.)
    """
    batch, kv_len = 2, 512
    run_a = get_run(batch, kv_len, num_split=8)
    q, q_pe, kv, k_pe = make_inputs(batch, H, kv_len, D, PE, seed=8)
    out_a = run_a(q, q_pe, kv, k_pe).clone()
    torch.npu.synchronize()

    run_b = compile_mla_decode_splitk(
        batch=batch, heads=H, kv_len=kv_len, dim=D, pe_dim=PE,
        softmax_scale=(D + PE) ** -0.5, num_split=8,
    )
    out_b = run_b(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    assert torch.equal(out_a, out_b), "two compiles of one shape disagree bitwise"


def test_effective_split_long_tail():
    """kv_len=8193 with block_N=128 leaves 65 = 5x13 blocks, so ns is reduced to 13.

    The shape test above proves the reduced split *numerically*; this pins the
    split-count arithmetic so the two cannot drift apart silently.
    """
    from example_mla_decode_splitk import effective_split

    assert effective_split(8193, 128, 16) == 13  # 65 blocks, 65 % 16 != 0 ... 13 | 65
    assert effective_split(8193, 64, 16) == 3    # 129 blocks, 129 = 3 * 43
    for ns, blocks in ((13, 65), (3, 129)):
        assert blocks % ns == 0

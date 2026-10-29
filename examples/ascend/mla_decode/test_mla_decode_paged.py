"""Correctness tests for the paged MLA decode kernel on Ascend.

The paged cache is built with a *shuffled* physical page order, so a kernel that
ignores ``block_table`` (i.e. reads pages linearly) produces wrong numbers and
fails. Sequence lengths vary per batch, and page sizes are chosen to exercise
both geometries: several tiles inside one page, and one tile spanning pages.

    TILELANG_DEFAULT_TARGET=ascend python -m pytest test_mla_decode_paged.py -q
"""

import math
import os
import sys

import pytest
import torch

import tilelang.testing

try:  # the kernel only compiles on an Ascend host; keep this importable elsewhere
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

# The example kernels live next to this test file, so a same-directory import is
# enough. pytest already prepends the test directory; the explicit insert keeps
# `python -m pytest <file>` working from any cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from example_mla_decode_paged import compile_mla_decode_paged

D, PE = 512, 64

# id, batch, seq_lens(list), block_size, block_N, num_split
CASES = [
    ("tile_spans_pages", 1, [128], 64, 128, 2),  # one tile = 2 pages
    ("tail_partial_page", 1, [100], 64, 128, 2),  # last page half live
    ("two_tiles_per_page", 1, [128], 128, 64, 2),  # block_size > block_N: offset != 0
    ("offset_in_page", 1, [200], 128, 64, 3),  # odd tile index -> offset 64
    ("var_len_batch", 3, [64, 129, 300], 64, 128, 4),
    ("short_seq", 1, [65], 64, 128, 2),
    ("seq_lt_tile", 1, [64], 64, 128, 8),  # nblk=1 < ns -> empty splits
    ("seq_exact_page", 1, [64], 64, 64, 1),
    ("ns1", 2, [128, 256], 64, 128, 1),
    ("multi_block", 1, [512], 64, 128, 4),
]


# P5 additions (task-3): the inherited 11 cases above are all thin on batch size.
# These widen the paged path exactly where README section 5 says "not verified":
#   * batch > 1 with shuffled physical pages and per-batch variable length
#   * empty splits per batch (some batches own fewer KV blocks than num_split)
#   * the page-128 / block_N-128 geometry (page == tile), missing before
#   * num_split beyond the block count (ns=32 with 4-8 blocks) and ns=1
#   * very short sequences (seq=1) where the mask covers a whole tile
#
# id, batch, seq_lens(list), block_size, block_N, num_split, seed
MULTIBATCH_CASES = [
    ("mb_b4_mixed", 4, [64, 129, 300, 512], 64, 128, 4, 11),
    ("mb_b4_empty_split", 4, [1, 64, 128, 300], 64, 128, 8, 12),
    ("mb_b8_mixed", 8, [33, 64, 129, 200, 257, 300, 384, 512], 128, 64, 4, 13),
    ("mb_b2_page_eq_tile", 2, [128, 512], 128, 128, 4, 14),
    ("mb_b2_page_eq_tile_ns1", 2, [65, 192], 64, 64, 1, 15),
    ("mb_b2_ns32_empty", 2, [512, 1024], 64, 128, 32, 16),
    ("mb_b4_one_block", 4, [64, 64, 64, 64], 64, 128, 4, 17),
    ("mb_b4_long_mixed", 4, [1024, 769, 512, 256], 128, 128, 8, 18),
    ("mb_b2_tiny_seq", 2, [1, 17], 64, 128, 2, 19),
]


def make_paged(batch, seq_lens, block_size, seed=0, spare_pages=8):
    """Build a paged cache whose physical pages are a random permutation."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    max_seqlen = max(seq_lens)
    max_pages = math.ceil(max_seqlen / block_size)
    num_pages = batch * max_pages + spare_pages

    perm = torch.randperm(num_pages, generator=g).to(torch.int32)
    table = perm[: batch * max_pages].view(batch, max_pages).contiguous()

    kv = torch.randn((num_pages * block_size, D), generator=g).half().npu()
    k_pe = torch.randn((num_pages * block_size, PE), generator=g).half().npu()
    seqlens = torch.tensor(seq_lens, dtype=torch.int32).npu()

    q = torch.randn((batch, 128, D), generator=g).half().npu()
    q_pe = torch.randn((batch, 128, PE), generator=g).half().npu()
    return q, q_pe, kv, k_pe, table.npu(), seqlens, max_pages


def ref_paged(q, q_pe, kv, k_pe, table, seqlens, block_size, scale):
    """Dense torch reference: gather the pages each batch actually points at."""
    batch = q.shape[0]
    out = torch.empty((batch, q.shape[1], D), dtype=torch.float16, device=q.device)
    kv_cpu = kv
    for b in range(batch):
        n = int(seqlens[b].item())
        npages = math.ceil(n / block_size)
        pages = table[b, :npages].tolist()
        slots = []
        for p in pages:
            slots.extend(range(p * block_size, p * block_size + block_size))
        slots = torch.tensor(slots[:n], dtype=torch.long, device=kv.device)
        k = kv_cpu[slots].float()  # [n, D]
        kp = k_pe[slots].float()
        query = torch.cat([q[b].float(), q_pe[b].float()], dim=-1)
        key = torch.cat([k, kp], dim=-1)
        scores = torch.einsum("hd,sd->hs", query, key) * scale
        attn = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hs,sd->hd", attn, k).to(torch.float16)
    return out


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "name,batch,seq_lens,block_size,block_N,num_split",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_paged(name, batch, seq_lens, block_size, block_N, num_split):
    scale = (D + PE) ** -0.5
    q, q_pe, kv, k_pe, table, seqlens, max_pages = make_paged(batch, seq_lens, block_size)

    run = compile_mla_decode_paged(
        batch=batch,
        heads=128,
        num_slots=kv.shape[0],
        dim=D,
        pe_dim=PE,
        block_size=block_size,
        max_pages=max_pages,
        block_N=block_N,
        num_split=num_split,
        softmax_scale=scale,
    )
    out = run(q, q_pe, kv, k_pe, table, seqlens)
    torch.npu.synchronize()

    expected = ref_paged(q, q_pe, kv, k_pe, table, seqlens, block_size, scale)
    actual = out.float()
    assert torch.isfinite(actual).all(), f"{name}: NaN/Inf"
    torch.testing.assert_close(actual, expected.float(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_ascend
@pytest.mark.parametrize(
    "name,batch,seq_lens,block_size,block_N,num_split,seed",
    MULTIBATCH_CASES,
    ids=[c[0] for c in MULTIBATCH_CASES],
)
def test_paged_multibatch(name, batch, seq_lens, block_size, block_N, num_split, seed):
    """batch > 1: per-batch variable length over shuffled physical pages."""
    scale = (D + PE) ** -0.5
    q, q_pe, kv, k_pe, table, seqlens, max_pages = make_paged(
        batch, seq_lens, block_size, seed=seed
    )

    run = compile_mla_decode_paged(
        batch=batch,
        heads=128,
        num_slots=kv.shape[0],
        dim=D,
        pe_dim=PE,
        block_size=block_size,
        max_pages=max_pages,
        block_N=block_N,
        num_split=num_split,
        softmax_scale=scale,
    )
    out = run(q, q_pe, kv, k_pe, table, seqlens)
    torch.npu.synchronize()

    expected = ref_paged(q, q_pe, kv, k_pe, table, seqlens, block_size, scale)
    actual = out.float()
    assert torch.isfinite(actual).all(), f"{name}: NaN/Inf"
    torch.testing.assert_close(actual, expected.float(), rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_ascend
def test_block_table_is_honoured_multibatch():
    """batch>1: repointing a page of ONE batch changes that batch only.

    A cross-batch gather bug (e.g. reading ``block_table[0]`` for every batch)
    would leave the other batches untouched but make batch 2 independent of its
    own table, so both halves of this assertion are needed.
    """
    scale = (D + PE) ** -0.5
    batch, seq, block_size, block_N, num_split = 4, 256, 64, 128, 4
    q, q_pe, kv, k_pe, table, seqlens, max_pages = make_paged(
        batch, [seq] * batch, block_size, seed=21
    )
    run = compile_mla_decode_paged(
        batch=batch, heads=128, num_slots=kv.shape[0], dim=D, pe_dim=PE,
        block_size=block_size, max_pages=max_pages, block_N=block_N,
        num_split=num_split, softmax_scale=scale,
    )
    out_a = run(q, q_pe, kv, k_pe, table, seqlens).clone()
    torch.npu.synchronize()

    # first physical page that no live table entry points at
    spare_page = batch * max_pages
    table2 = table.clone()
    table2[2, 0] = spare_page
    out_b = run(q, q_pe, kv, k_pe, table2, seqlens)
    torch.npu.synchronize()

    changed = not torch.allclose(out_a[2].float(), out_b[2].float(), rtol=1e-2, atol=1e-2)
    assert changed, "repointing batch 2's page did not change batch 2: block_table ignored"

    for b in (0, 1, 3):
        assert torch.allclose(out_a[b].float(), out_b[b].float(), rtol=1e-2, atol=1e-2), (
            f"repointing batch 2's page changed batch {b}: per-batch table indexing is wrong"
        )


@tilelang.testing.requires_ascend
def test_block_table_is_honoured():
    """A wrong gather must not accidentally pass: permuting the table changes O."""
    scale = (D + PE) ** -0.5
    q, q_pe, kv, k_pe, table, seqlens, max_pages = make_paged(1, [256], 64, seed=1)
    run = compile_mla_decode_paged(
        batch=1, heads=128, num_slots=kv.shape[0], dim=D, pe_dim=PE,
        block_size=64, max_pages=max_pages, block_N=128, num_split=4,
        softmax_scale=scale,
    )
    out_a = run(q, q_pe, kv, k_pe, table, seqlens).clone()
    torch.npu.synchronize()

    # Point one logical page at an unused spare page instead of swapping two live
    # pages: attention is invariant to the *order* of the KV positions, so a swap
    # would barely change the result even for a correct kernel.
    spare_page = 4  # first page past the 4 live pages of this case
    table2 = table.clone()
    table2[0, 0] = spare_page
    out_b = run(q, q_pe, kv, k_pe, table2, seqlens)
    torch.npu.synchronize()
    assert not torch.allclose(out_a, out_b, rtol=1e-2, atol=1e-2), (
        "repointing a page did not change the output: block_table is being ignored"
    )

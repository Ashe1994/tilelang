"""Correctness tests for the paged MLA decode kernel on Ascend.

The paged cache is built with a *shuffled* physical page order, so a kernel that
ignores ``block_table`` (i.e. reads pages linearly) produces wrong numbers and
fails. Sequence lengths vary per batch, and page sizes are chosen to exercise
both geometries: several tiles inside one page, and one tile spanning pages.

    TILELANG_DEFAULT_TARGET=ascend pytest test_mla_decode_paged_ascend.py -q
"""

import math
import os
import sys

import pytest
import torch
import torch_npu  # noqa: F401

ASCEND_DIR = "/data/h00662772/tilelang/examples/ascend"
if ASCEND_DIR not in sys.path:
    sys.path.insert(0, ASCEND_DIR)
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from example_mla_decode_paged_ascend import compile_mla_decode_paged

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

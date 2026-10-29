"""Paged MLA decode forward kernel on Ascend 950 (Cube + Vector cores).

Ascend port of ``examples/deepseek_mla/example_mla_decode_paged.py`` (the CUDA
``num_split`` path), keeping the platform constraints of the dense Ascend port::

    O[b,h,:] = sum_s softmax( (Q[b,h,:]KV[b,s,:] + Q_pe[b,h,:]K_pe[b,s,:]) * scale ) * KV[b,s,:]

With a paged KV cache the ``s`` axis is materialised through ``block_table``::

    page  = block_table[b, (kv_blk * block_N + pc * PAGE_ROWS) // block_size]
    slot  = page * block_size + (kv_blk * block_N + pc * PAGE_ROWS) % block_size

and the live length is per batch (``cache_seqlens[b]``). Four consequences drive
the port:

  * **one tile may span several pages, but never straddles an address range.** A
    ``block_N`` tile is assembled from ``NUM_PC`` page chunks of ``PAGE_ROWS``
    tokens; both page geometries work (``block_size > block_N``: several tiles in
    one page, ``block_size <= block_N``: one tile spans whole pages).
  * **each page chunk needs its own buffers.** Two Ascend constraints force it:
    a GM->L1 copy with a runtime source offset must land on a zero-based region
    covering the full row axis (so a page cannot fill a row slice of a shared
    tile), and ``T.gemm`` sizes K from the whole buffer rather than from a
    region (so a score accumulator cannot be written column-by-column). Hence
    ``k_l1``/``s_l0c``/``s_ub``/``p_l1``/``p_ub`` are per-page lists.
  * **the rescale still happens once per tile, not once per page.** The row max
    and row sum are combined across the page chunks before ``o_ub`` is touched,
    and the folded rescale is applied on the very first PV chunk only. This is
    where the ``block_N`` win survives a 64-token page size.
  * **the trip count is runtime.** Live KV blocks are split over cores like the
    CUDA kernel: ``per = nblk // NS`` plus one extra for the first ``nblk % NS``
    splits. A split that gets zero blocks publishes ``lse = -inf`` and zeros,
    which the merge weights to exactly 0.

Everything else -- split-KV, the phase-1/merge pair, fused QK accumulate, fused
rescale, ``pick_tiling`` -- is shared with ``example_mla_decode_splitk``; the
merge kernel is imported rather than copied.
"""

import tilelang
import tilelang.ascend.language as T

from example_mla_decode_splitk import mla_decode_merge, pick_tiling

# The paged kernel carries one score tile per page chunk, so it trims the ML row
# from 32 to 8 floats (still one 32 B DMA granule) to stay inside the UB budget.
ML_WIDTH_PAGED = 8


def mla_decode_paged_fwd(
    batch,
    heads,
    num_slots,
    dim,
    pe_dim,
    block_size,
    max_pages,
    block_H=128,
    block_N=128,
    chunk_d=64,
    softmax_scale=None,
    num_split=16,
    threads=1024,
    num_stages=2,
    fuse_rescale=True,
    ml_width=8,
    tiles_per_core=None,
):
    """Phase 1: per-split partial output + log-sum-exp over a paged KV cache.

    ``ml_width`` is the row width of the ``ML`` tensor. Only column 0 is used; the
    row is padded to a DMA granularity. The paged kernel carries two score tiles
    and needs the 6 KiB that the dense kernel spends on a 32-wide row.
    """
    scale = float(softmax_scale) if softmax_scale is not None else (dim + pe_dim) ** -0.5
    dtype = "float16"
    accum_dtype = "float32"

    ROWS = block_H // 2
    NUM_M_TILES = heads // block_H
    NUM_DC = dim // chunk_d
    NS = num_split
    NUM_BLOCKS = batch * NUM_M_TILES * NS

    # pages per tile, and the rows of a tile that each page contributes
    NUM_PC = max(1, block_N // block_size)
    PAGE_ROWS = block_N if NUM_PC == 1 else block_size

    assert block_H == 128, "block_H must be 128 (T.dual_copy L0C->UB requires M=128)"
    assert heads % block_H == 0, "heads must be a multiple of block_H"
    assert dim % chunk_d == 0, "dim must be a multiple of chunk_d"
    assert block_size % block_N == 0 or block_N % block_size == 0, (
        "a block_N tile must nest inside a page or span whole pages"
    )
    assert NUM_PC * PAGE_ROWS == block_N, "tile assembly must cover exactly block_N rows"
    assert block_size % 16 == 0 and pe_dim % 16 == 0 and dim % 16 == 0
    assert num_slots >= batch * max_pages * block_size, "num_slots must cover every table entry"
    # Upper bound on the tiles any single core can own, as a compile-time constant.
    TILES_PER_CORE = tiles_per_core or -(-(-(-max_pages * block_size // block_N)) // max(1, NS))
    assert TILES_PER_CORE >= 1

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, dim), dtype),
        Q_pe: T.Tensor((batch, heads, pe_dim), dtype),
        KV: T.Tensor((num_slots, dim), dtype),
        K_pe: T.Tensor((num_slots, pe_dim), dtype),
        block_table: T.Tensor((batch, max_pages), "int32"),
        cache_seqlens: T.Tensor((batch,), "int32"),
        O_partial: T.Tensor((NS, batch, heads, dim), accum_dtype),
        ML: T.Tensor((NS, batch, heads, ml_width), accum_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            b = bx // (NUM_M_TILES * NS)
            rem_blk = bx % (NUM_M_TILES * NS)
            h0 = (rem_blk // NS) * block_H
            sid = rem_blk % NS

            # ------------------------------ L1 ------------------------------
            q_l1 = T.alloc_l1((block_H, dim), dtype)
            q_pe_l1 = T.alloc_l1((block_H, pe_dim), dtype)
            k_l1 = [T.alloc_l1((PAGE_ROWS, dim), dtype) for _ in range(NUM_PC)]
            k_pe_l1 = [T.alloc_l1((PAGE_ROWS, pe_dim), dtype) for _ in range(NUM_PC)]
            p_l1 = [T.alloc_l1((block_H, PAGE_ROWS), dtype) for _ in range(NUM_PC)]
            v_l1 = [
                [T.alloc_l1((chunk_d, PAGE_ROWS), dtype) for _ in range(NUM_DC)]
                for _ in range(NUM_PC)
            ]

            # ------------------------------ L0C -----------------------------
            s_l0c = [T.alloc_l0c((block_H, PAGE_ROWS), accum_dtype) for _ in range(NUM_PC)]
            o_l0c = T.alloc_l0c((block_H, chunk_d), accum_dtype)

            # ------------------------------ UB ------------------------------
            s_ub = [T.alloc_shared((ROWS, PAGE_ROWS), accum_dtype) for _ in range(NUM_PC)]
            p_ub = [T.alloc_shared((ROWS, PAGE_ROWS), dtype) for _ in range(NUM_PC)]
            o_ub = [T.alloc_shared((ROWS, chunk_d), accum_dtype) for _ in range(NUM_DC)]
            o_tmp_ub = T.alloc_shared((ROWS, chunk_d), accum_dtype)
            ml_ub = T.alloc_shared((ROWS, ml_width), accum_dtype)

            m_ub = T.alloc_shared((ROWS,), accum_dtype)
            l_ub = T.alloc_shared((ROWS,), accum_dtype)
            alpha_ub = T.alloc_shared((ROWS,), accum_dtype)


            T.copy(Q[b, h0:h0 + block_H, 0:dim], q_l1[0:block_H, 0:dim])
            T.copy(Q_pe[b, h0:h0 + block_H, 0:pe_dim], q_pe_l1[0:block_H, 0:pe_dim])

            with T.SimtVF(threads=threads):
                for r in T.Parallel(ROWS):
                    m_ub[r] = -T.infinity(accum_dtype)
                    l_ub[r] = 0.0
                for ob in o_ub:
                    for r, c in T.Parallel(ROWS, chunk_d):
                        ob[r, c] = 0.0

            # ---- split of the live KV blocks over the NS cores ----
            seqlen = cache_seqlens[b]
            nblk = T.ceildiv(seqlen, block_N)
            per = nblk // NS
            extra = nblk % NS
            loop_range = per + T.if_then_else(sid < extra, 1, 0)
            first = per * sid + T.min(sid, extra)

            for kk in T.Pipelined(loop_range, num_stages=num_stages):
                kv_blk = first + kk
                pos = kv_blk * block_N

                # enumerate() keeps these Python-level unrolls; `for pc in range(N)`
                # would become a TIR loop whose index cannot subscript a Python list.
                for pc, _kb in enumerate(k_l1):
                    p_pos = pos + pc * PAGE_ROWS
                    # clamp: the last tile can reach past the live pages
                    page = block_table[b, T.min(p_pos // block_size, max_pages - 1)]
                    kv_start = page * block_size + p_pos % block_size

                    T.copy(KV[kv_start:kv_start + PAGE_ROWS, 0:dim], k_l1[pc])
                    T.copy(K_pe[kv_start:kv_start + PAGE_ROWS, 0:pe_dim], k_pe_l1[pc])
                    for i, vb in enumerate(v_l1[pc]):
                        T.copy(
                            KV[kv_start:kv_start + PAGE_ROWS, i * chunk_d:(i + 1) * chunk_d],
                            vb,
                            transpose=True,
                        )

                # QK per page chunk: the pe half and the content half accumulate
                # into the same score tile (clear on the first only).
                for pc, kb in enumerate(k_l1):
                    T.gemm(
                        q_pe_l1[0:block_H, 0:pe_dim],
                        k_pe_l1[pc],
                        s_l0c[pc],
                        transpose_B=True,
                        clear_accum=True,
                    )
                    T.gemm(q_l1[0:block_H, 0:dim], kb, s_l0c[pc], transpose_B=True, clear_accum=False)
                for pc, sc in enumerate(s_l0c):
                    T.dual_copy(sc, s_ub[pc][0:ROWS, 0:PAGE_ROWS])

                with T.SimtVF(threads=threads):
                    # scale + mask, positions are global
                    for pc, sb in enumerate(s_ub):
                        for r, c in T.Parallel(ROWS, PAGE_ROWS):
                            sb[r, c] = T.if_then_else(
                                pos + pc * PAGE_ROWS + c < seqlen,
                                sb[r, c] * scale,
                                -T.infinity(accum_dtype),
                            )

                    # One reducer accumulates across every page chunk, so the
                    # combined row max needs no extra register set per page.
                    red_max = T.alloc_reducer((ROWS,), accum_dtype, op="max")
                    T.reducer_init(red_max)
                    for pc, sb in enumerate(s_ub):
                        for r, c in T.Parallel(ROWS, PAGE_ROWS):
                            T.reducer_update(red_max[r], sb[r, c])
                    tile_max = T.alloc_fragment((ROWS,), accum_dtype)
                    T.finalize_reducer(red_max, tile_max)

                    for r in T.Parallel(ROWS):
                        new_m = T.max(m_ub[r], tile_max[r])
                        alpha_ub[r] = T.exp(m_ub[r] - new_m)
                        m_ub[r] = new_m

                    # exp and the fp16 cast happen in one pass: the separate
                    # p_ub = s_ub copy was a full extra sweep of the tile.
                    for pc, sb in enumerate(s_ub):
                        pbw = p_ub[pc]
                        for r, c in T.Parallel(ROWS, PAGE_ROWS):
                            e = T.exp(sb[r, c] - m_ub[r])
                            sb[r, c] = e
                            pbw[r, c] = e

                    red_sum = T.alloc_reducer((ROWS,), accum_dtype, op="sum")
                    T.reducer_init(red_sum)
                    for pc, sb in enumerate(s_ub):
                        for r, c in T.Parallel(ROWS, PAGE_ROWS):
                            T.reducer_update(red_sum[r], sb[r, c])
                    tile_sum = T.alloc_fragment((ROWS,), accum_dtype)
                    T.finalize_reducer(red_sum, tile_sum)

                    for r in T.Parallel(ROWS):
                        l_ub[r] = l_ub[r] * alpha_ub[r] + tile_sum[r]

                    if not fuse_rescale:
                        for ob in o_ub:
                            for r, c in T.Parallel(ROWS, chunk_d):
                                ob[r, c] = ob[r, c] * alpha_ub[r]

                for pc, pb in enumerate(p_ub):
                    T.dual_copy(pb[0:ROWS, 0:PAGE_ROWS], p_l1[pc][0:block_H, 0:PAGE_ROWS])

                # PV: every page chunk of one output-column chunk accumulates into
                # the SAME L0C (clear on the first gemm only), so a tile still
                # costs one L0C->UB transfer and one UB accumulate per output
                # chunk -- the page split does not multiply them. This is what
                # removes the fixpipe / vector doubling that page_size=64 showed.
                for i, _junk in enumerate(o_ub):
                    for pc, pb in enumerate(p_ub):
                        T.gemm(p_l1[pc], v_l1[pc][i], o_l0c, transpose_B=True,
                               clear_accum=(pc == 0))
                    T.dual_copy(o_l0c, o_tmp_ub[0:ROWS, 0:chunk_d])
                    with T.SimtVF(threads=threads):
                        if fuse_rescale:
                            for r, c in T.Parallel(ROWS, chunk_d):
                                o_ub[i][r, c] = o_ub[i][r, c] * alpha_ub[r] + o_tmp_ub[r, c]
                        else:
                            for r, c in T.Parallel(ROWS, chunk_d):
                                o_ub[i][r, c] = o_ub[i][r, c] + o_tmp_ub[r, c]

            # ---- publish partials: normalised output + row log-sum-exp ----
            # A split with zero KV blocks keeps l_ub == 0; guard the divide so it
            # publishes zeros. Its lse is -inf, so the merge gives it weight 0.
            with T.SimtVF(threads=threads):
                for ob in o_ub:
                    for r, c in T.Parallel(ROWS, chunk_d):
                        ob[r, c] = T.if_then_else(l_ub[r] > 0.0, ob[r, c] / l_ub[r], 0.0)

            for i, ob in enumerate(o_ub):
                T.dual_copy(
                    ob[0:ROWS, 0:chunk_d],
                    O_partial[sid, b, h0:h0 + block_H, i * chunk_d:(i + 1) * chunk_d],
                )

            with T.SimtVF(threads=threads):
                for r, c in T.Parallel(ROWS, ml_width):
                    ml_ub[r, c] = 0.0
                for r in T.Parallel(ROWS):
                    ml_ub[r, 0] = m_ub[r] + T.log(l_ub[r])
            T.dual_copy(ml_ub[0:ROWS, 0:ml_width], ML[sid, b, h0:h0 + block_H, 0:ml_width])

    return main


def compile_mla_decode_paged(
    batch,
    heads,
    num_slots,
    dim,
    pe_dim,
    block_size,
    max_pages,
    block_N=128,
    chunk_d=None,
    softmax_scale=None,
    num_split=16,
    threads=1024,
    num_stages=2,
    fuse_rescale=True,
    merge_rows=16,
    merge_stages=4,
    merge_chunk_d=None,
    tiles_per_core=None,
):
    """Compile phase 1 + the shared merge kernel, return one callable."""
    _, chunk_d, merge_chunk_d = pick_tiling(
        max_pages * block_size, dim, block_N, chunk_d, merge_chunk_d
    )

    partial = tilelang.compile(
        mla_decode_paged_fwd(
            batch=batch,
            heads=heads,
            num_slots=num_slots,
            dim=dim,
            pe_dim=pe_dim,
            block_size=block_size,
            max_pages=max_pages,
            block_H=128,
            block_N=block_N,
            chunk_d=chunk_d,
            softmax_scale=softmax_scale,
            num_split=num_split,
            threads=threads,
            num_stages=num_stages,
            fuse_rescale=fuse_rescale,
            ml_width=ML_WIDTH_PAGED,
            tiles_per_core=tiles_per_core,
        ),
        out_idx=[6, 7],
    )
    merge = tilelang.compile(
        mla_decode_merge(
            batch=batch,
            heads=heads,
            kv_len=num_split * block_N,  # only a shape hint; ns_explicit is authoritative
            dim=dim,
            pe_dim=pe_dim,
            block_H=128,
            block_N=block_N,
            chunk_d=merge_chunk_d,
            num_split=num_split,
            ns_explicit=num_split,
            rows_per_block=merge_rows,
            num_stages=merge_stages,
            ml_width=ML_WIDTH_PAGED,
        ),
        out_idx=[2],
    )

    def run(q, q_pe, kv, k_pe, block_table, cache_seqlens):
        o_partial, ml = partial(q, q_pe, kv, k_pe, block_table, cache_seqlens)
        return merge(o_partial, ml)

    return run

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
rescale, ``pick_tiling`` -- keeps the same design as the native decode kernel in
``example_mla_decode.py``; the merge kernel lives in this module so the paged
delivery is self-contained.
"""

import tilelang
import tilelang.ascend.language as T

# Row width of the ``ML`` staging tensor used by the merge kernel (column 0 holds
# ``m + log(l)``; the rest is padding to one 32 B DMA granule).
ML_WIDTH = 8


def pick_tiling(kv_len, dim, block_N=None, chunk_d=None, merge_chunk_d=None):
    """Pick the measured-best tiling for a shape.

    Defaults are the measured optimum after the SimdVF pass landed:
    ``block_N=64, chunk_d=128``. Historically ``block_N=128`` was faster (fewer
    rescale sweeps per KV element), but with the SimdVF rescale the 64-token tile
    wins: it halves the score buffers, which frees UB for cross-stage overlap and
    is the only tiling where the full-SIMD vector_mode fits at ``kv_len=8192``.
    The merge tiles columns independently, so it keeps 128-wide DMAs.
    """
    if block_N is None:
        # Measured after the SimdVF pass landed: a 64-token score tile beats 128
        # on the reference shape (70.8 -> 54.2 us e2e; the merge/fp16 work that
        # builds on it reaches 46.4 us). The smaller tile halves
        # the score buffers, which both frees UB for the scheduler to overlap
        # more stages and is the only tiling where the full-SIMD `vector_mode`
        # fits at kv_len=8192 (block_N=128 needs 222208 B > the 221184 B limit).
        block_N = 64 if kv_len >= 64 else 32
    if chunk_d is None:
        # Keep few, wide PV chunks: at block_N=64 this is 4 chunks per tile and
        # measured ~16% faster than chunk_d=64 (8 chunks). The 128-token tile
        # keeps 64 so its score/output staging stays inside L0C and UB.
        chunk_d = 64 if block_N >= 128 else 128
    if merge_chunk_d is None:
        merge_chunk_d = 128 if dim % 128 == 0 else chunk_d
    assert dim % chunk_d == 0, "dim must be a multiple of chunk_d"
    assert dim % merge_chunk_d == 0, "dim must be a multiple of merge_chunk_d"
    return block_N, chunk_d, merge_chunk_d


def effective_split(kv_len, block_N, num_split):
    """Split count actually used: bounded by the KV block count, must divide it."""
    kv_blocks = (kv_len + block_N - 1) // block_N
    ns = min(num_split, kv_blocks)
    while kv_blocks % ns != 0:
        ns -= 1
    return ns



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
    block_N=64,
    chunk_d=128,
    softmax_scale=None,
    num_split=16,
    threads=1024,
    num_stages=2,
    fuse_rescale=True,
    ml_width=8,
    tiles_per_core=None,
    partial_dtype="float32",
    vector_mode="simd_rc",
    stage_cols=None,
):
    """Phase 1: per-split partial output + log-sum-exp over a paged KV cache.

    ``ml_width`` is the row width of the ``ML`` tensor. Only column 0 is used; the
    row is padded to a DMA granularity. The paged kernel carries two score tiles
    and needs the 6 KiB that the dense kernel spends on a 32-wide row.

    ``partial_dtype`` is the dtype of the published ``O_partial`` ("float32" or
    "float16"). The merge kernel defined in this module must be compiled with the
    same value.

    ``vector_mode`` selects the programming model of the PV rescale/accumulate
    pass, the largest elementwise pass of a tile: ``"simd_rc"`` (default) runs
    ``o = o*alpha + pv`` as ``T.SimdVF`` vector instructions (64 fp32 lanes per
    instruction), ``"simt"`` restores the original thread-parallel code for A/B
    testing. It falls back to ``T.SimtVF`` automatically when ``chunk_d`` is not
    a whole number of 64-lane registers. The two SIMD *exp* modes of the dense
    kernel are not available here: they need UB headroom this kernel does not
    have (the attempt overflowed UB at ``page == block_N == 128``).
    """
    scale = float(softmax_scale) if softmax_scale is not None else (dim + pe_dim) ** -0.5
    dtype = "float16"
    accum_dtype = "float32"
    assert partial_dtype in (accum_dtype, dtype), (
        'partial_dtype must be "float32" or "float16"'
    )

    ROWS = block_H // 2
    NUM_M_TILES = heads // block_H
    NUM_DC = dim // chunk_d
    NS = num_split
    NUM_BLOCKS = batch * NUM_M_TILES * NS

    # fp16 publish staging width, in output columns. The dense kernel can afford a
    # full-width [ROWS, chunk_d] fp16 staging buffer; this paged kernel cannot:
    # the full-width 16 KiB buffer leaves the auto-scheduler no UB headroom to
    # multi-buffer phase 1 (e2e 57.0 -> 68.7 us), and even 32-column slices are
    # slower than writing fp32 (57.0 -> 59.4 us). fp32 therefore stays this
    # kernel's default and the fp16 path is only for callers that want the merge
    # traffic halved; ``stage_cols`` overrides the slice width for experiments.
    STAGE_COLS = min(chunk_d, 32) if stage_cols is None else min(stage_cols, chunk_d)
    NUM_SLICES = chunk_d // STAGE_COLS

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

    # The PV rescale/accumulate is the one phase-1 pass that both wins and fits
    # the tight paged UB budget; 64 fp32 lanes per vector register means the
    # output-column chunk must be a whole number of registers.
    assert vector_mode in ("simt", "simd_rc"), (
        "the paged kernel supports vector_mode 'simt'|'simd_rc'; the SIMD exp "
        "modes need the dense kernel's UB headroom"
    )
    simd_rc = vector_mode == "simd_rc" and chunk_d % 64 == 0

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, dim), dtype),
        Q_pe: T.Tensor((batch, heads, pe_dim), dtype),
        KV: T.Tensor((num_slots, dim), dtype),
        K_pe: T.Tensor((num_slots, pe_dim), dtype),
        block_table: T.Tensor((batch, max_pages), "int32"),
        cache_seqlens: T.Tensor((batch,), "int32"),
        O_partial: T.Tensor((NS, batch, heads, dim), partial_dtype),
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
            if partial_dtype != accum_dtype:
                # fp16 publish staging: one output-column slice at a time, 4 KiB.
                o_out_ub = T.alloc_shared((ROWS, STAGE_COLS), partial_dtype)

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
                    if simd_rc:
                        with T.SimdVF():
                            if fuse_rescale:
                                for r, c in T.Parallel(ROWS, chunk_d):
                                    o_ub[i][r, c] = o_ub[i][r, c] * alpha_ub[r] + o_tmp_ub[r, c]
                            else:
                                for r, c in T.Parallel(ROWS, chunk_d):
                                    o_ub[i][r, c] = o_ub[i][r, c] + o_tmp_ub[r, c]
                    else:
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
            if partial_dtype == accum_dtype:
                with T.SimtVF(threads=threads):
                    for ob in o_ub:
                        for r, c in T.Parallel(ROWS, chunk_d):
                            ob[r, c] = T.if_then_else(l_ub[r] > 0.0, ob[r, c] / l_ub[r], 0.0)

                for i, ob in enumerate(o_ub):
                    T.dual_copy(
                        ob[0:ROWS, 0:chunk_d],
                        O_partial[sid, b, h0:h0 + block_H, i * chunk_d:(i + 1) * chunk_d],
                    )
            else:
                # fp16 partial: normalise into the 4 KiB fp16 staging buffer and
                # stream it out column slice by column slice (see STAGE_COLS). The
                # empty-split guard is kept so a zero-block split still publishes
                # zeros. enumerate() keeps this a Python-level unroll (04-A5).
                for i, ob in enumerate(o_ub):
                    for s, _slice in enumerate(range(NUM_SLICES)):
                        with T.SimtVF(threads=threads):
                            for r, c in T.Parallel(ROWS, STAGE_COLS):
                                o_out_ub[r, c] = T.if_then_else(
                                    l_ub[r] > 0.0,
                                    ob[r, s * STAGE_COLS + c] / l_ub[r],
                                    0.0,
                                )
                        T.dual_copy(
                            o_out_ub[0:ROWS, 0:STAGE_COLS],
                            O_partial[
                                sid, b, h0:h0 + block_H,
                                i * chunk_d + s * STAGE_COLS:
                                i * chunk_d + (s + 1) * STAGE_COLS,
                            ],
                        )

            with T.SimtVF(threads=threads):
                for r, c in T.Parallel(ROWS, ml_width):
                    ml_ub[r, c] = 0.0
                for r in T.Parallel(ROWS):
                    ml_ub[r, 0] = m_ub[r] + T.log(l_ub[r])
            T.dual_copy(ml_ub[0:ROWS, 0:ml_width], ML[sid, b, h0:h0 + block_H, 0:ml_width])

    return main


def mla_decode_merge(
    batch,
    heads,
    kv_len,
    dim,
    pe_dim,
    block_H=128,
    block_N=64,
    chunk_d=128,
    num_split=8,
    threads=256,
    rows_per_block=16,
    num_stages=4,
    ns_explicit=None,
    ml_width=ML_WIDTH,
    partial_dtype="float16",
    merge_vector_mode="simd",
):
    """Phase 2: combine the per-split partials into the final fp16 output.

    ``rows_per_block`` splits the 128-row head block across more blocks. With the
    default (one block per column chunk) the merge runs on ``heads/block_H *
    dim/chunk_d`` blocks only -- 4 for the MLA decode shape -- so the split loop
    was latency bound rather than bandwidth bound.

    ``num_stages`` pipelines that split loop, overlapping the next split's DMA
    with the current split's vector accumulate.

    ``merge_vector_mode`` selects how the per-split elementwise passes
    (``acc += tmp * wts[i, r]`` and the final ``out_h = acc / row_sum``) run:

      "simt"     -- ``T.SimtVF``, one scalar op per lane (the original kernel)
      "simd"     -- both passes in ``T.SimdVF`` (64 fp32 lanes per instruction)
      "simd_acc" -- only the accumulate pass in ``T.SimdVF``; the fp16 output
                    normalisation stays in ``T.SimtVF``

    The SimdVF passes need whole 64-lane vector registers along the column axis
    (``chunk_d % 64 == 0``); anything else falls back to SimtVF automatically.
    """
    dtype = "float16"
    accum_dtype = "float32"
    assert partial_dtype in (accum_dtype, dtype), (
        'partial_dtype must be "float32" or "float16"'
    )
    assert merge_vector_mode in ("simt", "simd", "simd_acc"), (
        'merge_vector_mode must be "simt", "simd" or "simd_acc"'
    )
    # SimdVF whole-register condition: 64 fp32 lanes along the tiled c axis.
    simd_merge = merge_vector_mode in ("simd", "simd_acc") and chunk_d % 64 == 0
    # A float16 O_partial needs an explicit fp16->fp32 widening: the generic
    # SimdVF lowering rejects the implicit `T.Cast("float32", tmp[...])`. The
    # intrinsic path below widens with vcvt and keeps the accumulator in the
    # de-interleaved lane layout vcvt produces (even lanes in the low half,
    # odd lanes in the high half of each 128-column group); the final
    # normalisation pass writes it back contiguously.
    simd_widen = (
        simd_merge and partial_dtype != accum_dtype and chunk_d % 128 == 0
    )
    if partial_dtype != accum_dtype and simd_merge and not simd_widen:
        simd_merge = False  # cannot widen generically: stay on the scalar path
    simd_norm = merge_vector_mode == "simd" and chunk_d % 64 == 0 and not simd_widen

    RP = block_H if rows_per_block is None else rows_per_block
    assert block_H % RP == 0, "block_H must be a multiple of rows_per_block"
    RSPLIT = block_H // RP
    NUM_M_TILES = heads // block_H
    NUM_DC = dim // chunk_d
    # ns_explicit bypasses the kv_len-derived split count: the paged kernel has a
    # runtime sequence length, so its split count is fixed by the caller instead.
    NS = effective_split(kv_len, block_N, num_split) if ns_explicit is None else ns_explicit
    NUM_BLOCKS = batch * NUM_M_TILES * NUM_DC * RSPLIT

    @T.prim_func
    def merge_main(
        O_partial: T.Tensor((NS, batch, heads, dim), partial_dtype),
        ML: T.Tensor((NS, batch, heads, ml_width), accum_dtype),
        O: T.Tensor((batch, heads, dim), dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            b = bx // (NUM_M_TILES * NUM_DC * RSPLIT)
            rem = bx % (NUM_M_TILES * NUM_DC * RSPLIT)
            h0 = (rem // (NUM_DC * RSPLIT)) * block_H + (rem % RSPLIT) * RP
            dc = (rem // RSPLIT) % NUM_DC

            acc = T.alloc_shared((RP, chunk_d), accum_dtype)
            # The streamed partial tile keeps the element type it has in GM, so an
            # fp16 O_partial moves half the bytes; the widening happens in the
            # accumulate below, not in an extra staging buffer.
            tmp = T.alloc_shared((RP, chunk_d), partial_dtype)
            out_h = T.alloc_shared((RP, chunk_d), dtype)
            ml = T.alloc_shared((RP, ml_width), accum_dtype)
            # The split axis is a TIR loop, not a Python unroll: unrolling one
            # copy+sync group per split pushes the scheduler past the dav-3510
            # intra-core flag pool ("total spilled mutex slots (46) exceed 32")
            # once num_split reaches 16.
            lse = T.alloc_shared((NS, RP), accum_dtype)
            wts = T.alloc_shared((NS, RP), accum_dtype)
            row_max = T.alloc_shared((RP,), accum_dtype)
            row_sum = T.alloc_shared((RP,), accum_dtype)

            # gather the per-split log-sum-exp and track the running maximum
            with T.SimtVF(threads=threads):
                for r in T.Parallel(RP):
                    row_max[r] = -T.infinity(accum_dtype)
            if num_stages > 1:
                for i in T.Pipelined(NS, num_stages=num_stages):
                    T.copy(ML[i, b, h0:h0 + RP, 0:ml_width], ml)
                    with T.SimtVF(threads=threads):
                        for r in T.Parallel(RP):
                            lse[i, r] = ml[r, 0]
                            row_max[r] = T.max(row_max[r], ml[r, 0])
            else:
                for i in T.serial(NS):
                    T.copy(ML[i, b, h0:h0 + RP, 0:ml_width], ml)
                    with T.SimtVF(threads=threads):
                        for r in T.Parallel(RP):
                            lse[i, r] = ml[r, 0]
                            row_max[r] = T.max(row_max[r], ml[r, 0])

            with T.SimtVF(threads=threads):
                for r in T.Parallel(RP):
                    row_sum[r] = 0.0
                for i in T.serial(NS):
                    for r in T.Parallel(RP):
                        wts[i, r] = T.exp(lse[i, r] - row_max[r])
                        row_sum[r] = row_sum[r] + wts[i, r]

                for r, c in T.Parallel(RP, chunk_d):
                    acc[r, c] = 0.0

            if num_stages > 1:
                for i in T.Pipelined(NS, num_stages=num_stages):
                    T.copy(
                        O_partial[i, b, h0:h0 + RP, dc * chunk_d:(dc + 1) * chunk_d],
                        tmp,
                    )
                    if simd_widen:
                        # fp16 partial on the vector unit: widen with vcvt and
                        # accumulate in the de-interleaved lane layout that
                        # vcvt produces (even columns in the low 64 lanes of a
                        # 128-column group, odd columns in the high 64).
                        with T.SimdVF():
                            mask32 = T.simd.pset(32)
                            for r in T.serial(RP):
                                w = T.simd.vld(wts[i, r], dist="BRC_B32")
                                for ck in T.serial(chunk_d // 128):
                                    v = T.simd.vld(tmp[r, ck * 128])
                                    lo = T.simd.vadd(
                                        T.simd.vld(acc[r, ck * 128]),
                                        T.simd.vmul(T.simd.vcvt(v, accum_dtype, part=0), w, mask32),
                                        mask32,
                                    )
                                    hi = T.simd.vadd(
                                        T.simd.vld(acc[r, ck * 128 + 64]),
                                        T.simd.vmul(T.simd.vcvt(v, accum_dtype, part=1), w, mask32),
                                        mask32,
                                    )
                                    T.simd.vsts(acc[r, ck * 128], lo, mask32)
                                    T.simd.vsts(acc[r, ck * 128 + 64], hi, mask32)
                    else:
                        with (T.SimdVF() if simd_merge else T.SimtVF(threads=threads)):
                            for r, c in T.Parallel(RP, chunk_d):
                                if partial_dtype == accum_dtype:
                                    acc[r, c] = acc[r, c] + tmp[r, c] * wts[i, r]
                                else:
                                    acc[r, c] = acc[r, c] + T.cast(tmp[r, c], accum_dtype) * wts[i, r]
            else:
                for i in T.serial(NS):
                    T.copy(
                        O_partial[i, b, h0:h0 + RP, dc * chunk_d:(dc + 1) * chunk_d],
                        tmp,
                    )
                    if simd_widen:
                        # fp16 partial on the vector unit: widen with vcvt and
                        # accumulate in the de-interleaved lane layout that
                        # vcvt produces (even columns in the low 64 lanes of a
                        # 128-column group, odd columns in the high 64).
                        with T.SimdVF():
                            mask32 = T.simd.pset(32)
                            for r in T.serial(RP):
                                w = T.simd.vld(wts[i, r], dist="BRC_B32")
                                for ck in T.serial(chunk_d // 128):
                                    v = T.simd.vld(tmp[r, ck * 128])
                                    lo = T.simd.vadd(
                                        T.simd.vld(acc[r, ck * 128]),
                                        T.simd.vmul(T.simd.vcvt(v, accum_dtype, part=0), w, mask32),
                                        mask32,
                                    )
                                    hi = T.simd.vadd(
                                        T.simd.vld(acc[r, ck * 128 + 64]),
                                        T.simd.vmul(T.simd.vcvt(v, accum_dtype, part=1), w, mask32),
                                        mask32,
                                    )
                                    T.simd.vsts(acc[r, ck * 128], lo, mask32)
                                    T.simd.vsts(acc[r, ck * 128 + 64], hi, mask32)
                    else:
                        with (T.SimdVF() if simd_merge else T.SimtVF(threads=threads)):
                            for r, c in T.Parallel(RP, chunk_d):
                                if partial_dtype == accum_dtype:
                                    acc[r, c] = acc[r, c] + tmp[r, c] * wts[i, r]
                                else:
                                    acc[r, c] = acc[r, c] + T.cast(tmp[r, c], accum_dtype) * wts[i, r]

            if simd_widen:
                # acc was stored de-interleaved by the widening pass: inside each
                # 128-column group, k = c//2 + (c%2)*64. Normalise and write the
                # contiguous fp16 output tile in one scalar pass (cheap: one pass
                # over the tile, versus once per split on the old path).
                with T.SimtVF(threads=threads):
                    for r, c in T.Parallel(RP, chunk_d):
                        cc = c % 128
                        k = (c // 128) * 128 + cc // 2 + (cc % 2) * 64
                        out_h[r, c] = acc[r, k] / row_sum[r]
            elif simd_norm:
                # fp16 output normalisation on the vector unit: the generic
                # SimdVF lowering rejects the implicit fp32->fp16 store cast
                # ("unsupported expr: T.Cast(float16, ...)"), so it is spelled
                # out with vdiv + vcvt(PK_B32) like the dense phase-1 exp pass.
                with T.SimdVF():
                    mask32 = T.simd.pset(32)
                    mask16 = T.simd.pset(16)
                    for r in T.serial(RP):
                        rs = T.simd.vld(row_sum[r], dist="BRC_B32")
                        for ck in T.serial(chunk_d // 64):
                            q = T.simd.vdiv(T.simd.vld(acc[r, ck * 64]), rs, mask32)
                            T.simd.vsts(
                                out_h[r, ck * 64],
                                T.simd.vcvt(q, dtype),
                                mask16,
                                dist="PK_B32",
                            )
            else:
                with T.SimtVF(threads=threads):
                    for r, c in T.Parallel(RP, chunk_d):
                        out_h[r, c] = acc[r, c] / row_sum[r]

            T.copy(out_h, O[b, h0:h0 + RP, dc * chunk_d:(dc + 1) * chunk_d])

    return merge_main


def compile_mla_decode_paged(
    batch,
    heads,
    num_slots,
    dim,
    pe_dim,
    block_size,
    max_pages,
    block_N=64,
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
    partial_dtype="float32",
    vector_mode="simd_rc",
    ml_width=ML_WIDTH_PAGED,
    merge_vector_mode="simd",
    stage_cols=None,
):
    """Compile phase 1 + the shared merge kernel, return one callable.

    Defaults are the measured optimum for the DeepSeek page geometry
    (``block_size=64``): ``block_N=64`` (one page per tile), ``chunk_d=128`` and
    ``vector_mode="simd_rc"``. The full-SIMD modes are rejected here on purpose:
    a per-page score tile is only 64 columns wide, so the explicit MicroAPI path
    measured no gain (phase 1 49.2 vs 49.1 us) and only adds UB pressure.

    ``partial_dtype`` must be the same on both phases: the shared merge kernel
    declares the ``O_partial`` input with this dtype. The ``out_idx`` positions of
    phase 1 are unchanged (only the element type of output 6 changes).

    ``vector_mode`` is forwarded to phase 1 (see ``mla_decode_paged_fwd``); the
    merge kernel is the shared phase-2 kernel defined in this module.
    """
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
            ml_width=ml_width,
            tiles_per_core=tiles_per_core,
            partial_dtype=partial_dtype,
            vector_mode=vector_mode,
            stage_cols=stage_cols,
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
            ml_width=ml_width,
            partial_dtype=partial_dtype,
            merge_vector_mode=merge_vector_mode,
        ),
        out_idx=[2],
    )

    def run(q, q_pe, kv, k_pe, block_table, cache_seqlens):
        o_partial, ml = partial(q, q_pe, kv, k_pe, block_table, cache_seqlens)
        return merge(o_partial, ml)

    return run

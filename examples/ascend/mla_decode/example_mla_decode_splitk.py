"""MLA decode forward kernel on Ascend 950 -- flash-decoding (split-KV) variant.

Same operator as ``example_mla_decode.py``::

    O[b,h,:] = sum_s softmax( (Q[b,h,:]KV[b,s,:] + Q_pe[b,h,:]K_pe[b,s,:]) * scale ) * KV[b,s,:]

The ``num_split == 1`` kernel partitions work as ``grid = batch * heads / block_H``,
so ``batch=1, heads=128`` runs on a single AI core and leaves the other cores idle
(measured upstream: latency is flat from batch 1 to batch 16). This variant splits
the KV axis instead, ``grid = batch * heads / block_H * num_split``, and merges the
partials in a second, pure-vector kernel:

    phase 1 (Cube cores): each core owns one KV range of one head block and
        produces an unnormalised partial output plus its log-sum-exp:
            O_p[s] = sum_{kv in range} exp(x - m_s) * V ,  lse_s = m_s + log(l_s)
    phase 2 (Vector cores): combine splits with softmax weights over split axis:
            w_s = exp(lse_s - max_s lse_s)
            O = (sum_s w_s * O_p[s]) / (sum_s w_s)

Why two kernels instead of one: a cross-core barrier would need every core to
reach it, which serialises the merge behind the slowest split and still needs the
partials to round-trip through GM. Two launches overlap nothing but cost one
extra kernel boundary.

Phase 1 writes ``ML``, a ``[num_split, batch, heads, ML_WIDTH]`` float32 tensor
whose column 0 holds ``lse``. The wide row exists because the partial write
reuses ``T.dual_copy``, whose UB->GM path merges the two AIV halves of the M axis
through one DMA; a 1-wide region is not a useful transfer shape, and 32 fp32
lanes (128 B) keep the destination row aligned.

``partial_dtype`` selects the dtype of ``O_partial``. The default (``float32``)
is the reference path. ``float16`` halves the phase-2 ``mte2`` traffic, but on
Ascend 950PR that measured no gain (the merge is latency bound, not byte bound:
``AIV vec`` is 91% of its duration against 69.9% for ``mte2``), so it stays an
opt-in knob for shapes where the merge really is DMA bound. The partials written
here are already **normalised** inside the split (``O_p[s]/l_s``), so the merge
is a convex combination of O(1) values and an fp16 round trip costs about one
fp16 epsilon of relative error on the final output, which is itself fp16.

Every platform constraint of the ``num_split == 1`` kernel still applies
(``block_H == 128``, NT-only L1 GEMM, V transposed into one independent L1 buffer
per output-column chunk, ``kv_len`` any value >= ``block_N``). Two extra
constraints: the effective split count is ``min(num_split, kv blocks)`` reduced
until it divides the KV block count, and each split runs at least one KV block.
"""

import tilelang
import tilelang.ascend.language as T

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
        # on the reference shape (68.8 -> 54.2 us e2e). The smaller tile halves
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


def mla_decode_splitk_fwd(
    batch,
    heads,
    kv_len,
    dim,
    pe_dim,
    block_H=128,
    block_N=64,
    chunk_d=128,
    softmax_scale=None,
    num_split=16,
    threads=1024,
    num_stages=2,
    fuse_qk=True,
    fuse_rescale=True,
    vector_mode=None,
    partial_dtype="float32",
    ml_width=ML_WIDTH,
):
    """Phase 1: per-split partial attention output + log-sum-exp.

    ``vector_mode`` selects how the per-KV-block elementwise vector work runs.
    The rescale/accumulate is ``o = o*alpha + pv`` over ``dim`` columns per KV
    block -- the largest elementwise pass -- and on the vector unit it is one
    vmul + one vadd per 64 fp32 lanes instead of one scalar op per thread:

      "simt"     -- everything in ``T.SimtVF`` (the original kernel)
      "simd_rc"  -- PV rescale/accumulate in ``T.SimdVF``
      "simd_exp" -- exp + fp16 cast in ``T.SimdVF``
      "simd"     -- both of the above

    ``None`` (the default) resolves to ``"simd"`` for ``block_N <= 64`` and to
    ``"simd_rc"`` for the 128-token tile: the SIMD exp pass overflowed UB at
    ``kv_len=8192, block_N=128`` (``dyn_ubuf_bytes=222208`` against the 221184 B
    limit), which is one of the reasons the default tiling moved to
    ``block_N=64``. An explicit value always wins. Measured on the benchmark
    shape with the default tiling: e2e 70.9 -> 54.2 us (-23.5%), rel error
    unchanged at 3.4e-04.

    The SimdVF exp needs explicit MicroAPI intrinsics: ``AscendSimdVFLowerParallel``
    accepts only load/add/sub/mul/div/exp/log/sqrt/fabs inside a SimdVF block, so
    ``T.exp`` and ``T.if_then_else`` are rejected there, while
    ``T.simd.vexpdif`` is a single fused ``exp(a-b)`` instruction. Every SimdVF
    pass also falls back to SimtVF automatically when the tiled width is not a
    whole number of vector registers (``block_N % 64`` / ``chunk_d % 64``), which
    keeps the ``block_N=32`` tilings working.

    ``fuse_qk`` accumulates the pe-half and content-half QK GEMMs in one L0C
    buffer (``clear_accum=False`` on the second) instead of producing two score
    tiles and adding them in UB: one fewer L0C->UB transfer per KV block and one
    fewer score buffer (16 KiB at block_N=64, 32 KiB at block_N=128).

    ``fuse_rescale`` folds the per-block ``o_ub *= alpha`` pass into the PV
    accumulate (``o = o*alpha + pv``), halving the accumulator traffic per block.
    """
    scale = float(softmax_scale) if softmax_scale is not None else (dim + pe_dim) ** -0.5
    dtype = "float16"
    accum_dtype = "float32"
    assert partial_dtype in (accum_dtype, dtype), (
        'partial_dtype must be "float32" or "float16"'
    )

    ROWS = block_H // 2
    NUM_M_TILES = heads // block_H
    KV_BLOCKS = (kv_len + block_N - 1) // block_N
    NUM_DC = dim // chunk_d
    NS = effective_split(kv_len, block_N, num_split)
    BPS = KV_BLOCKS // NS
    NUM_BLOCKS = batch * NUM_M_TILES * NS

    # Auto mode: the full-SIMD pass (exp + rescale on the vector unit) is only
    # known to fit the UB budget at block_N <= 64; the 128-token tile keeps the
    # rescale-only pass. An explicit vector_mode= always wins.
    if vector_mode is None:
        vector_mode = "simd" if block_N <= 64 else "simd_rc"

    # A SimdVF pass holds 64 fp32 lanes per vector register, so the tiled axis
    # must be a whole number of registers; otherwise fall back to SimtVF.
    assert vector_mode in ("simt", "simd", "simd_exp", "simd_rc"), \
        "vector_mode must be simt|simd|simd_exp|simd_rc"
    simd_exp = vector_mode in ("simd", "simd_exp") and block_N % 64 == 0
    simd_rc = vector_mode in ("simd", "simd_rc") and chunk_d % 64 == 0

    assert block_H == 128, "block_H must be 128 (T.dual_copy L0C->UB requires M=128)"
    assert heads % block_H == 0, "heads must be a multiple of block_H"
    assert dim % chunk_d == 0, "dim must be a multiple of chunk_d"
    assert kv_len >= block_N, "kv_len must be at least block_N"
    assert pe_dim % 16 == 0 and dim % 16 == 0

    @T.prim_func
    def main(
        Q: T.Tensor((batch, heads, dim), dtype),
        Q_pe: T.Tensor((batch, heads, pe_dim), dtype),
        KV: T.Tensor((batch, kv_len, dim), dtype),
        K_pe: T.Tensor((batch, kv_len, pe_dim), dtype),
        O_partial: T.Tensor((NS, batch, heads, dim), partial_dtype),
        ML: T.Tensor((NS, batch, heads, ml_width), accum_dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            b = bx // (NUM_M_TILES * NS)
            rem = bx % (NUM_M_TILES * NS)
            h0 = (rem // NS) * block_H
            sid = rem % NS

            # ------------------------------ L1 ------------------------------
            q_l1 = T.alloc_l1((block_H, dim), dtype)
            q_pe_l1 = T.alloc_l1((block_H, pe_dim), dtype)
            k_l1 = T.alloc_l1((block_N, dim), dtype)
            k_pe_l1 = T.alloc_l1((block_N, pe_dim), dtype)
            p_l1 = T.alloc_l1((block_H, block_N), dtype)
            # V^T per output-column chunk; independent buffers are required
            # because a dynamic L1 row slice is not a valid GEMM B operand.
            v_l1 = [T.alloc_l1((chunk_d, block_N), dtype) for _ in range(NUM_DC)]

            # ------------------------------ L0C -----------------------------
            s_l0c = T.alloc_l0c((block_H, block_N), accum_dtype)
            o_l0c = T.alloc_l0c((block_H, chunk_d), accum_dtype)

            # ------------------------------ UB ------------------------------
            s_ub = T.alloc_shared((ROWS, block_N), accum_dtype)
            if not fuse_qk:
                s_pe_ub = T.alloc_shared((ROWS, block_N), accum_dtype)
            p_ub = T.alloc_shared((ROWS, block_N), dtype)
            # One contiguous accumulator per output-column chunk: the write-back
            # then reuses the same T.dual_copy shape as the num_split=1 kernel.
            o_ub = [T.alloc_shared((ROWS, chunk_d), accum_dtype) for _ in range(NUM_DC)]
            o_tmp_ub = T.alloc_shared((ROWS, chunk_d), accum_dtype)
            ml_ub = T.alloc_shared((ROWS, ml_width), accum_dtype)
            if partial_dtype != accum_dtype:
                # fp16 publish staging, one output-column chunk at a time:
                # ROWS * chunk_d * 2 B = 8 KiB at ROWS=64, chunk_d=64.
                o_out_ub = T.alloc_shared((ROWS, chunk_d), partial_dtype)

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

            for kv_local in T.Pipelined(BPS, num_stages=num_stages):
                kv = sid * BPS + kv_local
                base = kv * block_N
                start = T.min(base, kv_len - block_N)
                lo = base - start
                hi = kv_len - start

                T.copy(KV[b, start:start + block_N, 0:dim], k_l1[0:block_N, 0:dim])
                T.copy(K_pe[b, start:start + block_N, 0:pe_dim], k_pe_l1[0:block_N, 0:pe_dim])
                for i, vb in enumerate(v_l1):
                    T.copy(
                        KV[b, start:start + block_N, i * chunk_d:(i + 1) * chunk_d],
                        vb[0:chunk_d, 0:block_N],
                        transpose=True,
                    )

                # QK: position-encoding half first, then content half, reusing one L0C.
                T.gemm(
                    q_pe_l1[0:block_H, 0:pe_dim],
                    k_pe_l1[0:block_N, 0:pe_dim],
                    s_l0c,
                    transpose_B=True,
                    clear_accum=True,
                )
                if not fuse_qk:
                    T.dual_copy(s_l0c, s_pe_ub[0:ROWS, 0:block_N])
                # Accumulate the content half straight into the same L0C, so the
                # two halves never need two score tiles in UB.
                T.gemm(
                    q_l1[0:block_H, 0:dim],
                    k_l1[0:block_N, 0:dim],
                    s_l0c,
                    transpose_B=True,
                    clear_accum=not fuse_qk,
                )
                T.dual_copy(s_l0c, s_ub[0:ROWS, 0:block_N])

                with T.SimtVF(threads=threads):
                    for r, c in T.Parallel(ROWS, block_N):
                        if fuse_qk:
                            scaled = s_ub[r, c] * scale
                        else:
                            scaled = (s_ub[r, c] + s_pe_ub[r, c]) * scale
                        s_ub[r, c] = T.if_then_else(
                            c >= lo,
                            T.if_then_else(c < hi, scaled, -T.infinity(accum_dtype)),
                            -T.infinity(accum_dtype),
                        )

                    red_max = T.alloc_reducer((ROWS,), accum_dtype, op="max")
                    T.reducer_init(red_max)
                    for r, c in T.Parallel(ROWS, block_N):
                        T.reducer_update(red_max[r], s_ub[r, c])
                    tile_max = T.alloc_fragment((ROWS,), accum_dtype)
                    T.finalize_reducer(red_max, tile_max)

                    for r in T.Parallel(ROWS):
                        new_m = T.max(m_ub[r], tile_max[r])
                        alpha_ub[r] = T.exp(m_ub[r] - new_m)
                        m_ub[r] = new_m

                    if not simd_exp:
                        # exp and the fp16 cast in one pass (the separate p_ub =
                        # s_ub copy was a full extra sweep of the score tile)
                        for r, c in T.Parallel(ROWS, block_N):
                            e = T.exp(s_ub[r, c] - m_ub[r])
                            s_ub[r, c] = e
                            p_ub[r, c] = e

                if simd_exp:
                    # exp(s - m) and the fp16 cast, straight on the vector unit.
                    # vexpdif is the fused exp(a-b) instruction; the fp16 result
                    # is packed-pair stored, so no separate conversion sweep.
                    with T.SimdVF():
                        mask32 = T.simd.pset(32)
                        mask16 = T.simd.pset(16)
                        for r in T.serial(ROWS):
                            mb = T.simd.vld(m_ub[r], dist="BRC_B32")
                            for ck in T.serial(block_N // 64):
                                e = T.simd.vexpdif(
                                    T.simd.vld(s_ub[r, ck * 64]),
                                    mb,
                                    mask32,
                                )
                                T.simd.vsts(s_ub[r, ck * 64], e, mask32)
                                T.simd.vsts(
                                    p_ub[r, ck * 64],
                                    T.simd.vcvt(e, dtype),
                                    mask16,
                                    dist="PK_B32",
                                )

                with T.SimtVF(threads=threads):
                    red_sum = T.alloc_reducer((ROWS,), accum_dtype, op="sum")
                    T.reducer_init(red_sum)
                    for r, c in T.Parallel(ROWS, block_N):
                        T.reducer_update(red_sum[r], s_ub[r, c])
                    tile_sum = T.alloc_fragment((ROWS,), accum_dtype)
                    T.finalize_reducer(red_sum, tile_sum)

                    for r in T.Parallel(ROWS):
                        l_ub[r] = l_ub[r] * alpha_ub[r] + tile_sum[r]

                    if not fuse_rescale:
                        for ob in o_ub:
                            for r, c in T.Parallel(ROWS, chunk_d):
                                ob[r, c] = ob[r, c] * alpha_ub[r]

                T.dual_copy(p_ub[0:ROWS, 0:block_N], p_l1[0:block_H, 0:block_N])

                # P.V over output-column chunks, unrolled at Python level.
                for i, vb in enumerate(v_l1):
                    T.gemm(p_l1[0:block_H, 0:block_N], vb, o_l0c, transpose_B=True, clear_accum=True)
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
                                # One pass instead of "rescale all chunks, then add".
                                for r, c in T.Parallel(ROWS, chunk_d):
                                    o_ub[i][r, c] = o_ub[i][r, c] * alpha_ub[r] + o_tmp_ub[r, c]
                            else:
                                for r, c in T.Parallel(ROWS, chunk_d):
                                    o_ub[i][r, c] = o_ub[i][r, c] + o_tmp_ub[r, c]

            # ---- publish partials: per-split output + row log-sum-exp ----
            # Normalise inside the split so the merge is the standard
            # flash-decoding weighted average over log-sum-exp values; an
            # unnormalised partial would need m_s as a second scalar.
            if partial_dtype == accum_dtype:
                with T.SimtVF(threads=threads):
                    for ob in o_ub:
                        for r, c in T.Parallel(ROWS, chunk_d):
                            ob[r, c] = ob[r, c] / l_ub[r]

                # enumerate() keeps this a Python-level unroll; a `for i in range(N)`
                # would become a TIR loop whose index cannot subscript a Python list.
                for i, ob in enumerate(o_ub):
                    T.dual_copy(
                        ob[0:ROWS, 0:chunk_d],
                        O_partial[sid, b, h0:h0 + block_H, i * chunk_d:(i + 1) * chunk_d],
                    )
            else:
                # fp16 partial: normalise straight into the fp16 staging buffer and
                # stream that out, so O_partial is written at half the DMA bytes.
                # One staging buffer is reused per chunk (WAR handled by the same
                # sync pattern as the PV loop above).
                for i, ob in enumerate(o_ub):
                    with T.SimtVF(threads=threads):
                        for r, c in T.Parallel(ROWS, chunk_d):
                            o_out_ub[r, c] = ob[r, c] / l_ub[r]
                    T.dual_copy(
                        o_out_ub[0:ROWS, 0:chunk_d],
                        O_partial[sid, b, h0:h0 + block_H, i * chunk_d:(i + 1) * chunk_d],
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
    partial_dtype="float32",
):
    """Phase 2: combine the per-split partials into the final fp16 output.

    ``rows_per_block`` splits the 128-row head block across more blocks. With the
    default (one block per column chunk) the merge runs on ``heads/block_H *
    dim/chunk_d`` blocks only -- 4 for the MLA decode shape -- so the split loop
    was latency bound rather than bandwidth bound.

    ``num_stages`` pipelines that split loop, overlapping the next split's DMA
    with the current split's vector accumulate.
    """
    dtype = "float16"
    accum_dtype = "float32"
    assert partial_dtype in (accum_dtype, dtype), (
        'partial_dtype must be "float32" or "float16"'
    )

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
                    with T.SimtVF(threads=threads):
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
                    with T.SimtVF(threads=threads):
                        for r, c in T.Parallel(RP, chunk_d):
                            if partial_dtype == accum_dtype:
                                acc[r, c] = acc[r, c] + tmp[r, c] * wts[i, r]
                            else:
                                acc[r, c] = acc[r, c] + T.cast(tmp[r, c], accum_dtype) * wts[i, r]

            with T.SimtVF(threads=threads):
                for r, c in T.Parallel(RP, chunk_d):
                    out_h[r, c] = acc[r, c] / row_sum[r]

            T.copy(out_h, O[b, h0:h0 + RP, dc * chunk_d:(dc + 1) * chunk_d])

    return merge_main


def compile_mla_decode_splitk(
    batch,
    heads,
    kv_len,
    dim,
    pe_dim,
    block_H=128,
    block_N=None,
    chunk_d=None,
    softmax_scale=None,
    num_split=16,
    threads=1024,
    num_stages=2,
    fuse_qk=True,
    fuse_rescale=True,
    merge_rows=16,
    merge_stages=4,
    merge_chunk_d=None,
    vector_mode=None,
    partial_dtype="float32",
    ml_width=ML_WIDTH,
):
    """Compile both phases and return one callable with the baseline signature.

    ``block_N``/``chunk_d``/``merge_chunk_d`` default to ``None``, which picks the
    measured-best tiling for the shape (see ``pick_tiling``) instead of one fixed
    value: the larger score tile only pays off once the KV axis is long enough to
    hold it.

    ``vector_mode`` selects the phase-1 vector programming model and
    ``partial_dtype`` the dtype of the phase-1 -> phase-2 staging tensor; both are
    pass-through knobs documented on ``mla_decode_splitk_fwd`` and
    ``mla_decode_merge``.
    """
    block_N, chunk_d, merge_chunk_d = pick_tiling(
        kv_len, dim, block_N, chunk_d, merge_chunk_d
    )
    partial = tilelang.compile(
        mla_decode_splitk_fwd(
            batch=batch,
            heads=heads,
            kv_len=kv_len,
            dim=dim,
            pe_dim=pe_dim,
            block_H=block_H,
            block_N=block_N,
            chunk_d=chunk_d,
            softmax_scale=softmax_scale,
            num_split=num_split,
            threads=threads,
            num_stages=num_stages,
            fuse_qk=fuse_qk,
            fuse_rescale=fuse_rescale,
            vector_mode=vector_mode,
            partial_dtype=partial_dtype,
            ml_width=ml_width,
        ),
        out_idx=[4, 5],
    )
    merge = tilelang.compile(
        mla_decode_merge(
            batch=batch,
            heads=heads,
            kv_len=kv_len,
            dim=dim,
            pe_dim=pe_dim,
            block_H=block_H,
            block_N=block_N,
            # The merge tiles the output columns independently of phase 1: its
            # only link to phase 1 is the O_partial layout, which is [.., dim].
            chunk_d=merge_chunk_d,
            num_split=num_split,
            rows_per_block=merge_rows,
            num_stages=merge_stages,
            partial_dtype=partial_dtype,
            ml_width=ml_width,
        ),
        out_idx=[2],
    )

    def run(q, q_pe, kv, k_pe):
        o_partial, ml = partial(q, q_pe, kv, k_pe)
        return merge(o_partial, ml)

    return run

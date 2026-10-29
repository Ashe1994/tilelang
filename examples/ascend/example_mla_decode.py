"""MLA decode forward kernel on Ascend 950 (Cube + Vector cores).

Multi-head Latent Attention decode:

    O[b,h,:] = sum_s softmax( (Q[b,h,:]·KV[b,s,:] + Q_pe[b,h,:]·K_pe[b,s,:]) * scale ) * KV[b,s,:]
    scale    = (dim + pe_dim) ** -0.5

V reuses KV. Query length is 1 per (batch, head), so the M axis of the first
GEMM is the head block.

Platform notes (measured on Ascend950PR / CANN 9.3.0):
  * block_H must be 128 -- T.dual_copy's L0C->UB path requires M=128.
  * L0B is only 64 KiB, so GEMMs take L1 operands and the compiler slices L1->L0.
  * Ascend L1 GEMM only supports NT (trans_A=False, trans_B=True), so V must enter
    L1 transposed and be split into independent buffers per output-column chunk
    (a dynamic row slice of one L1 buffer is not a valid B operand).
  * dim is chunked along the output width to bound the UB accumulator; the chunk
    loop is unrolled at Python level, so any `dim` that is a multiple of `chunk_d`
    is supported.
  * kv_len need not be a multiple of block_N: the last block is shifted back to
    end at kv_len, and columns already consumed by the previous block are masked
    to -inf before the softmax.
"""

import tilelang
import tilelang.ascend.language as T


def mla_decode_fwd(
    batch,
    heads,
    kv_len,
    dim,
    pe_dim,
    block_H=128,
    block_N=64,
    chunk_d=128,
    softmax_scale=None,
    threads=512,
    num_stages=2,
):
    """Build the MLA decode TIR program. All arguments are compile-time constants."""
    scale = float(softmax_scale) if softmax_scale is not None else (dim + pe_dim) ** -0.5
    dtype = "float16"
    accum_dtype = "float32"

    ROWS = block_H // 2  # rows handled by each of the two AIVs
    NUM_M_TILES = heads // block_H
    NUM_BLOCKS = batch * NUM_M_TILES
    KV_BLOCKS = (kv_len + block_N - 1) // block_N
    NUM_DC = dim // chunk_d

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
        O: T.Tensor((batch, heads, dim), dtype),
    ):
        with T.Kernel(NUM_BLOCKS) as bx:
            b = bx // NUM_M_TILES
            h0 = (bx % NUM_M_TILES) * block_H

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
            s_pe_ub = T.alloc_shared((ROWS, block_N), accum_dtype)
            p_ub = T.alloc_shared((ROWS, block_N), dtype)
            o_ub = T.alloc_shared((ROWS, dim), accum_dtype)
            o_tmp_ub = T.alloc_shared((ROWS, chunk_d), accum_dtype)
            o_half_ub = T.alloc_shared((ROWS, chunk_d), dtype)

            m_ub = T.alloc_shared((ROWS,), accum_dtype)
            l_ub = T.alloc_shared((ROWS,), accum_dtype)
            alpha_ub = T.alloc_shared((ROWS,), accum_dtype)

            T.copy(Q[b, h0:h0 + block_H, 0:dim], q_l1[0:block_H, 0:dim])
            T.copy(Q_pe[b, h0:h0 + block_H, 0:pe_dim], q_pe_l1[0:block_H, 0:pe_dim])

            with T.SimtVF(threads=threads):
                for r in T.Parallel(ROWS):
                    m_ub[r] = -T.infinity(accum_dtype)
                    l_ub[r] = 0.0
                for r, c in T.Parallel(ROWS, dim):
                    o_ub[r, c] = 0.0

            for kv in T.Pipelined(KV_BLOCKS, num_stages=num_stages):
                # Shift the last block back so it ends exactly at kv_len; the
                # overlapping columns are masked out below instead of reading OOB.
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
                T.dual_copy(s_l0c, s_pe_ub[0:ROWS, 0:block_N])
                T.gemm(q_l1[0:block_H, 0:dim], k_l1[0:block_N, 0:dim], s_l0c, transpose_B=True, clear_accum=True)
                T.dual_copy(s_l0c, s_ub[0:ROWS, 0:block_N])

                with T.SimtVF(threads=threads):
                    # scale, and mask columns outside [lo, hi)
                    for r, c in T.Parallel(ROWS, block_N):
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

                    for r, c in T.Parallel(ROWS, block_N):
                        s_ub[r, c] = T.exp(s_ub[r, c] - m_ub[r])

                    red_sum = T.alloc_reducer((ROWS,), accum_dtype, op="sum")
                    T.reducer_init(red_sum)
                    for r, c in T.Parallel(ROWS, block_N):
                        T.reducer_update(red_sum[r], s_ub[r, c])
                    tile_sum = T.alloc_fragment((ROWS,), accum_dtype)
                    T.finalize_reducer(red_sum, tile_sum)

                    for r in T.Parallel(ROWS):
                        l_ub[r] = l_ub[r] * alpha_ub[r] + tile_sum[r]

                    for r, c in T.Parallel(ROWS, block_N):
                        p_ub[r, c] = s_ub[r, c]
                    for r, c in T.Parallel(ROWS, dim):
                        o_ub[r, c] = o_ub[r, c] * alpha_ub[r]

                T.dual_copy(p_ub[0:ROWS, 0:block_N], p_l1[0:block_H, 0:block_N])

                # P·V over output-column chunks, unrolled at Python level.
                for i, vb in enumerate(v_l1):
                    T.gemm(p_l1[0:block_H, 0:block_N], vb, o_l0c, transpose_B=True, clear_accum=True)
                    T.dual_copy(o_l0c, o_tmp_ub[0:ROWS, 0:chunk_d])
                    with T.SimtVF(threads=threads):
                        for r, c in T.Parallel(ROWS, chunk_d):
                            o_ub[r, i * chunk_d + c] = o_ub[r, i * chunk_d + c] + o_tmp_ub[r, c]

            with T.SimtVF(threads=threads):
                for r, c in T.Parallel(ROWS, dim):
                    o_ub[r, c] = o_ub[r, c] / l_ub[r]

            for i, _vb in enumerate(v_l1):
                with T.SimtVF(threads=threads):
                    for r, c in T.Parallel(ROWS, chunk_d):
                        o_half_ub[r, c] = o_ub[r, i * chunk_d + c]
                T.dual_copy(
                    o_half_ub[0:ROWS, 0:chunk_d],
                    O[b, h0:h0 + block_H, i * chunk_d:(i + 1) * chunk_d],
                )

    return main


def compile_mla_decode(**kwargs):
    """Convenience wrapper. Backend comes from TILELANG_DEFAULT_TARGET, never here."""
    return tilelang.compile(mla_decode_fwd(**kwargs), out_idx=[4])

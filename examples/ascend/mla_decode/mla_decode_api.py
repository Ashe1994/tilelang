"""Unified entry point for the Ascend MLA decode kernels.

The operator is the same either way::

    O[b,h,:] = sum_s softmax( (Q[b,h,:] @ KV[b,s,:] + Q_pe[b,h,:] @ K_pe[b,s,:]) * scale ) * KV[b,s,:]

Only the *addressing* of the KV cache differs:

* dense  -- ``KV`` is ``[batch, kv_len, dim]`` and position ``s`` is row ``s``.
* paged  -- ``KV`` is ``[num_slots, dim]`` and ``block_table[b, p]`` names the
            physical page holding logical positions ``[p*block_size, (p+1)*block_size)``,
            with ``cache_seqlens[b]`` the live length.

Both phase-1 kernels write the same ``O_partial``/``ML`` pair and reuse the same
merge kernel, so this module only picks the phase-1 kernel and fills in the
measured-best defaults for the shape.  Callers pass the same keyword arguments
for either layout and only change the input tensors::

    # dense
    run = compile_mla_decode(batch=1, heads=128, kv_len=8192, dim=512, pe_dim=64)
    out = run(q, q_pe, kv, k_pe)

    # paged (block_size-sized pages of the DeepSeek KV cache)
    run = compile_mla_decode(
        batch=1, heads=128, dim=512, pe_dim=64,
        num_slots=kv.shape[0], block_size=64, max_pages=ceil(max_seqlen / 64),
    )
    out = run(q, q_pe, kv, k_pe, block_table, cache_seqlens)

Defaults, measured on Ascend 950PR (see ``README.md`` for the numbers):

* ``block_N=128, chunk_d=64`` when the KV axis holds a 128-token tile, else
  ``block_N=64, chunk_d=128`` for very short dense sequences.
* ``num_split=16`` (the measured optimum at ``kv_len=8192``); the split count is
  reduced to divide the KV block count (dense) or to the tile count implied by
  ``max_pages * block_size`` (paged), so short sequences do not pay for empty
  splits.
* ``threads=1024``, ``num_stages=2`` for phase 1; ``merge_rows=16``,
  ``merge_stages=4`` for the merge kernel.
* ``vector_mode="simd_rc"`` runs the PV rescale-accumulate pass in
  ``T.SimdVF`` (register-level SIMD) instead of ``T.SimtVF``, which is the
  measured-best phase-1 programming model on Ascend 950PR (e2e -10%); "simt"
  restores the original thread-parallel code for A/B testing, and the stronger
  dense-only modes are ``"simd_exp"``/``"simd"``.
* ``partial_dtype="float32"`` keeps the phase-1 -> phase-2 staging tensor in
  fp32; ``"float16"`` halves that traffic but measured no gain on Ascend 950PR
  (the merge is latency-, not byte-bound), so fp32 is the default.

``resolve_config`` performs the dispatch and defaulting without touching the
device, which keeps the policy unit-testable.
"""

from example_mla_decode_paged import compile_mla_decode_paged
from example_mla_decode_splitk import (
    compile_mla_decode_splitk,
    effective_split,
    pick_tiling,
)

# Measured-best split count for the long-KV shape; ``effective_split`` trims it
# for shorter sequences.
DEFAULT_NUM_SPLIT = 16
# The paged kernel carries one score tile per page chunk; 128 is the measured
# optimum for page sizes 16..128 on Ascend 950PR.
DEFAULT_PAGED_BLOCK_N = 128


def resolve_config(
    batch,
    heads,
    dim,
    pe_dim,
    *,
    kv_len=None,
    num_slots=None,
    block_size=None,
    max_pages=None,
    paged=None,
    softmax_scale=None,
    block_N=None,
    chunk_d=None,
    num_split=None,
    threads=1024,
    num_stages=2,
    fuse_qk=True,
    fuse_rescale=True,
    merge_rows=16,
    merge_stages=4,
    merge_chunk_d=None,
    tiles_per_core=None,
    partial_dtype="float32",
    vector_mode="simd_rc",
):
    """Resolve the dispatch and the measured-best defaults for a shape.

    ``paged`` defaults to ``True`` when any of ``num_slots``/``block_size``/
    ``max_pages`` is given.  Returns a plain dict so tests can assert the policy
    without compiling a kernel.
    """
    if paged is None:
        paged = any(v is not None for v in (num_slots, block_size, max_pages))
    if kv_len is not None and any(v is not None for v in (num_slots, block_size, max_pages)):
        raise ValueError(
            "kv_len describes a dense cache; do not mix it with "
            "num_slots/block_size/max_pages (omit the dense geometry for paged)"
        )
    scale = float(softmax_scale) if softmax_scale is not None else (dim + pe_dim) ** -0.5

    if paged:
        if num_slots is None or block_size is None or max_pages is None:
            raise ValueError(
                "paged MLA decode needs num_slots, block_size and max_pages "
                f"(got num_slots={num_slots}, block_size={block_size}, max_pages={max_pages})"
            )
        block_N = DEFAULT_PAGED_BLOCK_N if block_N is None else block_N
        tiling_block_N, tiling_chunk_d, tiling_merge_chunk_d = pick_tiling(
            max_pages * block_size, dim, block_N, chunk_d, merge_chunk_d
        )
        if num_split is None:
            num_split = effective_split(max_pages * block_size, tiling_block_N, DEFAULT_NUM_SPLIT)
    else:
        if kv_len is None:
            raise ValueError("dense MLA decode needs kv_len (or pass a paged geometry)")
        if any(v is not None for v in (num_slots, block_size, max_pages)):
            raise ValueError(
                "num_slots/block_size/max_pages select the paged layout; "
                "do not mix them with the dense kv_len"
            )
        tiling_block_N, tiling_chunk_d, tiling_merge_chunk_d = pick_tiling(
            kv_len, dim, block_N, chunk_d, merge_chunk_d
        )
        if num_split is None:
            num_split = effective_split(kv_len, tiling_block_N, DEFAULT_NUM_SPLIT)

    return {
        "paged": bool(paged),
        "batch": batch,
        "heads": heads,
        "dim": dim,
        "pe_dim": pe_dim,
        "kv_len": kv_len,
        "num_slots": num_slots,
        "block_size": block_size,
        "max_pages": max_pages,
        "softmax_scale": scale,
        "block_N": tiling_block_N,
        "chunk_d": tiling_chunk_d,
        "merge_chunk_d": tiling_merge_chunk_d,
        "num_split": num_split,
        "threads": threads,
        "num_stages": num_stages,
        "fuse_qk": fuse_qk,
        "fuse_rescale": fuse_rescale,
        "merge_rows": merge_rows,
        "merge_stages": merge_stages,
        "tiles_per_core": tiles_per_core,
        "partial_dtype": partial_dtype,
        "vector_mode": vector_mode,
    }


def compile_mla_decode(
    batch,
    heads,
    dim,
    pe_dim,
    *,
    kv_len=None,
    num_slots=None,
    block_size=None,
    max_pages=None,
    paged=None,
    softmax_scale=None,
    block_N=None,
    chunk_d=None,
    num_split=None,
    threads=1024,
    num_stages=2,
    fuse_qk=True,
    fuse_rescale=True,
    merge_rows=16,
    merge_stages=4,
    merge_chunk_d=None,
    tiles_per_core=None,
    partial_dtype="float32",
    vector_mode="simd_rc",
):
    """Compile the MLA decode kernel best suited to the shape.

    The returned callable has the layout's signature -- ``run(q, q_pe, kv,
    k_pe)`` for dense and ``run(q, q_pe, kv, k_pe, block_table, cache_seqlens)``
    for paged -- and exposes ``paged``, ``num_split``, ``block_N``,
    ``chunk_d``, ``partial_dtype`` and ``vector_mode`` for inspection.
    """
    cfg = resolve_config(
        batch,
        heads,
        dim,
        pe_dim,
        kv_len=kv_len,
        num_slots=num_slots,
        block_size=block_size,
        max_pages=max_pages,
        paged=paged,
        softmax_scale=softmax_scale,
        block_N=block_N,
        chunk_d=chunk_d,
        num_split=num_split,
        threads=threads,
        num_stages=num_stages,
        fuse_qk=fuse_qk,
        fuse_rescale=fuse_rescale,
        merge_rows=merge_rows,
        merge_stages=merge_stages,
        merge_chunk_d=merge_chunk_d,
        tiles_per_core=tiles_per_core,
        partial_dtype=partial_dtype,
        vector_mode=vector_mode,
    )

    if cfg["paged"]:
        run = compile_mla_decode_paged(
            batch=cfg["batch"],
            heads=cfg["heads"],
            num_slots=cfg["num_slots"],
            dim=cfg["dim"],
            pe_dim=cfg["pe_dim"],
            block_size=cfg["block_size"],
            max_pages=cfg["max_pages"],
            block_N=cfg["block_N"],
            chunk_d=cfg["chunk_d"],
            softmax_scale=cfg["softmax_scale"],
            num_split=cfg["num_split"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
            fuse_rescale=cfg["fuse_rescale"],
            merge_rows=cfg["merge_rows"],
            merge_stages=cfg["merge_stages"],
            merge_chunk_d=cfg["merge_chunk_d"],
            tiles_per_core=cfg["tiles_per_core"],
            partial_dtype=cfg["partial_dtype"],
            vector_mode=cfg["vector_mode"],
        )
    else:
        run = compile_mla_decode_splitk(
            batch=cfg["batch"],
            heads=cfg["heads"],
            kv_len=cfg["kv_len"],
            dim=cfg["dim"],
            pe_dim=cfg["pe_dim"],
            block_N=cfg["block_N"],
            chunk_d=cfg["chunk_d"],
            softmax_scale=cfg["softmax_scale"],
            num_split=cfg["num_split"],
            threads=cfg["threads"],
            num_stages=cfg["num_stages"],
            fuse_qk=cfg["fuse_qk"],
            fuse_rescale=cfg["fuse_rescale"],
            merge_rows=cfg["merge_rows"],
            merge_stages=cfg["merge_stages"],
            merge_chunk_d=cfg["merge_chunk_d"],
            partial_dtype=cfg["partial_dtype"],
            vector_mode=cfg["vector_mode"],
        )

    run.paged = cfg["paged"]
    run.num_split = cfg["num_split"]
    run.block_N = cfg["block_N"]
    run.chunk_d = cfg["chunk_d"]
    run.partial_dtype = cfg["partial_dtype"]
    run.vector_mode = cfg["vector_mode"]
    run.config = cfg
    return run

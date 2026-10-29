"""Confirm the shipped defaults (no flags passed) reproduce the best numbers."""

import inspect
import statistics
import sys
from pathlib import Path

import torch
from tilelang.profiler import do_bench

try:  # a CUDA-only build has no torch_npu; the benchmark itself needs the NPU
    import torch_npu  # noqa: F401
except ImportError:  # pragma: no cover
    torch_npu = None

# Importable from any working directory: kernels and tests live one level up.
PATH = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PATH))

from example_mla_decode_splitk import compile_mla_decode_splitk

B, H, S, D, PE = 1, 128, 8192, 512, 64


def make_inputs(seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn((B, H, D), generator=g).half().npu()
    q_pe = torch.randn((B, H, PE), generator=g).half().npu()
    kv = torch.randn((B, S, D), generator=g).half().npu()
    k_pe = torch.randn((B, S, PE), generator=g).half().npu()
    return q, q_pe, kv, k_pe


def ref(q, q_pe, kv, k_pe, scale):
    query = torch.cat([q.float(), q_pe.float()], dim=-1)
    key = torch.cat([kv.float(), k_pe.float()], dim=-1)
    scores = torch.einsum("bhd,bsd->bhs", query, key) * scale
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum("bhs,bsd->bhd", attn, kv.float()).to(torch.float16)


def main():
    sig = inspect.signature(compile_mla_decode_splitk)
    defaults = {k: v.default for k, v in sig.parameters.items()
                if v.default is not inspect.Parameter.empty}
    print(f"DEFAULTS {defaults}")

    scale = (D + PE) ** -0.5
    flops = 2 * B * H * S * (D + PE) + 2 * B * H * S * D
    q, q_pe, kv, k_pe = make_inputs()

    from example_mla_decode import compile_mla_decode

    base = compile_mla_decode(batch=B, heads=H, kv_len=S, dim=D, pe_dim=PE, softmax_scale=scale)
    base(q, q_pe, kv, k_pe)

    run = compile_mla_decode_splitk(
        batch=B, heads=H, kv_len=S, dim=D, pe_dim=PE, softmax_scale=scale, num_split=16
    )
    out = run(q, q_pe, kv, k_pe)
    torch.npu.synchronize()
    expected = ref(q, q_pe, kv, k_pe, scale).float()
    rel = (out.float() - expected).abs().max().item() / expected.abs().max().item()

    wall = lambda fn: do_bench(fn, backend="event", _n_warmup=10, _n_repeat=50) * 1000.0
    b = statistics.median([wall(lambda: base(q, q_pe, kv, k_pe)) for _ in range(3)])
    m = statistics.median([wall(lambda: run(q, q_pe, kv, k_pe)) for _ in range(3)])
    print(f"DEFAULTRUN rel={rel:.5f} baseline_us={b:.1f} e2e_us={m:.1f} "
          f"speedup={b / m:.2f} tflops={flops / m / 1e6:.2f}")


if __name__ == "__main__":
    main()

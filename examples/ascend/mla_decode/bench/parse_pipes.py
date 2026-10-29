"""Summarise msprof 8-CSV captures: compute vs data-movement shares.

Usage:
    python parse_pipes.py label=/path/to/OPPROF_dir [label2=/path2 ...]

For each capture it prints, for the target kernel (all blocks agree to <1%):
  * Task Duration, Block Dim, frequency
  * per-pipe BUSY time in us and as a share of the task duration
      AIC: cube (MAD), mte1 (L1->L0), mte2 (GM->L1), fixpipe (L0C->UB), scalar
      AIV: vec, mte2, mte3
  * achieved compute and bandwidth, vs the platform peaks

AIC and AIV are separate cores running concurrently, so the two groups do not sum
to 100%; within a group, the shares are fractions of that core's own cycles.
"""

import csv
import glob
import os
import re
import sys

# Ascend 950DT-class peaks quoted by the skill reference.
PEAK_CUBE_FP16_TFLOPS = 432.0
PEAK_GM_GBPS = 4000.0

D, PE = 512, 64
FLOPS = 2 * 1 * 128 * 8192 * (D + PE) + 2 * 1 * 128 * 8192 * D


def load(kernel_dir, name):
    fs = glob.glob(os.path.join(kernel_dir, name + "_*.csv")) or glob.glob(os.path.join(kernel_dir, name + ".csv"))
    if not fs:
        return []
    with open(fs[0]) as f:
        return list(csv.DictReader(f))


def num(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


def main():
    for spec in sys.argv[1:]:
        label, root = spec.split("=", 1)
        dirs = sorted(glob.glob(os.path.join(root, "OPPROF_*")))
        if not dirs:
            print(f"{label}: no OPPROF dir under {root}")
            continue
        d = dirs[-1]
        # the CSV location depends on whether --kernel-name was used: either
        # <dir>/<kernel>/0/*.csv or <dir>/*.csv
        hits = glob.glob(os.path.join(d, "**", "OpBasicInfo*.csv"), recursive=True)
        if not hits:
            print(f"{label}: no OpBasicInfo.csv under {d}")
            continue
        k = os.path.dirname(hits[0])
        target = os.path.relpath(k, d)

        ob = load(k, "OpBasicInfo")
        dur = num(ob[0], "Task Duration(us)")
        block_dim = ob[0].get("Block Dim", "?")
        freq = ob[0].get("Current Freq", "?")

        pu = load(k, "PipeUtilization")
        cube = [r for r in pu if r["sub_block_id"] == "cube0"]
        vec = [r for r in pu if r["sub_block_id"] == "vector0"]
        c = cube[0] if cube else {}
        v = vec[0] if vec else {}

        me = load(k, "Memory")
        mc = [r for r in me if r["sub_block_id"] == "cube0"]
        m = mc[0] if mc else {}

        print(f"\n===== {label}  ({target}) =====")
        print(f"  Task Duration {dur:.1f} us | Block Dim {block_dim} | Freq {freq}")

        rows = [
            ("AIC cube (MAD)", "aic_cube_ratio", "aic_cube_time(us)"),
            ("AIC mte2 (GM->L1)", "aic_mte2_ratio", "aic_mte2_time(us)"),
            ("AIC mte1 (L1->L0)", "aic_mte1_ratio", "aic_mte1_time(us)"),
            ("AIC fixpipe (L0C->UB)", "aic_fixpipe_ratio", "aic_fixpipe_time(us)"),
            ("AIC scalar", "aic_scalar_ratio", "aic_scalar_time(us)"),
            ("AIV vec", "aiv_vec_ratio", "aiv_vec_time(us)"),
            ("AIV mte3 (UB->GM)", "aiv_mte3_ratio", "aiv_mte3_time(us)"),
            ("AIV mte2 (GM->UB)", "aiv_mte2_ratio", "aiv_mte2_time(us)"),
        ]
        print(f"  {'pipe':<22}{'busy us':>9}{'of dur':>9}")
        aic_sum = aiv_sum = 0.0
        for name, rk, tk in rows:
            r = num(c if name.startswith("AIC") else v, rk)
            t = num(c if name.startswith("AIC") else v, tk)
            share = t / dur if dur else 0.0
            if name.startswith("AIC") and "cube" in name:
                aic_sum += share
            if name.startswith("AIV") and name == "AIV vec":
                aiv_sum += share
            print(f"  {name:<22}{t:9.2f}{share:9.1%}")
        rc = load(k, "ResourceConflictRatio")
        rc_c = [r for r in rc if r["sub_block_id"] == "cube0"]
        rc_v = [r for r in rc if r["sub_block_id"] == "vector0"]
        if rc_c and rc_v:
            print(f"  waits  : aiv_vec_wait {num(rc_v[0], 'aiv_vec_wait_ratio'):.1%}  "
                  f"aic_cube_wait {num(rc_c[0], 'aic_cube_wait_ratio'):.1%}  "
                  f"aic_mte1_wait {num(rc_c[0], 'aic_mte1_wait_ratio'):.1%}  "
                  f"aic_mte2_wait {num(rc_c[0], 'aic_mte2_wait_ratio'):.1%}")
        print(f"  -> compute (cube) = {num(c, 'aic_cube_time(us)') / dur:.1%} of duration;  "
              f"vector = {num(v, 'aiv_vec_time(us)') / dur:.1%}")

        # achieved compute / bandwidth
        tflops = FLOPS / dur / 1e6
        l1_read = num(m, "aic_l1_read_bw(GB/s)")
        mte2_bw = num(c, "aic_mte2_active_bw(GB/s)")
        rd_kb = num(m, "read_main_memory_datas(KB)")
        wr_kb = num(m, "write_main_memory_datas(KB)")
        # whole-kernel traffic: every block reads its own share
        blocks = int(block_dim) if str(block_dim).isdigit() else 1
        gbps = (rd_kb + wr_kb) * 1024 * blocks / (dur * 1e-6) / 1e9
        print(f"  compute: {tflops:6.1f} TFLOPS  ({tflops / PEAK_CUBE_FP16_TFLOPS:.1%} of "
              f"{PEAK_CUBE_FP16_TFLOPS:.0f} TFLOPS peak)")
        print(f"  memory : per-block read {rd_kb:.0f} KB + write {wr_kb:.0f} KB, "
              f"x{blocks} blocks -> {gbps:.0f} GB/s ({gbps / PEAK_GM_GBPS:.1%} of "
              f"{PEAK_GM_GBPS:.0f} GB/s peak)")
        print(f"           aic mte2 active bw {mte2_bw:.0f} GB/s, L1 read bw {l1_read:.0f} GB/s")


if __name__ == "__main__":
    main()

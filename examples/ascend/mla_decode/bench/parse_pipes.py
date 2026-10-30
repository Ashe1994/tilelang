"""Summarise msprof 8-CSV captures: compute, data-movement and cache shares.

Usage:
    python parse_pipes.py label=/path/to/OPPROF_dir [label2=/path2 ...]

For each capture it prints, for the target kernel (all blocks agree to <1%):
  * Task Duration, Block Dim, frequency
  * per-pipe BUSY time in us and as a share of the task duration
      AIC: cube (MAD), mte1 (L1->L0), mte2 (GM->L1), fixpipe (L0C->UB), scalar
      AIV: vec, mte3 (UB->GM), mte2 (GM->UB)
  * wait ratios, L2 read hit rate, GM->L1 / GM->UB traffic and bandwidth usage,
    L0 bandwidth and the MTE instruction counts
  * achieved compute and bandwidth, vs the platform peaks

AIC and AIV are separate cores running concurrently, so the two groups do not sum
to 100%; within a group, the shares are fractions of that core's own cycles.

CSV fields used: OpBasicInfo (duration/freq), PipeUtilization (pipe ratios),
Memory (GM/L1 traffic + bw), MemoryL0 (L0 bw), L2Cache (hit rate),
ResourceConflictRatio (wait ratios).  All are produced by
``msprof op --aic-metrics=PipeUtilization,Memory,MemoryL0,L2Cache,ResourceConflictRatio``.
"""

import csv
import glob
import os
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


def has(row, key):
    return row.get(key, "NA") not in ("", "NA")


def pct(row, key):
    return num(row, key) / 100.0


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
        mv = [r for r in me if r["sub_block_id"] == "vector0"]
        m = mc[0] if mc else (mv[0] if mv else {})

        l0 = load(k, "MemoryL0")
        l0c = [r for r in l0 if r["sub_block_id"] == "cube0"]
        l0r = l0c[0] if l0c else {}

        l2 = load(k, "L2Cache")
        l2c = [r for r in l2 if r["sub_block_id"] == "cube0"]
        l2v = [r for r in l2 if r["sub_block_id"] == "vector0"]
        l2r = l2c[0] if l2c else (l2v[0] if l2v else {})

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
        for name, rk, tk in rows:
            r = c if name.startswith("AIC") else v
            t = num(r, tk)
            print(f"  {name:<22}{t:9.2f}{t / dur if dur else 0:9.1%}")

        rc = load(k, "ResourceConflictRatio")
        rc_c = [r for r in rc if r["sub_block_id"] == "cube0"]
        rc_v = [r for r in rc if r["sub_block_id"] == "vector0"]
        wait_bits = []
        if rc_v:
            wait_bits += [
                f"aiv_vec_wait {num(rc_v[0], 'aiv_vec_wait_ratio'):.1%}",
                f"aiv_mte2_wait {num(rc_v[0], 'aiv_mte2_wait_ratio'):.1%}",
                f"aiv_mte3_wait {num(rc_v[0], 'aiv_mte3_wait_ratio'):.1%}",
            ]
        if rc_c:
            wait_bits += [
                f"aic_cube_wait {num(rc_c[0], 'aic_cube_wait_ratio'):.1%}",
                f"aic_mte1_wait {num(rc_c[0], 'aic_mte1_wait_ratio'):.1%}",
                f"aic_mte2_wait {num(rc_c[0], 'aic_mte2_wait_ratio'):.1%}",
            ]
        if wait_bits:
            print("  waits  : " + "  ".join(wait_bits))

        if has(l2r, "aic_read_hit_rate(%)"):
            print(
                f"  L2 read hit: aic {pct(l2r, 'aic_read_hit_rate(%)'):.1%} "
                f"(close hit/miss {num(l2r, 'aic_read_close_hit'):.0f}/{num(l2r, 'aic_read_close_miss'):.0f}, "
                f"far hit/miss {num(l2r, 'aic_read_far_hit'):.0f}/{num(l2r, 'aic_read_far_miss'):.0f})"
            )
        if has(l2r, "aiv_read_hit_rate(%)") and num(l2r, "aiv_read_hit_rate(%)") > 0:
            print(f"               aiv {pct(l2r, 'aiv_read_hit_rate(%)'):.1%}")

        if has(m, "GM_to_L1_datas(KB)"):
            print(
                f"  GM->L1: {num(m, 'GM_to_L1_datas(KB)'):.0f} KB/block, pipe bw usage "
                f"{num(m, 'GM_to_L1_bw_usage_rate(%)'):.1f}% | main-mem read bw "
                f"{num(m, 'aic_main_mem_read_bw(GB/s)'):.0f} GB/s | mte2 inst "
                f"{num(m, 'aic_mte2_instructions'):.0f} (busy {num(m, 'aic_mte2_ratio'):.1%}), "
                f"mte1 inst {num(m, 'aic_mte1_instructions'):.0f}"
            )
        if has(m, "GM_to_UB_datas(KB)"):
            print(
                f"  GM->UB: {num(m, 'GM_to_UB_datas(KB)'):.0f} KB/block, pipe bw usage "
                f"{num(m, 'GM_to_UB_bw_usage_rate(%)'):.1f}% | main-mem read bw "
                f"{num(m, 'aiv_main_mem_read_bw(GB/s)'):.0f} GB/s | mte2 inst "
                f"{num(m, 'aiv_mte2_instructions'):.0f}"
            )
        if has(l0r, "aic_l0c_write_bw_cube(GB/s)"):
            print(
                f"  L0 bw: A {num(l0r, 'aic_l0a_read_bw(GB/s)'):.0f}/"
                f"{num(l0r, 'aic_l0a_write_bw(GB/s)'):.0f}, B {num(l0r, 'aic_l0b_read_bw(GB/s)'):.0f}/"
                f"{num(l0r, 'aic_l0b_write_bw(GB/s)'):.0f}, C {num(l0r, 'aic_l0c_read_bw_cube(GB/s)'):.0f}/"
                f"{num(l0r, 'aic_l0c_write_bw_cube(GB/s)'):.0f} GB/s (read/write)"
            )

        aic_busy = sum(
            num(c, key)
            for key in ("aic_cube_time(us)", "aic_mte2_time(us)", "aic_mte1_time(us)",
                        "aic_fixpipe_time(us)", "aic_scalar_time(us)")
        )
        aiv_busy = sum(
            num(v, key) for key in ("aiv_vec_time(us)", "aiv_mte2_time(us)", "aiv_mte3_time(us)")
        )
        print(f"  group busy: AIC {aic_busy / dur if dur else 0:.1%} of dur, "
              f"AIV {aiv_busy / dur if dur else 0:.1%}")

        rd_kb = num(m, "read_main_memory_datas(KB)")
        wr_kb = num(m, "write_main_memory_datas(KB)")
        blocks = int(block_dim) if str(block_dim).isdigit() else 1
        gbps = (rd_kb + wr_kb) * 1024 * blocks / (dur * 1e-6) / 1e9
        tflops = FLOPS / dur / 1e6
        print(f"  compute: {tflops:6.1f} TFLOPS  ({tflops / PEAK_CUBE_FP16_TFLOPS:.1%} of "
              f"{PEAK_CUBE_FP16_TFLOPS:.0f} TFLOPS peak)")
        print(f"  memory : per-block read {rd_kb:.0f} KB + write {wr_kb:.0f} KB, "
              f"x{blocks} blocks -> {gbps:.0f} GB/s ({gbps / PEAK_GM_GBPS:.1%} of "
              f"{PEAK_GM_GBPS:.0f} GB/s peak)")


if __name__ == "__main__":
    main()

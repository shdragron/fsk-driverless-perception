"""Collate the PnP evaluation JSONs into the comparison tables the ablation was run to produce.

Two questions:
  1. Does dropping keypoints hurt the 3D pose, even though it flatters the pose-mAP number?
  2. Does it hurt *more* as the image degrades -- i.e. is the reduced model less robust?

Reports median and p90 relative depth error. The p90 matters more than the median here: a
perception stack fails on its tail, not its average.
"""
import argparse
import json
from pathlib import Path

KINDS = ["blur", "noise", "dark"]
LEVELS = ["0.25", "0.5", "1.0"]
BUCKETS = ["near", "mid", "far", "all"]


def load(results_dir, n, tag):
    p = results_dir / f"{n}kpt_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("metrics", {})


def fmt(metrics, bucket, key):
    if not metrics or bucket not in metrics:
        return "  --  "
    return f"{metrics[bucket][key] * 100:5.1f}%"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="/data/brt_cone_pose/results", type=Path)
    ap.add_argument("--kpts", default="8,6,4")
    args = ap.parse_args()
    kpts = [int(k) for k in args.kpts.split(",")]

    print("=" * 72)
    print("CLEAN IMAGES -- depth error by distance (median / p90)")
    print("=" * 72)
    print(f"{'model':<8}" + "".join(f"{b:>16}" for b in BUCKETS))
    for n in kpts:
        m = load(args.results, n, "clean")
        cells = "".join(f"  {fmt(m, b, 'median_rel_err')} /{fmt(m, b, 'p90_rel_err')}" for b in BUCKETS)
        print(f"{n}kpt   {cells}")

    print()
    print("=" * 72)
    print("ROBUSTNESS -- depth error (all cones) as the image degrades")
    print("=" * 72)
    for kind in KINDS:
        print(f"\n{kind.upper()}")
        print(f"{'model':<8}{'clean':>14}" + "".join(f"{l:>14}" for l in LEVELS))
        for n in kpts:
            clean = load(args.results, n, "clean")
            row = f"{n}kpt   {fmt(clean, 'all', 'median_rel_err')} /{fmt(clean, 'all', 'p90_rel_err')}"
            for lvl in LEVELS:
                m = load(args.results, n, f"{kind}_{lvl}")
                row += f"  {fmt(m, 'all', 'median_rel_err')} /{fmt(m, 'all', 'p90_rel_err')}"
            print(row)

    print()
    print("=" * 72)
    print("DEGRADATION -- how much p90 error grows from clean to worst corruption")
    print("=" * 72)
    print(f"{'model':<8}" + "".join(f"{k:>12}" for k in KINDS))
    for n in kpts:
        clean = load(args.results, n, "clean")
        if not clean or "all" not in clean:
            continue
        base = clean["all"]["p90_rel_err"]
        cells = ""
        for kind in KINDS:
            m = load(args.results, n, f"{kind}_1.0")
            if m and "all" in m:
                cells += f"{m['all']['p90_rel_err'] / base:>11.2f}x"
            else:
                cells += f"{'--':>12}"
        print(f"{n}kpt   {cells}")

    # Error alone flatters a degraded model: cones the detector misses never enter the error
    # stats at all. Under noise@0.5 the 8kpt model lost 74% of its far cones, and the survivors
    # are the easy ones. Track how many cones each condition still measures.
    print()
    print("=" * 72)
    print("SURVIVING CONES -- share of clean-image cones still measured (detection + PnP)")
    print("=" * 72)
    print(f"{'model':<8}{'clean':>9}" + "".join(f"{k + '@1.0':>12}" for k in KINDS))
    for n in kpts:
        clean = load(args.results, n, "clean")
        if not clean or "all" not in clean:
            continue
        base_n = clean["all"]["n"]
        cells = ""
        for kind in KINDS:
            m = load(args.results, n, f"{kind}_1.0")
            cells += f"{100 * m['all']['n'] / base_n:>11.0f}%" if m and "all" in m else f"{'--':>12}"
        print(f"{n}kpt   {base_n:>9,}{cells}")


if __name__ == "__main__":
    main()

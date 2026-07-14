"""Plot the keypoint-count ablation: pose-mAP says fewer keypoints is better, PnP says the
opposite, and the gap widens as the image degrades.

Three panels:
  1. the metric inversion -- pose mAP vs PnP depth error, ranked oppositely
  2. robustness -- depth error as noise and motion blur increase
  3. survivorship -- how many cones are still detected at all (the error curves above are
     computed only over survivors, so they flatter the worst conditions)
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KPTS = [8, 6, 4]
COLORS = {8: "#2166ac", 6: "#7b3294", 4: "#d6604d"}
LEVELS = ["0.25", "0.5", "1.0"]


def load_metric(results, n, tag):
    p = results / f"{n}kpt_{tag}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())["metrics"].get("all")


def load_map(n):
    p = Path(f"/home/moon/runs/pose/cone-pose-{n}kpt/results.csv")
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p)))
    return float({k.strip(): v for k, v in rows[-1].items()}["metrics/mAP50-95(P)"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", default="/data/brt_cone_pose/results", type=Path)
    ap.add_argument("--out", default="/data/brt_cone_pose/results/ablation.png", type=Path)
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # --- 1. the inversion ---
    ax = axes[0]
    maps = [load_map(n) for n in KPTS]
    errs = [(load_metric(args.results, n, "clean") or {}).get("median_rel_err") for n in KPTS]
    x = range(len(KPTS))
    ax.plot(x, [m * 100 if m else None for m in maps], "o-", color="#4393c3", lw=2, label="pose mAP50-95 (%)")
    ax2 = ax.twinx()
    ax2.plot(x, [e * 100 if e else None for e in errs], "s--", color="#d6604d", lw=2, label="PnP depth error (%)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n}kpt" for n in KPTS])
    ax.set_ylabel("pose mAP50-95 (%)", color="#4393c3")
    ax2.set_ylabel("median depth error (%)", color="#d6604d")
    ax.set_title("Fewer keypoints score better on mAP,\nbut recover worse 3D depth")
    ax.grid(alpha=0.3)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center left", fontsize=8)

    # --- 2. robustness ---
    ax = axes[1]
    for kind, style in (("noise", "-"), ("blur", "--")):
        for n in KPTS:
            xs, ys = [0], []
            clean = load_metric(args.results, n, "clean")
            ys.append(clean["median_rel_err"] * 100 if clean else None)
            for i, lvl in enumerate(LEVELS, 1):
                m = load_metric(args.results, n, f"{kind}_{lvl}")
                if m:
                    xs.append(i)
                    ys.append(m["median_rel_err"] * 100)
            ax.plot(xs, ys, style, color=COLORS[n], marker="o", ms=4,
                    label=f"{n}kpt {kind}" if kind == "noise" else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["clean", "0.25", "0.5", "1.0"])
    ax.set_xlabel("corruption level  (solid = noise, dashed = motion blur)")
    ax.set_ylabel("median depth error (%)")
    ax.set_title("Depth error under degradation\n(the gap widens as it gets worse)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --- 3. survivorship ---
    ax = axes[2]
    for kind, style in (("noise", "-"), ("blur", "--")):
        for n in KPTS:
            clean = load_metric(args.results, n, "clean")
            if not clean:
                continue
            base = clean["n"]
            xs, ys = [0], [100.0]
            for i, lvl in enumerate(LEVELS, 1):
                m = load_metric(args.results, n, f"{kind}_{lvl}")
                if m:
                    xs.append(i)
                    ys.append(100 * m["n"] / base)
            ax.plot(xs, ys, style, color=COLORS[n], marker="o", ms=4,
                    label=f"{n}kpt {kind}" if kind == "noise" else None)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["clean", "0.25", "0.5", "1.0"])
    ax.set_xlabel("corruption level")
    ax.set_ylabel("cones still detected (%)")
    ax.set_title("Survivorship: error curves above only\ncount cones the detector still finds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

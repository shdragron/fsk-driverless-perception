"""Approximate the MIT original's 7-keypoint layout from BRT's 8.

RektNet's layout is an apex plus three left/right pairs:

        0            apex
      1   2          mid_L_top, mid_R_top
      3   4          mid_L_bot, mid_R_bot
      5   6          bot_L,     bot_R

BRT has no apex -- it pairs every level. The apex is therefore *synthesised* as the midpoint of
BRT's top pair (kpt0, kpt1), which sits on the cone's centreline but slightly below the true tip.

This is an approximation and it is not free: the synthetic apex carries the localisation error of
two other keypoints, and it is not where an annotator would have put the tip. The MIT-original row
in the results is worth reading with that in mind -- it reproduces the *architecture*, not the
labelling.

The remaining six map straight across: BRT's mid pair -> RektNet's mid_top, BRT's extra pair (large
cones) or base pair -> mid_bot, BRT's base pair -> bot. Large-cone-only points are dropped, since
the original has no slot for them.
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

# BRT index -> RektNet slot. The apex (slot 0) is synthesised, not copied.
#   RektNet 1,2 = mid_L_top, mid_R_top   <- BRT 2,3 (mid pair)
#   RektNet 3,4 = mid_L_bot, mid_R_bot   <- BRT 6,7 if present, else interpolated
#   RektNet 5,6 = bot_L, bot_R           <- BRT 4,5 (base pair)
CLASSES = {0: "BLUE", 1: "ORANGE_BIG", 2: "ORANGE", 3: "UNDEFINED", 4: "YELLOW"}
# Under a horizontal flip: apex maps to itself, each L/R pair swaps.
FLIP_IDX = [0, 2, 1, 4, 3, 6, 5]


def to_7kpt(kpts):
    """(8, 3) BRT keypoints -> (7, 3) RektNet-style. Returns None if the cone is unusable."""
    xy, vis = kpts[:, :2], kpts[:, 2]

    # Apex: midpoint of the top pair. Needs both to exist.
    if vis[0] == 0 or vis[1] == 0:
        return None
    apex = (xy[0] + xy[1]) / 2

    out = np.zeros((7, 3), dtype=np.float64)
    out[0] = [apex[0], apex[1], 2]
    out[1], out[2] = [*xy[2], vis[2]], [*xy[3], vis[3]]   # mid pair -> mid_top

    # mid_bot: the extra pair on large cones; on small cones there is no such stripe boundary, so
    # interpolate halfway between mid and base rather than invent a landmark.
    if vis[6] > 0 and vis[7] > 0:
        out[3], out[4] = [*xy[6], vis[6]], [*xy[7], vis[7]]
    else:
        for slot, (mid, base) in ((3, (2, 4)), (4, (3, 5))):
            if vis[mid] == 0 or vis[base] == 0:
                return None
            out[slot] = [*(xy[mid] + xy[base]) / 2, 2]

    out[5], out[6] = [*xy[4], vis[4]], [*xy[5], vis[5]]   # base pair -> bot

    if (out[:, 2] == 0).any():
        return None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="brt-clean-8kpt root")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    total = dropped = 0
    for split in ("train", "val", "test"):
        src_img = args.source / "images" / split
        src_lbl = args.source / "labels" / split
        if not src_img.is_dir():
            sys.exit(f"missing {src_img}")

        dst_img = args.out / "images" / split
        dst_lbl = args.out / "labels" / split
        for d in (dst_img, dst_lbl):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)

        for img in sorted(src_img.iterdir()):
            (dst_img / img.name).symlink_to(img.resolve())

        for lbl in sorted(src_lbl.glob("*.txt")):
            lines = []
            for line in lbl.read_text().split("\n"):
                if not line.strip():
                    continue
                t = [float(x) for x in line.split()]
                total += 1
                seven = to_7kpt(np.array(t[5:]).reshape(8, 3))
                if seven is None:
                    dropped += 1
                    continue
                vals = [f"{v:.6f}" for v in t[1:5]]
                for x, y, v in seven:
                    vals += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
                lines.append(f"{int(t[0])} " + " ".join(vals))
            (dst_lbl / lbl.name).write_text("\n".join(lines) + ("\n" if lines else ""))

    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "kpt_shape: [7, 3]\n"
        f"flip_idx: {FLIP_IDX}\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in CLASSES.items())
    )

    print(f"{total - dropped:,} cones written, {dropped:,} dropped "
          f"({100 * dropped / max(total, 1):.1f}% -- top pair missing)")
    print(f"wrote {args.out}/data.yaml")


if __name__ == "__main__":
    main()

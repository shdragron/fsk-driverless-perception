"""Build a RektNet-style cone-crop dataset from BRT Cone Pose, so RektNet and YOLO-pose
can be compared on identical data.

RektNet consumes a single cone cropped to 80x80 and regresses keypoints inside that crop;
BRT ships full frames with boxes plus keypoints. This crops each GT box out and rewrites the
keypoints into crop-local pixel coordinates, producing the CSV layout RektNet's loader wants.

Crops are cut from the same frames and splits the YOLO-pose models see, so the two pipelines are
compared on identical cones.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

# BRT keypoint order, with the large-cone-only pair last.
BRT_NAMES = ["top_L", "top_R", "mid_L", "mid_R", "bot_L", "bot_R", "extra_L", "extra_R"]
# The MIT/RektNet layout produced by make_mit_7kpt: an apex plus three left/right pairs.
MIT_NAMES = ["apex", "mid_L_top", "mid_R_top", "mid_L_bot", "mid_R_bot", "bot_L", "bot_R"]
LARGE_CLASS_ID = 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--brt-root", required=True, type=Path, help="brt-cone-pose-11k root")
    p.add_argument("--out", required=True, type=Path, help="Output dir for crops + CSVs")
    p.add_argument("--n-kpt", default=8, type=int, choices=[4, 6, 7, 8],
                   help="Keypoints to keep, matching the YOLO-pose variant being compared")
    p.add_argument("--pad", default=0.10, type=float,
                   help="Box padding, mimicking a real detector's slightly loose boxes")
    p.add_argument("--min-size", default=16, type=int,
                   help="Skip cones smaller than this; below it the crop is mostly interpolation")
    args = p.parse_args()

    # 7 comes from make_mit_7kpt, whose labels already hold exactly the 7 RektNet points -- a
    # different layout, not a subset of BRT's, so it gets its own names.
    keep = {8: list(range(8)), 7: list(range(7)), 6: list(range(6)),
            4: [0, 1, 4, 5]}[args.n_kpt]
    names = MIT_NAMES if args.n_kpt == 7 else [BRT_NAMES[i] for i in keep]

    for split in ("train", "val", "test"):
        img_dir = args.brt_root / "images" / split
        lbl_dir = args.brt_root / "labels" / split
        crop_dir = args.out / "images" / split
        crop_dir.mkdir(parents=True, exist_ok=True)

        rows, skipped_small, skipped_invisible = [], 0, 0
        for lbl_path in sorted(lbl_dir.glob("*.txt")):
            img_path = next(img_dir.glob(f"{lbl_path.stem}.*"), None)
            if img_path is None:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            H, W = img.shape[:2]

            for idx, line in enumerate(lbl_path.read_text().split("\n")):
                if not line.strip():
                    continue
                t = [float(x) for x in line.split()]
                cls = int(t[0])
                cx, cy, bw, bh = np.array(t[1:5]) * [W, H, W, H]
                kpts = np.array(t[5:]).reshape(-1, 3)[keep]

                # Keep the visibility flags rather than dropping partially-annotated cones. kpt6/7
                # only exist on large orange cones (5.3% of the set) -- a small cone simply has no
                # fourth stripe boundary. Requiring all 8 would throw away 94.7% of the data, so
                # instead the flags are carried through and the loss masks the absent points.
                if (kpts[:, 2] > 0).sum() < 4:
                    skipped_invisible += 1  # too few points left to supervise anything
                    continue

                pw, ph = bw * (1 + args.pad), bh * (1 + args.pad)
                x1, y1 = int(round(cx - pw / 2)), int(round(cy - ph / 2))
                x2, y2 = int(round(cx + pw / 2)), int(round(cy + ph / 2))
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < args.min_size or y2 - y1 < args.min_size:
                    skipped_small += 1
                    continue

                crop = img[y1:y2, x1:x2]
                # Keypoints are absolute in the full frame; re-express them inside the crop.
                local = kpts[:, :2] * [W, H] - [x1, y1]
                vis = kpts[:, 2]

                # A *visible* keypoint outside its own box is a bad label; an invisible one is
                # just padding and its coordinates are meaningless, so only check the real ones.
                seen = vis > 0
                if seen.any() and (
                    (local[seen] < 0).any()
                    or (local[seen, 0] > x2 - x1).any()
                    or (local[seen, 1] > y2 - y1).any()
                ):
                    continue

                name = f"{lbl_path.stem}_{idx}.jpg"
                cv2.imwrite(str(crop_dir / name), crop)
                row = {"image": name, "cls": cls, "is_large": int(cls == LARGE_CLASS_ID)}
                for n, (x, y), v in zip(names, local, vis):
                    row[n] = f"[{x:.2f}, {y:.2f}]"
                    row[f"{n}_vis"] = int(v)
                rows.append(row)

        csv_path = args.out / f"{split}.csv"
        cols = ["image", "cls", "is_large"]
        for n in names:
            cols += [n, f"{n}_vis"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"{split:<6} {len(rows):>7,} crops  "
              f"(skipped: {skipped_small:,} too small, {skipped_invisible:,} occluded kpts)  -> {csv_path}")


if __name__ == "__main__":
    main()

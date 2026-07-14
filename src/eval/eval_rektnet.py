"""Evaluate RektNet with the same PnP depth metric used for the YOLO-pose models.

RektNet only does keypoints -- it is handed a cone crop and never has to find the cone. To keep
the comparison honest, it is fed crops cut from ground-truth boxes, i.e. a perfect detector.
That is a real advantage over YOLO-pose, which must detect and localise in one shot, so any
YOLO-pose win here is a win against a favourably-handicapped RektNet.

Two robustness axes:
  --corrupt    degrades the full frame before cropping (same corruptions as the YOLO-pose eval)
  --box-noise  jitters the GT box, standing in for a real detector's imperfect output -- the
               failure mode a two-stage pipeline has and a one-stage one does not
"""
import argparse
import ast
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

# The cone's 3D model is imported, not redefined: if RektNet and YOLO-pose were scored against
# even slightly different object points, the comparison between them would be meaningless.
from src.eval.eval_pose import (LARGE_CLASS_ID, cone_object_points, corrupt_image,
                                distance_bucket, solve_depth)
from src.models.keypoint_net import KeypointNet

INPUT_SIZE = (80, 80)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--num-kpt", required=True, type=int, choices=[4, 6, 8])
    ap.add_argument("--brt-root", required=True, type=Path, help="Full frames + YOLO labels")
    ap.add_argument("--corrupt", default="none",
                    choices=["none", "blur", "noise", "dark",
                             "sun", "overcast", "shadow", "backlight"])
    ap.add_argument("--level", default=0.0, type=float)
    ap.add_argument("--box-noise", default=0.0, type=float,
                    help="Jitter the GT box by this fraction, imitating detector error")
    ap.add_argument("--pad", default=0.10, type=float, help="Must match the training crops")
    ap.add_argument("--focal", default=1000.0, type=float)
    ap.add_argument("--limit", default=0, type=int)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu")
    ck = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = KeypointNet(num_kpt=args.num_kpt, image_size=INPUT_SIZE).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    rng = np.random.default_rng(17)
    obj_small = cone_object_points(args.num_kpt, "small")
    obj_large = cone_object_points(args.num_kpt, "large")

    img_dir = args.brt_root / "images" / "test"
    lbl_dir = args.brt_root / "labels" / "test"
    images = sorted(img_dir.iterdir())
    if args.limit:
        images = images[: args.limit]

    errors, pnp_failures = defaultdict(list), 0
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        img = corrupt_image(img, args.corrupt, args.level)

        crops, metas = [], []
        for line in lbl_path.read_text().split("\n"):
            if not line.strip():
                continue
            t = [float(x) for x in line.split()]
            cls = int(t[0])
            kpts = np.array(t[5:]).reshape(-1, 3)[: args.num_kpt]

            # Feed PnP only the keypoints this cone actually has. Skipping any cone with an
            # absent keypoint would, at 8kpt, discard the 94.7% of cones without kpt6/7 -- and
            # those are the small, distant, hard ones. RektNet would then be scored on an easier
            # set than YOLO-pose, and the two numbers would not be comparable.
            valid = kpts[:, 2] > 0
            if valid.sum() < 4:
                continue  # PnP needs four correspondences

            cx, cy, bw, bh = np.array(t[1:5]) * [W, H, W, H]
            gt_px = kpts[valid, :2] * [W, H]
            obj_full = obj_large if cls == LARGE_CLASS_ID else obj_small
            obj = obj_full[valid]
            gt_depth = solve_depth(obj, gt_px, np.array(
                [[args.focal, 0, W / 2], [0, args.focal, H / 2], [0, 0, 1]], dtype=np.float64))
            if gt_depth is None:
                continue

            # The crop the detector would hand over -- optionally mis-placed.
            jx, jy, js = 0.0, 0.0, 1.0
            if args.box_noise > 0:
                jx = rng.normal(0, args.box_noise * 0.15) * bw
                jy = rng.normal(0, args.box_noise * 0.15) * bh
                js = 1 + rng.normal(0, args.box_noise * 0.15)
            pw, ph = bw * (1 + args.pad) * js, bh * (1 + args.pad) * js
            x1 = int(round(cx + jx - pw / 2)); y1 = int(round(cy + jy - ph / 2))
            x2 = int(round(cx + jx + pw / 2)); y2 = int(round(cy + jy + ph / 2))
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 16 or y2 - y1 < 16:
                continue

            crop = cv2.resize(img[y1:y2, x1:x2], INPUT_SIZE)
            crops.append(crop.transpose(2, 0, 1) / 255.0)
            metas.append((cls, obj, valid, gt_depth, (x1, y1, x2 - x1, y2 - y1), bh))

        if not crops:
            continue

        with torch.no_grad():
            batch = torch.from_numpy(np.stack(crops)).float().to(device)
            _, pred = model(batch)
        pred = pred.cpu().numpy()

        for (cls, obj, valid, gt_depth, (x1, y1, cw, ch), bh), pts in zip(metas, pred):
            # Keypoints come back normalized to the crop; map them back to full-frame pixels, and
            # use the same subset the reference pose was solved from.
            px = (pts * [cw, ch] + [x1, y1])[valid]
            K = np.array([[args.focal, 0, W / 2], [0, args.focal, H / 2], [0, 0, 1]], dtype=np.float64)
            pd_depth = solve_depth(obj, px, K)
            if pd_depth is None:
                pnp_failures += 1
                continue
            rel = abs(pd_depth - gt_depth) / gt_depth
            errors[distance_bucket(bh)].append(rel)
            errors["all"].append(rel)

    tag = f"rektnet-{args.num_kpt}kpt"
    if args.corrupt != "none":
        tag += f" / {args.corrupt}@{args.level}"
    if args.box_noise:
        tag += f" / box-noise {args.box_noise}"
    print(f"\n=== {tag} ===")
    out = {}
    for b in ("near", "mid", "far", "all"):
        e = np.array(errors[b])
        if not len(e):
            continue
        out[b] = {"n": len(e), "median_rel_err": float(np.median(e)),
                  "p90_rel_err": float(np.percentile(e, 90))}
        print(f"  {b:<5} n={len(e):>6,}  median={np.median(e)*100:5.1f}%  p90={np.percentile(e,90)*100:6.1f}%")
    if pnp_failures:
        print(f"  PnP failures: {pnp_failures:,}")
    out["pnp_failures"] = pnp_failures

    if args.out:
        args.out.write_text(json.dumps(
            {"model": "rektnet", "n_kpt": args.num_kpt, "corrupt": args.corrupt,
             "level": args.level, "box_noise": args.box_noise, "metrics": out}, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

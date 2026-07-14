"""Zoom in on individual cones so the keypoint predictions are actually legible.

The full-frame view shows which cones get detected at all, but the cones are ~40px tall in a
2344px frame, so the keypoints themselves are invisible. This crops a few representative cones
(near / mid / far) and blows them up side by side across the three models, with the PnP depth
error each one produces.

Hollow circles are ground truth, filled are predictions, and the grey line between them is the
error. What to look for: the 4kpt model's individual points often sit *closer* to truth (that is
its higher pose-mAP), yet its depth error is worse -- because with no spare correspondences,
PnP has nothing to average the remaining error against.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.eval.eval_pose import (LARGE_CLASS_ID, cone_object_points, corrupt_image, solve_depth)

KPT_COLORS = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
              (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]
ZOOM = 6  # cones are tiny; blow them up


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--condition", default="clean", help="clean | noise_0.5 | blur_1.0 ...")
    ap.add_argument("--out", default="/data/brt_cone_pose/results/zoom.jpg", type=Path)
    ap.add_argument("--focal", default=1000.0, type=float)
    ap.add_argument("--n-cones", default=4, type=int)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path("/data/brt_cone_pose/brt-cone-pose-11k")
    best, best_n = None, 0
    for lbl in sorted((root / "labels" / "test").glob("*.txt"))[:200]:
        n = len([l for l in lbl.read_text().split("\n") if l.strip()])
        if n > best_n:
            best, best_n = lbl, n
    img_path = next((root / "images" / "test").glob(f"{best.stem}.*"))

    img0 = cv2.imread(str(img_path))
    H, W = img0.shape[:2]
    K = np.array([[args.focal, 0, W / 2], [0, args.focal, H / 2], [0, 0, 1]], dtype=np.float64)

    if args.condition == "clean":
        kind, lvl = "none", 0.0
    else:
        kind, lvl = args.condition.rsplit("_", 1)
        lvl = float(lvl)
    img = corrupt_image(img0.copy(), kind, lvl)

    gt = []
    for line in best.read_text().split("\n"):
        if not line.strip():
            continue
        t = [float(x) for x in line.split()]
        k = np.array(t[5:]).reshape(-1, 3)
        gt.append((int(t[0]), np.array(t[1:5]), k[:, :2], k[:, 2]))

    # Sample cones across the size range: a big near one, a tiny far one, and two between.
    gt.sort(key=lambda g: -g[1][3])
    idxs = np.linspace(0, len(gt) - 1, args.n_cones).astype(int)
    chosen = [gt[i] for i in idxs]

    models = {}
    for n in (8, 6, 4):
        w = Path(f"/home/moon/runs/pose/cone-pose-{n}kpt/weights/best.pt")
        if w.exists():
            models[n] = YOLO(str(w))

    preds = {}
    for n, model in models.items():
        res = model.predict(img, verbose=False, device="cpu")[0]
        preds[n] = (res.boxes.xywh.cpu().numpy() if len(res.boxes) else np.zeros((0, 4)),
                    res.keypoints.xy.cpu().numpy() if res.keypoints is not None else np.zeros((0, n, 2)))

    rows = []
    for cls, box, kpts_n, vis in chosen:
        cx, cy, bw, bh = box * [W, H, W, H]
        pad = max(bw, bh) * 0.35
        x1, y1 = int(cx - bw / 2 - pad), int(cy - bh / 2 - pad)
        x2, y2 = int(cx + bw / 2 + pad), int(cy + bh / 2 + pad)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)

        tiles = []
        for n, model in models.items():
            crop = cv2.resize(img[y1:y2, x1:x2], None, fx=ZOOM, fy=ZOOM,
                              interpolation=cv2.INTER_NEAREST)
            k_n, v_n = kpts_n[:n], vis[:n]
            valid = v_n > 0
            gt_px = (k_n * [W, H] - [x1, y1]) * ZOOM

            pb, pk = preds[n]
            err_txt = "no detection"
            if len(pb):
                d = np.linalg.norm(pb[:, :2] - [cx, cy], axis=1)
                j = int(np.argmin(d))
                if d[j] <= 0.5 * bw:
                    pd_px = (pk[j] - [x1, y1]) * ZOOM
                    obj = cone_object_points(n, "large" if cls == LARGE_CLASS_ID else "small")
                    gd = solve_depth(obj[valid], k_n[valid] * [W, H], K)
                    pd = solve_depth(obj[valid], pk[j][valid], K)
                    if gd and pd:
                        err_txt = f"depth err {abs(pd - gd) / gd * 100:.1f}%"

                    for kk in range(n):
                        if not valid[kk]:
                            continue
                        c = KPT_COLORS[kk % len(KPT_COLORS)]
                        g, p = gt_px[kk].astype(int), pd_px[kk].astype(int)
                        cv2.line(crop, tuple(g), tuple(p), (190, 190, 190), 1)
                        cv2.circle(crop, tuple(g), 6, c, 1)      # truth: hollow
                        cv2.circle(crop, tuple(p), 4, c, -1)     # prediction: filled

            bar = np.full((30, crop.shape[1], 3), 30, np.uint8)
            cv2.putText(bar, f"{n}kpt  {err_txt}", (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(np.vstack([bar, crop]))

        h = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 8, cv2.BORDER_CONSTANT, value=(20, 20, 20))
                 for t in tiles]
        rows.append(np.hstack(tiles))

    w = max(r.shape[1] for r in rows)
    rows = [cv2.copyMakeBorder(r, 0, 10, 0, w - r.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
            for r in rows]
    grid = np.vstack(rows)

    header = np.full((34, grid.shape[1], 3), 20, np.uint8)
    cv2.putText(header, f"condition: {args.condition}   |   hollow = ground truth, filled = prediction",
                (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    grid = np.vstack([header, grid])

    cv2.imwrite(str(args.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")


if __name__ == "__main__":
    main()

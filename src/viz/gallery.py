"""Qualitative gallery: many cones, many conditions, all three models side by side.

The single-frame views answer "does detection survive?"; this answers "what does the keypoint
prediction actually look like, and what depth does it produce?" across enough samples that the
reader is not looking at one lucky cone.

Each row is one cone under one condition; each column is a model. Hollow circles are ground
truth, filled are predictions, the grey line is the drift, and the header carries the depth
error PnP recovers from those points. Cones are drawn from several frames and several distances.

Writes straight into the repo's results/ directory.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from src.eval.eval_pose import (LARGE_CLASS_ID, cone_object_points, corrupt_image, solve_depth)

KPT_COLORS = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
              (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]
TILE = 190  # every cone crop is rendered to this square, so the grid lines up


def parse_condition(name):
    if name == "clean":
        return "none", 0.0
    kind, lvl = name.rsplit("_", 1)
    return kind, float(lvl)


def render_cone(img, gt_entry, model_out, n_kpt, K, W, H):
    """Return a TILE x TILE crop of one cone with GT/predicted keypoints, plus its depth error."""
    cls, box, kpts_n, vis = gt_entry
    cx, cy, bw, bh = box * [W, H, W, H]
    pad = max(bw, bh) * 0.4
    x1, y1 = max(0, int(cx - bw / 2 - pad)), max(0, int(cy - bh / 2 - pad))
    x2, y2 = min(W, int(cx + bw / 2 + pad)), min(H, int(cy + bh / 2 + pad))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None, None

    src = img[y1:y2, x1:x2]
    scale = TILE / max(src.shape[0], src.shape[1])
    crop = cv2.resize(src, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

    k_n, v_n = kpts_n[:n_kpt], vis[:n_kpt]
    valid = v_n > 0
    if valid.sum() < 4:
        return None, None

    pred_boxes, pred_kpts = model_out
    err = None
    if len(pred_boxes):
        d = np.linalg.norm(pred_boxes[:, :2] - [cx, cy], axis=1)
        j = int(np.argmin(d))
        if d[j] <= 0.6 * bw:
            obj = cone_object_points(n_kpt, "large" if cls == LARGE_CLASS_ID else "small")
            gd = solve_depth(obj[valid], k_n[valid] * [W, H], K)
            pd = solve_depth(obj[valid], pred_kpts[j][valid], K)
            if gd and pd:
                err = abs(pd - gd) / gd

            gt_px = (k_n * [W, H] - [x1, y1]) * scale
            pd_px = (pred_kpts[j] - [x1, y1]) * scale
            for k in range(n_kpt):
                if not valid[k]:
                    continue
                c = KPT_COLORS[k % len(KPT_COLORS)]
                g, p = gt_px[k].astype(int), pd_px[k].astype(int)
                cv2.line(crop, tuple(g), tuple(p), (190, 190, 190), 1)
                cv2.circle(crop, tuple(g), 5, c, 1)
                cv2.circle(crop, tuple(p), 3, c, -1)

    canvas = np.full((TILE, TILE, 3), 22, np.uint8)
    h, w = crop.shape[:2]
    canvas[:h, :w] = crop[:TILE, :TILE]
    return canvas, err


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--conditions", default="clean,sun_1.0,overcast_1.0,backlight_1.0,shadow_1.0,noise_0.5,blur_1.0")
    ap.add_argument("--cones-per-condition", default=4, type=int)
    ap.add_argument("--out", default="results/gallery.jpg", type=Path)
    ap.add_argument("--focal", default=1000.0, type=float)
    args = ap.parse_args()

    from ultralytics import YOLO

    root = Path("/data/brt_cone_pose/brt-cone-pose-11k")
    models = {}
    for n in (8, 6, 4):
        w = Path(f"/home/moon/runs/pose/cone-pose-{n}kpt/weights/best.pt")
        if w.exists():
            models[n] = YOLO(str(w))
    if not models:
        sys.exit("no trained weights found")

    # Draw cones from several different frames so one odd image cannot skew the impression.
    frames = []
    for lbl in sorted((root / "labels" / "test").glob("*.txt")):
        lines = [l for l in lbl.read_text().split("\n") if l.strip()]
        if len(lines) >= 8:
            frames.append((lbl, lines))
        if len(frames) >= 12:
            break

    rows, labels = [], []
    for cond in args.conditions.split(","):
        kind, lvl = parse_condition(cond)

        picked = 0
        for lbl, lines in frames:
            if picked >= args.cones_per_condition:
                break
            img_path = next((root / "images" / "test").glob(f"{lbl.stem}.*"), None)
            if img_path is None:
                continue
            img0 = cv2.imread(str(img_path))
            if img0 is None:
                continue
            H, W = img0.shape[:2]
            K = np.array([[args.focal, 0, W / 2], [0, args.focal, H / 2], [0, 0, 1]], np.float64)
            img = corrupt_image(img0.copy(), kind, lvl)

            gt = []
            for line in lines:
                t = [float(x) for x in line.split()]
                k = np.array(t[5:]).reshape(-1, 3)
                gt.append((int(t[0]), np.array(t[1:5]), k[:, :2], k[:, 2]))
            gt.sort(key=lambda g: -g[1][3])

            outs = {}
            for n, model in models.items():
                r = model.predict(img, verbose=False, device="cpu")[0]
                outs[n] = (r.boxes.xywh.cpu().numpy() if len(r.boxes) else np.zeros((0, 4)),
                           r.keypoints.xy.cpu().numpy() if r.keypoints is not None else np.zeros((0, n, 2)))

            # One cone from this frame: alternate near / mid / far so the row isn't all big cones.
            idx = [0, len(gt) // 3, 2 * len(gt) // 3, len(gt) - 1][picked % 4]
            tiles, errs = [], []
            for n in models:
                tile, err = render_cone(img, gt[idx], outs[n], n, K, W, H)
                if tile is None:
                    break
                tiles.append(tile)
                errs.append(err)
            if len(tiles) != len(models):
                continue

            strip = []
            for n, tile, err in zip(models, tiles, errs):
                bar = np.full((24, TILE, 3), 32, np.uint8)
                txt = f"{n}kpt " + (f"{err*100:.0f}%" if err is not None else "miss")
                color = ((90, 220, 90) if err < 0.05 else (0, 190, 255) if err < 0.15 else (60, 60, 255)) \
                    if err is not None else (140, 140, 140)
                cv2.putText(bar, txt, (5, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
                strip.append(np.vstack([bar, tile]))
            rows.append(np.hstack(strip))
            labels.append(cond if picked == 0 else "")
            picked += 1

    if not rows:
        sys.exit("nothing rendered")

    # Left gutter carries the condition name once per block.
    gut = 118
    out_rows = []
    for row, lab in zip(rows, labels):
        g = np.full((row.shape[0], gut, 3), 18, np.uint8)
        if lab:
            cv2.putText(g, lab, (6, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (230, 230, 230), 1, cv2.LINE_AA)
        out_rows.append(np.hstack([g, row]))

    grid = np.vstack(out_rows)
    header = np.full((32, grid.shape[1], 3), 18, np.uint8)
    cv2.putText(header, "hollow = ground truth   filled = prediction   header = PnP depth error",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1, cv2.LINE_AA)
    grid = np.vstack([header, grid])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"wrote {args.out}  ({grid.shape[1]}x{grid.shape[0]}, {len(rows)} cones)")


if __name__ == "__main__":
    main()

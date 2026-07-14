"""Run every trained model over a folder of real photos and lay the results out for comparison.

The dataset evaluations all live inside BRT's domain -- outdoor track, FS-spec cones, the same
handful of team cameras. This runs the models somewhere they have never been: an indoor car park,
training cones with a different stripe pattern, a phone camera. There is no ground truth here, so
what it measures is detection behaviour under domain shift: what gets found, what gets missed,
what gets hallucinated.

Emits one side-by-side panel per photo plus a summary table of per-model detection counts.
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

KPT_COLORS = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
              (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]
CLASS_COLORS = {0: (220, 120, 40), 1: (40, 120, 240), 2: (40, 160, 250),
                3: (150, 150, 150), 4: (60, 210, 240)}
CLASS_NAMES = {0: "BLUE", 1: "ORANGE_BIG", 2: "ORANGE", 3: "UNDEF", 4: "YELLOW"}


def draw(img, result):
    out = img.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out, 0, {}

    boxes = result.boxes.xyxy.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    kpts = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

    counts = {}
    s = max(1.0, img.shape[1] / 1000)
    for i, (box, cls, conf) in enumerate(zip(boxes, clses, confs)):
        name = CLASS_NAMES.get(int(cls), str(cls))
        counts[name] = counts.get(name, 0) + 1
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(int(cls), (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(2 * s)))

        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * s, 2)
        cv2.rectangle(out, (x1, max(0, y1 - th - 7)), (x1 + tw + 5, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, max(10, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * s, (255, 255, 255), max(1, int(1.4 * s)), cv2.LINE_AA)

        if kpts is not None and i < len(kpts):
            pts = kpts[i]
            n = len(pts)
            left, right = list(range(0, n, 2)), list(range(1, n, 2))
            for chain in (left, right):
                for a, b in zip(chain, chain[1:]):
                    cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                             (255, 255, 255), max(1, int(s)), cv2.LINE_AA)
            for a, b in zip(left, right):
                cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                         (170, 170, 170), max(1, int(s)), cv2.LINE_AA)
            for k, (px, py) in enumerate(pts):
                cv2.circle(out, (int(px), int(py)), max(3, int(3 * s)),
                           KPT_COLORS[k % len(KPT_COLORS)], -1)
    return out, len(boxes), counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--images", required=True, type=Path, help="Folder of photos")
    ap.add_argument("--weights", required=True, nargs="+", type=Path)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--width", default=560, type=int, help="Per-panel width in the output")
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO

    files = sorted(p for p in args.images.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"no images in {args.images}")

    titles = args.labels or [w.parent.parent.name for w in args.weights]
    models = [(t, YOLO(str(w))) for t, w in zip(titles, args.weights)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        h, w = img.shape[:2]
        img = cv2.resize(img, (args.width, int(h * args.width / w)))

        panels, row = [], {"image": f.stem}
        for title, model in models:
            res = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                                device=args.device, verbose=False)[0]
            panel, n, counts = draw(img, res)
            bar = np.full((30, panel.shape[1], 3), 30, np.uint8)
            summary = " ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "none"
            cv2.putText(bar, f"{title}  {n}  [{summary}]", (6, 21),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
            panels.append(np.vstack([bar, panel]))
            row[title] = n
        rows.append(row)

        grid = np.hstack(panels)
        head = np.full((28, grid.shape[1], 3), 18, np.uint8)
        cv2.putText(head, f.stem, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (220, 220, 220), 1, cv2.LINE_AA)
        cv2.imwrite(str(args.out_dir / f"{f.stem}.jpg"), np.vstack([head, grid]),
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f"{f.stem:<12} " + "  ".join(f"{t}={row[t]}" for t, _ in models))

    csv_path = args.out_dir / "detections.csv"
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=["image"] + [t for t, _ in models])
        wtr.writeheader()
        wtr.writerows(rows)

    print(f"\n{'model':<10} {'total detections':>18}")
    for t, _ in models:
        print(f"{t:<10} {sum(r[t] for r in rows):>18}")
    print(f"\nwrote {len(rows)} panels + {csv_path}")


if __name__ == "__main__":
    main()

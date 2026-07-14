"""Run the trained pose models on an arbitrary photo and draw what they see.

Unlike the dataset evaluations there is no ground truth here, so this shows the raw prediction:
box, class, confidence, and the keypoints. Useful for checking whether the model transfers to a
scene that looks nothing like the training set (indoors, different cone type, phone camera).
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

KPT_COLORS = [(66, 133, 244), (219, 68, 55), (244, 180, 0), (15, 157, 88),
              (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36)]
CLASS_COLORS = {0: (220, 120, 40), 1: (40, 120, 240), 2: (40, 160, 250),
                3: (150, 150, 150), 4: (60, 210, 240)}
CLASS_NAMES = {0: "BLUE", 1: "ORANGE_BIG", 2: "ORANGE", 3: "UNDEFINED", 4: "YELLOW"}


def draw(img, result, skeleton=True):
    out = img.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out, 0

    boxes = result.boxes.xyxy.cpu().numpy()
    clses = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    kpts = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None

    scale = max(1.0, img.shape[1] / 1200)
    for i, (box, cls, conf) in enumerate(zip(boxes, clses, confs)):
        x1, y1, x2, y2 = box.astype(int)
        color = CLASS_COLORS.get(int(cls), (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, max(2, int(2 * scale)))

        label = f"{CLASS_NAMES.get(int(cls), cls)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, 2)
        cv2.rectangle(out, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(out, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6 * scale, (255, 255, 255), max(1, int(1.6 * scale)), cv2.LINE_AA)

        if kpts is not None and i < len(kpts):
            pts = kpts[i]
            if skeleton:
                # Left column, right column, and the rungs between them -- the cone silhouette.
                n = len(pts)
                left = list(range(0, n, 2))
                right = list(range(1, n, 2))
                for chain in (left, right):
                    for a, b in zip(chain, chain[1:]):
                        cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                                 (255, 255, 255), max(1, int(scale)), cv2.LINE_AA)
                for a, b in zip(left, right):
                    cv2.line(out, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                             (180, 180, 180), max(1, int(scale)), cv2.LINE_AA)
            for k, (px, py) in enumerate(pts):
                cv2.circle(out, (int(px), int(py)), max(4, int(4 * scale)),
                           KPT_COLORS[k % len(KPT_COLORS)], -1)
                cv2.circle(out, (int(px), int(py)), max(4, int(4 * scale)), (255, 255, 255), 1)

    return out, len(boxes)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", required=True, type=Path)
    ap.add_argument("--weights", required=True, nargs="+", type=Path,
                    help="One or more checkpoints; each gets a panel")
    ap.add_argument("--labels", nargs="*", default=None, help="Panel titles, one per checkpoint")
    ap.add_argument("--conf", default=0.25, type=float)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    from ultralytics import YOLO

    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")

    # Phone photos are huge; shrink for a sane output while keeping the aspect.
    h, w = img.shape[:2]
    if w > 1400:
        img = cv2.resize(img, (1400, int(h * 1400 / w)))

    panels = []
    titles = args.labels or [p.parent.parent.name for p in args.weights]
    for wt, title in zip(args.weights, titles):
        model = YOLO(str(wt))
        res = model.predict(img, conf=args.conf, imgsz=args.imgsz,
                            device=args.device, verbose=False)[0]
        panel, n = draw(img, res)
        bar = np.full((40, panel.shape[1], 3), 28, np.uint8)
        cv2.putText(bar, f"{title}   {n} cone(s)", (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(np.vstack([bar, panel]))
        print(f"{title}: {n} detections")
        for cls, conf in zip(res.boxes.cls.cpu().numpy().astype(int),
                             res.boxes.conf.cpu().numpy()):
            print(f"   {CLASS_NAMES.get(int(cls), cls):<12} conf={conf:.3f}")

    grid = np.hstack(panels)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

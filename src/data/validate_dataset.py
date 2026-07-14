"""Validate the BRT Cone Pose dataset and write a corrected data.yaml for YOLO-Pose training.

The dataset ships a data.yaml with no `flip_idx`. Ultralytics defaults to fliplr=0.5, and
without flip_idx a horizontal flip mirrors the pixels while leaving the keypoint order
untouched -- every left keypoint silently becomes a right one on half the training images.
The cone keypoints are left/right pairs (even = left, odd = right), so flipping must swap
each pair. This also rewrites `path` to an absolute path, since Ultralytics resolves the
shipped relative path against its own datasets dir rather than the extracted location.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

NUM_KPT = 8
# Keypoints are left/right pairs down the cone silhouette: (0,1) top, (2,3) mid,
# (4,5) base, (6,7) extra stripe on large cones. A mirror swaps each pair.
FLIP_IDX = [1, 0, 3, 2, 5, 4, 7, 6]
CLASSES = {0: "BLUE", 1: "ORANGE_BIG", 2: "ORANGE", 3: "UNDEFINED", 4: "YELLOW"}


def validate_labels(labels_dir, sample_size, seed):
    """Spot-check label geometry. Returns (class counts, visible-keypoint histogram)."""
    paths = sorted(labels_dir.glob("*.txt"))
    if not paths:
        sys.exit(f"No labels found in {labels_dir}")
    if sample_size and len(paths) > sample_size:
        paths = random.Random(seed).sample(paths, sample_size)

    expected_tokens = 5 + NUM_KPT * 3
    class_counts, visible_hist, problems = Counter(), Counter(), []

    for path in paths:
        for lineno, line in enumerate(path.read_text().split("\n"), 1):
            if not line.strip():
                continue
            tokens = line.split()
            if len(tokens) != expected_tokens:
                problems.append(f"{path.name}:{lineno} has {len(tokens)} tokens, expected {expected_tokens}")
                continue

            class_id = int(tokens[0])
            class_counts[class_id] += 1

            coords = [float(t) for t in tokens[1:]]
            if any(not 0.0 <= c <= 1.0 for c in coords[:4]):
                problems.append(f"{path.name}:{lineno} box not normalized")

            visible = 0
            for k in range(NUM_KPT):
                x, y, v = coords[4 + k * 3: 7 + k * 3]
                if int(v) != 0:
                    visible += 1
                    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                        problems.append(f"{path.name}:{lineno} kpt{k} out of range")
            visible_hist[visible] += 1

    return class_counts, visible_hist, problems, len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path,
                        help="Extracted dataset dir (contains images/ and labels/)")
    parser.add_argument("--sample", default=500, type=int,
                        help="Label files to validate per split; 0 checks all")
    parser.add_argument("--seed", default=17, type=int)
    args = parser.parse_args()

    root = args.root.resolve()
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            if not (root / kind / split).is_dir():
                sys.exit(f"Missing {kind}/{split} under {root}")

    for split in ("train", "val", "test"):
        n_img = len(list((root / "images" / split).glob("*")))
        n_lbl = len(list((root / "labels" / split).glob("*.txt")))
        status = "OK" if n_img == n_lbl else "MISMATCH"
        print(f"{split:<6} images={n_img:>6,}  labels={n_lbl:>6,}  [{status}]")

    class_counts, visible_hist, problems, n_checked = validate_labels(
        root / "labels" / "train", args.sample, args.seed
    )
    print(f"\nValidated {n_checked:,} train label files:")
    total = sum(class_counts.values())
    for class_id, name in CLASSES.items():
        n = class_counts[class_id]
        share = 100 * n / total if total else 0
        print(f"  {class_id} {name:<12} {n:>7,}  ({share:4.1f}%)")
    print("  visible keypoints per cone: " +
          ", ".join(f"{k}kpt={v:,}" for k, v in sorted(visible_hist.items())))

    if problems:
        print(f"\n{len(problems)} malformed labels (first 5):")
        for p in problems[:5]:
            print(f"  {p}")
    else:
        print("  no malformed labels")

    yaml_path = root / "data.yaml"
    yaml_path.write_text(
        f"path: {root}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"kpt_shape: [{NUM_KPT}, 3]\n"
        # Without this, fliplr augmentation mirrors images without swapping L/R keypoints.
        f"flip_idx: {FLIP_IDX}\n\n"
        "names:\n" + "".join(f"  {i}: {n}\n" for i, n in CLASSES.items())
    )
    print(f"\nWrote {yaml_path} (absolute path + flip_idx)")


if __name__ == "__main__":
    main()

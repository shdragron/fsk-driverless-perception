"""Derive reduced-keypoint variants of the BRT Cone Pose dataset for the ablation.

BRT ships 8 keypoints as left/right pairs descending the cone silhouette:

      0  1     top pair
      2  3     mid pair
      6  7     extra pair (large cones only; zero-padded on the other 96%)
      4  5     base pair

The 8 -> 6 cut drops kpt6/7, which only ORANGE_BIG actually uses, so it is nearly lossless.
The 8 -> 4 cut keeps only the top and base pairs, dropping the mid pair -- that discards real
signal and takes PnP from overdetermined to minimal, which is the point of the ablation.

Images are symlinked, not copied, so the variants cost megabytes rather than gigabytes.
"""
import argparse
import shutil
import sys
from pathlib import Path

# Which of the original 8 keypoint indices each variant keeps, in output order.
VARIANTS = {
    6: [0, 1, 2, 3, 4, 5],  # drop the large-cone-only pair
    4: [0, 1, 4, 5],        # top pair + base pair; drop the mid pair
}
CLASSES = {0: "BLUE", 1: "ORANGE_BIG", 2: "ORANGE", 3: "UNDEFINED", 4: "YELLOW"}


def flip_idx_for(n):
    """Even indices are left-side points, odd are their right-side partners; a mirror swaps each pair."""
    return [i + 1 if i % 2 == 0 else i - 1 for i in range(n)]


def subset_label(line, keep):
    """Rewrite one YOLO-pose line, keeping only the listed keypoint indices."""
    tokens = line.split()
    box, kpts = tokens[:5], tokens[5:]
    out = list(box)
    for k in keep:
        out.extend(kpts[k * 3: k * 3 + 3])
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="brt-cone-pose-11k root")
    parser.add_argument("--out-root", required=True, type=Path, help="Where variant dirs are created")
    args = parser.parse_args()

    source = args.source.resolve()
    if not (source / "labels" / "train").is_dir():
        sys.exit(f"Not a BRT dataset root: {source}")

    for n_kpt, keep in VARIANTS.items():
        dest = (args.out_root / f"brt-cone-pose-{n_kpt}kpt").resolve()
        if dest.exists():
            shutil.rmtree(dest)

        n_boxes = 0
        for split in ("train", "val", "test"):
            (dest / "images" / split).mkdir(parents=True)
            (dest / "labels" / split).mkdir(parents=True)

            for img in sorted((source / "images" / split).iterdir()):
                (dest / "images" / split / img.name).symlink_to(img)

            for lbl in sorted((source / "labels" / split).glob("*.txt")):
                lines = [subset_label(l, keep) for l in lbl.read_text().split("\n") if l.strip()]
                n_boxes += len(lines)
                (dest / "labels" / split / lbl.name).write_text("\n".join(lines) + ("\n" if lines else ""))

        (dest / "data.yaml").write_text(
            f"path: {dest}\n"
            "train: images/train\n"
            "val: images/val\n"
            "test: images/test\n\n"
            f"kpt_shape: [{n_kpt}, 3]\n"
            f"flip_idx: {flip_idx_for(n_kpt)}\n\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in CLASSES.items())
        )
        print(f"{n_kpt}kpt: kept {keep} -> {dest}  ({n_boxes:,} boxes)")


if __name__ == "__main__":
    main()

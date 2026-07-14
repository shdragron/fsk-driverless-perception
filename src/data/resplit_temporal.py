"""Re-split BRT Cone Pose by contiguous frame blocks, because the shipped split leaks.

BRT shuffles frames at random. FSOCO frames come from continuous driving footage, so
`mms_00185` and `mms_00186` are the same scene 1/30 s apart -- and the shipped split puts one in
train and the other in test. 54.8% of frames have a neighbour (+/-1 or +/-2) in a different
split. Test scores measured that way are inflated: the model has effectively seen the test set.

This assigns *contiguous runs* of frame numbers, per team, to one split each, with a gap of
`--gap` frames between blocks so that even neighbouring frames across a boundary are excluded.
Ratios are preserved per team, so every team stays represented in every split.
"""
import argparse
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

# Roboflow encodes the *original* extension in the stem, so both `..._jpg.rf.<hash>.jpg` and
# `..._png.rf.<hash>.jpg` occur. Matching only `_jpg` silently drops 60% of the dataset.
FRAME_RE = re.compile(r"^(.*?)_(\d+)_(?:jpg|png|jpeg)\.rf\.[0-9a-f]+\.(?:jpg|png|jpeg)$", re.I)


def parse_frame(name):
    """('mms_00185_jpg.rf.abc.jpg') -> ('mms', 185); None if the name doesn't fit."""
    m = FRAME_RE.match(name)
    return (m.group(1), int(m.group(2))) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path, help="brt-cone-pose-11k root")
    ap.add_argument("--out", required=True, type=Path, help="Destination for the clean split")
    ap.add_argument("--val-frac", default=0.10, type=float)
    ap.add_argument("--test-frac", default=0.10, type=float)
    ap.add_argument("--gap", default=3, type=int,
                    help="Frames dropped at each block boundary, so no near-duplicate straddles it")
    ap.add_argument("--n-kpt", default=8, type=int, choices=[4, 6, 8])
    args = ap.parse_args()

    # Gather every frame from every existing split -- we are re-partitioning the whole pool.
    pool = defaultdict(list)  # team -> [(frame_idx, img_path, lbl_path)]
    unparsed = 0
    for split in ("train", "val", "test"):
        img_dir = args.source / "images" / split
        lbl_dir = args.source / "labels" / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.iterdir():
            parsed = parse_frame(img.name)
            if parsed is None:
                unparsed += 1
                continue
            team, idx = parsed
            lbl = lbl_dir / f"{img.stem}.txt"
            if lbl.exists():
                pool[team].append((idx, img.resolve(), lbl))
    if unparsed:
        print(f"warning: {unparsed} filenames did not match the team_frame pattern", file=sys.stderr)
    if not pool:
        sys.exit(f"no frames found under {args.source}")

    assign = {"train": [], "val": [], "test": []}
    dropped = 0

    for team, frames in sorted(pool.items()):
        frames.sort()
        n = len(frames)
        n_val = int(round(n * args.val_frac))
        n_test = int(round(n * args.test_frac))
        n_train = n - n_val - n_test

        # One contiguous block each: [train | gap | val | gap | test]. Contiguous is the point --
        # a random draw is exactly what created the leak.
        blocks = [("train", 0, n_train), ("val", n_train, n_train + n_val),
                  ("test", n_train + n_val, n)]
        for i, (split, lo, hi) in enumerate(blocks):
            # Drop `gap` frames at each internal boundary so adjacent frames can't straddle it.
            if i > 0:
                lo += args.gap
            if i < len(blocks) - 1:
                hi -= args.gap
            if hi <= lo:
                continue
            dropped += args.gap * (1 if i > 0 else 0) + args.gap * (1 if i < len(blocks) - 1 else 0)
            assign[split].extend(frames[lo:hi])

        print(f"  {team:<12} {n:>5} frames -> train {n_train:>5}  val {n_val:>4}  test {n_test:>4}")

    for split, items in assign.items():
        img_dir = args.out / "images" / split
        lbl_dir = args.out / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True)
        lbl_dir.mkdir(parents=True)
        for _, img, lbl in items:
            (img_dir / img.name).symlink_to(img)
            shutil.copy(lbl, lbl_dir / lbl.name)

    flip = [i + 1 if i % 2 == 0 else i - 1 for i in range(args.n_kpt)]
    (args.out / "data.yaml").write_text(
        f"path: {args.out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        f"kpt_shape: [{args.n_kpt}, 3]\n"
        f"flip_idx: {flip}\n\n"
        "names:\n  0: BLUE\n  1: ORANGE_BIG\n  2: ORANGE\n  3: UNDEFINED\n  4: YELLOW\n"
    )

    print(f"\ntrain {len(assign['train']):,} | val {len(assign['val']):,} | test {len(assign['test']):,} "
          f"| dropped {dropped:,} at block boundaries")
    print(f"wrote {args.out}/data.yaml")


if __name__ == "__main__":
    main()

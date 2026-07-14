"""Train RektNet-V on BRT cone crops, for a head-to-head against single-stage YOLO26-pose.

RektNet-V is RektNet with a visibility mask (hence the V), 8 keypoints, and its geometric loss
re-derived for BRT's apex-less layout. Passing --num-kpt 7 instead runs the unmodified original,
on the synthesised-apex labels from make_mit_7kpt.

RektNet's own images are gone (the MIT bucket's billing lapsed), so this trains it on crops cut
from the same BRT frames the YOLO-pose models see. Same cones, same splits -- the only thing
that differs is the architecture and the fact that RektNet is handed the box for free.

The geometric (cross-ratio) loss is ON by default here, unlike the original CLI whose gammas
default to 0. That loss is RektNet's headline contribution, and benchmarking it without one would
be a strawman. The weights are the ones the paper reports converging on -- gamma_vert=0.038,
gamma_horz=0.055 (Eq. 2) -- not the [0, 0.15] search prior in train_eval_hyper.py.

Keypoints carry a visibility mask. kpt6/7 exist only on large orange cones (5.3% of the set); a
small cone has no fourth stripe boundary. Supervising those slots on a cone that lacks them would
teach the model to invent a landmark. The original never faced this -- its 7 keypoints were
always all present -- so this is an extension, not a reproduction.
"""
import argparse
import ast
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset

from src.models.cross_ratio_loss import BRTCrossRatioLoss
from src.models.keypoint_net import KeypointNet

INPUT_SIZE = (80, 80)


class BRTCropDataset(Dataset):
    """Cone crops with keypoints in crop-local pixels, resized to RektNet's 80x80 input."""

    def __init__(self, csv_path, images_dir, num_kpt, augment=False):
        self.images_dir = Path(images_dir)
        self.num_kpt = num_kpt
        self.augment = augment
        with open(csv_path) as f:
            self.rows = list(csv.DictReader(f))
        cols = [c for c in self.rows[0] if c not in ("image", "cls", "is_large")]
        self.kpt_cols = [c for c in cols if not c.endswith("_vis")][:num_kpt]
        # Written by make_cone_crops; absent in older CSVs, in which case every point is visible.
        self.has_vis = f"{self.kpt_cols[0]}_vis" in self.rows[0]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = cv2.imread(str(self.images_dir / row["image"]))
        h, w = img.shape[:2]
        pts = np.array([ast.literal_eval(row[c]) for c in self.kpt_cols], dtype=np.float32)
        if self.has_vis:
            vis = np.array([float(row[f"{c}_vis"]) > 0 for c in self.kpt_cols], dtype=np.float32)
        else:
            vis = np.ones(self.num_kpt, dtype=np.float32)

        img = cv2.resize(img, INPUT_SIZE)
        pts = pts / [w, h]  # normalize before augmenting so flips are trivial

        if self.augment and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            pts[:, 0] = 1.0 - pts[:, 0]
            # Mirroring swaps each left/right keypoint pair; without this the labels are wrong.
            order = [i + 1 if i % 2 == 0 else i - 1 for i in range(self.num_kpt)]
            pts = pts[order]
            vis = vis[order]

        hm = self._heatmap(pts, vis)
        img = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float()
        return (img, torch.from_numpy(hm).float(),
                torch.from_numpy(pts).float(), torch.from_numpy(vis).float())

    def _heatmap(self, pts, vis):
        """Gaussian target heatmap, one channel per keypoint. Absent points get an empty channel."""
        H, W = INPUT_SIZE
        hm = np.zeros((self.num_kpt, H, W), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        for k, (x, y) in enumerate(pts):
            if not vis[k]:
                continue  # no such landmark on this cone -- leave the channel at zero and mask it
            px, py = x * W, y * H
            hm[k] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.0 ** 2))
            s = hm[k].sum()
            if s > 0:
                hm[k] /= s
        return hm


def evaluate(model, loader, device, num_kpt):
    """Mean L2 error in crop pixels, over the keypoints that actually exist on each cone.

    Note this is a diagnostic, not a benchmark: the paper reports no per-keypoint pixel error to
    compare against. Its published target is depth -- mean error < 0.5 m with < 5 cm std out to
    20 m -- which is what eval_rektnet measures.
    """
    model.eval()
    errs = []
    scale = torch.tensor(INPUT_SIZE, dtype=torch.float32)
    with torch.no_grad():
        for img, _, pts, vis in loader:
            img = img.to(device)
            _, pred = model(img)
            d = torch.norm((pred.cpu() - pts) * scale, dim=2)   # (B, K)
            errs.append(d[vis > 0].numpy())
    return float(np.concatenate(errs).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="Dir with train/val/test CSVs + images/")
    p.add_argument("--num-kpt", required=True, type=int, choices=[4, 6, 7, 8],
               help="8 requires the visibility mask: kpt6/7 exist only on large cones")
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--patience", default=8, type=int,
                   help="Early-stop tolerance; 8 is the upstream value")
    p.add_argument("--batch", default=64, type=int)
    p.add_argument("--lr", default=1e-3, type=float,
                   help="The original default of 0.1 diverges on this data; 1e-3 is Adam's usual range")
    p.add_argument("--lr-gamma", default=0.995, type=float)
    # The values the paper reports converging on (Eq. 2): "a Bayesian optimization framework was
    # used to determine values for the loss constants, resulting in gamma_vert = 0.038 and
    # gamma_horz = 0.055". [0, 0.15] in the upstream train_eval_hyper.py is the *search prior*,
    # not a result -- these are the results, and they are asymmetric.
    p.add_argument("--geo-gamma-vert", default=0.038, type=float,
                   help="Vertical (collinearity) weight; 0.038 is the paper's tuned value")
    p.add_argument("--geo-gamma-horz", default=0.055, type=float,
                   help="Horizontal (parallelism) weight; 0.055 is the paper's tuned value")
    p.add_argument("--no-geo", action="store_true", help="Ablate the cross-ratio loss")
    p.add_argument("--workers", default=8, type=int)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(17)
    np.random.seed(17)

    train_ds = BRTCropDataset(args.data / "train.csv", args.data / "images/train", args.num_kpt, augment=True)
    val_ds = BRTCropDataset(args.data / "val.csv", args.data / "images/val", args.num_kpt)
    print(f"train {len(train_ds):,} crops | val {len(val_ds):,} crops | {args.num_kpt} keypoints")

    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=args.workers)

    model = KeypointNet(num_kpt=args.num_kpt, image_size=INPUT_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    loss_fn = BRTCrossRatioLoss(
        loss_type="l1_softargmax",
        include_geo=not args.no_geo,
        geo_loss_gamma_vert=args.geo_gamma_vert,
        geo_loss_gamma_horz=args.geo_gamma_horz,
        num_kpt=args.num_kpt,
    )
    print(f"geometric loss: {'OFF' if args.no_geo else f'ON (vert={args.geo_gamma_vert}, horz={args.geo_gamma_horz})'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    best_err, best_epoch, stale = float("inf"), -1, 0
    for epoch in range(args.epochs):
        model.train()
        t0, totals = time.time(), np.zeros(3)
        for img, hm, pts, vis in train_dl:
            img, hm, pts, vis = img.to(device), hm.to(device), pts.to(device), vis.to(device)
            optimizer.zero_grad()
            pred_hm, pred_pts = model(img)
            loc, geo, loss = loss_fn(pred_hm, pred_pts, hm, pts, vis)
            loss.backward()
            optimizer.step()
            totals += [loc.item(), float(geo), loss.item()]
        scheduler.step()

        n = len(train_dl)
        val_err = evaluate(model, val_dl, device, args.num_kpt)
        flag = ""
        if val_err < best_err:
            best_err, best_epoch, flag = val_err, epoch, "  *best"
            stale = 0
            torch.save({"epoch": epoch, "model": model.state_dict(), "num_kpt": args.num_kpt}, args.out)
        else:
            stale += 1
        print(f"epoch {epoch:>3}  loc={totals[0]/n:.4f}  geo={totals[1]/n:.4f}  "
              f"val_px={val_err:.3f}  ({time.time()-t0:.0f}s){flag}", flush=True)

        # The upstream trainer stops after 8 epochs with no improvement; honour that.
        if stale >= args.patience:
            print(f"early stop: no improvement for {args.patience} epochs")
            break

    print(f"\nbest val error {best_err:.3f} px at epoch {best_epoch} -> {args.out}")


if __name__ == "__main__":
    main()

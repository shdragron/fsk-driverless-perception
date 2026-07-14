"""Train RektNet on BRT cone crops, so it can be compared head-to-head with YOLO26-pose.

RektNet's own images are gone (the MIT bucket's billing lapsed), so this trains it on crops cut
from the same BRT frames the YOLO-pose models see. Same cones, same splits -- the only thing
that differs is the architecture and the fact that RektNet is handed the box for free.

The geometric (cross-ratio) loss is ON by default here, unlike the original CLI which defaults
the gammas to 0. That loss is RektNet's headline contribution, and the original authors searched
it over [0, 0.15] (train_eval_hyper.py); benchmarking RektNet without it would be a strawman.
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
        self.kpt_cols = [c for c in self.rows[0] if c not in ("image", "cls", "is_large")][:num_kpt]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        img = cv2.imread(str(self.images_dir / row["image"]))
        h, w = img.shape[:2]
        pts = np.array([ast.literal_eval(row[c]) for c in self.kpt_cols], dtype=np.float32)

        img = cv2.resize(img, INPUT_SIZE)
        pts = pts / [w, h]  # normalize before augmenting so flips are trivial

        if self.augment and np.random.rand() < 0.5:
            img = cv2.flip(img, 1)
            pts[:, 0] = 1.0 - pts[:, 0]
            # Mirroring swaps each left/right keypoint pair; without this the labels are wrong.
            order = [i + 1 if i % 2 == 0 else i - 1 for i in range(self.num_kpt)]
            pts = pts[order]

        hm = self._heatmap(pts)
        img = torch.from_numpy(img.transpose(2, 0, 1) / 255.0).float()
        return img, torch.from_numpy(hm).float(), torch.from_numpy(pts).float()

    def _heatmap(self, pts):
        """Gaussian target heatmap, one channel per keypoint."""
        H, W = INPUT_SIZE
        hm = np.zeros((self.num_kpt, H, W), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        for k, (x, y) in enumerate(pts):
            px, py = x * W, y * H
            hm[k] = np.exp(-((xx - px) ** 2 + (yy - py) ** 2) / (2 * 2.0 ** 2))
            s = hm[k].sum()
            if s > 0:
                hm[k] /= s
        return hm


def evaluate(model, loader, device, num_kpt):
    """Mean per-keypoint L2 error in pixels of the 80x80 crop."""
    model.eval()
    errs = []
    with torch.no_grad():
        for img, _, pts in loader:
            img = img.to(device)
            _, pred = model(img)
            d = torch.norm((pred.cpu() - pts) * torch.tensor(INPUT_SIZE, dtype=torch.float32), dim=2)
            errs.append(d.numpy())
    return float(np.concatenate(errs).mean())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path, help="Dir with train/val/test CSVs + images/")
    p.add_argument("--num-kpt", required=True, type=int, choices=[4, 6])
    p.add_argument("--epochs", default=200, type=int)
    p.add_argument("--batch", default=64, type=int)
    p.add_argument("--lr", default=1e-3, type=float,
                   help="The original default of 0.1 diverges on this data; 1e-3 is Adam's usual range")
    p.add_argument("--lr-gamma", default=0.995, type=float)
    p.add_argument("--geo-gamma-vert", default=0.075, type=float,
                   help="Midpoint of the [0, 0.15] range the RektNet authors tuned over")
    p.add_argument("--geo-gamma-horz", default=0.075, type=float)
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
    best_err, best_epoch = float("inf"), -1
    for epoch in range(args.epochs):
        model.train()
        t0, totals = time.time(), np.zeros(3)
        for img, hm, pts in train_dl:
            img, hm, pts = img.to(device), hm.to(device), pts.to(device)
            optimizer.zero_grad()
            pred_hm, pred_pts = model(img)
            loc, geo, loss = loss_fn(pred_hm, pred_pts, hm, pts)
            loss.backward()
            optimizer.step()
            totals += [loc.item(), float(geo), loss.item()]
        scheduler.step()

        n = len(train_dl)
        val_err = evaluate(model, val_dl, device, args.num_kpt)
        flag = ""
        if val_err < best_err:
            best_err, best_epoch, flag = val_err, epoch, "  *best"
            torch.save({"epoch": epoch, "model": model.state_dict(), "num_kpt": args.num_kpt}, args.out)
        print(f"epoch {epoch:>3}  loc={totals[0]/n:.4f}  geo={totals[1]/n:.4f}  "
              f"val_px={val_err:.3f}  ({time.time()-t0:.0f}s){flag}", flush=True)

    print(f"\nbest val error {best_err:.3f} px at epoch {best_epoch} -> {args.out}")


if __name__ == "__main__":
    main()

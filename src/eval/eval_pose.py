"""Compare cone-keypoint models by the accuracy of the 3D pose they enable, not by pixel error.

Keypoints exist to feed PnP for monocular depth, so the metric that matters is the depth PnP
recovers. This also sidesteps the fact that RektNet's 7 keypoints and BRT's 8 are defined
differently -- both resolve to a cone position in camera space, which is directly comparable.

There is no ground-truth depth in BRT (it is 2D-annotated imagery pooled from many teams'
cameras), so absolute metres are meaningless. Instead we take PnP on the *ground-truth*
keypoints as the reference pose and measure how far each model's *predicted* keypoints move
it. A fixed nominal intrinsic matrix is applied to every model identically: it makes absolute
depth arbitrary but leaves the model-vs-model comparison sound, which is what the ablation asks.

Robustness is probed by perturbing the input and watching the depth error grow:
  --box-noise  jitters the crop fed to RektNet, mimicking an imperfect upstream detector
  --corrupt    blurs / darkens / adds sensor noise, mimicking motion and bad light
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

# FS Driverless Specification 2026, DS 1.3 (and FSAE DD.1.3): the cones are WEMAS Kegel E320/E500.
#   small  228 x 228 x 325 mm
#   large  285 x 285 x 505 mm
# The 228 x 228 footprint is SQUARE -- these are truncated pyramids, not cones.
CONE_DIMS_M = {
    "small": (0.228, 0.325),
    "large": (0.285, 0.505),
}
LARGE_CLASS_ID = 1  # ORANGE_BIG

# A square cone's silhouette width depends on viewing azimuth: face-on you see the flat face
# (half-width w/2), at 45 degrees the diagonal (w/2 * sqrt(2)) -- 41% wider. PnP cannot know the
# azimuth; it is one of the unknowns being solved for. Modelling the cone as round with half-width
# w/2 therefore under-reads it at every angle but zero, and PnP compensates by pushing the cone
# further away.
#
# Averaging |cos t| + |sin t| over t in [0, pi/2] gives 4/pi ~ 1.273, so a round cone of that
# effective radius matches the square one's mean silhouette across all viewing angles.
SILHOUETTE_FACTOR = 4 / np.pi

# Stripe heights as a fraction of cone height, base = 0. Measured from BRT's own labels (median
# keypoint position within its box; n = 88,208 small, 4,776 large) because neither the FS rules nor
# WEMAS publish them. AMZ hit the same gap and measured a physical cone (arXiv:1905.05150, S3.2.3).
#
# The label heights hold across distance -- comparing near/mid/far cones, each keypoint's position
# within its box moves by at most 0.02, well inside the +/-0.05 spread. So the annotation is
# consistent enough for a single template to be meaningful.
LEVELS = {
    # A small cone has no fourth stripe, so "extra" is a placeholder: kpt6/7 always carry
    # visibility 0 there and are masked out of the loss, the heatmap, and PnP. It only needs to be
    # a valid number so the template can be built.
    "small": {"top": 0.703, "mid": 0.451, "bot": 0.178, "extra": 0.089},
    # A large cone's stripes sit higher, and it has a fourth pair -- which is BELOW the third:
    # kpt6/7 at 0.266, kpt4/5 at 0.428. The index order is not the height order.
    "large": {"top": 0.742, "mid": 0.581, "bot": 0.428, "extra": 0.266},
}


def cone_object_points(n_kpt, size="small"):
    """3D cone-frame coordinates for the BRT keypoint layout, origin at the cone's base centre.

    Keypoints are left/right silhouette pairs going down the cone; the silhouette half-width at a
    given height is what the camera sees, so each pair sits at +/-x of the cone's radius there.

    Outer dimensions come from the FS rules and the square-to-round silhouette correction is exact.
    The stripe heights are not in the rules; they are measured from the labels (see LEVELS).

    What remains unmodelled: the real cone is a truncated pyramid with a flared square base, not a
    straight taper, so the radius at each stripe is approximate. That scales absolute depth. The
    relative metric used here divides it out -- reference and prediction are solved with the same
    template -- but a ROS stack consuming the metres should measure a physical cone.
    """
    width, height = CONE_DIMS_M[size]
    half = (width / 2) * SILHOUETTE_FACTOR   # square cone -> equivalent round-cone radius

    # Stripe heights, as a fraction of cone height, measured from the labels themselves: the median
    # keypoint position within its own box, over 88,208 small cones and 4,776 large ones. Neither
    # the FS rules nor WEMAS publish these, and estimating them by eye was wrong by up to 0.33.
    #
    # Small and large cones differ -- the large cone's stripes sit higher and it has a fourth pair.
    # Note the ordering on a large cone: kpt6/7 (0.266) are BELOW kpt4/5 (0.428), not above.
    lv = LEVELS[size]
    if n_kpt == 8:
        levels = [lv["top"], lv["mid"], lv["bot"], lv["extra"]]  # pairs (0,1) (2,3) (4,5) (6,7)
        order = [0, 1, 2, 3, 4, 5, 6, 7]
    elif n_kpt == 6:
        levels = [lv["top"], lv["mid"], lv["bot"]]
        order = [0, 1, 2, 3, 4, 5]
    elif n_kpt == 4:
        levels = [lv["top"], lv["bot"]]
        order = [0, 1, 2, 3]
    elif n_kpt == 7:
        # RektNet layout: apex + three left/right pairs. The apex is synthesised as the midpoint of
        # BRT's top pair (make_mit_7kpt.py), so it lies on the centreline at that pair's height --
        # not at the cone's tip. Modelling it at the tip would bias every PnP solve.
        pts = [[0.0, 0.0, lv["top"] * height]]
        mid_bot = (lv["mid"] + lv["bot"]) / 2   # make_mit_7kpt interpolates this on small cones
        for frac in (lv["mid"], mid_bot, lv["bot"]):
            r = half * (1 - frac)
            z = frac * height
            pts.append([-r, 0.0, z])
            pts.append([+r, 0.0, z])
        # order: apex, mid_L_top, mid_R_top, mid_L_bot, mid_R_bot, bot_L, bot_R
        return np.array(pts, dtype=np.float64)
    else:
        raise ValueError(f"unsupported keypoint count: {n_kpt}")

    pts = []
    for frac in levels:
        r = half * (1 - frac)  # taper: narrower near the tip
        z = frac * height
        pts.append([-r, 0.0, z])
        pts.append([+r, 0.0, z])
    return np.array(pts, dtype=np.float64)[order][:n_kpt]


def solve_depth(object_points, image_points, K):
    """Return the cone's depth (metres along the optical axis), or None if PnP fails.

    SQPNP, not ITERATIVE: a cone's keypoints are left/right silhouette pairs, so every one of them
    lies in the same plane (y=0 in the cone frame). OpenCV 5 raises on ITERATIVE with coplanar
    points -- and since the exception is caught below, that failure would look like "PnP didn't
    converge" and silently drop every cone. SQPNP handles the planar case and recovers the pose
    exactly (verified: 5.000 m round-trip, against EPNP's 6.55 m).
    """
    if len(image_points) < 4:
        return None
    try:
        ok, rvec, tvec = cv2.solvePnP(
            np.ascontiguousarray(object_points, dtype=np.float64),
            np.ascontiguousarray(image_points, dtype=np.float64),
            K, None,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None
    if not ok:
        return None
    depth = float(np.asarray(tvec).ravel()[2])
    # PnP on near-planar, near-symmetric points occasionally flips the solution behind the
    # camera or throws it to infinity; those are failures, not measurements.
    if not np.isfinite(depth) or depth <= 0 or depth > 200:
        return None
    return depth


def load_labels(path, n_kpt):
    """Parse a YOLO-pose label file into (class_id, box_xywh, keypoints_xy, visibility)."""
    out = []
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        t = [float(x) for x in line.split()]
        cls, box = int(t[0]), np.array(t[1:5])
        kpts = np.array(t[5:]).reshape(n_kpt, 3)
        out.append((cls, box, kpts[:, :2], kpts[:, 2]))
    return out


def corrupt_image(img, kind, level):
    """Apply a graded image degradation. level in [0, 1]."""
    if level <= 0 or kind == "none":
        return img
    f = img.astype(np.float32)

    if kind == "blur":
        # Horizontal motion blur, not Gaussian: a car's camera smears the scene along its
        # direction of travel. A Gaussian kernel small enough to be plausible barely touched
        # these 2344px-wide frames (28 dB PSNR at full strength), making the sweep meaningless.
        k = int(2 * round(level * 20) + 1)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        return cv2.filter2D(img, -1, kernel)

    if kind == "noise":
        return np.clip(f + np.random.normal(0, level * 50, img.shape), 0, 255).astype(np.uint8)

    if kind == "dark":
        return np.clip(f * (1 - 0.8 * level), 0, 255).astype(np.uint8)

    # --- lighting conditions a race car actually drives through ---

    if kind == "sun":
        # Low sun straight into the lens: a bright bloom plus a global veiling glare that
        # washes out the cone/track contrast. This is the classic FS afternoon-session failure.
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # Sun sits just above the horizon, off to one side.
        cx, cy, r = w * 0.72, h * 0.28, max(h, w) * 0.35
        glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))[..., None]
        bloomed = f + level * 200 * glow                       # the disc itself
        veil = level * 60                                      # scattered light lifts the blacks
        return np.clip(bloomed * (1 - 0.25 * level) + veil, 0, 255).astype(np.uint8)

    if kind == "overcast":
        # Cloud cover: no directional light, so contrast collapses toward mid-grey and the
        # colour cast goes cool. Cones stay lit but stop standing out from the tarmac.
        contrast = 1 - 0.55 * level
        out = (f - 128) * contrast + 128 + 10 * level
        out[..., 0] *= 1 + 0.08 * level                        # BGR: lift blue
        out[..., 2] *= 1 - 0.06 * level                        # drop red
        return np.clip(out, 0, 255).astype(np.uint8)

    if kind == "shadow":
        # Driving under a bridge / tree line: part of the frame is in deep shade while the rest
        # stays lit, so no single exposure works and cones in the shadow band lose all contrast.
        h, w = img.shape[:2]
        band = np.ones((h, w), np.float32)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        band[:, x0:x1] = 1 - 0.75 * level
        band = cv2.GaussianBlur(band, (0, 0), sigmaX=w * 0.02)  # soft penumbra
        return np.clip(f * band[..., None], 0, 255).astype(np.uint8)

    if kind == "backlight":
        # Sun behind the cones: they become near-silhouettes, colour information is destroyed
        # (which also breaks the blue/yellow class the planner depends on).
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
        desat = f * (1 - 0.7 * level) + gray * 0.7 * level     # colour drains away
        crushed = np.clip((desat - 40 * level) * (1 - 0.45 * level), 0, 255)
        return np.clip(crushed + 70 * level * (gray / 255.0), 0, 255).astype(np.uint8)

    raise ValueError(kind)


def jitter_box(box, level, rng):
    """Shift and rescale a normalized xywh box to imitate detector error."""
    cx, cy, w, h = box
    cx += rng.normal(0, level * 0.15) * w
    cy += rng.normal(0, level * 0.15) * h
    scale = 1 + rng.normal(0, level * 0.15)
    return np.array([cx, cy, max(w * scale, 1e-3), max(h * scale, 1e-3)])


def distance_bucket(box_h_px):
    """Bucket by apparent cone height -- the only distance proxy available without depth GT."""
    if box_h_px >= 60:
        return "near"
    if box_h_px >= 30:
        return "mid"
    return "far"


def eval_yolo_pose(weights, data_root, n_kpt, K, corrupt, level, device, limit):
    from ultralytics import YOLO

    model = YOLO(weights)
    img_dir = data_root / "images" / "test"
    lbl_dir = data_root / "labels" / "test"
    obj_small = cone_object_points(n_kpt, "small")
    obj_large = cone_object_points(n_kpt, "large")

    errors = defaultdict(list)
    pnp_failures = 0
    images = sorted(img_dir.iterdir())[:limit] if limit else sorted(img_dir.iterdir())

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        H, W = img.shape[:2]
        img = corrupt_image(img, corrupt, level)

        gt = load_labels(lbl_path, n_kpt)
        if not gt:
            continue

        res = model.predict(img, verbose=False, device=device)[0]
        if res.keypoints is None or len(res.boxes) == 0:
            continue
        pred_boxes = res.boxes.xywh.cpu().numpy()
        pred_kpts = res.keypoints.xy.cpu().numpy()

        for cls, box, kpts_n, vis in gt:
            gt_px = kpts_n * [W, H]
            valid = vis > 0
            if valid.sum() < 4:
                continue

            obj = obj_large if cls == LARGE_CLASS_ID else obj_small
            gt_depth = solve_depth(obj[valid], gt_px[valid], K)
            if gt_depth is None:
                continue

            # Match the prediction to this cone by box centre.
            gt_c = box[:2] * [W, H]
            d = np.linalg.norm(pred_boxes[:, :2] - gt_c, axis=1)
            j = int(np.argmin(d))
            if d[j] > 0.5 * box[2] * W:
                continue  # no prediction on this cone

            pd_depth = solve_depth(obj[valid], pred_kpts[j][valid], K)
            if pd_depth is None:
                pnp_failures += 1
                continue

            bucket = distance_bucket(box[3] * H)
            rel = abs(pd_depth - gt_depth) / gt_depth
            errors[bucket].append(rel)
            errors["all"].append(rel)

    return errors, pnp_failures


def summarize(errors, pnp_failures, label):
    print(f"\n=== {label} ===")
    if not errors["all"]:
        print("  no valid measurements")
        return {}
    out = {}
    for bucket in ("near", "mid", "far", "all"):
        e = np.array(errors[bucket])
        if len(e) == 0:
            continue
        out[bucket] = {
            "n": len(e),
            "median_rel_err": float(np.median(e)),
            "p90_rel_err": float(np.percentile(e, 90)),
        }
        print(f"  {bucket:<5} n={len(e):>6,}  median={np.median(e)*100:5.1f}%  p90={np.percentile(e,90)*100:6.1f}%")
    if pnp_failures:
        print(f"  PnP failures: {pnp_failures:,}")
    out["pnp_failures"] = pnp_failures
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="Trained YOLO-pose .pt")
    p.add_argument("--data-root", required=True, type=Path)
    p.add_argument("--n-kpt", required=True, type=int, choices=[4, 6, 7, 8])
    p.add_argument("--corrupt", default="none",
                   choices=["none", "blur", "noise", "dark",
                            "sun", "overcast", "shadow", "backlight"])
    p.add_argument("--level", default=0.0, type=float, help="Corruption strength, 0-1")
    p.add_argument("--device", default="0")
    p.add_argument("--limit", default=0, type=int, help="Cap test images (0 = all)")
    p.add_argument("--focal", default=1000.0, type=float,
                   help="Nominal focal length in px; identical across models, so comparisons hold")
    p.add_argument("--out", type=Path, help="Write results as JSON")
    args = p.parse_args()

    img_dir = args.data_root / "images" / "test"
    first = next(iter(sorted(img_dir.iterdir())), None)
    if first is None:
        sys.exit(f"No test images in {img_dir}")
    h, w = cv2.imread(str(first)).shape[:2]
    K = np.array([[args.focal, 0, w / 2], [0, args.focal, h / 2], [0, 0, 1]], dtype=np.float64)

    errors, failures = eval_yolo_pose(
        args.weights, args.data_root, args.n_kpt, K, args.corrupt, args.level, args.device, args.limit
    )
    tag = f"{args.n_kpt}kpt" + (f" / {args.corrupt}@{args.level}" if args.corrupt != "none" else "")
    result = summarize(errors, failures, tag)

    if args.out:
        args.out.write_text(json.dumps(
            {"n_kpt": args.n_kpt, "corrupt": args.corrupt, "level": args.level, "metrics": result},
            indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()

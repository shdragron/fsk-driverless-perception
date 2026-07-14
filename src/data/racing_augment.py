"""Lighting augmentations for the conditions a race car actually drives through.

Ultralytics' stock augmentation covers geometry (flip, scale, mosaic) and a mild HSV jitter.
That HSV jitter is why plain brightness changes already do nothing to these models -- but it
does not cover the *structured* lighting failures that break cone perception in practice: sun
straight into the lens, a shadow band across the track, or backlit cones that lose their colour
entirely (and with it the blue/yellow class the planner depends on).

Held out on purpose: `noise` and `blur`. Those stay evaluation-only, so the robustness numbers
measure generalisation to corruptions the model has never seen rather than memorisation of its
own augmentation pipeline. Train on what you test on and the robustness result is worthless.

Usage -- attach to a trainer before .train():

    from src.data.racing_augment import attach_racing_augment
    model = YOLO("yolo26n-pose.pt")
    attach_racing_augment(model, p=0.5)
    model.train(...)
"""
import cv2
import numpy as np

# Only lighting. noise/blur are deliberately absent -- see the module docstring.
TRAIN_CORRUPTIONS = ("sun", "overcast", "shadow", "backlight")


def apply_sun(img, level, rng):
    """Low sun into the lens: a bright disc plus veiling glare that lifts the blacks."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Put the sun somewhere plausible: near the horizon, either side of frame.
    cx = w * rng.uniform(0.15, 0.85)
    cy = h * rng.uniform(0.10, 0.40)
    r = max(h, w) * rng.uniform(0.25, 0.45)
    glow = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r ** 2))[..., None]
    out = f + level * 200 * glow
    return np.clip(out * (1 - 0.25 * level) + level * 60, 0, 255).astype(np.uint8)


def apply_overcast(img, level, rng):
    """Flat cloud light: contrast collapses toward mid-grey and the cast goes cool."""
    f = img.astype(np.float32)
    out = (f - 128) * (1 - 0.55 * level) + 128 + 10 * level
    out[..., 0] *= 1 + 0.08 * level   # BGR: lift blue
    out[..., 2] *= 1 - 0.06 * level   # drop red
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shadow(img, level, rng):
    """A shaded band across the frame -- bridge, tree line, grandstand."""
    h, w = img.shape[:2]
    f = img.astype(np.float32)
    band = np.ones((h, w), np.float32)
    width = rng.uniform(0.2, 0.5)
    x0 = int(w * rng.uniform(0, 1 - width))
    x1 = int(x0 + w * width)
    band[:, x0:x1] = 1 - 0.75 * level
    band = cv2.GaussianBlur(band, (0, 0), sigmaX=w * 0.02)  # soft penumbra
    return np.clip(f * band[..., None], 0, 255).astype(np.uint8)


def apply_backlight(img, level, rng):
    """Sun behind the cones: near-silhouettes, colour drained -- blue/yellow stop being separable."""
    f = img.astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)[..., None]
    desat = f * (1 - 0.7 * level) + gray * 0.7 * level
    crushed = np.clip((desat - 40 * level) * (1 - 0.45 * level), 0, 255)
    return np.clip(crushed + 70 * level * (gray / 255.0), 0, 255).astype(np.uint8)


_FNS = {"sun": apply_sun, "overcast": apply_overcast,
        "shadow": apply_shadow, "backlight": apply_backlight}


class RacingLighting:
    """Picks one lighting condition per image, at a random strength, with probability p.

    Keypoint labels are untouched: every transform here is photometric, so cone geometry -- and
    therefore the keypoints and boxes -- is unchanged.
    """

    def __init__(self, p=0.5, kinds=TRAIN_CORRUPTIONS, level_range=(0.3, 1.0), seed=17):
        self.p = p
        self.kinds = tuple(kinds)
        self.level_range = level_range
        self.rng = np.random.default_rng(seed)

    def __call__(self, labels):
        if self.rng.random() >= self.p:
            return labels
        kind = self.kinds[self.rng.integers(len(self.kinds))]
        level = self.rng.uniform(*self.level_range)
        labels["img"] = _FNS[kind](labels["img"], level, self.rng)
        return labels


def attach_racing_augment(model, p=0.5, kinds=TRAIN_CORRUPTIONS, seed=17):
    """Insert RacingLighting into the training dataset's transform chain.

    Hooks `on_train_start`, at which point the trainer's dataloader (and its dataset's transform
    Compose) exists. Photometric-only, so it can go anywhere in the chain; appending keeps it
    after the geometric transforms, closest to what the camera would actually produce.
    """
    aug = RacingLighting(p=p, kinds=kinds, seed=seed)

    def _hook(trainer):
        dataset = trainer.train_loader.dataset
        transforms = getattr(dataset, "transforms", None)
        if transforms is None or not hasattr(transforms, "transforms"):
            raise RuntimeError("could not find the dataset transform chain to attach to")
        # Sit before Format, which converts the image to a tensor -- after that, cv2 ops fail.
        chain = transforms.transforms
        idx = next((i for i, t in enumerate(chain) if type(t).__name__ == "Format"), len(chain))
        chain.insert(idx, aug)
        names = ", ".join(k for k in aug.kinds)
        print(f"racing-lighting augment: p={p}, kinds=[{names}] (noise/blur held out for eval)")

    model.add_callback("on_train_start", _hook)
    return model

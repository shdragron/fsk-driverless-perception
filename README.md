# FSK Driverless Perception

Cone detection and keypoint estimation for Formula Student Driverless, in one model.

> **This is a fork of [cv-core/MIT-Driverless-CV-TrainingInfra](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra)** (Apache-2.0), restructured around a single-stage YOLO26-pose pipeline. See [NOTICE](NOTICE) for what was kept, what was changed, and the licensing that follows.

## What this changes

The original pipeline is two stages: **CVC-YOLOv3** detects cone boxes, then **RektNet** takes each
crop and regresses 7 keypoints on it. Those keypoints exist to feed PnP, which recovers a cone's
3D position from a single camera.

This fork replaces both with **one YOLO26n-pose model** that emits boxes and keypoints together,
then asks how many keypoints are actually needed. Fewer means a smaller
head and an easier regression — but PnP needs at least four correspondences, and every one you
remove takes away the redundancy it uses to average out error.

RektNet is retained as the two-stage baseline, retrained on the same data with its geometric loss
intact, so the comparison is architecture-vs-architecture rather than model-vs-memory.

## Data caveats

**1. The BRT split leaks — 88.9% of it.** BRT Cone Pose shuffles frames at random, but FSOCO frames
come from continuous driving footage: `mms_00185` and `mms_00186` are the same scene 1/30 s apart.
Measured on the shipped split, **88.9% of frames have a near-neighbour (±1–3) in a different
split**. Test scores on it are inflated; the model has effectively seen the test set.

`src/data/resplit_temporal.py` re-partitions by *contiguous frame blocks per team*, with a gap at
each boundary, taking adjacent-frame leakage to **0.0%**. Everything here uses that split.

**2. BRT ships no `flip_idx`, and Ultralytics defaults to `fliplr=0.5`.** Without it, a horizontal
flip mirrors the pixels while leaving keypoint order untouched — silently training half your data
on swapped left/right landmarks. No error, no warning, just a worse model. Every yaml written here
sets `flip_idx`.

**3. Depth error here is relative, not metres.** BRT has no ground-truth depth (2D annotations,
pooled from many teams' cameras, unknown intrinsics). PnP on the *ground-truth* keypoints is the
reference pose; the metric is how far a model's *predicted* keypoints move it, with one nominal
focal length applied identically to every model. **Model-vs-model comparison is sound. The absolute
percentages are not metres**, and they are not comparable to the paper's published 0.5 m figure.

## Setup

```bash
conda create -n fsk python=3.11 -y && conda activate fsk
pip install -r requirements.txt
```

YOLO26 checkpoints download on first use; they are not vendored (see [NOTICE](NOTICE) §2).

## Data

Get the [BRT Cone Pose Dataset](https://github.com/Bauman-Racing-Team/BRT-Cone-Pose-Dataset)
(~3.2 GB — images are FSOCO's, annotations are BRT's; respect both licenses), then:

```bash
# Re-split to kill the temporal leak, and write the flip_idx BRT omits
python -m src.data.resplit_temporal --source <brt-cone-pose-11k> --out <data>/brt-clean-8kpt
python -m src.data.make_keypoint_variants --source <data>/brt-clean-8kpt --out-root <data>
python -m src.data.validate_dataset --root <data>/brt-clean-8kpt
```

The original MIT datasets are gone: the GCS bucket returns 403 (`The billing account for the owning
project is disabled`). BRT Cone Pose is what makes this work reproducible at all.

## Run

```bash
bash scripts/run_all.sh      # train 8/6/4-kpt, evaluate depth, train + evaluate RektNet
```

## Keypoint layout, and why 8 is not always 8

BRT's keypoints are left/right pairs down the cone silhouette:

```
   0  1     top pair
   2  3     mid pair
   6  7     extra pair  <- large orange cones only
   4  5     base pair
```

**kpt6/7 exist on only 5.3% of cones.** A small cone physically has no fourth stripe boundary. So:

- **YOLO-pose** handles this natively — absent keypoints have `visibility=0` and drop out of the loss.
- **RektNet** did not: it is heatmap-based and the upstream code assumes all keypoints exist. Feeding
  it 8 keypoints naively means dropping every cone that lacks kpt6/7 — **94.7% of the dataset**, and
  not at random: the discarded cones are the occluded, distant, hard ones. The remaining set is
  easier than the one YOLO-pose is scored on, so the two are no longer comparable.

  `src/train_rektnet.py` therefore adds a **visibility mask**: absent points are excluded from the
  location loss, from the heatmap target, and from any geometric term that involves them. This is an
  extension, not a reproduction — **the original paper never addresses keypoint visibility.**

## Fidelity to the original RektNet

Against the paper (arXiv:2007.13971) and the upstream code:

| | Original | Here | |
|---|---|---|---|
| Geometric loss weights | **γvert=0.038, γhorz=0.055** (paper Eq. 2, Bayesian-optimised) | same | ✅ |
| Geometric loss form | collinearity + parallelism of unit vectors. Despite the name, **there is no projective cross-ratio term** | same, re-derived for BRT's chain | ✅ |
| Input size / loss / optimiser | 80×80, `l1_softargmax`, Adam | same | ✅ |
| Early stopping | tolerance 8 | same | ✅ |
| Learning rate | 0.1 | **1e-3** | ⚠️ deviation — 0.1 with Adam diverges on this data |
| Augmentation | **none** | horizontal flip + pair swap | ⚠️ addition; paper is silent |
| Keypoint visibility | not addressed — all 7 assumed present | masked | ⚠️ extension, see above |
| Reported accuracy | **depth: <0.5 m mean, <5 cm std to 20 m**. No pixel-error figure exists | depth via PnP | — |

Note the `[0, 0.15]` range in upstream `train_eval_hyper.py` is the *search prior*, not a result.
The results are 0.038 / 0.055, and they are asymmetric.

## Robustness

Corruptions are applied to the full frame before inference:

| | models |
|---|---|
| `noise` | high-ISO sensor noise |
| `blur` | directional motion blur (a car smears the scene along its heading) |
| `sun` | low sun into the lens — bloom plus veiling glare |
| `overcast` | flat cloud light; contrast collapses toward mid-grey |
| `shadow` | a shaded band across the frame (bridge, tree line) |
| `backlight` | sun behind the cones — silhouettes, colour destroyed |

Plus `--box-noise`, for RektNet only: it jitters the ground-truth box before cropping, standing in
for a real detector's imperfect output. **That failure mode exists only in a two-stage pipeline.**

`src/data/racing_augment.py` can train with the *lighting* conditions (`--racing-aug`). Noise and
blur are deliberately **held out of training** — train on what you test on and the robustness number
measures memorisation, not generalisation.

**Read error alongside survivor counts.** Under heavy corruption the hard cones stop being detected
at all and leave the error statistic entirely, which makes the error *improve*. `summarize.py` prints
both for this reason.

## Deployment

`src/eval/benchmark.py` measures latency, throughput and peak memory — the numbers that decide
whether this runs on the car.

The two pipelines scale differently:

- **single-stage**: one forward pass per *frame*, regardless of cone count
- **two-stage**: one detection pass per frame, then one RektNet pass **per cone**

The per-cone cost lands hardest when the frame is busiest — entering a slalom with twenty cones in
view. The benchmark sweeps cone count rather than reporting a single figure.

```bash
python -m src.eval.benchmark \
    --pose-weights <8kpt> <6kpt> <4kpt> --pose-labels 8kpt 6kpt 4kpt \
    --rektnet-weights <rektnet.pt> --device cuda:0 --half
```

## Layout

```
src/
  data/     resplit_temporal, make_keypoint_variants, validate_dataset,
            make_cone_crops, racing_augment
  models/   keypoint_net, resnet, cross_ratio_loss    (RektNet; from upstream)
  eval/     eval_pose, eval_rektnet, summarize, benchmark
  viz/      plot_metrics, gallery, zoom, predict_image, predict_batch
  train_pose.py      YOLO26-pose (single-stage)
  train_rektnet.py   RektNet (two-stage baseline)
scripts/run_all.sh
results/
```

## Status

Results are being regenerated on the leak-free split. Numbers measured on BRT's shipped split are
not published here: the 88.9% temporal leak inflates them.

## License

**AGPL-3.0** ([LICENSE](LICENSE)) — required because this depends on
[Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0.

Upstream code is Apache-2.0; its text is preserved at [LICENSE.apache-2.0](LICENSE.apache-2.0) and
its attribution in [NOTICE](NOTICE). Apache-2.0 is one-way compatible with AGPL-3.0, so the combined
work is distributed under the AGPL.

Cite the original paper (arXiv:2007.13971) and, if you use the data, BRT and FSOCO — see
[NOTICE](NOTICE) §4.

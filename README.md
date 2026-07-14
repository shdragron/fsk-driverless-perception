# FSK Driverless Perception

Cone detection and keypoint estimation for Formula Student Driverless, in one model.

> **This is a fork of [cv-core/MIT-Driverless-CV-TrainingInfra](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra)** (Apache-2.0), restructured around a single-stage YOLO26-pose pipeline. See [NOTICE](NOTICE) for what was kept, what was changed, and the licensing that follows from it.

## What this changes

The original pipeline is two stages: **CVC-YOLOv3** detects cone boxes, then **RektNet** takes each
crop and regresses 7 keypoints on it. The keypoints exist to feed PnP, which recovers each cone's
3D position from a single camera.

This fork replaces both with **one YOLO26n-pose model** that emits boxes and keypoints together,
and then asks the obvious follow-up: *how many keypoints do you actually need?* Fewer keypoints
mean a smaller head and an easier regression problem — but PnP needs at least four correspondences,
and every one you remove takes away the redundancy it uses to average out error.

RektNet is retained as the two-stage baseline, retrained on the same data, so the comparison is
architecture-vs-architecture rather than model-vs-memory.

## Two things worth knowing before you trust any number here

**1. The BRT split leaks, badly.** BRT Cone Pose shuffles frames at random, but FSOCO frames come
from continuous driving footage — `mms_00185` and `mms_00186` are the same scene 1/30 s apart.
**88.9% of frames have a near-neighbour in a different split.** Test scores measured on the shipped
split are inflated: the model has effectively seen the test set.

`src/data/resplit_temporal.py` re-partitions by *contiguous frame blocks per team*, with a gap at
each boundary, bringing adjacent-frame leakage to **0%**. Every result here uses that split. If you
use BRT for anything, re-split it first.

**2. Depth error is relative, not absolute.** BRT has no ground-truth depth — it is a 2D-annotated
dataset pooled from many teams' cameras, so true intrinsics are unknown. PnP on the *ground-truth*
keypoints is the reference pose, and the metric is how far each model's *predicted* keypoints move
it, with a fixed nominal focal length applied identically to every model. **Model-vs-model
comparison is sound; the absolute percentages are not metres.**

## Setup

```bash
conda create -n fsk python=3.11 -y && conda activate fsk
pip install -r requirements.txt
```

Pretrained YOLO26 checkpoints download themselves on first use. They are not vendored here (see
[NOTICE](NOTICE) §2).

## Data

Get the [BRT Cone Pose Dataset](https://github.com/Bauman-Racing-Team/BRT-Cone-Pose-Dataset)
(~3.2 GB; images are FSOCO's, annotations are BRT's — respect both licenses), then:

```bash
# Re-split to remove temporal leakage, and inject the flip_idx BRT omits
python -m src.data.resplit_temporal --source <brt-cone-pose-11k> --out <data>/brt-clean-8kpt
python -m src.data.make_keypoint_variants --source <data>/brt-clean-8kpt --out-root <data>
python -m src.data.validate_dataset --root <data>/brt-clean-8kpt
```

> **`flip_idx`**: BRT's `data.yaml` omits it, and Ultralytics defaults to `fliplr=0.5`. Without it,
> a horizontal flip mirrors the pixels while leaving keypoint order untouched — silently training
> half your data on swapped left/right landmarks, with no error and no warning. The scripts here
> always write `flip_idx`.

## Run

```bash
bash scripts/run_all.sh          # train 8/6/4-kpt, evaluate, train + evaluate RektNet
```

Or piecemeal:

```bash
python -m src.train_pose --task pose --data <data>/brt-clean-8kpt/data.yaml --epochs 100 --batch 32
python -m src.eval.eval_pose --weights <best.pt> --data-root <data>/brt-clean-8kpt --n-kpt 8
python -m src.eval.summarize --results results/metrics
python -m src.viz.plot_metrics --results results/metrics --out results/ablation.png
```

## Robustness

Corruptions are applied to the full frame before inference, at three strengths each:

| corruption | what it models |
|---|---|
| `noise` | high-ISO sensor noise |
| `blur` | directional motion blur (a car smears the scene along its heading) |
| `sun` | low sun into the lens — bloom plus veiling glare |
| `overcast` | flat cloud light; contrast collapses toward mid-grey |
| `shadow` | a shaded band across the frame (bridge, tree line) |
| `backlight` | sun behind the cones — silhouettes, colour destroyed |

Plus `--box-noise` for RektNet only: it jitters the ground-truth box before cropping, standing in
for a real detector's imperfect output. That is a failure mode a two-stage pipeline has and a
one-stage one does not.

**Read the error numbers alongside the survivor counts.** Under heavy corruption the hard cones
stop being detected at all and drop out of the error statistic entirely — which makes the error
*improve*. `summarize.py` prints both for this reason.

## Layout

```
src/
  data/     resplit_temporal, make_keypoint_variants, validate_dataset, make_cone_crops
  models/   keypoint_net, resnet, cross_ratio_loss    (RektNet; from upstream)
  eval/     eval_pose, eval_rektnet, summarize
  viz/      plot_metrics, gallery, zoom
  train_pose.py      YOLO26-pose (single-stage)
  train_rektnet.py   RektNet (two-stage baseline)
scripts/run_all.sh
results/
```

## License

**AGPL-3.0** (see [LICENSE](LICENSE)) — required because this depends on
[Ultralytics](https://github.com/ultralytics/ultralytics), which is AGPL-3.0.

Upstream code is Apache-2.0; its license text is preserved verbatim at
[LICENSE.apache-2.0](LICENSE.apache-2.0) and its attribution in [NOTICE](NOTICE). Apache-2.0 is
one-way compatible with AGPL-3.0, so the combined work is distributed under the AGPL.

If you use this, cite the original MIT Driverless paper (arXiv:2007.13971) and, if you use the
data, the BRT and FSOCO datasets — see [NOTICE](NOTICE) §4.

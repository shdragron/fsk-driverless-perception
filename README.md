# FSK Driverless Perception

Single-stage cone detection and keypoint estimation for Formula Student Driverless.

The MIT pipeline detects cones with YOLOv3, then runs RektNet on each crop to regress keypoints.
This collapses both into **one YOLO26n-pose model** that emits boxes and keypoints in a single pass,
and measures how many keypoints PnP depth recovery actually needs.

> Fork of [cv-core/MIT-Driverless-CV-TrainingInfra](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra) (Apache-2.0). See [NOTICE](NOTICE).

---

## Model

| | detector | keypoints | params |
|---|---|---|---|
| **single-stage** | YOLO26n-pose | same pass | 2.75 – 2.98 M |
| **two-stage** | YOLO26n | **RektNet-V**, one pass **per cone** | 2.6 M + 3.0 M |
| **MIT original** | YOLOv3 | RektNet, 7 kpt, per cone | 103.8 M + 3.0 M |

The first two share a YOLO26n backbone, so comparing them isolates the architectural question — one
pass or two — from detector generation. The MIT original is there to show what that generation gap
is worth. Keypoint count (4 / 6 / 8) is ablated throughout.

Keypoints feed PnP, which recovers each cone's 3D position from a single camera. PnP needs four
correspondences; anything beyond that is redundancy it uses to average out error.

### RektNet-V

RektNet's architecture — 80×80 crop, heatmap, soft-argmax, cross-ratio loss at the paper's tuned
weights (γvert 0.038, γhorz 0.055) — with three changes BRT's labels require:

- **Visibility masking** (the *V*). kpt6/7 exist on only 5.3% of cones; a small cone has no fourth
  stripe boundary. RektNet assumes every keypoint is present, so training on 8 would mean dropping
  every cone without them — the small, distant, occluded ones. Absent points are masked out of the
  loss, the heatmap target, and PnP instead. This is also what keeps both pipelines scored on the
  same cones.
- **8 keypoints**, which the mask makes reachable.
- **Re-derived geometry.** RektNet's chains are rooted at an apex; BRT has none, so the collinearity
  and parallelism terms are rebuilt for its all-pairs layout.

The MIT-original row runs the **unmodified 7-keypoint RektNet**, on a 7-keypoint approximation of
BRT: the apex is synthesised as the midpoint of BRT's top pair. It reproduces the architecture, not
the labelling.

## Dataset

| | |
|---|---|
| images | **[FSOCO](https://fsoco.github.io/fsoco-dataset/)** |
| box + keypoint annotations | **[BRT Cone Pose](https://github.com/Bauman-Racing-Team/BRT-Cone-Pose-Dataset)** (Bauman Racing Team, CC BY 4.0) |
| frames | 11,308 — train 9,165 / val 1,045 / test 1,098 |
| cones | 103,653 |
| classes | blue 41,015 · yellow 44,865 · orange 12,690 · orange_big 5,083 |
| keypoints | 8 per cone, left/right pairs |

```
   0  1   top          kpt6/7 exist on large orange cones only (5.3%).
   2  3   mid          A small cone has no fourth stripe boundary, so
   6  7   extra        they carry a visibility flag and are masked out
   4  5   base         of the loss, the heatmap target, and PnP.
```

**Re-split before use.** BRT shuffles frames at random, but FSOCO frames come from continuous
footage — `mms_00185` and `mms_00186` are the same scene 1/30 s apart. 88.9% of frames have a
near-neighbour in a different split, which inflates test scores. `resplit_temporal.py` partitions by
contiguous per-team frame blocks with boundary gaps → **0.0% leakage**. All results below use it.

**`flip_idx` is missing from BRT's yaml.** Ultralytics defaults to `fliplr=0.5`; without it a
horizontal flip mirrors the pixels but not the keypoint order, silently training half the data on
swapped left/right landmarks. Every yaml written here sets it.

## Accuracy

Depth error is **relative, not metres** — the reference pose is PnP on the ground-truth keypoints,
and the metric is how far predicted keypoints move it, with one nominal focal length applied to
every model.

### Single-stage

| kpt | box mAP50 | pose mAP50-95 | depth err (median) | p90 |
|---|---|---|---|---|
| 8 | 0.9597 | 0.8772 | — | — |
| 6 | — | — | — | — |
| 4 | — | — | — | — |

### Two-stage (YOLO26n + RektNet-V)

| kpt | pose mAP50-95 | depth err (median) | p90 |
|---|---|---|---|
| 8 | — | — | — |
| 6 | — | — | — |
| 4 | — | — | — |

### MIT original (YOLOv3 + RektNet, 7 kpt)

| | box mAP50 | pose mAP50-95 | depth err (median) | p90 |
|---|---|---|---|---|
| 7 | — | — | — | — |

*(sweep in progress)*

## Robustness

Corruptions applied to the full frame at inference. Median depth error:

| | single-stage 8 / 6 / 4 | two-stage 8 / 6 / 4 | MIT original |
|---|---|---|---|
| clean | — | — | — |
| `noise` high-ISO sensor noise | — | — | — |
| `blur` directional motion blur | — | — | — |
| `sun` bloom + veiling glare | — | — | — |
| `overcast` flat light, low contrast | — | — | — |
| `shadow` shaded band across frame | — | — | — |
| `backlight` silhouettes, colour lost | — | — | — |
| `box-noise` jittered detection box | n/a | — | — |

`box-noise` has no single-stage column: a mis-placed crop is a failure mode only a two-stage
pipeline has.

Lighting doubles as *training* augmentation (`--racing-aug`); noise and blur are held out of
training so the numbers measure generalisation.

**Read error alongside survivor counts.** Under heavy corruption the hard cones stop being detected
and leave the error statistic, which makes the error *improve*. `summarize.py` prints both.

*(sweep in progress)*

## Compute — PyTorch

RTX 3060, 640×640, FP32, batch 1.

### Single-stage — one pass per frame

| kpt | params | mean | p95 | FPS |
|---|---|---|---|---|
| 8 | 2.98 M | 66.9 ms | 68.9 ms | 14.9 |
| 6 | 2.86 M | 55.6 ms | 65.8 ms | 18.0 |
| 4 | 2.75 M | 51.4 ms | 53.8 ms | 19.5 |

### Two-stage — detector + one RektNet pass per cone

| cones in frame | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| YOLO26n + RektNet-V (ms) | — | — | — | — | — |
| YOLOv3 + RektNet (7kpt) (ms) | — | — | — | — | — |

*(pending RektNet training)*

## Compute — TensorRT

Target hardware is **Jetson Orin**; measured on RTX 3060, so absolute latency differs but the
model-vs-model and single-vs-two-stage comparisons carry over.

### Single-stage — FP16

| kpt | mean | p95 | FPS | speedup vs PyTorch |
|---|---|---|---|---|
| 8 | — | — | — | — |
| 6 | — | — | — | — |
| 4 | — | — | — | — |

### Two-stage — FP16

| cones in frame | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| total (ms) | — | — | — | — | — |
| FPS | — | — | — | — | — |

*(pending)*

## Run

```bash
conda create -n fsk python=3.11 -y && conda activate fsk
pip install -r requirements.txt

python -m src.data.resplit_temporal --source <brt-cone-pose-11k> --out <data>/brt-clean-8kpt
python -m src.data.make_keypoint_variants --source <data>/brt-clean-8kpt --out-root <data>
bash scripts/run_all.sh
```

## RektNet-V vs the original

| | original | here |
|---|---|---|
| geometric loss weights | γvert 0.038, γhorz 0.055 (paper Eq. 2) | same |
| loss form | collinearity + parallelism (no actual cross-ratio term, despite the name) | same, re-derived for BRT's chain |
| input / loss / optimiser | 80×80, `l1_softargmax`, Adam | same |
| early stopping | tolerance 8 | same |
| learning rate | 0.1 | **1e-3** — 0.1 diverges with Adam here |
| augmentation | none | **+ horizontal flip** |
| keypoint visibility | not addressed | **+ masked** |
| reported accuracy | depth < 0.5 m mean, < 5 cm std to 20 m | depth via PnP |

## Layout

```
src/
  data/   resplit_temporal, make_keypoint_variants, validate_dataset,
          make_cone_crops, racing_augment
  models/ keypoint_net, resnet, cross_ratio_loss     (RektNet, from upstream)
  eval/   eval_pose, eval_rektnet, summarize, benchmark
  viz/    plot_metrics, gallery, zoom, predict_image, predict_batch
  train_pose.py       single-stage
  train_rektnet.py    RektNet-V, and the 7-kpt original
scripts/run_all.sh
```

## License

**AGPL-3.0** ([LICENSE](LICENSE)) — required by the Ultralytics dependency. Upstream Apache-2.0 text
and attribution preserved in [LICENSE.apache-2.0](LICENSE.apache-2.0) and [NOTICE](NOTICE).

Cite arXiv:2007.13971 (MIT Driverless) and, if using the data, BRT and FSOCO.

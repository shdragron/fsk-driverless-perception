# FSK Driverless Perception

Single-stage cone detection and keypoint estimation for Formula Student Driverless.

The MIT pipeline detects cones with YOLOv3, then runs RektNet on each crop to regress keypoints.
This collapses both into **one YOLO26n-pose model** that emits boxes and keypoints in a single
pass, and measures how many keypoints PnP depth recovery actually needs.

> Fork of [cv-core/MIT-Driverless-CV-TrainingInfra](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra) (Apache-2.0). See [NOTICE](NOTICE).

---

## Model

Three configurations, of which only two are runnable:

| | detector | keypoints | params | |
|---|---|---|---|---|
| **MIT original** | YOLOv3 | RektNet, 7 kpt, per cone | 62 M + 3.0 M | not reproducible — see below |
| **two-stage** | YOLO26n | RektNet, per cone | 2.6 M + 3.0 M | |
| **single-stage** | YOLO26n-pose | same pass | 2.75 – 2.98 M | proposed |

The MIT original cannot be retrained: both its datasets (`YOLO_Dataset.zip`, `RektNet_Dataset.zip`)
are on a GCS bucket that now returns 403 (*billing account disabled*). RektNet's labels and
pretrained weights survive in a fork, but the images do not, so there is nothing to train on.

The comparison here is therefore **single-stage vs two-stage on the same YOLO26n backbone**. That
isolates the architectural question — one pass or two — rather than confounding it with the eight
years of detector progress between YOLOv3 and YOLO26.

Keypoint count (4 / 6 / 8) is ablated on both.

Keypoints feed PnP, which recovers each cone's 3D position from a single camera. PnP needs four
correspondences; anything beyond that is redundancy it uses to average out error — which is what
the keypoint-count ablation tests.

BRT's keypoints are left/right pairs down the cone:

```
   0  1   top          kpt6/7 exist on large orange cones only (5.3%).
   2  3   mid          A small cone has no fourth stripe boundary, so
   6  7   extra        they carry a visibility flag and are masked out
   4  5   base         of the loss, the heatmap, and PnP.
```

## Dataset

[BRT Cone Pose](https://github.com/Bauman-Racing-Team/BRT-Cone-Pose-Dataset) — 11,523 FSOCO frames,
8 keypoints per cone, 5 classes. The original MIT datasets are gone (GCS bucket returns 403:
*billing account disabled*).

Two fixes it needs before use:

**Temporal leakage.** BRT shuffles frames at random, but FSOCO frames come from continuous footage —
`mms_00185` and `mms_00186` are the same scene 1/30 s apart. **88.9% of frames have a near-neighbour
in a different split.** `resplit_temporal.py` partitions by contiguous per-team frame blocks with
boundary gaps → **0.0% leakage**.

Measured cost of the leak, 8kpt:

| | leaked split | clean split |
|---|---|---|
| box mAP50 | 0.9684 | **0.9597** |
| pose mAP50-95 | 0.9039 | **0.8772** |

The keypoint metric falls three times as far as the box metric — memorising an adjacent frame helps
most with the exact pixel location of a landmark.

**Missing `flip_idx`.** Ultralytics defaults to `fliplr=0.5`; without `flip_idx` a horizontal flip
mirrors the pixels but not the keypoint order, silently training half the data on swapped left/right
landmarks. Every yaml written here sets it.

## Accuracy

*(regenerating on the clean split — table lands when the sweep finishes)*

| model | box mAP50 | pose mAP50-95 | PnP depth err (median) | p90 |
|---|---|---|---|---|
| 8kpt | 0.9597 | 0.8772 | — | — |
| 6kpt | — | — | — | — |
| 4kpt | — | — | — | — |
| RektNet 8kpt | — | — | — | — |

Depth error is **relative, not metres**: BRT has no ground-truth depth, so PnP on the ground-truth
keypoints is the reference and the metric is how far predicted keypoints move it, with one nominal
focal length applied to every model. Model-vs-model is sound; the absolute figure is not comparable
to the paper's 0.5 m.

## Robustness

Corruptions applied to the full frame at inference:

| | |
|---|---|
| `noise` | high-ISO sensor noise |
| `blur` | directional motion blur |
| `sun` | low sun into the lens — bloom + veiling glare |
| `overcast` | flat light, contrast collapses |
| `shadow` | shaded band across the frame |
| `backlight` | silhouetted cones, colour destroyed |
| `box-noise` | jittered detection box — **two-stage only** |

Lighting can also be used as *training* augmentation (`--racing-aug`). Noise and blur are held out
of training so the robustness numbers measure generalisation, not memorisation.

*(results pending)*

**Read error alongside survivor counts.** Under heavy corruption the hard cones stop being detected
and leave the error statistic, which makes the error *improve*. `summarize.py` prints both.

## Compute — PyTorch

RTX 3060, 640×640, FP32, batch 1:

| model | params | mean | p95 | FPS |
|---|---|---|---|---|
| 8kpt | 2.98 M | 66.9 ms | 68.9 ms | 14.9 |
| 6kpt | 2.86 M | 55.6 ms | 65.8 ms | 18.0 |
| 4kpt | 2.75 M | 51.4 ms | 53.8 ms | 19.5 |

The two pipelines scale differently: single-stage is **one pass per frame**; two-stage is one
detection pass **plus one RektNet pass per cone**. The benchmark sweeps cone count rather than
reporting a single figure.

*(two-stage numbers pending RektNet training)*

## Compute — TensorRT

*(pending)*

Target hardware is **Jetson Orin**. Numbers measured here are on an RTX 3060 — absolute latency will
differ, but the model-vs-model and single-vs-two-stage comparisons carry over.

```bash
python -m src.eval.benchmark --pose-weights <...> --device cuda:0 --half
```

## Run

```bash
conda create -n fsk python=3.11 -y && conda activate fsk
pip install -r requirements.txt

python -m src.data.resplit_temporal --source <brt-cone-pose-11k> --out <data>/brt-clean-8kpt
python -m src.data.make_keypoint_variants --source <data>/brt-clean-8kpt --out-root <data>
bash scripts/run_all.sh
```

## Fidelity to the original RektNet

| | original | here |
|---|---|---|
| geometric loss weights | γvert 0.038, γhorz 0.055 (paper Eq. 2) | same |
| loss form | collinearity + parallelism (no actual cross-ratio term, despite the name) | same, re-derived for BRT's chain |
| input / loss / optimiser | 80×80, `l1_softargmax`, Adam | same |
| early stopping | tolerance 8 | same |
| learning rate | 0.1 | **1e-3** — 0.1 diverges with Adam here |
| augmentation | none | **+ horizontal flip** |
| keypoint visibility | not addressed | **+ masked** |
| reported accuracy | depth < 0.5 m mean, < 5 cm std to 20 m. No pixel-error figure | depth via PnP |

## Layout

```
src/
  data/   resplit_temporal, make_keypoint_variants, validate_dataset,
          make_cone_crops, racing_augment
  models/ keypoint_net, resnet, cross_ratio_loss     (RektNet, from upstream)
  eval/   eval_pose, eval_rektnet, summarize, benchmark
  viz/    plot_metrics, gallery, zoom, predict_image, predict_batch
  train_pose.py       single-stage
  train_rektnet.py    two-stage (RektNet)
scripts/run_all.sh
```

## License

**AGPL-3.0** ([LICENSE](LICENSE)) — required by the Ultralytics dependency. Upstream Apache-2.0 text
and attribution preserved in [LICENSE.apache-2.0](LICENSE.apache-2.0) and [NOTICE](NOTICE).

Cite arXiv:2007.13971 (MIT Driverless) and, if using the data, BRT and FSOCO.

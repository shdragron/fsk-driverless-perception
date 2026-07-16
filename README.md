# FSK Driverless Perception

Single-stage cone detection and keypoint estimation for Formula Student Driverless.

The MIT pipeline detects cones with YOLOv3, then runs RektNet on each crop to regress keypoints.
This collapses both into **one YOLO26n-pose model** that emits boxes and keypoints in a single pass,
and measures how many keypoints PnP depth recovery actually needs.

> Fork of [cv-core/MIT-Driverless-CV-TrainingInfra](https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra) (Apache-2.0). See [NOTICE](NOTICE).

---

## Findings

On the leak-free split, with the measured cone template:

1. **Two-stage is more accurate — but only with a perfect crop.** Given ground-truth boxes RektNet-V
   roughly halves single-stage depth error (2.3% vs 4.3% clean). Jitter the box (`box-noise`, the
   failure mode only two-stage has) and it degrades to 9.2% — worse than single-stage. The
   advantage is entirely contingent on detection, which it cannot control.
2. **Two-stage is slowest exactly when the scene is busiest.** Its cost is one RektNet pass per cone,
   so at 30 cones (a slalom entry) it is 1.79× slower than single-stage's flat 27 ms.
3. **pose mAP and PnP depth rank the models oppositely.** Fewer keypoints score better on pose mAP
   yet recover worse depth under noise, where extra keypoints give PnP redundancy to average over.

The single-stage one-pass model is the better fit for real racing — many cones, imperfect detection
— while two-stage wins a controlled setting it rarely gets.

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

**pose mAP and PnP depth rank the models in opposite orders.** More keypoints score *worse* on
pose mAP and recover 3D depth *better* — because pose mAP measures whether each landmark lands in
the right place, while PnP measures whether the landmarks together resolve a pose, and extra
correspondences are redundancy PnP uses to average out error.

### Single-stage YOLO26n-pose

| kpt | box mAP50 | pose mAP50-95 | PnP depth err (median) | p90 |
|---|---|---|---|---|
| 8 | 0.960 | 0.877 *(worst)* | **4.3%** *(best)* | 13.6% |
| 6 | 0.963 | 0.919 | 4.6% | 14.7% |
| 4 | 0.961 | **0.943** *(best)* | 4.5% | 13.7% |

Detection is identical across all three (box mAP50 0.960–0.963); the whole difference is in what
the keypoints are for. On clean frames the depth gap is small (4.3 vs 4.5%), but it opens up under
noise — see below.

### Two-stage (YOLO26n + RektNet-V)

RektNet is fed crops cut from **ground-truth** boxes — a perfect detector — so these are its
best-case numbers.

| kpt | PnP depth err (median, clean) |
|---|---|
| 8 | **2.3%** |
| 6 | 2.3% |
| 4 | 2.2% |

Given a clean crop, two-stage roughly halves the single-stage error (2.3% vs 4.3%), and keypoint
count barely matters — RektNet only ever sees one cone, so it has no distant-cone problem to solve.
But that win depends entirely on the crop being right; see `box-noise` below.

### MIT original (YOLOv3 + RektNet-7kpt)

*(separate run — `scripts/run_mit_original.sh`; YOLOv3 is 103.8 M params and takes ~17 h)*

> **Depth error is relative, not metres.** Camera intrinsics are unknown (BRT pools many teams'
> cameras) and the cone template's stripe heights are measured from the labels, not surveyed. Both
> scale every PnP distance by one factor, which the metric divides out — reference and prediction
> use the same template. So model-vs-model holds; absolute metres do not, and cannot be compared
> with the paper's 0.5 m. For real metres, measure a cone and replace the template.

## Robustness

Single-stage, median PnP depth error. Corruption levels chosen for the highest strength that still
detects most cones (survivor rate in parentheses) — beyond that the error stops being meaningful.

| corruption | 8kpt | 6kpt | 4kpt |
|---|---|---|---|
| clean | **4.3%** (100%) | 4.6% (100%) | 4.5% (100%) |
| `noise` @0.25 (high-ISO) | **5.9%** (93%) | 6.8% (89%) | 6.2% (90%) |
| `noise` @0.5 | **6.7%** (67%) | 8.8% (62%) | 8.6% (54%) |
| `blur` @0.5 (motion) | **4.3%** (83%) | 4.9% (85%) | 4.6% (83%) |
| `sun` @1.0 (bloom) | **6.6%** (57%) | 7.0% (56%) | **6.6%** (56%) |
| `overcast` @1.0 | **5.3%** (95%) | 6.1% (88%) | **5.3%** (88%) |
| `shadow` @1.0 | 4.8% (97%) | 5.1% (99%) | **4.7%** (99%) |
| `backlight` @1.0 | **5.3%** (98%) | 5.7% (99%) | 5.8% (99%) |

**8kpt wins where it counts — sensor noise.** At `noise`@0.5 it is 6.7% against 8.6–8.8%, and the
gap grows with noise (0.3 → 0.9 → 2.1 pp). Noise perturbs each landmark independently, so PnP's
spare correspondences average it out; the 8-keypoint model has four to spare, the 4-keypoint model
none. Lighting and blur separate the models far less — there the whole cone shifts together, which
redundancy cannot fix.

**6kpt is consistently the worst of the three**, not the middle. Dropping the large-cone pair left
it an awkward halfway layout that helps neither the pose-mAP nor the PnP side.

### box-noise — the two-stage pipeline's own failure mode

RektNet assumes the crop is right. A real detector's box is not. Jittering the ground-truth box
before cropping shows what that costs:

| box-noise | 2-stage 8kpt | 6kpt | 4kpt |
|---|---|---|---|
| 0.25 | 3.0% | 3.1% | 2.9% |
| 0.5 | 4.7% | 4.7% | 4.7% |
| 1.0 | **9.2%** | 9.1% | 9.6% |

A perfect crop gives two-stage 2.3%; a badly-placed one pushes it to 9.2% — **worse than
single-stage** (≈5% under comparable corruption). The two-stage advantage is entirely contingent
on detection quality, which is exactly what it cannot control. Single-stage has no such term: it
detects and localises in one pass, so there is no crop to misplace.

Lighting doubles as *training* augmentation (`--racing-aug`); noise and blur are held out of
training so these numbers measure generalisation, not memorisation. Read error alongside the
survivor rate: under heavy corruption the hard cones stop being detected and leave the statistic,
which makes the error *improve* — `noise`@1.0 drops to ~10% survival and its median is meaningless.

## Compute

RTX 3060, FP16, 640×640, batch 1. Absolute latency is hardware-specific — the deployment target is
a Jetson Orin, where it will differ — but the single-vs-two-stage scaling carries over.

### Single-stage — one pass per frame, regardless of cone count

| kpt | params | mean | p95 | FPS |
|---|---|---|---|---|
| 8 | 2.98 M | 34.9 ms | 67.8 ms | 28.7 |
| 6 | 2.86 M | 27.4 ms | 47.2 ms | 36.5 |
| 4 | 2.75 M | 27.0 ms | 50.2 ms | 37.1 |

### Two-stage — detector + one RektNet pass **per cone**

| cones in frame | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| total (ms) | 31.8 | 34.9 | 39.1 | 43.2 | **48.2** |
| FPS | 31.4 | 28.7 | 25.6 | 23.1 | **20.7** |

Two-stage cost grows with cone count — one RektNet pass each — while single-stage is flat. At 30
cones (a slalom entry, the busiest and most safety-critical frame) two-stage is **1.79× slower**
(48.2 ms vs 27.0 ms). This is where the per-cone cost hurts most.

*Deployment note:* TensorRT export works (`benchmark.py`), but TensorRT's pip package pulls CUDA-13
libraries that break this env's CUDA-12 PyTorch, so engine builds are left for the Orin, where they
belong. The numbers above are native PyTorch FP16.

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
  train_pose.py       single-stage pose, or --task detect for a box-only cone detector
  train_rektnet.py    RektNet-V, and the 7-kpt original
scripts/run_all.sh
```

## License

**AGPL-3.0** ([LICENSE](LICENSE)) — required by the Ultralytics dependency. Upstream Apache-2.0 text
and attribution preserved in [LICENSE.apache-2.0](LICENSE.apache-2.0) and [NOTICE](NOTICE).

Cite arXiv:2007.13971 (MIT Driverless) and, if using the data, BRT and FSOCO.

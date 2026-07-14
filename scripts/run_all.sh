#!/usr/bin/env bash
# Full pipeline on the leak-free split:
#   1. single-stage YOLO26n-pose at 8/6/4 keypoints
#   2. PnP depth evaluation under every corruption
#   3. two-stage RektNet-V at 8/6/4 keypoints
#   4. its evaluation, plus the box-noise sweep only a two-stage pipeline is exposed to
#   5. the MIT original -- YOLOv3 + unmodified 7-keypoint RektNet
#
# Strictly sequential. Two dataloader fleets do not fit in 31 GB of RAM together; that already
# cost one training run to the kernel OOM killer.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_ROOT:-/data/brt_cone_pose}"
RESULTS="$REPO/results/metrics"
RUNS="${RUNS_DIR:-$HOME/runs/pose}"
EPOCHS="${EPOCHS:-100}"
RN_EPOCHS="${RN_EPOCHS:-60}"

cd "$REPO"
mkdir -p "$RESULTS"

CORRUPTIONS=(noise blur sun overcast shadow backlight)
LEVELS=(0.25 0.5 1.0)

echo "[$(date +%H:%M)] === 1/4  train YOLO26-pose (8/6/4 kpt) ==="
for n in 8 6 4; do
    if [ -f "$RUNS/clean-${n}kpt/weights/best.pt" ]; then
        echo "[$(date +%H:%M)] ${n}kpt already trained -- skipping"
        continue
    fi
    echo "[$(date +%H:%M)] --- train ${n}kpt ---"
    python -m src.train_pose --task pose \
        --data "$DATA/brt-clean-${n}kpt/data.yaml" \
        --epochs "$EPOCHS" --imgsz 640 --batch 32 --workers 8 \
        --name "clean-${n}kpt"
done

echo "[$(date +%H:%M)] === 2/4  evaluate depth (PnP) ==="
for n in 8 6 4; do
    W="$RUNS/clean-${n}kpt/weights/best.pt"
    [ -f "$W" ] || { echo "[$(date +%H:%M)] SKIP ${n}kpt -- no weights"; continue; }
    ROOT="$DATA/brt-clean-${n}kpt"

    python -m src.eval.eval_pose --weights "$W" --data-root "$ROOT" \
        --n-kpt "$n" --out "$RESULTS/${n}kpt_clean.json"
    for kind in "${CORRUPTIONS[@]}"; do
        for lvl in "${LEVELS[@]}"; do
            python -m src.eval.eval_pose --weights "$W" --data-root "$ROOT" \
                --n-kpt "$n" --corrupt "$kind" --level "$lvl" \
                --out "$RESULTS/${n}kpt_${kind}_${lvl}.json"
        done
    done
done

# 8 as well as 6/4: the visibility mask makes it trainable. Without it, requiring all 8 keypoints
# would drop the 94.7% of cones that have no kpt6/7, so upstream RektNet could only ever do 6.
echo "[$(date +%H:%M)] === 3/4  train RektNet-V (two-stage) ==="
for n in 8 6 4; do
    CROPS="$DATA/rektnet-clean-${n}kpt"
    if [ ! -f "$CROPS/test.csv" ]; then
        echo "[$(date +%H:%M)] --- build ${n}kpt crops ---"
        python -m src.data.make_cone_crops --brt-root "$DATA/brt-clean-8kpt" \
            --out "$CROPS" --n-kpt "$n"
    fi
    # Geometric loss stays ON. Ablating it would only make RektNet weaker, and the paper already
    # established its value; the question here is how the *best* RektNet compares to YOLO-pose.
    OUT="$DATA/rektnet_runs/rektnet-v-${n}kpt.pt"
    if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M)] RektNet-V ${n}kpt exists -- skipping"
    else
        echo "[$(date +%H:%M)] --- train RektNet-V ${n}kpt ---"
        # Gammas default to the paper's tuned values (0.038 / 0.055).
        python -m src.train_rektnet --data "$CROPS" --num-kpt "$n" \
            --epochs "$RN_EPOCHS" --batch 64 --workers 6 --out "$OUT"
    fi
done

echo "[$(date +%H:%M)] === 4/4  evaluate RektNet-V ==="
for n in 8 6 4; do
    W="$DATA/rektnet_runs/rektnet-v-${n}kpt.pt"
    [ -f "$W" ] || continue
    base="rektnet-v-${n}kpt"
    python -m src.eval.eval_rektnet --weights "$W" --num-kpt "$n" \
        --brt-root "$DATA/brt-clean-8kpt" --out "$RESULTS/${base}_clean.json"
    for kind in noise blur sun; do
        for lvl in 0.5 1.0; do
            python -m src.eval.eval_rektnet --weights "$W" --num-kpt "$n" \
                --brt-root "$DATA/brt-clean-8kpt" --corrupt "$kind" --level "$lvl" \
                --out "$RESULTS/${base}_${kind}_${lvl}.json"
        done
    done
    # The failure mode a two-stage pipeline has and a one-stage one does not: the detector hands
    # over an imperfect box.
    for bn in 0.25 0.5 1.0; do
        python -m src.eval.eval_rektnet --weights "$W" --num-kpt "$n" \
            --brt-root "$DATA/brt-clean-8kpt" --box-noise "$bn" \
            --out "$RESULTS/${base}_boxnoise_${bn}.json"
    done
done

# ---------------------------------------------------------------------------
# 5. MIT original: YOLOv3 detector + RektNet on 7 keypoints.
#
# The apex is synthesised from BRT's top pair (see make_mit_7kpt.py), so this reproduces the
# architecture, not the labelling. YOLOv3 is 103.8 M parameters against YOLO26n's 2.57 M -- a 40x
# gap that is the point of including it.
# ---------------------------------------------------------------------------
echo "[$(date +%H:%M)] === 5/5  MIT original (YOLOv3 + RektNet-7kpt) ==="

MIT7="$DATA/brt-clean-7kpt-mit"
if [ ! -f "$MIT7/data.yaml" ]; then
    echo "[$(date +%H:%M)] --- build 7kpt (apex synthesised) ---"
    python -m src.data.make_mit_7kpt --source "$DATA/brt-clean-8kpt" --out "$MIT7"
fi

# YOLOv3 detector. Batch 16, not 32: at 103.8 M parameters it will not fit otherwise.
if [ ! -f "$RUNS/../detect/mit-yolov3/weights/best.pt" ]; then
    echo "[$(date +%H:%M)] --- train YOLOv3 detector ---"
    python -m src.train_pose --task detect --model yolov3.yaml \
        --data "$MIT7/data.yaml" --epochs "$EPOCHS" --imgsz 640 --batch 16 --workers 8 \
        --name mit-yolov3
fi

RN7="$DATA/rektnet-clean-7kpt"
if [ ! -f "$RN7/test.csv" ]; then
    echo "[$(date +%H:%M)] --- build 7kpt crops ---"
    python -m src.data.make_cone_crops --brt-root "$MIT7" --out "$RN7" --n-kpt 7
fi

OUT7="$DATA/rektnet_runs/rektnet-7kpt-mit.pt"
if [ ! -f "$OUT7" ]; then
    echo "[$(date +%H:%M)] --- train RektNet 7kpt ---"
    python -m src.train_rektnet --data "$RN7" --num-kpt 7 \
        --epochs "$RN_EPOCHS" --batch 64 --workers 6 --out "$OUT7"
fi

if [ -f "$OUT7" ]; then
    echo "[$(date +%H:%M)] --- evaluate MIT original ---"
    python -m src.eval.eval_rektnet --weights "$OUT7" --num-kpt 7 \
        --brt-root "$MIT7" --out "$RESULTS/mit-7kpt_clean.json"
    for kind in noise blur sun; do
        for lvl in 0.5 1.0; do
            python -m src.eval.eval_rektnet --weights "$OUT7" --num-kpt 7 \
                --brt-root "$MIT7" --corrupt "$kind" --level "$lvl" \
                --out "$RESULTS/mit-7kpt_${kind}_${lvl}.json"
        done
    done
    for bn in 0.25 0.5 1.0; do
        python -m src.eval.eval_rektnet --weights "$OUT7" --num-kpt 7 \
            --brt-root "$MIT7" --box-noise "$bn" \
            --out "$RESULTS/mit-7kpt_boxnoise_${bn}.json"
    done
fi

echo "[$(date +%H:%M)] === done -- summarising ==="
python -m src.eval.summarize --results "$RESULTS"
python -m src.viz.plot_metrics --results "$RESULTS" --out "$REPO/results/ablation.png"
echo "[$(date +%H:%M)] ALL DONE"

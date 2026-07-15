#!/usr/bin/env bash
# The MIT original -- YOLOv3 detector + unmodified 7-keypoint RektNet -- as a separate run.
#
# Split out of run_all.sh because YOLOv3 is 103.8 M parameters against YOLO26n's 2.57 M and takes
# ~17 h to train, which would have doubled the wait for the single-vs-two-stage result that is the
# actual question. This answers a different one: what the eight years between YOLOv3 and YOLO26 are
# worth.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_ROOT:-/data/brt_cone_pose}"
RESULTS="$REPO/results/metrics"
RUNS="${RUNS_DIR:-$HOME/runs/pose}"
EPOCHS="${EPOCHS:-100}"
RN_EPOCHS="${RN_EPOCHS:-60}"

cd "$REPO"
mkdir -p "$RESULTS"

source /home/moon/anaconda3/etc/profile.d/conda.sh
conda activate yolocone

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


echo "[$(date +%H:%M)] MIT ORIGINAL DONE"

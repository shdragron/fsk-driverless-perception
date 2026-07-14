#!/usr/bin/env bash
# Full pipeline on the leak-free split: train the three keypoint variants, evaluate depth via
# PnP under every corruption, then train and evaluate RektNet as the two-stage baseline.
#
# Runs strictly sequentially. Two jobs sharing this 12 GB card is survivable; two dataloader
# fleets sharing 31 GB of RAM is not -- that combination already got a training run shot by the
# kernel OOM killer once.
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
echo "[$(date +%H:%M)] === 3/4  train RektNet (two-stage baseline) ==="
for n in 8 6 4; do
    CROPS="$DATA/rektnet-clean-${n}kpt"
    if [ ! -f "$CROPS/test.csv" ]; then
        echo "[$(date +%H:%M)] --- build ${n}kpt crops ---"
        python -m src.data.make_cone_crops --brt-root "$DATA/brt-clean-8kpt" \
            --out "$CROPS" --n-kpt "$n"
    fi
    # Geometric loss stays ON. Ablating it would only make RektNet weaker, and the paper already
    # established its value; the question here is how the *best* RektNet compares to YOLO-pose.
    OUT="$DATA/rektnet_runs/rektnet-${n}kpt.pt"
    if [ -f "$OUT" ]; then
        echo "[$(date +%H:%M)] rektnet ${n}kpt exists -- skipping"
    else
        echo "[$(date +%H:%M)] --- train rektnet ${n}kpt ---"
        # Gammas default to the paper's tuned values (0.038 / 0.055).
        python -m src.train_rektnet --data "$CROPS" --num-kpt "$n" \
            --epochs "$RN_EPOCHS" --batch 64 --workers 6 --out "$OUT"
    fi
done

echo "[$(date +%H:%M)] === 4/4  evaluate RektNet ==="
for n in 8 6 4; do
    W="$DATA/rektnet_runs/rektnet-${n}kpt.pt"
    [ -f "$W" ] || continue
    base="rektnet-${n}kpt"
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

echo "[$(date +%H:%M)] === done -- summarising ==="
python -m src.eval.summarize --results "$RESULTS"
python -m src.viz.plot_metrics --results "$RESULTS" --out "$REPO/results/ablation.png"
echo "[$(date +%H:%M)] ALL DONE"

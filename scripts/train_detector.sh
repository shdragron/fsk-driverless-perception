#!/usr/bin/env bash
# Train a cone detector on its own -- the box-only YOLO26n, no keypoints.
#
# The two-stage pipeline needs a detector in front of RektNet, and the benchmark had been using
# the stock COCO yolo26n.pt, which has never seen a cone. This trains the real thing on the same
# leak-free split (the pose labels carry boxes; detect just ignores the keypoints). It also gives
# a clean box-mAP baseline to compare the pose model's detection head against.
#
# Waits for the GPU -- a render job or another train can be holding it.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_ROOT:-/data/brt_cone_pose}"
EPOCHS="${EPOCHS:-100}"
cd "$REPO"

source /home/moon/anaconda3/etc/profile.d/conda.sh
conda activate yolocone

echo "[$(date +%H:%M)] waiting for >=10 GB free VRAM..."
while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$free" -ge 10000 ] && break
    sleep 120
done
echo "[$(date +%H:%M)] ${free} MiB free -- training detector"

python -m src.train_pose --task detect \
    --data "$DATA/brt-clean-8kpt/data.yaml" \
    --epochs "$EPOCHS" --imgsz 640 --batch 32 --workers 8 \
    --name clean-detect

echo "[$(date +%H:%M)] DETECTOR DONE -> ~/runs/detect/clean-detect/weights/best.pt"

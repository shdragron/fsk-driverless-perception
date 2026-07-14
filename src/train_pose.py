"""Train YOLO26 on cone data -- either detection (FSOCO) or pose (BRT Cone Pose).

Pose mode replaces the two-stage YOLOv3 -> RektNet pipeline with a single model that emits
boxes and the 8 cone keypoints together. Detection mode is the drop-in replacement for
CVC-YOLOv3 alone, leaving RektNet downstream.

The RTX 3060 shares memory with other jobs, so batch defaults are conservative; pass
--batch -1 to let Ultralytics autotune to ~60% VRAM once the card is free.
"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["detect", "pose"], required=True)
    parser.add_argument("--data", required=True, type=Path, help="Path to data.yaml")
    parser.add_argument("--model", default=None,
                        help="Pretrained checkpoint (default: yolo26n.pt / yolo26n-pose.pt)")
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--imgsz", default=640, type=int,
                        help="Cones are small; going below 640 loses the distant ones")
    parser.add_argument("--batch", default=16, type=int, help="-1 autotunes to ~60%% VRAM")
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--name", default=None, help="Run name under runs/")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if not args.data.is_file():
        sys.exit(f"data.yaml not found: {args.data}")

    # Pose training silently mislearns left/right keypoints when flip_idx is missing and
    # fliplr is on, so refuse to start rather than waste a run.
    if args.task == "pose":
        text = args.data.read_text()
        if "flip_idx" not in text:
            sys.exit(
                f"{args.data} has no flip_idx. Horizontal-flip augmentation would mirror "
                "images without swapping left/right cone keypoints, corrupting the labels.\n"
                "Run prepare_brt_pose.py to regenerate the yaml, or set fliplr=0.0."
            )

    from ultralytics import YOLO

    model_path = args.model or ("yolo26n-pose.pt" if args.task == "pose" else "yolo26n.pt")
    model = YOLO(model_path)

    model.train(
        task=args.task,
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        name=args.name or f"yolo26n-cone-{args.task}",
        resume=args.resume,
        seed=17,  # matches the seed the RektNet/YOLOv3 code pins
        val=True,
        plots=True,
    )

    metrics = model.val()
    print(f"\nmAP50-95 (box): {metrics.box.map:.4f}   mAP50 (box): {metrics.box.map50:.4f}")
    if args.task == "pose":
        print(f"mAP50-95 (pose): {metrics.pose.map:.4f}   mAP50 (pose): {metrics.pose.map50:.4f}")


if __name__ == "__main__":
    main()

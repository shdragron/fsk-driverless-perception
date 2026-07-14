"""Measure what actually matters on the car: latency, throughput, and memory.

A perception stack that is 2% more accurate and 40 ms slower is a worse stack. The original paper
makes the same point -- its whole framing is "low-latency" -- and reports the two-stage pipeline
at roughly 132-170 ms per frame.

Two things get measured here, because they answer different questions:

  single-stage (YOLO26-pose)   one forward pass per *frame*, regardless of cone count
  two-stage (YOLO + RektNet)   one detection pass per frame, then one RektNet pass per *cone*

The second scales with how many cones are in view, which is exactly when you can least afford it:
entering a slalom, twenty cones on screen, is both the busiest frame and the one where a
per-cone cost hurts most. Reporting a single number for the two-stage pipeline hides that, so
this sweeps cone count.

Memory is reported as peak allocated, not reserved -- reserved reflects the caching allocator,
not what the model needs.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

INPUT_SIZE = (80, 80)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_module(fn, warmup, iters, device):
    """Return (mean_ms, p50, p95) after discarding warm-up runs."""
    for _ in range(warmup):
        fn()
    sync(device)

    times = []
    for _ in range(iters):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        times.append((time.perf_counter() - t0) * 1000)
    t = np.array(times)
    return float(t.mean()), float(np.percentile(t, 50)), float(np.percentile(t, 95))


def bench_pose(weights, imgsz, device, warmup, iters, half):
    from ultralytics import YOLO

    model = YOLO(str(weights))
    model.to(device)
    if half and device.type == "cuda":
        model.model.half()
    net = model.model.eval()

    dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
    x = torch.zeros(1, 3, imgsz, imgsz, device=device, dtype=dtype)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        mean, p50, p95 = time_module(lambda: net(x), warmup, iters, device)

    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
    params = sum(p.numel() for p in net.parameters()) / 1e6
    return {"mean_ms": mean, "p50_ms": p50, "p95_ms": p95,
            "peak_mem_mb": peak, "params_m": params}


def bench_rektnet(weights, num_kpt, device, warmup, iters, half, cone_counts):
    from src.models.keypoint_net import KeypointNet

    ck = torch.load(weights, map_location="cpu", weights_only=True)
    net = KeypointNet(num_kpt=num_kpt, image_size=INPUT_SIZE).to(device).eval()
    net.load_state_dict(ck["model"])
    if half and device.type == "cuda":
        net.half()

    dtype = torch.float16 if (half and device.type == "cuda") else torch.float32
    out = {"params_m": sum(p.numel() for p in net.parameters()) / 1e6, "by_cone_count": {}}

    for n in cone_counts:
        # Cones are cropped and batched, which is the sane way to run this -- one crop at a time
        # would be far worse.
        x = torch.zeros(n, 3, *INPUT_SIZE, device=device, dtype=dtype)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            mean, p50, p95 = time_module(lambda: net(x), warmup, iters, device)
        peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else float("nan")
        out["by_cone_count"][n] = {"mean_ms": mean, "p50_ms": p50, "p95_ms": p95,
                                   "peak_mem_mb": peak}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pose-weights", nargs="*", type=Path, default=[],
                    help="YOLO26-pose checkpoints (the single-stage pipeline)")
    ap.add_argument("--pose-labels", nargs="*", default=None)
    ap.add_argument("--rektnet-weights", type=Path, default=None,
                    help="RektNet checkpoint (the per-cone stage of the two-stage pipeline)")
    ap.add_argument("--rektnet-kpt", default=8, type=int)
    ap.add_argument("--detector-weights", type=Path, default=None,
                    help="Plain detector for the two-stage pipeline; defaults to yolo26n.pt")
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true", help="FP16 -- what you would actually deploy")
    ap.add_argument("--warmup", default=20, type=int)
    ap.add_argument("--iters", default=100, type=int)
    ap.add_argument("--cone-counts", default="1,5,10,20,30", type=str)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu"
                          else "cpu")
    cone_counts = [int(c) for c in args.cone_counts.split(",")]
    results = {"device": str(device), "imgsz": args.imgsz, "half": args.half, "single_stage": {}}

    if device.type == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}  |  fp16: {args.half}  |  imgsz: {args.imgsz}")
    else:
        print(f"device: CPU  |  imgsz: {args.imgsz}")

    print(f"\n{'SINGLE-STAGE (per frame, cone count does not matter)':<50}")
    print(f"{'model':<14} {'params(M)':>10} {'mean(ms)':>10} {'p95(ms)':>9} {'FPS':>7} {'mem(MB)':>9}")
    labels = args.pose_labels or [w.parent.parent.name for w in args.pose_weights]
    for w, label in zip(args.pose_weights, labels):
        r = bench_pose(w, args.imgsz, device, args.warmup, args.iters, args.half)
        results["single_stage"][label] = r
        print(f"{label:<14} {r['params_m']:>10.2f} {r['mean_ms']:>10.2f} {r['p95_ms']:>9.2f} "
              f"{1000/r['mean_ms']:>7.1f} {r['peak_mem_mb']:>9.1f}")

    if args.rektnet_weights:
        det = args.detector_weights or Path("yolo26n.pt")
        print(f"\n{'TWO-STAGE (detector + RektNet per cone)':<50}")
        d = bench_pose(det, args.imgsz, device, args.warmup, args.iters, args.half)
        rn = bench_rektnet(args.rektnet_weights, args.rektnet_kpt, device,
                           args.warmup, args.iters, args.half, cone_counts)
        results["two_stage"] = {"detector": d, "rektnet": rn}

        print(f"detector       {d['params_m']:>10.2f} {d['mean_ms']:>10.2f} {d['p95_ms']:>9.2f}"
              f" {1000/d['mean_ms']:>7.1f} {d['peak_mem_mb']:>9.1f}")
        print(f"\n{'cones':>6} {'rektnet(ms)':>12} {'total(ms)':>11} {'FPS':>7}")
        for n in cone_counts:
            r = rn["by_cone_count"][n]
            total = d["mean_ms"] + r["mean_ms"]
            print(f"{n:>6} {r['mean_ms']:>12.2f} {total:>11.2f} {1000/total:>7.1f}")

        if results["single_stage"]:
            best = min(results["single_stage"].values(), key=lambda r: r["mean_ms"])
            worst_n = max(cone_counts)
            two = d["mean_ms"] + rn["by_cone_count"][worst_n]["mean_ms"]
            print(f"\nat {worst_n} cones: two-stage {two:.1f} ms vs single-stage "
                  f"{best['mean_ms']:.1f} ms  ({two / best['mean_ms']:.2f}x)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

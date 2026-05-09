"""
python profiling/profile_inference.py \
    --weights yolov5l-xs-1.pt \
    --source VisDrone/sample_val \
    --img 1536 \
    --device 0 \
    --iters 30 --warmup 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from contextlib import contextmanager

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.experimental import attempt_load  # noqa: E402
from utils.datasets import LoadImages  # noqa: E402
from utils.general import check_img_size, non_max_suppression  # noqa: E402
from utils.torch_utils import select_device  # noqa: E402


# NVTX helpers. torch.cuda.nvtx is a thin wrapper around the NVTX C API; the
# Nsight Systems UI groups all events into the corresponding range.
@contextmanager
def nvtx_range(name: str):
    """Context manager that pushes an NVTX range when CUDA is available."""
    push = torch.cuda.is_available()
    if push:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if push:
            torch.cuda.nvtx.range_pop()


def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# Core profiling loop.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="yolov5l-xs-1.pt")
    parser.add_argument("--source", default="VisDrone/sample_val")
    parser.add_argument("--img", type=int, default=1536, help="inference resolution")
    parser.add_argument("--device", default="0")
    parser.add_argument("--half", action="store_true", help="FP16 autocast inference")
    parser.add_argument("--iters", type=int, default=30,
                        help="number of measured iterations (cycled over images)")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--out", default="profiling/runs/baseline")
    parser.add_argument("--torch-profiler", action="store_true",
                        help="also run torch.profiler and export Chrome trace")
    parser.add_argument("--no-summary", action="store_true",
                        help="skip writing summary.json/torch artefacts (use when running under nsys/ncu)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress stdout logs so profiler CSV output stays parseable")
    args = parser.parse_args()

    out_dir = Path(args.out)
    if not args.no_summary:
        out_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(args.device)
    if not args.quiet:
        print(f"[profile] device={device} half={args.half}")

    with nvtx_range("model_load"):
        model = attempt_load(args.weights, map_location=device)
        model.eval()
        if args.half and device.type != "cpu":
            model.half()
        stride = int(model.stride.max())
        imgsz = check_img_size(args.img, s=stride)

    with nvtx_range("dataset_init"):
        dataset = LoadImages(args.source, img_size=imgsz, stride=stride, auto=True)
    images = []  # cache decoded numpy frames so disk IO is excluded from timing
    for path, img, im0, _, _ in dataset:
        images.append((path, img, im0))
        if len(images) >= 8:
            break
    if not images:
        raise RuntimeError(f"No images found under {args.source}")
    if not args.quiet:
        print(f"[profile] cached {len(images)} images at {imgsz}px")

    # warmup
    dummy = torch.zeros(1, 3, imgsz, imgsz, device=device,
                        dtype=torch.float16 if args.half else torch.float32)
    with torch.no_grad():
        for _ in range(args.warmup):
            with nvtx_range("warmup_iter"):
                model(dummy)
    cuda_sync()

    # timed loop
    timings = {"preprocess": [], "inference": [], "nms": [], "h2d": [], "d2h": [], "total": []}
    num_dets = []

    if args.torch_profiler:
        from torch.profiler import profile, ProfilerActivity, schedule
        prof_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=2, active=args.iters, repeat=1),
            record_shapes=True,
            with_stack=False,
            profile_memory=True,
        )
        prof_ctx.__enter__()
    else:
        prof_ctx = None

    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None

    with torch.no_grad(), nvtx_range("timed_loop"):
        for i in range(args.iters):
            path, img_np, im0 = images[i % len(images)]

            with nvtx_range(f"iter_{i}"):
                # preprocess on CPU + H2D copy
                t0 = time.perf_counter()
                with nvtx_range("preprocess_cpu"):
                    img_cpu = torch.from_numpy(img_np)
                with nvtx_range("h2d_copy"):
                    img_dev = img_cpu.to(device, non_blocking=True)
                    img_dev = img_dev.half() if args.half else img_dev.float()
                    img_dev /= 255.0
                    if img_dev.ndim == 3:
                        img_dev = img_dev[None]
                    cuda_sync()
                t1 = time.perf_counter()

                # model forward
                with nvtx_range("inference_forward"):
                    pred = model(img_dev)[0]
                    cuda_sync()
                t2 = time.perf_counter()

                # post-processing / NMS
                with nvtx_range("nms"):
                    det = non_max_suppression(pred, args.conf, args.iou)
                    cuda_sync()
                t3 = time.perf_counter()

                # D2H copy (boxes back to CPU)
                with nvtx_range("d2h_copy"):
                    cpu_dets = [d.cpu() for d in det]
                    cuda_sync()
                t4 = time.perf_counter()

            timings["preprocess"].append((t1 - t0) * 1000)
            timings["inference"].append((t2 - t1) * 1000)
            timings["nms"].append((t3 - t2) * 1000)
            timings["d2h"].append((t4 - t3) * 1000)
            timings["total"].append((t4 - t0) * 1000)
            num_dets.append(int(sum(d.shape[0] for d in cpu_dets)))

            if prof_ctx is not None:
                prof_ctx.step()

    if prof_ctx is not None:
        prof_ctx.__exit__(None, None, None)
        if not args.no_summary:
            trace_path = out_dir / "torch_trace.json"
            prof_ctx.export_chrome_trace(str(trace_path))
            # also dump key averages text
            with open(out_dir / "torch_key_averages.txt", "w") as fh:
                fh.write(prof_ctx.key_averages().table(sort_by="cuda_time_total", row_limit=30))
            if not args.quiet:
                print(f"[profile] torch trace -> {trace_path}")

    # summary
    def stats(xs):
        a = np.asarray(xs)
        return {"mean_ms": float(a.mean()), "median_ms": float(np.median(a)),
                "p95_ms": float(np.percentile(a, 95)), "std_ms": float(a.std())}

    summary = {
        "weights": args.weights,
        "imgsz": imgsz,
        "half": args.half,
        "device": str(device),
        "iters": args.iters,
        "stages": {k: stats(v) for k, v in timings.items() if v},
        "mean_detections_per_image": float(np.mean(num_dets)),
    }
    if torch.cuda.is_available():
        summary["peak_memory_MB"] = torch.cuda.max_memory_allocated() / 1024 ** 2
        summary["reserved_memory_MB"] = torch.cuda.max_memory_reserved() / 1024 ** 2
        summary["device_name"] = torch.cuda.get_device_name(0)

    if not args.quiet:
        print(json.dumps(summary, indent=2))
    if args.no_summary:
        if not args.quiet:
            print("[profile] --no-summary set; skipped writing files")
    else:
        with open(out_dir / "summary.json", "w") as fh:
            json.dump(summary, fh, indent=2)
        if not args.quiet:
            print(f"[profile] wrote {out_dir/'summary.json'}")


if __name__ == "__main__":
    main()

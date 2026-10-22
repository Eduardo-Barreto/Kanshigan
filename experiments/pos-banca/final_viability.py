"""Stage-timed viability of the full pipeline, cold and warm, with process VRAM.

infer.py starts its timer after decoding the video and before loading the model, so
its "end_to_end_fps" leaves decoding out and puts model loading in. This script runs
the same stages with the same functions, in one fresh process, and times each one:

  decode     load_frames (cv2 decode of the whole clip)
  model_load YOLO(weights) (and the tracker's ReID model, if any)
  process    per-frame dohyo calibration, crop ROI, detection (crop + YOLO + dohyo
             filter), tracking, kinematics and events

Run 0 is the cold start (first CUDA context, first cuDNN autotune, cold file cache
for the weights); runs 1..N are warm. Reported FPS:

  e2e_fps        frames / (decode + model_load + process)
  infer_py_fps   frames / (model_load + process), what infer.py reports
  process_fps    frames / process, the steady-state rate of a loaded service

Peak VRAM is read two ways: torch max_memory_allocated/reserved (tensors and the
caching allocator) and NVML per-process usage sampled every 5 ms in a thread, which
includes the CUDA context and is what nvidia-smi shows for the process.

The overlay video is not rendered: infer.py renders it outside its timer too.

Usage (run once per configuration, fresh process each time):
    uv run --project ../pre-banca --with nvidia-ml-py python final_viability.py \\
        --video ../../data/processed/clips/br/gold_zb01.mp4 \\
        --weights ../../results/training/yolo26n_kanshigan/weights/best.pt \\
        --label gold_crop_yolo26n_gpu --out ../../results/E6_final_measurements/viability
    CUDA_VISIBLE_DEVICES= uv run ... --label ..._cpu        # CPU run
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

NVML_POLL_S = 0.005


class NvmlPeak:
    """Background sampler of this process's GPU memory, as NVML reports it."""

    def __init__(self) -> None:
        self.peak_mb = 0.0
        self.available = False
        self._stop = threading.Event()
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.available = True
        except Exception as exc:  # no driver / no GPU: report as unavailable, never guess
            self.error = repr(exc)

    def _poll(self) -> None:
        pid = os.getpid()
        while not self._stop.is_set():
            for proc in self._nvml.nvmlDeviceGetComputeRunningProcesses(self._handle):
                if proc.pid == pid and proc.usedGpuMemory:
                    self.peak_mb = max(self.peak_mb, proc.usedGpuMemory / 2**20)
            time.sleep(NVML_POLL_S)

    def __enter__(self):
        self.peak_mb = 0.0
        self._stop.clear()
        if self.available:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        if self.available:
            self._thread.join()


def build_tracker(name: str, device: str):
    if name in ("deepocsort", "botsort"):
        from tracker_comparison import _build_appearance_tracker

        return _build_appearance_tracker(name, device)
    from tracking import build_tracker as build_motion

    return build_motion(name)


def one_run(video: Path, weights: Path, tracker_name: str, on_cuda: bool) -> dict:
    import torch
    from ultralytics import YOLO

    from crop import clip_roi
    from events import EventConfig, detect_events
    from infer import detect_robots, load_frames, per_frame_calibrations
    from metrics import kinematics_from_track
    from tracking import run_tracker

    if on_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    device = "cuda:0" if on_cuda else "cpu"

    t0 = time.perf_counter()
    frames, fps = load_frames(video)
    t1 = time.perf_counter()
    model = YOLO(str(weights))
    tracker = build_tracker(tracker_name, device)
    t2 = time.perf_counter()
    calibrations = per_frame_calibrations(frames)
    roi = clip_roi(frames[:: max(1, len(frames) // 12)])
    detections = [detect_robots(model, f, cal, roi) for f, cal in zip(frames, calibrations)]
    tracks = run_tracker(tracker, detections, frames, fps)
    kinematics = {rid: kinematics_from_track(t, lambda fr: calibrations[min(fr, len(calibrations) - 1)], fps) for rid, t in tracks.items()}
    events = detect_events(kinematics["A"], kinematics["B"], EventConfig()) if {"A", "B"} <= kinematics.keys() else []
    if on_cuda:
        torch.cuda.synchronize()
    t3 = time.perf_counter()

    n = len(frames)
    decode, load, process = t1 - t0, t2 - t1, t3 - t2
    return {
        "n_frames": n,
        "frame_size": [frames[0].shape[1], frames[0].shape[0]],
        "roi_xywh": list(roi),
        "decode_s": round(decode, 3),
        "model_load_s": round(load, 3),
        "process_s": round(process, 3),
        "e2e_fps": round(n / (decode + load + process), 2),
        "infer_py_fps": round(n / (load + process), 2),
        "process_fps": round(n / process, 2),
        "decode_fps": round(n / decode, 2),
        "torch_peak_allocated_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if on_cuda else None,
        "torch_peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20, 1) if on_cuda else None,
        "events": [{"kind": e.kind, "frame": e.frame} for e in events],
    }


def summarize(runs: list[dict], key: str) -> dict:
    values = [r[key] for r in runs]
    return {"mean": round(statistics.mean(values), 2), "std": round(statistics.stdev(values), 2) if len(values) > 1 else 0.0}


def hardware() -> dict:
    import torch

    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name")), platform.processor())
    info = {"cpu": cpu, "cpu_logical_cores": os.cpu_count(), "torch_threads": torch.get_num_threads(),
            "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "python": platform.python_version()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_total_mb"] = round(torch.cuda.get_device_properties(0).total_memory / 2**20)
    import ultralytics

    import boxmot

    info["ultralytics"], info["boxmot"] = ultralytics.__version__, boxmot.__version__
    return info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tracker", default="ocsort")
    ap.add_argument("--warm-runs", type=int, default=3)
    ap.add_argument("--label", required=True)
    ap.add_argument("--note", default="", help="what the input is (cropped clip, native, ...)")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "E6_final_measurements" / "viability")
    args = ap.parse_args()

    import torch

    on_cuda = torch.cuda.is_available()
    sampler = NvmlPeak()
    runs = []
    for i in range(args.warm_runs + 1):
        with sampler:
            run = one_run(args.video.resolve(), args.weights.resolve(), args.tracker, on_cuda)
        run["nvml_process_peak_mb"] = round(sampler.peak_mb, 1) if (sampler.available and on_cuda) else None
        run["kind"] = "cold" if i == 0 else "warm"
        runs.append(run)
        print(f"{args.label} run {i} ({run['kind']}): e2e {run['e2e_fps']} fps, infer.py-equivalent {run['infer_py_fps']}, process {run['process_fps']}, decode {run['decode_s']} s, load {run['model_load_s']} s, nvml {run['nvml_process_peak_mb']} MB", flush=True)

    warm = [r for r in runs if r["kind"] == "warm"]
    report = {
        "label": args.label,
        "note": args.note,
        "video": str(args.video.resolve().relative_to(ROOT)) if args.video.resolve().is_relative_to(ROOT) else str(args.video),
        "weights": str(args.weights.resolve().relative_to(ROOT)),
        "tracker": args.tracker,
        "device": "cuda:0" if on_cuda else "cpu",
        "crop_in_timer": "the per-clip dohyo ROI and the per-frame crop run inside process_s; "
                         "a pre-cropped clip (prep_clips.py) had its crop done offline, outside any timer",
        "hardware": hardware(),
        "nvml_available": sampler.available,
        "runs": runs,
        "cold": runs[0],
        "warm_summary": {k: summarize(warm, k) for k in ("e2e_fps", "infer_py_fps", "process_fps", "decode_s", "model_load_s", "process_s")},
        "warm_peak_nvml_mb": max((r["nvml_process_peak_mb"] or 0) for r in warm) if on_cuda else None,
        "warm_peak_torch_reserved_mb": max((r["torch_peak_reserved_mb"] or 0) for r in warm) if on_cuda else None,
        "warm_peak_torch_allocated_mb": max((r["torch_peak_allocated_mb"] or 0) for r in warm) if on_cuda else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{args.label}.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["warm_summary"]))


if __name__ == "__main__":
    main()

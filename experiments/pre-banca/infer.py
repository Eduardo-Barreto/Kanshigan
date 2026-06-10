"""End-to-end inference pipeline. Runs a clip from raw video to metrics JSON.

Stages, matching the methodology:
  [1] decode native frames        [2] per-frame dohyo calibration
  [3] pixel-to-cm scale            [4] YOLO detection, filtered to the dohyo
  [5] OC-SORT tracking             [6] kinematics    [7] event detection

Output is a structured JSON (trajectories, metrics, events, timestamps in ms) and
an overlay video. Viability (end-to-end FPS) is measured over the whole pipeline,
not just the detector forward pass, so the number reflects real cost.

Usage:
    uv run python infer.py <video.mp4> --weights runs/detect/train/weights/best.pt \\
        --tracker ocsort --out results/run
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from crop import clip_roi, crop_frame
from dohyo import detect_calibration, point_in_ellipse
from events import EventConfig, detect_events
from metrics import Kinematics, kinematics_from_track
from schema import DOHYO_DIAMETER_CM, Calibration, Track
from tracking import build_tracker, run_tracker

IMG_SIZE = 640
MAX_ROBOTS = 2


def load_frames(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    return frames, fps


def per_frame_calibrations(frames: list[np.ndarray]) -> list[Calibration]:
    """Detect the dohyo each frame; carry the last valid fit over misses.

    The handheld camera moves, so calibration is per frame. Carrying the last good
    fit over an occasional miss keeps the cm projection continuous.
    """
    raw: list[Calibration | None] = []
    last: Calibration | None = None
    for frame in frames:
        found = detect_calibration(frame)
        if found is not None:
            last = found[0]
        raw.append(last)
    first_valid = next((c for c in raw if c is not None), None)
    if first_valid is None:
        raise RuntimeError("dohyo not detected in any frame")
    return [c if c is not None else first_valid for c in raw]


def detect_robots(model, frame: np.ndarray, cal: Calibration, roi: tuple[int, int, int, int]) -> tuple[list, list]:
    """YOLO on the dohyo crop, boxes mapped back to native, kept inside the dohyo.

    Detecting on the crop makes the robots large enough to survive motion blur and
    removes the background; the box is then offset back to native coordinates so
    tracking and metrics stay in the world frame.
    """
    crop = crop_frame(frame, roi)
    result = model(crop, imgsz=IMG_SIZE, verbose=False)[0]
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return [], []
    xywh = boxes.xywh.cpu().numpy()
    confs = boxes.conf.cpu().numpy()
    x0, y0, _, _ = roi
    kept = []
    for (cx, cy, w, h), conf in zip(xywh, confs):
        ncx, ncy = cx + x0, cy + y0
        if not point_in_ellipse(ncx, ncy, cal, margin=1.15):
            continue
        kept.append(((ncx - w / 2, ncy - h / 2, w, h), float(conf)))
    kept.sort(key=lambda kc: kc[1], reverse=True)
    kept = kept[:MAX_ROBOTS]
    return [b for b, _ in kept], [c for _, c in kept]


def _kinematics_json(k: Kinematics) -> dict:
    return {
        "robot_id": k.robot_id,
        "frames": k.frames.tolist(),
        "t_s": np.round(k.t_s, 4).tolist(),
        "x_cm": np.round(k.x_cm, 2).tolist(),
        "y_cm": np.round(k.y_cm, 2).tolist(),
        "speed_cms": np.round(k.speed_cms, 2).tolist(),
        "accel_cms2": np.round(k.accel_cms2, 2).tolist(),
        "path_length_cm": round(k.path_length_cm, 1),
        "max_speed_cms": round(float(k.speed_cms.max()), 1),
    }


def run(video: Path, weights: Path, tracker_name: str, out_dir: Path, cfg: EventConfig) -> dict:
    import torch
    from ultralytics import YOLO

    frames, fps = load_frames(video)
    out_dir.mkdir(parents=True, exist_ok=True)

    on_cuda = torch.cuda.is_available()
    if on_cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = YOLO(str(weights))
    weights_mb = round(sum(p.numel() * p.element_size() for p in model.model.parameters()) / 1e6, 1)
    calibrations = per_frame_calibrations(frames)
    roi = clip_roi(frames[:: max(1, len(frames) // 12)])

    detections = [detect_robots(model, f, cal, roi) for f, cal in zip(frames, calibrations)]
    tracker = build_tracker(tracker_name)
    tracks = run_tracker(tracker, detections, frames, fps)

    kinematics = {
        rid: kinematics_from_track(track, lambda fr: calibrations[min(fr, len(calibrations) - 1)], fps)
        for rid, track in tracks.items()
    }
    events = []
    if "A" in kinematics and "B" in kinematics:
        events = detect_events(kinematics["A"], kinematics["B"], cfg)
    elapsed = time.perf_counter() - started

    payload = {
        "video": video.name,
        "fps": round(fps, 2),
        "n_frames": len(frames),
        "dohyo_diameter_cm": DOHYO_DIAMETER_CM,
        "tracker": tracker_name,
        "trajectories": [_kinematics_json(k) for k in kinematics.values()],
        "events": [asdict(e) for e in events],
        "viability": {
            "end_to_end_fps": round(len(frames) / elapsed, 2),
            "wall_time_s": round(elapsed, 2),
            "model_weights_mb": weights_mb,
            "peak_vram_allocated_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if on_cuda else None,
            "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / 1e6, 1) if on_cuda else None,
        },
    }
    out_json = out_dir / f"{video.stem}.json"
    out_json.write_text(json.dumps(payload, indent=2))
    _write_mot(tracks, out_dir / f"{video.stem}_tracks.txt")
    _render_overlay(frames, calibrations, tracks, events, fps, out_dir / f"{video.stem}_overlay.mp4")
    print(f"wrote {out_json} ({payload['viability']['end_to_end_fps']} fps)")
    return payload


def _write_mot(tracks: dict[str, Track], path: Path) -> None:
    """Track boxes as 'frame,id,x,y,w,h' in pixels, for tracking evaluation."""
    id_map = {"A": 0, "B": 1}
    rows = []
    for rid, track in tracks.items():
        oid = id_map.get(rid, 0)
        for p in track.points:
            x, y, w, h = p.bbox_xywh_px
            rows.append(f"{p.frame},{oid},{x:.1f},{y:.1f},{w:.1f},{h:.1f}")
    path.write_text("\n".join(rows) + "\n")


def _render_overlay(frames, calibrations, tracks, events, fps, out_path: Path) -> None:
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    colors = {"A": (0, 220, 0), "B": (0, 120, 255)}
    event_by_frame: dict[int, list] = {}
    for e in events:
        event_by_frame.setdefault(e.frame, []).append(e)

    for idx, (frame, cal) in enumerate(zip(frames, calibrations)):
        out = frame.copy()
        cv2.ellipse(out, (int(cal.center_x_px), int(cal.center_y_px)),
                    (int(cal.axis_w_px / 2), int(cal.axis_h_px / 2)), cal.angle_deg, 0, 360, (0, 255, 255), 2)
        for rid, track in tracks.items():
            pt = next((p for p in track.points if p.frame == idx), None)
            if pt is None:
                continue
            x, y, bw, bh = pt.bbox_xywh_px
            cv2.rectangle(out, (int(x), int(y)), (int(x + bw), int(y + bh)), colors.get(rid, (255, 255, 255)), 2)
            cv2.putText(out, rid, (int(x), int(y) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colors.get(rid, (255, 255, 255)), 2)
        for e in event_by_frame.get(idx, []):
            cv2.putText(out, e.kind.upper(), (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)
        writer.write(out)
    writer.release()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--tracker", default="ocsort", choices=("ocsort", "bytetrack"))
    ap.add_argument("--out", type=Path, default=Path("results/run"))
    args = ap.parse_args()
    run(args.video, args.weights, args.tracker, args.out, EventConfig())


if __name__ == "__main__":
    main()

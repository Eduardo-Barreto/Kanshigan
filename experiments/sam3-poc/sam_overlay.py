"""Diagnostic: render what SAM 3 annotates on every frame of a clip.

Unlike annotate_to_yolo (which only writes frames where it found boxes), this draws
SAM's detections on EVERY frame of the SAM input video, with a per-frame count, so
the input and the annotated output can be watched side by side. Used to show where
SAM fails (e.g., the small clustered overhead JP robots).

Usage:
    uv run python sam_overlay.py <sam_input.mp4> <out.mp4> [--prompt toy]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from pathlib import Path

import cv2
import numpy as np
import torch

from annotate_to_yolo import CHUNK_FRAMES, masks_to_bboxes
from sam3.model_builder import build_sam3_video_predictor


def load(path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    return frames, fps


def run_chunk(predictor, frames_rgb, fps, tmp, prompt):
    h, w = frames_rgb[0].shape[:2]
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames_rgb:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    r = predictor.handle_request(request=dict(type="start_session", resource_path=str(tmp)))
    sid = r["session_id"]
    r = predictor.handle_request(request=dict(type="add_prompt", session_id=sid, frame_index=0, text=prompt))
    outs = {0: r["outputs"]}
    for x in predictor.handle_stream_request(request=dict(type="propagate_in_video", session_id=sid)):
        outs[x["frame_index"]] = x["outputs"]
    masks = {}
    for fi, out in outs.items():
        m = out.get("out_binary_masks")
        if m is None:
            continue
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        masks[fi] = m
    predictor.handle_request(request=dict(type="close_session", session_id=sid))
    return masks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sam_input")
    ap.add_argument("out")
    ap.add_argument("--prompt", default="toy")
    ap.add_argument("--score-thresh", type=float, default=None, help="lower SAM detection threshold")
    args = ap.parse_args()

    frames_bgr, fps = load(args.sam_input)
    frames_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr]
    h, w = frames_bgr[0].shape[:2]

    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])
    if args.score_thresh is not None:
        from annotate_to_yolo import _set_detection_threshold

        _set_detection_threshold(predictor, args.score_thresh)
    import tempfile

    masks_per_frame = {}
    with tempfile.TemporaryDirectory() as td:
        for start in range(0, len(frames_rgb), CHUNK_FRAMES):
            chunk = frames_rgb[start : start + CHUNK_FRAMES]
            for li, m in run_chunk(predictor, chunk, fps, Path(td) / "c.mp4", args.prompt).items():
                masks_per_frame[start + li] = m
            torch.cuda.empty_cache()

    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for idx, frame in enumerate(frames_bgr):
        out = frame.copy()
        boxes = masks_to_bboxes(masks_per_frame[idx], w, h) if idx in masks_per_frame else []
        for (x, y, bw, bh) in boxes:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 220, 0), 2)
        label = f"SAM (prompt '{args.prompt}'): {len(boxes)} robo(s)"
        color = (0, 220, 0) if len(boxes) == 2 else (0, 165, 255) if len(boxes) == 1 else (0, 0, 255)
        cv2.rectangle(out, (0, 0), (w, 28), (0, 0, 0), -1)
        cv2.putText(out, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        writer.write(out)
    writer.release()
    n2 = sum(1 for m in masks_per_frame.values() if len(masks_to_bboxes(m, w, h)) == 2)
    print(f"wrote {args.out}: {len(frames_bgr)} frames, {n2} with 2 robots")


if __name__ == "__main__":
    main()

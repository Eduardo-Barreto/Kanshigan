"""Run SAM 3 on a prepared clip and emit YOLO bbox labels + frames.

Input:
    data/processed/sam_input/<clip_id>.mp4    decimated 480x270 clip for SAM
    data/processed/clips/<subset>/<clip_id>.mp4  native res clip (used to dump high-res frames)

Output:
    data/annotations/<split>/images/<clip_id>_<frame>.jpg
    data/annotations/<split>/labels/<clip_id>_<frame>.txt   YOLO: cls cx cy w h normalized

Usage:
    uv run python annotate.py <clip_id> --split train
"""

import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch

from sam3.model_builder import build_sam3_video_predictor

ROOT = Path(__file__).resolve().parents[2]
SAM_DIR = ROOT / "data" / "processed" / "sam_input"
CLIPS_DIR = ROOT / "data" / "processed" / "clips"
ANN_DIR = ROOT / "data" / "annotations"

PROMPT = "toy"
MAX_OBJECTS = 2
MIN_BBOX_AREA_RATIO = 0.0005
MAX_BBOX_AREA_RATIO = 0.20
CHUNK_FRAMES = 60  # SAM 3 holds per-frame feature maps; chunking caps VRAM on 8GB GPUs


def find_clip_subset(clip_id: str) -> str:
    for sub in ("br", "jp"):
        if (CLIPS_DIR / sub / f"{clip_id}.mp4").exists():
            return sub
    raise FileNotFoundError(f"native clip not found for {clip_id}")


def load_video_rgb(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def masks_to_bboxes(masks: np.ndarray, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    """Each mask -> bounding box of largest connected component, filtered by area."""
    if masks.ndim == 4:
        masks = masks[:, 0]
    bboxes = []
    img_area = img_w * img_h
    for m in masks:
        m_u8 = (m > 0).astype(np.uint8)
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        area_ratio = (w * h) / img_area
        if area_ratio < MIN_BBOX_AREA_RATIO or area_ratio > MAX_BBOX_AREA_RATIO:
            continue
        # Reject detections outside the dohyo: the clip is cropped to the arena, so
        # a box centered in the frame corners (outside the inscribed ellipse) is a
        # false positive on the surrounding mat/crowd, common at low thresholds.
        cx, cy = x + w / 2, y + h / 2
        if ((cx - img_w / 2) / (0.5 * img_w)) ** 2 + ((cy - img_h / 2) / (0.5 * img_h)) ** 2 > 1.0:
            continue
        bboxes.append((x, y, w, h))
    return bboxes


def scale_bbox(box: tuple[int, int, int, int], src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    x, y, w, h = box
    sx, sy = dst_w / src_w, dst_h / src_h
    return int(x * sx), int(y * sy), int(w * sx), int(h * sy)


def to_yolo_line(box: tuple[int, int, int, int], img_w: int, img_h: int, cls: int = 0) -> str:
    x, y, w, h = box
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def _write_chunk(frames_rgb: list[np.ndarray], fps: float, path: Path) -> None:
    h, w = frames_rgb[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames_rgb:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def _run_session(predictor, video_path: Path, prompt: str) -> dict[int, np.ndarray]:
    resp = predictor.handle_request(request=dict(type="start_session", resource_path=str(video_path)))
    session_id = resp["session_id"]
    resp = predictor.handle_request(
        request=dict(type="add_prompt", session_id=session_id, frame_index=0, text=prompt)
    )
    out_per_frame = {0: resp["outputs"]}
    for r in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        out_per_frame[r["frame_index"]] = r["outputs"]

    masks = {}
    for fidx, out in out_per_frame.items():
        m = out.get("out_binary_masks")
        if m is None:
            continue
        if isinstance(m, torch.Tensor):
            m = m.cpu().numpy()
        masks[fidx] = m

    predictor.handle_request(request=dict(type="close_session", session_id=session_id))
    return masks


def _set_detection_threshold(predictor, thresh: float) -> None:
    """Lower SAM 3's new-detection and detection-score thresholds.

    Defaults (new_det_thresh=0.7, score_threshold_detection=0.5) are tuned for
    salient objects; the plain black Robot Sumo robots, especially in the JP
    overhead view, score lower for the text concept, so a lower threshold is needed
    to detect both. Set on the in-process model (single-GPU predictor).
    """
    model = getattr(predictor, "model", None)
    if model is None:
        return
    if hasattr(model, "new_det_thresh"):
        model.new_det_thresh = thresh
    if hasattr(model, "score_threshold_detection"):
        model.score_threshold_detection = thresh


def run_sam(sam_frames: list[np.ndarray], sam_fps: float, prompt: str = PROMPT, score_thresh: float | None = None) -> dict[int, np.ndarray]:
    """Masks per global frame index, processing in chunks to bound VRAM.

    SAM 3's video predictor keeps feature maps for every frame in a session, so a
    long clip exhausts an 8 GB GPU. Each chunk runs as its own session (re-prompted
    at its frame 0), and the cache is cleared between chunks. Robot Sumo rounds are
    only seconds long, so an 80-frame (8 s at 10 fps) window comfortably spans one.
    """
    predictor = build_sam3_video_predictor(gpus_to_use=[torch.cuda.current_device()])
    if score_thresh is not None:
        _set_detection_threshold(predictor, score_thresh)
    result: dict[int, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for start in range(0, len(sam_frames), CHUNK_FRAMES):
            chunk = sam_frames[start : start + CHUNK_FRAMES]
            chunk_path = Path(tmp) / f"chunk_{start:05d}.mp4"
            _write_chunk(chunk, sam_fps, chunk_path)
            print(f"  chunk {start}-{start + len(chunk) - 1}")
            for local_idx, masks in _run_session(predictor, chunk_path, prompt).items():
                result[start + local_idx] = masks
            torch.cuda.empty_cache()
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id")
    ap.add_argument("--split", default="train", help="output subdir under data/annotations (train|val|gold or scratch name)")
    ap.add_argument("--prompt", default=PROMPT, help="SAM 3 text prompt")
    ap.add_argument("--score-thresh", type=float, default=None, help="lower SAM detection threshold (e.g. 0.15 for JP)")
    args = ap.parse_args()

    clip_id = args.clip_id
    subset = find_clip_subset(clip_id)
    sam_clip = SAM_DIR / f"{clip_id}.mp4"
    native_clip = CLIPS_DIR / subset / f"{clip_id}.mp4"

    if not sam_clip.exists():
        raise FileNotFoundError(f"SAM input missing: {sam_clip}. Run prep_clips.py first.")

    out_images = ANN_DIR / args.split / "images"
    out_labels = ANN_DIR / args.split / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    print(f"[{clip_id}] loading SAM frames {sam_clip}")
    sam_frames, sam_fps = load_video_rgb(sam_clip)
    sam_h, sam_w = sam_frames[0].shape[:2]

    # Native frames are read on demand: a full-length 1080p60 clip would not fit
    # in RAM if loaded whole, and only the labeled frames are ever needed.
    native_cap = cv2.VideoCapture(str(native_clip))
    native_fps = native_cap.get(cv2.CAP_PROP_FPS)
    native_w = int(native_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_h = int(native_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    native_count = int(native_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[{clip_id}] running SAM 3 ({len(sam_frames)} frames @ {sam_fps:.1f}fps, {sam_w}x{sam_h})")
    masks_per_frame = run_sam(sam_frames, sam_fps, args.prompt, args.score_thresh)

    stride = max(1, round(native_fps / sam_fps))
    written = 0
    for sam_idx, masks in sorted(masks_per_frame.items()):
        bboxes_small = masks_to_bboxes(masks, sam_w, sam_h)[:MAX_OBJECTS]
        if not bboxes_small:
            continue

        native_idx = min(sam_idx * stride, native_count - 1)
        native_cap.set(cv2.CAP_PROP_POS_FRAMES, native_idx)
        ok, native_bgr = native_cap.read()
        if not ok:
            continue

        lines = [
            to_yolo_line(scale_bbox(box, sam_w, sam_h, native_w, native_h), native_w, native_h)
            for box in bboxes_small
        ]
        stem = f"{clip_id}_{sam_idx:04d}"
        cv2.imwrite(str(out_images / f"{stem}.jpg"), native_bgr)
        (out_labels / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        written += 1

    native_cap.release()
    print(f"[{clip_id}] wrote {written} labeled frames to {args.split}")


if __name__ == "__main__":
    main()

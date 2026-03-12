import subprocess
import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sam3.model_builder import build_sam3_video_predictor

VIDEO_PATH = "/home/barreto/tcc/betternrobos9s.mp4"
OUTPUT_DIR = Path(__file__).parent / "output"
PROMPT = "toy"
TARGET_FPS = 5
CROP_SIZE = (320, 180)
MAX_ROBOTS = 20

COLORS = np.array(
    [[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [255, 50, 255]],
    dtype=np.uint8,
)
ALPHA = 0.45
ROI_COLOR = (0, 255, 200)
ROI_THICKNESS = 2


def detect_dohyo(frame_bgr: np.ndarray) -> tuple[int, int, int, int] | None:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    h, w = frame_bgr.shape[:2]
    x, y, cw, ch = cv2.boundingRect(largest)

    pad = int(0.05 * max(w, h))
    x = max(0, x - pad)
    y = max(0, y - pad)
    cw = min(w - x, cw + 2 * pad)
    ch = min(h - y, ch + 2 * pad)

    return x, y, cw, ch


def preprocess_video(video_path: str, output_path: str, fps: int, crop_size: tuple[int, int]):
    """Extract frames, detect dohyo per frame, crop + resize, save as video."""
    cap = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = max(1, round(src_fps / fps))

    raw_frames_bgr = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_skip == 0:
            raw_frames_bgr.append(frame)
        frame_idx += 1
    cap.release()
    print(f"  Extracted {len(raw_frames_bgr)} frames (skip={frame_skip})")

    # Detect dohyo per frame
    rois = []
    fallback_roi = None
    for i, frame in enumerate(raw_frames_bgr):
        roi = detect_dohyo(frame)
        if roi is not None:
            fallback_roi = roi
        rois.append(roi if roi is not None else fallback_roi)

    if fallback_roi is None:
        raise RuntimeError("Could not detect dohyo in any frame")

    # Fill any initial None frames
    for i in range(len(rois)):
        if rois[i] is None:
            rois[i] = fallback_roi

    # Crop + resize each frame
    cw, ch = crop_size
    cropped_frames = []
    for frame, roi in zip(raw_frames_bgr, rois):
        rx, ry, rw, rh = roi
        crop = frame[ry : ry + rh, rx : rx + rw]
        resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_LINEAR)
        cropped_frames.append(resized)

    # Save as video
    writer = cv2.VideoWriter(
        output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, crop_size
    )
    for frame in cropped_frames:
        writer.write(frame)
    writer.release()

    # Also return original frames as RGB and ROIs for mapping back
    original_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in raw_frames_bgr]
    return original_rgb, rois


def load_video_frames(video_path: str) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frames, fps


def masks_to_full_frame(
    output: dict,
    roi: tuple[int, int, int, int],
    crop_size: tuple[int, int],
    full_h: int,
    full_w: int,
) -> list[np.ndarray]:
    masks = output.get("out_binary_masks")
    if masks is None:
        return []

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    if masks.ndim == 4:
        masks = masks[:, 0]

    rx, ry, rw, rh = roi
    result = []
    for i in range(masks.shape[0]):
        small_mask = masks[i]
        resized = cv2.resize(
            small_mask.astype(np.uint8), (rw, rh), interpolation=cv2.INTER_NEAREST
        )
        full_mask = np.zeros((full_h, full_w), dtype=np.uint8)
        y_end = min(ry + rh, full_h)
        x_end = min(rx + rw, full_w)
        full_mask[ry:y_end, rx:x_end] = resized[: y_end - ry, : x_end - rx]
        result.append(full_mask)

    return result


def filter_closest_to_center(
    full_masks: list[np.ndarray],
    roi: tuple[int, int, int, int],
    max_objects: int,
) -> list[np.ndarray]:
    rx, ry, rw, rh = roi
    roi_area = rw * rh
    max_mask_ratio = 0.15

    valid_masks = []
    for mask in full_masks:
        mask_area = np.count_nonzero(mask)
        if mask_area > roi_area * max_mask_ratio:
            continue
        valid_masks.append(mask)

    if len(valid_masks) <= max_objects:
        return valid_masks

    center_x, center_y = rx + rw / 2, ry + rh / 2

    distances = []
    for mask in valid_masks:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            distances.append(float("inf"))
            continue
        cx, cy = xs.mean(), ys.mean()
        dist = np.sqrt((cx - center_x) ** 2 + (cy - center_y) ** 2)
        distances.append(dist)

    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
    return [valid_masks[i] for i in sorted_indices[:max_objects]]


def overlay_full_frame(
    frame: np.ndarray,
    full_masks: list[np.ndarray],
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    result = frame.copy()

    for obj_idx, mask in enumerate(full_masks):
        binary = mask > 0
        if not binary.any():
            continue
        color = COLORS[obj_idx % len(COLORS)]
        result[binary] = (ALPHA * color + (1 - ALPHA) * result[binary]).astype(
            np.uint8
        )
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color.tolist(), 2)

    if roi:
        rx, ry, rw, rh = roi
        cv2.rectangle(result, (rx, ry), (rx + rw, ry + rh), ROI_COLOR, ROI_THICKNESS)

    return result


def save_frame_png(frame: np.ndarray, path: Path, title: str):
    plt.figure(figsize=(10, 6))
    plt.imshow(frame)
    plt.title(title)
    plt.axis("off")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Preprocess - detect dohyo per frame, crop + resize
    print("Preprocessing video (dynamic ROI per frame)...")
    cropped_path = str(OUTPUT_DIR / "cropped_input.mp4")
    original_frames, rois = preprocess_video(
        VIDEO_PATH, cropped_path, TARGET_FPS, CROP_SIZE
    )
    orig_h, orig_w = original_frames[0].shape[:2]
    n_frames = len(original_frames)
    print(f"  {n_frames} frames, ROI detected in each")

    # Save ROI preview
    preview = original_frames[0].copy()
    rx, ry, rw, rh = rois[0]
    cv2.rectangle(preview, (rx, ry), (rx + rw, ry + rh), ROI_COLOR, ROI_THICKNESS + 1)
    save_frame_png(preview, OUTPUT_DIR / "roi_preview.png", "Detected dohyo ROI (frame 0)")

    # Step 2: Run SAM 3 on cropped video
    gpus_to_use = [torch.cuda.current_device()]
    print("Building SAM 3 video predictor...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    print("Starting inference session...")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=cropped_path)
    )
    session_id = response["session_id"]

    print(f"Adding text prompt: '{PROMPT}' on frame 0...")
    response = predictor.handle_request(
        request=dict(
            type="add_prompt",
            session_id=session_id,
            frame_index=0,
            text=PROMPT,
        )
    )
    frame0_output = response["outputs"]
    n_det = len(frame0_output.get("out_obj_ids", []))
    print(f"  Frame 0: {n_det} detections")

    print("Propagating through video...")
    outputs_per_frame = {0: frame0_output}
    for resp in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]
    print(f"  Propagated {len(outputs_per_frame)} frames")

    # Step 3: Map masks to full frame, filter, render
    print("Building full-frame annotated video...")
    out_path = str(OUTPUT_DIR / "annotated.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), TARGET_FPS, (orig_w, orig_h)
    )

    stride = max(1, TARGET_FPS)
    for frame_idx in range(n_frames):
        frame = original_frames[frame_idx]
        roi = rois[frame_idx]

        if frame_idx in outputs_per_frame:
            full_masks = masks_to_full_frame(
                outputs_per_frame[frame_idx], roi, CROP_SIZE, orig_h, orig_w
            )
            full_masks = filter_closest_to_center(full_masks, roi, MAX_ROBOTS)
            annotated = overlay_full_frame(frame, full_masks, roi)
        else:
            annotated = overlay_full_frame(frame, [], roi)

        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

        if frame_idx % stride == 0:
            save_frame_png(
                annotated,
                OUTPUT_DIR / f"frame_{frame_idx:03d}.png",
                f"Frame {frame_idx}",
            )

    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

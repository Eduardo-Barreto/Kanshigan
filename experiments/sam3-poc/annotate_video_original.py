import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from sam3.model_builder import build_sam3_video_predictor

VIDEO_PATH = "/home/barreto/tcc/experiments/sam3-test/input_5fps.mp4"
OUTPUT_DIR = Path("/home/barreto/tcc/experiments/sam3-test/output")
PROMPT = "object on metal platform"

COLORS = np.array(
    [[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [255, 50, 255]],
    dtype=np.uint8,
)
ALPHA = 0.45


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


def overlay_masks(frame: np.ndarray, output: dict) -> np.ndarray:
    result = frame.copy()

    masks = output.get("out_binary_masks")
    if masks is None:
        masks = output.get("out_masks")
    if masks is None:
        return result

    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()

    # masks shape: (N, 1, H, W) or (N, H, W)
    if masks.ndim == 4:
        masks = masks[:, 0]

    for obj_idx in range(masks.shape[0]):
        binary = masks[obj_idx] > 0.0
        if binary.shape != frame.shape[:2]:
            continue
        color = COLORS[obj_idx % len(COLORS)]
        result[binary] = (ALPHA * color + (1 - ALPHA) * result[binary]).astype(
            np.uint8
        )
        # Draw contour
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color.tolist(), 2)

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

    gpus_to_use = [torch.cuda.current_device()]

    print("Building SAM 3 video predictor...")
    predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

    print(f"Loading video: {VIDEO_PATH}")
    video_frames, fps = load_video_frames(VIDEO_PATH)
    print(f"  {len(video_frames)} frames, {fps:.0f} FPS")

    print("Starting inference session...")
    response = predictor.handle_request(
        request=dict(type="start_session", resource_path=VIDEO_PATH)
    )
    session_id = response["session_id"]

    # Detect on frame 0 with text prompt
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

    # Debug: inspect output format
    print(f"  Output keys: {list(frame0_output.keys())}")
    for k, v in frame0_output.items():
        if isinstance(v, torch.Tensor):
            print(f"    {k}: tensor shape={v.shape}, dtype={v.dtype}")
        elif isinstance(v, np.ndarray):
            print(f"    {k}: ndarray shape={v.shape}")
        else:
            print(f"    {k}: {type(v).__name__} = {v}")

    # Save frame 0
    annotated_0 = overlay_masks(video_frames[0], frame0_output)
    save_frame_png(annotated_0, OUTPUT_DIR / "frame_000.png", f"Frame 0 - prompt: '{PROMPT}'")
    print(f"  Saved: {OUTPUT_DIR / 'frame_000.png'}")

    # Propagate through entire video
    print("Propagating through video...")
    outputs_per_frame = {}
    for resp in predictor.handle_stream_request(
        request=dict(type="propagate_in_video", session_id=session_id)
    ):
        outputs_per_frame[resp["frame_index"]] = resp["outputs"]
    print(f"  Propagated {len(outputs_per_frame)} frames")

    # Save key frames (every second)
    stride = max(1, int(fps))
    key_indices = list(range(0, len(video_frames), stride))
    for idx in key_indices:
        if idx in outputs_per_frame:
            annotated = overlay_masks(video_frames[idx], outputs_per_frame[idx])
            save_frame_png(
                annotated, OUTPUT_DIR / f"frame_{idx:03d}.png", f"Frame {idx}"
            )
    print(f"  Saved {len(key_indices)} key frames to {OUTPUT_DIR}")

    # Build annotated video
    print("Building annotated video...")
    h, w = video_frames[0].shape[:2]
    out_path = str(OUTPUT_DIR / "annotated.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    for frame_idx in range(len(video_frames)):
        frame = video_frames[frame_idx]
        if frame_idx in outputs_per_frame:
            frame = overlay_masks(frame, outputs_per_frame[frame_idx])
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

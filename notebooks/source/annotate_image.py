import cv2
import torch
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

VIDEO_PATH = "/home/barreto/tcc/experiments/sam3-test/input_10fps.mp4"
OUTPUT_DIR = Path("/home/barreto/tcc/experiments/sam3-test/output/image_model")
PROMPT = "object on metal platform"

COLORS = np.array(
    [[255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50], [255, 50, 255]],
    dtype=np.uint8,
)
ALPHA = 0.45


def overlay_masks(frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
    result = frame.copy()
    if masks.ndim == 4:
        masks = masks[:, 0]

    for i in range(masks.shape[0]):
        binary = masks[i] > 0
        if binary.shape != frame.shape[:2]:
            continue
        color = COLORS[i % len(COLORS)]
        result[binary] = (ALPHA * color + (1 - ALPHA) * result[binary]).astype(
            np.uint8
        )
        contours, _ = cv2.findContours(
            binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(result, contours, -1, color.tolist(), 2)
    return result


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load pre-converted 10fps video
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    print(f"Loaded {len(frames)} frames at {fps:.0f} FPS")

    # Build model
    print("Building SAM 3 image model...")
    model = build_sam3_image_model()
    processor = Sam3Processor(model)

    # Process frame by frame
    print(f"Running prompt '{PROMPT}' on each frame...")
    annotated_frames = []
    for i, frame in enumerate(frames):
        pil_img = Image.fromarray(frame)
        state = processor.set_image(pil_img)
        output = processor.set_text_prompt(state=state, prompt=PROMPT)

        masks = output["masks"]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()

        n_det = masks.shape[0] if masks is not None and len(masks.shape) > 0 else 0
        annotated = overlay_masks(frame, masks) if n_det > 0 else frame.copy()
        annotated_frames.append(annotated)

        if i % 10 == 0:
            print(f"  frame {i}/{len(frames)}: {n_det} detections")

    # Save key frames as PNG
    stride = max(1, len(frames) // 10)
    for i in range(0, len(frames), stride):
        path = OUTPUT_DIR / f"frame_{i:03d}.png"
        plt.figure(figsize=(10, 6))
        plt.imshow(annotated_frames[i])
        plt.title(f"Frame {i}")
        plt.axis("off")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
    print(f"Saved key frames to {OUTPUT_DIR}")

    # Build video
    print("Building annotated video...")
    h, w = frames[0].shape[:2]
    out_path = str(OUTPUT_DIR / "annotated.mp4")
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
    )
    for frame in annotated_frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""Annotate a folder of gold images with the SAM 3 image model, for annotator validation.

To measure whether SAM 3 is a good annotator, we run it on the exact human-reviewed
gold frames and compare its boxes to the gold (evaluate.py --sam-labels). Annotating
the gold images directly (instead of re-running the video predictor on the clip) keeps
the frames perfectly aligned with the gold labels, which are a renumbered subset.

Usage:
    uv run python annotate_gold_images.py --images <gold>/images --out <sam_labels_dir> \
        --prompt robot --score-thresh 0.15
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

MAX_OBJECTS = 2
MIN_BBOX_AREA_RATIO = 0.0005
MAX_BBOX_AREA_RATIO = 0.40  # gold images are cropped to the dohyo, so robots are larger


def masks_to_bboxes(masks: np.ndarray, img_w: int, img_h: int) -> list[tuple[int, int, int, int]]:
    if masks is None or len(masks) == 0:
        return []
    if masks.ndim == 4:
        masks = masks[:, 0]
    out = []
    img_area = img_w * img_h
    for m in masks:
        m_u8 = (m > 0).astype(np.uint8)
        contours, _ = cv2.findContours(m_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        ratio = (w * h) / img_area
        if ratio < MIN_BBOX_AREA_RATIO or ratio > MAX_BBOX_AREA_RATIO:
            continue
        out.append((x, y, w, h, w * h))
    out.sort(key=lambda b: b[4], reverse=True)
    return [b[:4] for b in out[:MAX_OBJECTS]]


def to_yolo_line(box: tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x, y, w, h = box
    return f"0 {(x + w / 2) / img_w:.6f} {(y + h / 2) / img_h:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def set_threshold(model, thresh: float | None) -> None:
    if thresh is None:
        return
    for obj in (model, getattr(model, "model", None)):
        if obj is None:
            continue
        if hasattr(obj, "new_det_thresh"):
            obj.new_det_thresh = thresh
        if hasattr(obj, "score_threshold_detection"):
            obj.score_threshold_detection = thresh


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--prompt", default="robot")
    ap.add_argument("--score-thresh", type=float, default=None)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = build_sam3_image_model()
    set_threshold(model, args.score_thresh)
    processor = Sam3Processor(model)

    images = sorted(args.images.glob("*.jpg"))
    written = 0
    for img_path in images:
        bgr = cv2.imread(str(img_path))
        h, w = bgr.shape[:2]
        state = processor.set_image(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        output = processor.set_text_prompt(state=state, prompt=args.prompt)
        masks = output["masks"]
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        boxes = masks_to_bboxes(masks, w, h)
        lines = [to_yolo_line(b, w, h) for b in boxes]
        (args.out / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
        if lines:
            written += 1
    print(f"wrote labels for {written}/{len(images)} images to {args.out}")


if __name__ == "__main__":
    main()

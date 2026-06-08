"""Transform the human-approved native gold into dohyo-crop space.

The gold was reviewed in native full frames. Training and inference run on dohyo
crops, so the test set must match that space. Rather than re-annotate (which would
discard the manual review), this crops each gold frame to the dohyo and maps the
approved boxes into crop coordinates deterministically. The native gold stays
untouched as the source of truth.

Usage:
    uv run python make_gold_crop.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

from crop import box_native_to_crop_yolo, clip_roi, crop_frame

ROOT = Path(__file__).resolve().parents[2]
GOLD = ROOT / "data" / "annotations" / "gold"
GOLD_CROP = ROOT / "data" / "annotations" / "gold_crop"


def _read_boxes_px(label: Path, w: int, h: int):
    boxes = []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        boxes.append((cx * w - bw * w / 2, cy * h - bh * h / 2, bw * w, bh * h))
    return boxes


def main() -> None:
    images = sorted((GOLD / "images").glob("*.jpg"))
    if not images:
        raise SystemExit("no native gold found; review the gold first")
    sample = [cv2.imread(str(p)) for p in images[:: max(1, len(images) // 12)]]
    roi = clip_roi(sample)
    x0, y0, cw, ch = roi
    print(f"gold dohyo crop {cw}x{ch} at ({x0},{y0})")

    (GOLD_CROP / "images").mkdir(parents=True, exist_ok=True)
    (GOLD_CROP / "labels").mkdir(parents=True, exist_ok=True)
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        boxes_px = _read_boxes_px(GOLD / "labels" / f"{img_path.stem}.txt", w, h)
        crop_boxes = [box_native_to_crop_yolo(b, roi) for b in boxes_px]
        lines = [f"0 {' '.join(f'{v:.6f}' for v in box)}" for box in crop_boxes if box[2] > 0 and box[3] > 0]
        cv2.imwrite(str(GOLD_CROP / "images" / img_path.name), crop_frame(img, roi))
        (GOLD_CROP / "labels" / f"{img_path.stem}.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {len(images)} cropped gold frames to {GOLD_CROP}")


if __name__ == "__main__":
    main()

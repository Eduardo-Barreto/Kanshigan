"""Render an annotation-overlay video so labels can be eyeballed for quality.

Draws each clip's YOLO boxes on its native frames and writes an mp4 to
results/preview/. Useful to watch SAM 3's output before training and to review the
gold set. Not part of the pipeline; a human-in-the-loop quality check.

Usage:
    uv run python preview_annotations.py --split train               # all clips
    uv run python preview_annotations.py --split val --clip-id br03_galena_gyarados_semi
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data" / "annotations"
PREVIEW = ROOT / "results" / "preview"
FPS = 10


def _frames_by_clip(split: str) -> dict[str, list[str]]:
    clips: dict[str, list[str]] = defaultdict(list)
    for label in sorted((ANN / split / "labels").glob("*.txt")):
        clip_id = label.stem.rsplit("_", 1)[0]
        clips[clip_id].append(label.stem)
    return clips


def render(split: str, clip_id: str, stems: list[str]) -> Path:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    first = cv2.imread(str(ANN / split / "images" / f"{stems[0]}.jpg"))
    h, w = first.shape[:2]
    out = PREVIEW / f"{split}_{clip_id}_annotated.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for stem in stems:
        img = cv2.imread(str(ANN / split / "images" / f"{stem}.jpg"))
        boxes = []
        for line in (ANN / split / "labels" / f"{stem}.txt").read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, bw, bh = (float(v) for v in line.split())
            boxes.append((cx, cy, bw, bh))
        # Detection labels carry no identity; color by left/right position per frame
        # so the overlay does not imply a tracked id (which is OC-SORT's job at inference).
        boxes.sort(key=lambda b: b[0])
        for i, (cx, cy, bw, bh) in enumerate(boxes):
            x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
            x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
            color = (0, 220, 0) if i == 0 else (0, 120, 255)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        cv2.putText(img, f"{clip_id} f{stem.rsplit('_', 1)[1]} (cor=esq/dir, sem ID)", (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        writer.write(img)
    writer.release()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("train", "val", "gold"), default="train")
    ap.add_argument("--clip-id")
    args = ap.parse_args()

    clips = _frames_by_clip(args.split)
    if args.clip_id:
        clips = {args.clip_id: clips[args.clip_id]}
    for clip_id, stems in clips.items():
        if not stems:
            continue
        out = render(args.split, clip_id, stems)
        print(f"wrote {out} ({len(stems)} frames)")


if __name__ == "__main__":
    main()

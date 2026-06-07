"""Build an identity-labeled gold (A/B per frame) from the approved gold boxes.

IDF1/HOTA need ground-truth identity, which detection labels lack. This links the
two boxes per frame across time by nearest-centroid continuity, names the
leftmost-at-frame-0 robot A, and writes a MOT file plus an overlay video so the
identities can be eyeballed. Linking swaps identity at a crossing exactly like any
motion tracker, so the overlay is meant for a human to spot and fix the few
crossings before the file is used as ground truth.

Usage:
    uv run python identity_gold.py            # split gold_crop -> data/annotations/gold_crop/identity.txt
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "data" / "annotations" / "gold_crop"
PREVIEW = ROOT / "results" / "preview"
NATIVE_STRIDE = 6  # gold annotated at 10 fps; inference runs at native 60 fps


def _frame_boxes(label: Path, w: int, h: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        boxes.append((cx * w - bw * w / 2, cy * h - bh * h / 2, bw * w, bh * h))
    return boxes


def _centroid(box):
    x, y, w, h = box
    return np.array([x + w / 2, y + h / 2])


def link_identities():
    images = sorted((SPLIT / "images").glob("*.jpg"))
    first = cv2.imread(str(images[0]))
    h, w = first.shape[:2]

    rows = []  # (native_frame, id, box)
    prev = None  # {id: centroid}
    for img_path in images:
        sam_idx = int(img_path.stem.rsplit("_", 1)[1])
        native_frame = sam_idx * NATIVE_STRIDE
        boxes = _frame_boxes(SPLIT / "labels" / f"{img_path.stem}.txt", w, h)
        if len(boxes) != 2:
            continue
        if prev is None:
            order = sorted(range(2), key=lambda i: _centroid(boxes[i])[0])  # leftmost = A
            assign = {"A": boxes[order[0]], "B": boxes[order[1]]}
        else:
            keep = np.linalg.norm(_centroid(boxes[0]) - prev["A"]) + np.linalg.norm(_centroid(boxes[1]) - prev["B"])
            swap = np.linalg.norm(_centroid(boxes[1]) - prev["A"]) + np.linalg.norm(_centroid(boxes[0]) - prev["B"])
            assign = {"A": boxes[0], "B": boxes[1]} if keep <= swap else {"A": boxes[1], "B": boxes[0]}
        prev = {rid: _centroid(box) for rid, box in assign.items()}
        for rid, box in assign.items():
            rows.append((native_frame, rid, box))
    return rows, (w, h), images


def main() -> None:
    rows, (w, h), images = link_identities()
    id_map = {"A": 0, "B": 1}
    mot = "\n".join(f"{f},{id_map[rid]},{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}" for f, rid, b in rows)
    (SPLIT / "identity.txt").write_text(mot + "\n")

    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = PREVIEW / "gold_identity_review.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    colors = {"A": (0, 220, 0), "B": (0, 120, 255)}
    by_frame: dict[int, list] = {}
    for f, rid, b in rows:
        by_frame.setdefault(f, []).append((rid, b))
    for img_path in images:
        img = cv2.imread(str(img_path))
        native_frame = int(img_path.stem.rsplit("_", 1)[1]) * NATIVE_STRIDE
        for rid, b in by_frame.get(native_frame, []):
            x, y, bw, bh = b
            cv2.rectangle(img, (int(x), int(y)), (int(x + bw), int(y + bh)), colors[rid], 2)
            cv2.putText(img, rid, (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[rid], 2)
        writer.write(img)
    writer.release()
    print(f"wrote {SPLIT / 'identity.txt'} and {out} ({len(images)} frames)")


if __name__ == "__main__":
    main()

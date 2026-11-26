"""Contact sheets of harvested clips with their SAM 3 boxes drawn, for a quick visual check.

For each clip: up to 8 labelled frames, evenly spread, in one image, with the share of
labelled frames that hold exactly two boxes in the file name. Also prints one TSV line
per clip (clip, labelled frames, frames with two boxes, share).

Usage (from experiments/pre-banca, for its OpenCV):
    uv run python ../harvest/qa_sheet.py <annotations/dataset dir> <out dir> [clip prefix ...]
"""

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

TILES = 8
TILE_WIDTH = 480


def tile(image_path: Path, label_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    h, w = image.shape[:2]
    for line in label_path.read_text().split("\n"):
        if not line.strip():
            continue
        _cls, cx, cy, bw, bh = map(float, line.split())
        x0, y0 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x1, y1 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), max(2, w // 300))
    cv2.putText(image, image_path.stem[-4:], (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    return cv2.resize(image, (TILE_WIDTH, int(TILE_WIDTH * h / w)))


def main() -> None:
    dataset, out = Path(sys.argv[1]), Path(sys.argv[2])
    prefixes = tuple(sys.argv[3:]) or ("",)
    out.mkdir(parents=True, exist_ok=True)
    frames = defaultdict(list)
    for label in sorted((dataset / "labels").glob("*.txt")):
        clip = label.stem.rsplit("_", 1)[0]
        if clip.startswith(prefixes):
            frames[clip].append(label)
    for clip, labels in sorted(frames.items()):
        pairs = sum(1 for label in labels if len(label.read_text().split("\n")) - 1 == 2)
        share = pairs / len(labels)
        print(f"{clip}\t{len(labels)}\t{pairs}\t{share:.2f}")
        picks = [labels[i] for i in np.linspace(0, len(labels) - 1, min(TILES, len(labels))).astype(int)]
        tiles = [tile(dataset / "images" / f"{label.stem}.jpg", label) for label in picks]
        height = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 4)
        rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
        cv2.imwrite(str(out / f"{int(share * 100):03d}_{clip}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()

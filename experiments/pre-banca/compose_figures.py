"""Compose the paper's qualitative figures from inference overlay videos.

Each figure is a labelled montage of frames taken from the overlay MP4s that
infer.py writes. Kept as a script (not a one-off command) so every figure in the
paper is reproducible: rerun this after regenerating any overlay and the PNGs
update in place.

Frames are chosen by reading the inference JSON and picking a frame where both
robots are tracked, so the montage never lands on a frame with a missing box.

Usage:
    uv run python compose_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "results" / "examples"
WORLDS = EXAMPLES / "worlds"
FIGURES = ROOT / "results" / "figures"

BANNER_H = 36
PAD = 6
FONT = ImageFont.truetype("/usr/share/fonts/liberation/LiberationSans-Bold.ttf", 19)


def both_robots_frame(json_path: Path) -> int:
    """Median frame index in which both robots are tracked, for a clean montage."""
    payload = json.loads(json_path.read_text())
    frame_sets = [set(t["frames"]) for t in payload["trajectories"]]
    shared = sorted(set.intersection(*frame_sets)) if len(frame_sets) > 1 else sorted(frame_sets[0])
    return shared[len(shared) // 2] if shared else 0


def tracking_loss_frame(json_path: Path, into: float = 0.3) -> int:
    """A frame inside the longest run where no robot is tracked: the model's failure.

    During the violent collision the motion blur drops every box, so the trajectories
    have a contiguous gap. Landing `into` the way through the longest such gap shows
    the robots still mid-arena and clearly present, with nothing detected: the honest
    counterpart to `both_robots_frame`, which only ever picks moments that work.
    """
    payload = json.loads(json_path.read_text())
    n = payload["n_frames"]
    tracked = set().union(*(set(t["frames"]) for t in payload["trajectories"]))
    best_start, best_len, run_start = 0, 0, None
    for f in range(n):
        if f not in tracked:
            run_start = f if run_start is None else run_start
            if f - run_start + 1 > best_len:
                best_start, best_len = run_start, f - run_start + 1
        else:
            run_start = None
    return best_start + int(best_len * into)


def grab(video: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {frame_idx} from {video}")
    return frame


def fit_width(img: np.ndarray, width: int) -> np.ndarray:
    h = round(img.shape[0] * width / img.shape[1])
    return cv2.resize(img, (width, h), interpolation=cv2.INTER_AREA)


def banner(img: np.ndarray, text: str) -> np.ndarray:
    bar = Image.new("RGB", (img.shape[1], BANNER_H), (0, 0, 0))
    ImageDraw.Draw(bar).text((10, 8), text, font=FONT, fill=(255, 255, 255))
    return np.vstack([np.asarray(bar)[:, :, ::-1], img])


def pad_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] >= height:
        return img
    fill = np.zeros((height - img.shape[0], img.shape[1], 3), dtype=np.uint8)
    return np.vstack([img, fill])


def hstack(panels: list[np.ndarray]) -> np.ndarray:
    height = max(p.shape[0] for p in panels)
    panels = [pad_to_height(p, height) for p in panels]
    sep = np.full((height, PAD, 3), 255, dtype=np.uint8)
    row: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i:
            row.append(sep)
        row.append(p)
    return np.hstack(row)


def vstack(rows: list[np.ndarray]) -> np.ndarray:
    width = max(r.shape[1] for r in rows)
    rows = [r if r.shape[1] == width else cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=0) for r in rows]
    sep = np.full((PAD, width, 3), 255, dtype=np.uint8)
    out: list[np.ndarray] = []
    for i, r in enumerate(rows):
        if i:
            out.append(sep)
        out.append(r)
    return np.vstack(out)


def panel(video: Path, json_path: Path, title: str, width: int = 640) -> np.ndarray:
    frame = grab(video, both_robots_frame(json_path))
    return banner(fit_width(frame, width), title)


def qualitative_br_jp() -> None:
    left = panel(EXAMPLES / "demo_br_overlay.mp4", EXAMPLES / "demo_br.json", "Câmera de mão (Brasil, autônomo)")
    right = panel(EXAMPLES / "demo_jp_overlay.mp4", EXAMPLES / "demo_jp.json", "Câmera cenital fixa (Japão, autônomo)")
    cv2.imwrite(str(FIGURES / "qualitative_br_jp.png"), hstack([left, right]))


def cross_category_rc() -> None:
    left = panel(EXAMPLES / "atena" / "atena_overlay.mp4", EXAMPLES / "atena" / "atena.json", "Sumô RC: Atena vs. Bulbassauro")
    right = panel(EXAMPLES / "rc1591" / "rc1591_overlay.mp4", EXAMPLES / "rc1591" / "rc1591.json", "Sumô RC: outra partida")
    cv2.imwrite(str(FIGURES / "cross_category_rc.png"), hstack([left, right]))


def worlds_model_vs_sam() -> None:
    cell = 640
    r1 = WORLDS / "w_round1.json"
    model_start = banner(fit_width(grab(WORLDS / "w_round1_overlay.mp4", both_robots_frame(r1)), cell), "Round 1, início (movimento lento) - nosso modelo rastreia A e B")
    model_clash = banner(fit_width(grab(WORLDS / "w_round1_overlay.mp4", tracking_loss_frame(r1)), cell), "Round 1, auge da colisão (movimento rápido) - nosso modelo perde os dois")
    sam_r1 = banner(fit_width(grab(WORLDS / "w_round1_SAM.mp4", 30), cell), "Round 1 - SAM 3 (recorte do dohyo)")
    sam_rp = banner(fit_width(grab(WORLDS / "w_replay_SAM.mp4", 40), cell), "Replay câmera lenta - SAM 3")
    grid = vstack([hstack([model_start, model_clash]), hstack([sam_r1, sam_rp])])
    cv2.imwrite(str(FIGURES / "worlds_model_vs_sam.png"), grid)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    qualitative_br_jp()
    cross_category_rc()
    worlds_model_vs_sam()
    print(f"wrote composites to {FIGURES}")


if __name__ == "__main__":
    main()

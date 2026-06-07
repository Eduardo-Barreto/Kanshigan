"""Automatically segment a raw match video into single rounds.

The BR tournament videos concatenate several rounds plus intro and sponsor cards.
Tracking and event metrics are only meaningful within one round (robots reset
between rounds), so this finds each round as a contiguous burst of in-arena motion:
high frame-to-frame difference while the dohyo is present. Intro and ending cards
have no dohyo and little motion, so they fall away. Boundaries are approximate
(~0.5 s); they are meant to be eyeballed and trimmed, not taken as ground truth.

Usage:
    uv run python segment_rounds.py data/raw/br/<id>.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from dohyo import detect_calibration

SAMPLE_FPS = 5
MOTION_ACTIVE = 4.0          # mean abs frame diff above this counts as active play
DOHYO_MIN_SCORE = 0.40
DOHYO_FRACTION = 0.25        # a round must show the dohyo in this fraction of frames
GAP_TOLERANCE_S = 0.6        # bridge brief lulls inside a round; resets are longer
MIN_ROUND_S = 2.0
MAX_ROUND_S = 12.0           # longer windows are merged rounds; flagged for splitting
PAD_S = 0.4


def round_windows(path: Path) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / SAMPLE_FPS))

    prev = None
    samples: list[tuple[float, bool, bool]] = []  # (t, active, has_dohyo)
    for i in range(0, n, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY)
        motion = float(np.mean(cv2.absdiff(gray, prev))) if prev is not None else 0.0
        prev = gray
        found = detect_calibration(frame)
        samples.append((i / fps, motion >= MOTION_ACTIVE, found is not None and found[1] >= DOHYO_MIN_SCORE))
    cap.release()

    active = [s for s in samples if s[1]]
    if not active:
        return []
    # group active samples into runs, bridging only brief lulls
    runs: list[list[tuple[float, bool, bool]]] = [[active[0]]]
    for s in active[1:]:
        if s[0] - runs[-1][-1][0] <= GAP_TOLERANCE_S:
            runs[-1].append(s)
        else:
            runs.append([s])

    span = 1.0 / SAMPLE_FPS
    windows = []
    for run in runs:
        start, end = run[0][0], run[-1][0]
        if (end - start) < MIN_ROUND_S:
            continue
        if sum(s[2] for s in run) / len(run) < DOHYO_FRACTION:  # excludes intro animation
            continue
        windows.append((max(0.0, start - PAD_S), end + span + PAD_S))
    return windows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    args = ap.parse_args()
    for k, (s, e) in enumerate(round_windows(args.video), 1):
        print(f"round {k}: {s:.1f}-{e:.1f}  ({e - s:.1f}s)")


if __name__ == "__main__":
    main()

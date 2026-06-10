"""Mark round start/end frames in a multi-round match clip.

The galena clips hold a whole match (2-3 rounds) with camera cuts between rounds.
Identity and tracking are per round (robots reset between them), so each round
needs explicit [start, end] frames. Auto-segmentation is too rough, so this lets a
human scrub the native video and mark boundaries by hand. Gold boxes are overlaid
when present (native frame = sam_idx * STRIDE), so the annotated robots and the
cuts between rounds are visible while marking.

Output: data/annotations/round_marks/<clip>.json
    {"clip", "fps", "n_frames", "rounds": [[start, end], ...]}  (native frame indices)

Navigation:
    right / '.'   +1 frame        left / ','    -1 frame
    ']'           +10 frames      '['           -10 frames
    '='           +60 frames      '-'           -60 frames
    '0'           jump to frame 0

Marking:
    's'   set round START at current frame
    'e'   set round END at current frame
    'u'   undo the last mark
    'w'   write marks to disk
    'q' / ESC   write and quit

Usage:
    uv run python mark_rounds.py br01_galena_gyarados
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = ROOT / "data" / "processed" / "clips"
ANN_DIR = ROOT / "data" / "annotations"
MARKS_DIR = ANN_DIR / "round_marks"
GOLD_SPLIT = "gold_candidates"
STRIDE = 6  # gold annotated at 10 fps; native is 60 fps

COLOR_BOX = (0, 220, 0)
COLOR_START = (0, 220, 0)
COLOR_END = (0, 120, 255)
TXT_BG = (0, 0, 0)
TXT_FG = (255, 255, 255)


def find_clip(clip_id: str) -> Path:
    for sub in ("br", "jp"):
        p = CLIPS_DIR / sub / f"{clip_id}.mp4"
        if p.exists():
            return p
    raise SystemExit(f"clip not found: {clip_id}")


def gold_boxes(clip_id: str, native_frame: int, w: int, h: int) -> list[tuple[int, int, int, int]]:
    if native_frame % STRIDE != 0:
        return []
    sam_idx = native_frame // STRIDE
    label = ANN_DIR / GOLD_SPLIT / "labels" / f"{clip_id}_{sam_idx:04d}.txt"
    if not label.exists():
        return []
    boxes = []
    for line in label.read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, bw, bh = (float(v) for v in line.split())
        boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h), int(bw * w), int(bh * h)))
    return boxes


def compute_rounds(marks: list[tuple[int, str]]) -> list[list[int]]:
    """Pair START/END marks in frame order into [start, end] rounds."""
    rounds = []
    open_start = None
    for frame, kind in sorted(marks):
        if kind == "S":
            open_start = frame
        elif kind == "E" and open_start is not None:
            rounds.append([open_start, frame])
            open_start = None
    if open_start is not None:
        rounds.append([open_start, -1])  # unterminated; flagged with -1
    return rounds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("clip_id")
    args = ap.parse_args()
    clip_id = args.clip_id
    video = find_clip(clip_id)

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    MARKS_DIR.mkdir(parents=True, exist_ok=True)
    marks_path = MARKS_DIR / f"{clip_id}.json"
    marks: list[tuple[int, str]] = []
    if marks_path.exists():
        prev = json.loads(marks_path.read_text())
        for s, e in prev.get("rounds", []):
            marks.append((s, "S"))
            if e >= 0:
                marks.append((e, "E"))

    cv2.namedWindow("mark_rounds", cv2.WINDOW_NORMAL)
    pos = -1
    frame = None

    def get_frame(target: int):
        nonlocal pos, frame
        target = max(0, min(n_frames - 1, target))
        if target != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, img = cap.read()
        if ok:
            frame, pos = img, target
        return frame

    def save():
        rounds = compute_rounds(marks)
        marks_path.write_text(json.dumps({"clip": clip_id, "fps": fps, "n_frames": n_frames, "rounds": rounds}, indent=2))
        return rounds

    get_frame(0)
    while True:
        h, w = frame.shape[:2]
        view = frame.copy()
        for x, y, bw, bh in gold_boxes(clip_id, pos, w, h):
            cv2.rectangle(view, (x, y), (x + bw, y + bh), COLOR_BOX, 2)
        for f, kind in marks:
            if f == pos:
                col = COLOR_START if kind == "S" else COLOR_END
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), col, 6)

        rounds = compute_rounds(marks)
        starts = sorted(f for f, k in marks if k == "S")
        ends = sorted(f for f, k in marks if k == "E")
        cv2.rectangle(view, (0, 0), (w, 56), TXT_BG, -1)
        cv2.putText(view, f"frame {pos}/{n_frames - 1}  t={pos / fps:5.2f}s  rounds={len(rounds)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TXT_FG, 1, cv2.LINE_AA)
        cv2.putText(view, f"S={starts}  E={ends}",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TXT_FG, 1, cv2.LINE_AA)
        cv2.imshow("mark_rounds", view)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue
        if key in (ord("q"), 27):
            save()
            break
        if key in (ord("."), 83):
            get_frame(pos + 1)
        elif key in (ord(","), 81):
            get_frame(pos - 1)
        elif key == ord("]"):
            get_frame(pos + 10)
        elif key == ord("["):
            get_frame(pos - 10)
        elif key == ord("="):
            get_frame(pos + 60)
        elif key == ord("-"):
            get_frame(pos - 60)
        elif key == ord("0"):
            get_frame(0)
        elif key == ord("s"):
            marks = [(f, k) for f, k in marks if f != pos]
            marks.append((pos, "S"))
        elif key == ord("e"):
            marks = [(f, k) for f, k in marks if f != pos]
            marks.append((pos, "E"))
        elif key == ord("u") and marks:
            marks.pop()
        elif key == ord("w"):
            save()

    cap.release()
    cv2.destroyAllWindows()
    rounds = save()
    print(f"wrote {marks_path}")
    for k, (s, e) in enumerate(rounds, 1):
        dur = (e - s) / fps if e >= 0 else -1
        print(f"  round {k}: {s}..{e}  ({dur:.1f}s)" if e >= 0 else f"  round {k}: {s}..UNTERMINATED")


if __name__ == "__main__":
    main()

"""Mark round start/end frames in a multi-round match clip.

The galena clips hold a whole match (2-3 rounds) with camera cuts between rounds.
Identity and tracking are per round (robots reset between them), so each round
needs explicit [start, end] frames. Auto-segmentation is too rough, so this lets a
human scrub the match and mark boundaries by hand.

Navigation is by GOLD frame, not native frame: the gold is annotated at 10 fps but
the video is 60 fps, so this steps in units of one gold frame (STRIDE native
frames) and shows the gold box at each step. Marks and the saved rounds are in gold
frame indices (the sam_idx in the label filenames), which is what the identity and
tracker steps consume.

Output: data/annotations/round_marks/<clip>.json
    {"clip", "fps", "stride", "max_idx", "rounds": [[start_idx, end_idx], ...]}

Navigation (steps are in gold frames):
    right / '.'   +1        left / ','    -1
    ']'           +10       '['           -10
    '='           +50       '-'           -50
    'm'           next gold frame that has a box
    'n'           prev gold frame that has a box
    '0'           jump to first frame

Marking:
    's'   set round START at current gold frame
    'e'   set round END at current gold frame
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


def label_path(clip_id: str, idx: int) -> Path:
    return ANN_DIR / GOLD_SPLIT / "labels" / f"{clip_id}_{idx:04d}.txt"


def gold_boxes(clip_id: str, idx: int, w: int, h: int) -> list[tuple[int, int, int, int]]:
    label = label_path(clip_id, idx)
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
    """Pair START/END marks in index order into [start, end] rounds."""
    rounds = []
    open_start = None
    for idx, kind in sorted(marks):
        if kind == "S":
            open_start = idx
        elif kind == "E" and open_start is not None:
            rounds.append([open_start, idx])
            open_start = None
    if open_start is not None:
        rounds.append([open_start, -1])
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
    max_idx = (n_frames - 1) // STRIDE

    labeled = sorted(
        int(p.stem.rsplit("_", 1)[1])
        for p in (ANN_DIR / GOLD_SPLIT / "labels").glob(f"{clip_id}_*.txt")
    )
    labeled_set = set(labeled)

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
    pos = -1  # last native frame read
    frame = None

    def show(idx: int):
        nonlocal pos, frame
        idx = max(0, min(max_idx, idx))
        native = idx * STRIDE
        if native != pos + 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, native)
        ok, img = cap.read()
        if ok:
            frame, pos = img, native
        return idx

    def next_labeled(idx: int, step: int) -> int:
        cands = [i for i in labeled if (i > idx if step > 0 else i < idx)]
        return (min(cands) if step > 0 else max(cands)) if cands else idx

    def save():
        rounds = compute_rounds(marks)
        marks_path.write_text(json.dumps(
            {"clip": clip_id, "fps": fps, "stride": STRIDE, "max_idx": max_idx, "rounds": rounds}, indent=2))
        return rounds

    idx = 0
    show(idx)
    while True:
        h, w = frame.shape[:2]
        view = frame.copy()
        boxes = gold_boxes(clip_id, idx, w, h)
        for x, y, bw, bh in boxes:
            cv2.rectangle(view, (x, y), (x + bw, y + bh), COLOR_BOX, 2)
        for f, kind in marks:
            if f == idx:
                cv2.rectangle(view, (0, 0), (w - 1, h - 1), COLOR_START if kind == "S" else COLOR_END, 6)

        rounds = compute_rounds(marks)
        starts = sorted(f for f, k in marks if k == "S")
        ends = sorted(f for f, k in marks if k == "E")
        tag = f"{len(boxes)} box" if boxes else "no box"
        cv2.rectangle(view, (0, 0), (w, 56), TXT_BG, -1)
        cv2.putText(view, f"gold {idx}/{max_idx}  t={idx / 10:5.2f}s  {tag}  labeled={len(labeled)}  rounds={len(rounds)}",
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TXT_FG, 1, cv2.LINE_AA)
        cv2.putText(view, f"S={starts}  E={ends}", (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TXT_FG, 1, cv2.LINE_AA)
        cv2.imshow("mark_rounds", view)
        key = cv2.waitKey(20) & 0xFF

        if key == 255:
            continue
        if key in (ord("q"), 27):
            save()
            break
        if key in (ord("."), 83):
            idx = show(idx + 1)
        elif key in (ord(","), 81):
            idx = show(idx - 1)
        elif key == ord("]"):
            idx = show(idx + 10)
        elif key == ord("["):
            idx = show(idx - 10)
        elif key == ord("="):
            idx = show(idx + 50)
        elif key == ord("-"):
            idx = show(idx - 50)
        elif key == ord("m"):
            idx = show(next_labeled(idx, +1))
        elif key == ord("n"):
            idx = show(next_labeled(idx, -1))
        elif key == ord("0"):
            idx = show(0)
        elif key == ord("s"):
            marks = [(f, k) for f, k in marks if f != idx]
            marks.append((idx, "S"))
        elif key == ord("e"):
            marks = [(f, k) for f, k in marks if f != idx]
            marks.append((idx, "E"))
        elif key == ord("u") and marks:
            marks.pop()
        elif key == ord("w"):
            save()

    cap.release()
    cv2.destroyAllWindows()
    rounds = save()
    print(f"wrote {marks_path}")
    for k, (s, e) in enumerate(rounds, 1):
        print(f"  round {k}: gold {s}..{e}  ({(e - s) / 10:.1f}s)" if e >= 0 else f"  round {k}: gold {s}..UNTERMINATED")


if __name__ == "__main__":
    main()

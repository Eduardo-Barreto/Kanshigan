"""How long does a round last? Three measurements, from loosest to tightest.

The paper motivates the 60 fps requirement with "most rounds end in under a second".
This checks that sentence against the footage we have, without new annotation:

  1. window: the segment_rounds.py windows in configs/clips*.yaml (seconds). They
     include the wait before the release and padding, so they are upper bounds.
  2. pipeline: the pipeline's own detections and tracks on every native 60 fps clip
     (detector + OC-SORT, as infer.py), read for three instants:
       move    round_start event (first sustained motion, events.py)
       contact first frame the two robot boxes intersect (pixel space)
       exit    first frame a robot center leaves the dohyo ellipse (margin 1.0) of
               that frame's calibration, the same ellipse the pipeline uses
     and the durations move->exit and contact->exit.
  3. labels: the same contact and exit instants read off the human-reviewed boxes
     (gold, gold_jp, train, val, candidates_jp; 10 fps, label index k = native frame
     6k), with the dohyo ellipse fitted on the clip. They validate the pipeline's
     instants where both exist. The labelled frames are only those SAM 3 found and a
     reviewer kept, so a round's release is often not labelled: label "move" is not
     reported, only contact and exit.

Usage:
    uv run --project ../pre-banca python round_durations.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

CONFIGS = ROOT / "experiments" / "pre-banca" / "configs"
ANN = ROOT / "data" / "annotations"
LABEL_SPLITS = ("gold_crop", "gold_jp", "train", "val", "candidates_jp")
LABEL_STRIDE = 6
WEIGHTS = ROOT / "results/training/yolov8s_kanshigan/weights/best.pt"
OUT = ROOT / "results" / "E6_final_measurements" / "rounds"
AGREEMENT_S = 0.1

Box = tuple[float, float, float, float]  # x, y, w, h


def boxes_intersect(a: Box, b: Box) -> bool:
    return min(a[0] + a[2], b[0] + b[2]) > max(a[0], b[0]) and min(a[1] + a[3], b[1] + b[3]) > max(a[1], b[1])


def first_contact(boxes_by_frame: dict[int, list[Box]]) -> int | None:
    for frame in sorted(boxes_by_frame):
        boxes = boxes_by_frame[frame]
        if len(boxes) >= 2 and boxes_intersect(boxes[0], boxes[1]):
            return frame
    return None


def first_exit(boxes_by_frame: dict[int, list[Box]], inside) -> int | None:
    """First frame some box center is outside the dohyo; inside(frame, cx, cy)."""
    for frame in sorted(boxes_by_frame):
        if any(not inside(frame, x + w / 2, y + h / 2) for x, y, w, h in boxes_by_frame[frame]):
            return frame
    return None


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    arr = np.asarray(values)
    return {
        "n": int(arr.size),
        "median": round(float(np.median(arr)), 3),
        "mean": round(float(arr.mean()), 3),
        "min": round(float(arr.min()), 3),
        "max": round(float(arr.max()), 3),
        "share_under_1s": round(float(np.mean(arr < 1.0)), 3),
    }


def consensus(rows: list[dict], tolerance_s: float = AGREEMENT_S) -> dict:
    """Rounds whose exit instant the pipeline and the reviewed labels agree on.

    Either source alone is noisy (a missed box, a box outside the ellipse at the start,
    a round whose ejected robot is not labelled); where both put the exit within
    tolerance_s of each other the instant is trusted, and only those rounds are summarized.
    """
    agreed = [r for r in rows if r.get("labels", {}).get("pipeline_minus_label_exit_s") is not None
              and abs(r["labels"]["pipeline_minus_label_exit_s"]) <= tolerance_s]
    return {
        "tolerance_s": tolerance_s,
        "clips": [r["clip"] for r in agreed],
        "pipeline_move_to_exit_s": stats([r["pipeline"]["move_to_exit_s"] for r in agreed if r["pipeline"]["move_to_exit_s"] is not None]),
        "pipeline_contact_to_exit_s": stats([r["pipeline"]["contact_to_exit_s"] for r in agreed if r["pipeline"]["contact_to_exit_s"] is not None]),
        "labels_contact_to_exit_s": stats([r["labels"]["contact_to_exit_s"] for r in agreed if r["labels"]["contact_to_exit_s"] is not None]),
    }


def clip_windows() -> list[dict]:
    rows = []
    for name in ("clips.yaml", "clips_jp_candidates.yaml"):
        for clip in yaml.safe_load((CONFIGS / name).read_text())["clips"]:
            rows.append({"clip": clip["id"], "subset": clip["subset"], "source": Path(clip["source"]).name,
                         "window_s": round(clip["end"] - clip["start"], 2)})
    return rows


def label_boxes(split: str, clip: str, width: int, height: int) -> dict[int, list[Box]]:
    out: dict[int, list[Box]] = {}
    for label in sorted((ANN / split / "labels").glob(f"{clip}_*.txt")):
        idx = int(label.stem.rsplit("_", 1)[1])
        boxes = []
        for line in label.read_text().splitlines():
            if line.strip():
                _, cx, cy, w, h = map(float, line.split())
                boxes.append(((cx - w / 2) * width, (cy - h / 2) * height, w * width, h * height))
        out[idx * LABEL_STRIDE] = boxes
    return out


def pipeline_clip(model, video: Path) -> dict:
    from crop import clip_roi
    from dohyo import point_in_ellipse
    from events import EventConfig, detect_events
    from infer import detect_robots, load_frames, per_frame_calibrations
    from metrics import kinematics_from_track
    from tracking import build_tracker, run_tracker

    frames, fps = load_frames(video)
    height, width = frames[0].shape[:2]
    calibrations = per_frame_calibrations(frames)
    roi = clip_roi(frames[:: max(1, len(frames) // 12)])
    detections = [detect_robots(model, f, c, roi) for f, c in zip(frames, calibrations)]
    tracks = run_tracker(build_tracker("ocsort"), detections, frames, fps)
    kin = {rid: kinematics_from_track(t, lambda fr: calibrations[min(fr, len(calibrations) - 1)], fps) for rid, t in tracks.items()}
    events = detect_events(kin["A"], kin["B"], EventConfig()) if {"A", "B"} <= kin.keys() else []
    by_kind = {e.kind: e.frame for e in events}

    track_boxes: dict[int, list[Box]] = {}
    for track in tracks.values():
        for p in track.points:
            track_boxes.setdefault(p.frame, []).append(p.bbox_xywh_px)
    inside = lambda fr, x, y: point_in_ellipse(x, y, calibrations[min(fr, len(calibrations) - 1)], margin=1.0)  # noqa: E731
    stable = calibrations[len(calibrations) // 2]
    label_inside = lambda _fr, x, y: point_in_ellipse(x, y, stable, margin=1.0)  # noqa: E731
    return {
        "fps": fps, "n_frames": len(frames), "size": (width, height),
        "move": by_kind.get("round_start"), "event_ring_out": by_kind.get("ring_out"),
        "event_first_contact": by_kind.get("first_contact"),
        "contact": first_contact(track_boxes), "exit": first_exit(track_boxes, inside),
        "label_inside": label_inside,
    }


def seconds(a: int | None, b: int | None, fps: float) -> float | None:
    """Duration from frame a to frame b; None if either is missing or b precedes a."""
    return round((b - a) / fps, 3) if a is not None and b is not None and b >= a else None


def signed_seconds(a: int | None, b: int | None, fps: float) -> float | None:
    return round((b - a) / fps, 3) if a is not None and b is not None else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    from ultralytics import YOLO

    model = YOLO(str(WEIGHTS))
    windows = clip_windows()
    rows = []
    for w in windows:
        video = ROOT / "data/processed/clips" / w["subset"] / f"{w['clip']}.mp4"
        if not video.exists():
            rows.append({**w, "error": "clip not on disk"})
            continue
        p = pipeline_clip(model, video)
        fps = p["fps"]
        row = {**w, "fps": fps, "n_frames": p["n_frames"], "clip_s": round(p["n_frames"] / fps, 2),
               "pipeline": {k: p[k] for k in ("move", "contact", "exit", "event_first_contact", "event_ring_out")}}
        row["pipeline"]["move_to_exit_s"] = seconds(p["move"], p["exit"], fps)
        row["pipeline"]["contact_to_exit_s"] = seconds(p["contact"], p["exit"], fps)
        row["pipeline"]["move_to_contact_s"] = seconds(p["move"], p["contact"], fps)
        split = next((s for s in LABEL_SPLITS if any((ANN / s / "labels").glob(f"{w['clip']}_*.txt"))), None)
        if split:
            lb = label_boxes(split, w["clip"], *p["size"])
            lc, le = first_contact(lb), first_exit(lb, p["label_inside"])
            row["labels"] = {"split": split, "frames": len(lb), "first_native_frame": min(lb), "last_native_frame": max(lb),
                             "contact": lc, "exit": le, "contact_to_exit_s": seconds(lc, le, fps),
                             "pipeline_minus_label_contact_s": signed_seconds(lc, p["contact"], fps),
                             "pipeline_minus_label_exit_s": signed_seconds(le, p["exit"], fps)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    ok = [r for r in rows if "pipeline" in r]
    summary = {}
    for subset in ("jp", "br", "all"):
        sel = [r for r in ok if subset == "all" or r["subset"] == subset]
        summary[subset] = {
            "window_s": stats([r["window_s"] for r in sel]),
            "pipeline_move_to_exit_s": stats([r["pipeline"]["move_to_exit_s"] for r in sel if r["pipeline"]["move_to_exit_s"] is not None]),
            "pipeline_contact_to_exit_s": stats([r["pipeline"]["contact_to_exit_s"] for r in sel if r["pipeline"]["contact_to_exit_s"] is not None]),
            "pipeline_move_to_contact_s": stats([r["pipeline"]["move_to_contact_s"] for r in sel if r["pipeline"]["move_to_contact_s"] is not None]),
            "labels_contact_to_exit_s": stats([r["labels"]["contact_to_exit_s"] for r in sel if r.get("labels", {}).get("contact_to_exit_s") is not None]),
            "abs_pipeline_minus_label_exit_s": stats([abs(r["labels"]["pipeline_minus_label_exit_s"]) for r in sel if r.get("labels", {}).get("pipeline_minus_label_exit_s") is not None]),
            "abs_pipeline_minus_label_contact_s": stats([abs(r["labels"]["pipeline_minus_label_contact_s"]) for r in sel if r.get("labels", {}).get("pipeline_minus_label_contact_s") is not None]),
            "clips_without_exit": [r["clip"] for r in sel if r["pipeline"]["exit"] is None],
        }
    br01 = json.loads((ANN / "round_marks" / "br01_galena_gyarados.json").read_text())
    report = {
        "weights": str(WEIGHTS.relative_to(ROOT)), "tracker": "ocsort", "label_stride": LABEL_STRIDE,
        "round_marks_br01": {"rounds_native_frames": br01["rounds"], "fps": br01["fps"],
                             "durations_s": [round((e - s) / br01["fps"], 2) for s, e in br01["rounds"]],
                             "note": "old mark_rounds.py format (no stride key): native frames, per diary 22"},
        "summary": summary, "consensus_exit": consensus(ok), "clips": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "round_durations.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"summary": summary, "consensus_exit": report["consensus_exit"]}, indent=2))


if __name__ == "__main__":
    main()

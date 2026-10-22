"""Tracker grid for the final paper: 2 detectors x 4 trackers on the BR identity gold.

Extends ../pre-banca/tracker_comparison.py, which fixed the detector (YOLO26n) and
varied the tracker. Here both detectors (YOLOv8s and YOLO26n) feed the same four
trackers, each detector's detections cached once and replayed into every tracker, so
inside a detector row any gap is the tracker's.

Each run is scored three ways, because the banca asked what the reported numbers hide:

  * gold "as_is": identity.txt exactly as evaluate.py reads it (118 boxes). It was
    linked before the crop-clamp fix of diary 16, so a few boxes lie partly or wholly
    outside the cropped clip the pipeline sees; no tracker can match those.
  * gold "visible": the same boxes clipped to the clip frame at evaluation time, boxes
    with no visible area dropped. The file on disk is not touched.
  * tracks "assigned" (after assign_ab keeps the two longest tracks, what infer.py
    emits) and "raw" (every tracker id, before that post-filter).

MOTA/IDF1/ID switches come from evaluate.evaluate_tracking (motmetrics), the numbers
the paper already reports. HOTA (with DetA, AssA, LocA) and a CLEAR/Identity
cross-check come from the official TrackEval metric classes, fed per gold frame with
IoU similarity. Only gold frames are scored (one every 6 native frames), the same
convention as evaluate.py.

Usage (pre-banca env plus TrackEval, pinned to its last commit):
    uv run --project ../pre-banca \\
        --with "git+https://github.com/JonathonLuiten/TrackEval@12c8791b303e0a0b50f753af204249e622d0281a" \\
        python final_tracking.py
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

from tracking import assign_ab, detections_to_array  # noqa: E402

DETECTORS = {
    "yolov8s": ROOT / "results/training/yolov8s_kanshigan/weights/best.pt",
    "yolo26n": ROOT / "results/training/yolo26n_kanshigan/weights/best.pt",
}
TRACKERS = ("ocsort", "bytetrack", "deepocsort", "botsort")
APPEARANCE = ("deepocsort", "botsort")
FPS_REPEATS = 3
OUT = ROOT / "results" / "E6_final_measurements" / "tracking"

MotRows = dict[int, dict[int, np.ndarray]]


def clip_gold_to_frame(gold: MotRows, width: int, height: int) -> MotRows:
    """Clip each xywh box to the frame; drop boxes left with no area."""
    out: MotRows = {}
    for frame, boxes in gold.items():
        for oid, (x, y, w, h) in boxes.items():
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            if x1 <= x0 or y1 <= y0:
                continue
            out.setdefault(frame, {})[oid] = np.array([x0, y0, x1 - x0, y1 - y0])
    return out


def iou_matrix_xywh(a: list[np.ndarray], b: list[np.ndarray]) -> np.ndarray:
    if not a or not b:
        return np.zeros((len(a), len(b)))
    a_arr, b_arr = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    ax0, ay0, ax1, ay1 = a_arr[:, 0:1], a_arr[:, 1:2], a_arr[:, 0:1] + a_arr[:, 2:3], a_arr[:, 1:2] + a_arr[:, 3:4]
    bx0, by0, bx1, by1 = b_arr[:, 0], b_arr[:, 1], b_arr[:, 0] + b_arr[:, 2], b_arr[:, 1] + b_arr[:, 3]
    inter = np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None) * np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None)
    union = a_arr[:, 2:3] * a_arr[:, 3:4] + b_arr[:, 2] * b_arr[:, 3] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def trackeval_data(gold: MotRows, pred: MotRows) -> dict:
    """The per-sequence dict TrackEval metrics consume, over gold frames only.

    Ids are remapped to contiguous 0-based integers, as TrackEval's datasets do in
    get_preprocessed_seq_data.
    """
    frames = sorted(gold)
    gt_map = {oid: i for i, oid in enumerate(sorted({o for f in frames for o in gold[f]}))}
    tr_map = {oid: i for i, oid in enumerate(sorted({o for f in frames for o in pred.get(f, {})}))}
    data = {"gt_ids": [], "tracker_ids": [], "similarity_scores": [], "num_timesteps": len(frames)}
    for frame in frames:
        g, p = gold[frame], pred.get(frame, {})
        data["gt_ids"].append(np.array([gt_map[o] for o in g], dtype=int))
        data["tracker_ids"].append(np.array([tr_map[o] for o in p], dtype=int))
        data["similarity_scores"].append(iou_matrix_xywh(list(g.values()), list(p.values())))
    data["num_gt_ids"], data["num_tracker_ids"] = len(gt_map), len(tr_map)
    data["num_gt_dets"] = int(sum(len(x) for x in data["gt_ids"]))
    data["num_tracker_dets"] = int(sum(len(x) for x in data["tracker_ids"]))
    return data


def trackeval_scores(gold: MotRows, pred: MotRows) -> dict:
    for alias, kind in (("float", float), ("int", int)):  # TrackEval predates NumPy 2
        if not hasattr(np, alias):
            setattr(np, alias, kind)
    from trackeval.metrics import CLEAR, HOTA, Identity

    data = trackeval_data(gold, pred)
    hota = HOTA().eval_sequence(data)
    clear = CLEAR({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    ident = Identity({"THRESHOLD": 0.5, "PRINT_CONFIG": False}).eval_sequence(data)
    return {
        "HOTA": round(float(np.mean(hota["HOTA"])), 4),
        "DetA": round(float(np.mean(hota["DetA"])), 4),
        "AssA": round(float(np.mean(hota["AssA"])), 4),
        "LocA": round(float(np.mean(hota["LocA"])), 4),
        "HOTA@0.5": round(float(hota["HOTA"][9]), 4),
        "te_MOTA": round(float(clear["MOTA"]), 4),
        "te_IDF1": round(float(ident["IDF1"]), 4),
        "te_IDSW": int(clear["IDSW"]),
        "te_TP": int(clear["CLR_TP"]),
        "te_FN": int(clear["CLR_FN"]),
        "te_FP": int(clear["CLR_FP"]),
    }


def motmetrics_scores(gold: MotRows, pred: MotRows) -> dict:
    """evaluate.evaluate_tracking's exact logic, on in-memory rows."""
    if not hasattr(np, "asfarray"):
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    import motmetrics as mm

    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(gold):
        g, h = gold[frame], pred.get(frame, {})
        gb, hb = list(g.values()), list(h.values())
        dists = mm.distances.iou_matrix(gb, hb, max_iou=0.5) if gb and hb else np.empty((len(gb), len(hb)))
        acc.update(list(g.keys()), list(h.keys()), dists, frameid=frame)
    s = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches", "num_false_positives", "num_misses"], name="x")
    return {
        "MOTA": round(float(s["mota"].iloc[0]), 4),
        "IDF1": round(float(s["idf1"].iloc[0]), 4),
        "ID_switches": int(s["num_switches"].iloc[0]),
        "FP": int(s["num_false_positives"].iloc[0]),
        "FN": int(s["num_misses"].iloc[0]),
    }


def tracks_to_rows(tracks: dict) -> MotRows:
    rows: MotRows = {}
    for oid, track in tracks.items():
        for p in track.points:
            rows.setdefault(p.frame, {})[oid] = np.array(p.bbox_xywh_px, dtype=float)
    return rows


def write_mot(rows: MotRows, path: Path) -> None:
    lines = [f"{f},{oid},{b[0]:.1f},{b[1]:.1f},{b[2]:.1f},{b[3]:.1f}" for f in sorted(rows) for oid, b in rows[f].items()]
    path.write_text("\n".join(lines) + "\n")


def run_raw(tracker, detections, frames, fps: float) -> dict:
    """tracking.run_tracker without the final assign_ab, so raw ids survive."""
    from schema import Track, TrackPoint

    raw: dict = {}
    for idx, ((bboxes, confs), image) in enumerate(zip(detections, frames)):
        for row in tracker.update(detections_to_array(bboxes, confs), image):
            x1, y1, x2, y2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            raw.setdefault(tid, Track(robot_id=str(tid))).points.append(TrackPoint(frame=idx, t_s=idx / fps, bbox_xywh_px=(x1, y1, x2 - x1, y2 - y1)))
    return raw


def gold_stats(gold: MotRows, visible: MotRows, n_frames: int, fps: float, width: int, height: int) -> dict:
    frames = sorted(gold)
    steps = sorted(set(np.diff(frames).tolist()))
    return {
        "file": "data/annotations/gold_crop/identity.txt",
        "clip": "data/processed/clips/br/gold_zb01.mp4",
        "clip_frames": n_frames,
        "clip_fps": fps,
        "clip_size": [width, height],
        "gold_frames": len(frames),
        "gold_boxes": int(sum(len(v) for v in gold.values())),
        "frame_range": [frames[0], frames[-1]],
        "frame_step": steps,
        "sampling_hz": round(fps / steps[0], 2) if len(steps) == 1 else None,
        "boxes_fully_outside_clip": int(sum(len(v) for v in gold.values()) - sum(len(v) for v in visible.values())),
        "boxes_partly_outside_clip": int(sum(
            1 for f, bs in gold.items() for oid, b in bs.items()
            if oid in visible.get(f, {}) and not np.allclose(visible[f][oid], b)
        )),
        "visible_boxes": int(sum(len(v) for v in visible.values())),
    }


def build(name: str, device: str):
    if name in APPEARANCE:
        from tracker_comparison import _build_appearance_tracker

        return _build_appearance_tracker(name, device)
    from tracking import build_tracker

    return build_tracker(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "data/processed/clips/br/gold_zb01.mp4")
    ap.add_argument("--gold", type=Path, default=ROOT / "data/annotations/gold_crop/identity.txt")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    import torch

    from crop import clip_roi
    from evaluate import _load_mot
    from infer import load_frames, per_frame_calibrations
    from tracker_comparison import cache_detections

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    frames, fps = load_frames(args.video)
    height, width = frames[0].shape[:2]
    gold = _load_mot(args.gold)
    visible = clip_gold_to_frame(gold, width, height)
    golds = {"as_is": gold, "visible": visible}
    report = {"gold": gold_stats(gold, visible, len(frames), fps, width, height), "device": device,
              "fps_repeats": FPS_REPEATS, "runs": []}
    print(json.dumps(report["gold"], indent=2))

    calibrations = per_frame_calibrations(frames)
    roi = clip_roi(frames[:: max(1, len(frames) // 12)])
    for det_name, weights in DETECTORS.items():
        detections = cache_detections(weights, frames, calibrations, roi)
        n_det = sum(len(b) for b, _ in detections)
        for trk in TRACKERS:
            timings, raw = [], None
            for _ in range(FPS_REPEATS):
                tracker = build(trk, device)
                started = time.perf_counter()
                result = run_raw(tracker, detections, frames, fps)
                timings.append(time.perf_counter() - started)
                raw = raw or result
            fps_runs = [len(frames) / t for t in timings]
            variants = {"raw": {int(k): v for k, v in raw.items()}, "assigned": {0 if k == "A" else 1: v for k, v in assign_ab(raw).items()}}
            row = {"detector": det_name, "tracker": trk, "kind": "appearance" if trk in APPEARANCE else "motion",
                   "n_detections": n_det, "n_raw_ids": len(raw),
                   "tracker_fps_runs": [round(f, 1) for f in fps_runs],
                   "tracker_fps_mean": round(statistics.mean(fps_runs), 1),
                   "tracker_fps_std": round(statistics.stdev(fps_runs), 1), "scores": {}}
            for tv, tracks in variants.items():
                pred = tracks_to_rows(tracks)
                write_mot(pred, args.out / f"{det_name}_{trk}_{tv}.txt")
                for gv, g in golds.items():
                    row["scores"][f"{tv}|{gv}"] = {**motmetrics_scores(g, pred), **trackeval_scores(g, pred)}
            report["runs"].append(row)
            s = row["scores"]["assigned|as_is"]
            print(f"{det_name} {trk:10s} MOTA={s['MOTA']:.3f} IDF1={s['IDF1']:.3f} HOTA={s['HOTA']:.3f} IDsw={s['ID_switches']} {row['tracker_fps_mean']} fps")

    (args.out / "tracker_grid.json").write_text(json.dumps(report, indent=2))
    with (args.out / "tracker_grid.csv").open("w", newline="") as fh:
        writer = None
        for r in report["runs"]:
            for variant, s in r["scores"].items():
                tracks_variant, gold_variant = variant.split("|")
                flat = {"detector": r["detector"], "tracker": r["tracker"], "kind": r["kind"], "tracks": tracks_variant,
                        "gold": gold_variant, "n_raw_ids": r["n_raw_ids"], **s,
                        "tracker_fps_mean": r["tracker_fps_mean"], "tracker_fps_std": r["tracker_fps_std"]}
                writer = writer or csv.DictWriter(fh, fieldnames=list(flat))
                if fh.tell() == 0:
                    writer.writeheader()
                writer.writerow(flat)
    print(f"-> {args.out / 'tracker_grid.json'}")


if __name__ == "__main__":
    main()

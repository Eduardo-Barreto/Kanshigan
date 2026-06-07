"""Evaluate a trained pipeline against the gold set.

The gold set (manual, human-trusted) is the only ground truth, so every number
here is measured against it, never projected. Detector quality is mAP; tracking is
MOTA and IDF1 (HOTA via TrackEval is left for the full paper); extracted-metric
quality is mean absolute position and speed error; events are precision, recall and
mean temporal error. With a two-clip gold set these are order-of-magnitude
estimates, a limitation stated in the paper rather than hidden.

Usage:
    uv run python evaluate.py --weights <best.pt>            # detector mAP
    uv run python evaluate.py --pred run.json --gold-mot gold.txt --gold-events gold.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA_YAML = HERE / "configs" / "kanshigan.yaml"


def evaluate_detector(weights: Path) -> dict:
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(DATA_YAML), split="test", verbose=False)
    return {
        "mAP@0.5": round(float(metrics.box.map50), 4),
        "mAP@0.5:0.95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def _load_mot(path: Path) -> dict[int, dict[int, np.ndarray]]:
    """Parse 'frame,id,x,y,w,h' rows into {frame: {id: [x, y, w, h]}}."""
    by_frame: dict[int, dict[int, np.ndarray]] = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        frame, oid, x, y, w, h = (float(v) for v in line.split(","))
        by_frame.setdefault(int(frame), {})[int(oid)] = np.array([x, y, w, h])
    return by_frame


def evaluate_tracking(pred_mot: Path, gold_mot: Path) -> dict:
    import motmetrics as mm

    gt = _load_mot(gold_mot)
    hyp = _load_mot(pred_mot)
    acc = mm.MOTAccumulator(auto_id=False)
    for frame in sorted(set(gt) | set(hyp)):
        g = gt.get(frame, {})
        h = hyp.get(frame, {})
        gids, gboxes = list(g.keys()), list(g.values())
        hids, hboxes = list(h.keys()), list(h.values())
        dists = mm.distances.iou_matrix(gboxes, hboxes, max_iou=0.5) if gboxes and hboxes else np.empty((len(gboxes), len(hboxes)))
        acc.update(gids, hids, dists, frameid=frame)
    summary = mm.metrics.create().compute(acc, metrics=["mota", "idf1", "num_switches"], name="pipeline")
    return {
        "MOTA": round(float(summary["mota"].iloc[0]), 4),
        "IDF1": round(float(summary["idf1"].iloc[0]), 4),
        "ID_switches": int(summary["num_switches"].iloc[0]),
    }


def evaluate_events(pred: dict, gold_events: dict, tol_ms: float = 150.0) -> dict:
    """Precision, recall and mean temporal error per event kind.

    A predicted event matches a gold event of the same kind within tol_ms.
    """
    pred_by_kind: dict[str, list[float]] = {}
    for e in pred["events"]:
        pred_by_kind.setdefault(e["kind"], []).append(e["t_ms"])

    out: dict[str, dict] = {}
    for kind, gold_times in gold_events.items():
        preds = sorted(pred_by_kind.get(kind, []))
        matched, errors = 0, []
        for gt in gold_times:
            if preds and min(abs(p - gt) for p in preds) <= tol_ms:
                nearest = min(preds, key=lambda p: abs(p - gt))
                errors.append(abs(nearest - gt))
                matched += 1
        n_pred = len(preds)
        out[kind] = {
            "precision": round(matched / n_pred, 3) if n_pred else 0.0,
            "recall": round(matched / len(gold_times), 3) if gold_times else 0.0,
            "mean_temporal_error_ms": round(float(np.mean(errors)), 1) if errors else None,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path)
    ap.add_argument("--pred", type=Path, help="inference JSON")
    ap.add_argument("--pred-mot", type=Path, help="inference tracks in frame,id,x,y,w,h")
    ap.add_argument("--gold-mot", type=Path)
    ap.add_argument("--gold-events", type=Path)
    args = ap.parse_args()

    report: dict = {}
    if args.weights:
        report["detector"] = evaluate_detector(args.weights)
    if args.pred_mot and args.gold_mot:
        report["tracking"] = evaluate_tracking(args.pred_mot, args.gold_mot)
    if args.pred and args.gold_events:
        pred = json.loads(args.pred.read_text())
        gold = json.loads(args.gold_events.read_text())
        report["events"] = evaluate_events(pred, gold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

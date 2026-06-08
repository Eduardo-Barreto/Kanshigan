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


def _load_yolo_boxes(label_file: Path) -> np.ndarray:
    """Read a YOLO label file into xyxy boxes (normalized), ignoring class."""
    boxes = []
    if label_file.exists():
        for line in label_file.read_text().splitlines():
            if not line.strip():
                continue
            _, cx, cy, w, h = (float(v) for v in line.split()[:5])
            boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return np.array(boxes, dtype=np.float64).reshape(-1, 4)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def evaluate_annotator(pred_labels: Path, gold_labels: Path, iou_thresh: float = 0.5) -> dict:
    """Agreement of a proposed annotation (e.g. SAM 3) with the human-reviewed gold.

    Greedy IoU matching per frame over the gold frames: a proposed box matches a gold
    box at IoU >= iou_thresh. Reports precision, recall, F1 and the mean IoU of matched
    pairs, which together say how much manual correction the annotator demands.
    """
    tp = fp = fn = 0
    matched_ious: list[float] = []
    gold_files = sorted(Path(gold_labels).glob("*.txt"))
    for gold_file in gold_files:
        gold = _load_yolo_boxes(gold_file)
        pred = _load_yolo_boxes(Path(pred_labels) / gold_file.name)
        taken = set()
        for g in gold:
            best_iou, best_j = 0.0, -1
            for j, p in enumerate(pred):
                if j in taken:
                    continue
                iou = _iou(g, p)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_iou >= iou_thresh:
                tp += 1
                taken.add(best_j)
                matched_ious.append(best_iou)
            else:
                fn += 1
        fp += len(pred) - len(taken)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "mean_iou_matched": round(float(np.mean(matched_ious)), 4) if matched_ious else 0.0,
        "frames": len(gold_files),
        "gold_boxes": tp + fn,
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
    if not hasattr(np, "asfarray"):  # motmetrics predates NumPy 2.0
        np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)
    import motmetrics as mm

    gt = _load_mot(gold_mot)
    hyp = _load_mot(pred_mot)
    acc = mm.MOTAccumulator(auto_id=False)
    # Evaluate only on annotated frames: the gold is sampled sparser than inference,
    # so scoring inference-only frames would count every one as a false positive.
    for frame in sorted(gt):
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
    ap.add_argument("--sam-labels", type=Path, help="proposed YOLO labels dir (e.g. SAM 3)")
    ap.add_argument("--gold-labels", type=Path, help="human-reviewed YOLO labels dir")
    args = ap.parse_args()

    report: dict = {}
    if args.weights:
        report["detector"] = evaluate_detector(args.weights)
    if args.sam_labels and args.gold_labels:
        report["annotator"] = evaluate_annotator(args.sam_labels, args.gold_labels)
    if args.pred_mot and args.gold_mot:
        report["tracking"] = evaluate_tracking(args.pred_mot, args.gold_mot)
    if args.pred and args.gold_events:
        pred = json.loads(args.pred.read_text())
        gold = json.loads(args.gold_events.read_text())
        report["events"] = evaluate_events(pred, gold)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

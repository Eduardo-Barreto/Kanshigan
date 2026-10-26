"""Detector accuracy on both golds, two threshold conventions, with a dohyo post-filter.

Two conventions coexist in the paper and give different precision/recall:

  * "val": Ultralytics model.val(). It runs NMS at conf 0.001, builds the PR curve and
    reports P and R at the single confidence that maximizes the class-mean F1
    (ap_per_class: smooth(f1.mean(0), 0.1).argmax()). mAP comes from the whole curve.
  * "fixed": conf 0.25 (Ultralytics' predict default and the tracker's min_conf),
    greedy IoU >= 0.5 matching per frame, counted by hand (same rule as
    pos-banca/data_scaling.py).

The dohyo post-filter drops predictions whose center lies outside the dohyo ellipse,
where the ellipse is the pipeline's own dohyo.detect_calibration on that gold image
(margin 1.0 = the ring itself; 1.15 = the margin infer.py's detect_robots uses).
The gold is never touched; only predictions are filtered.

The COCO baseline (E3) is the stock yolov8s.pt. Its 80 COCO classes do not include a
robot, so it is scored twice: as Ultralytics scores it by default against the 1-class
gold (only COCO class 0, "person", can match class 0 "robot") and class-agnostic
(single_cls=True, every COCO class counts as robot).

Usage:
    uv run --project ../pre-banca python final_detection.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

from data_scaling import gold_boxes, inside_dohyo, iou  # noqa: E402

CONFIGS = ROOT / "experiments" / "pre-banca" / "configs"
GOLDS = {"gold_br": ("gold_crop", CONFIGS / "kanshigan.yaml"), "gold_jp": ("gold_jp", CONFIGS / "kanshigan_goldjp.yaml")}
DETECTORS = {
    "yolov8s_kanshigan": ROOT / "results/training/yolov8s_kanshigan/weights/best.pt",
    "yolo26n_kanshigan": ROOT / "results/training/yolo26n_kanshigan/weights/best.pt",
    "yolov8s_coco": ROOT / "experiments/pre-banca/yolov8s.pt",
}
CONF = 0.25
IOU_MATCH = 0.5
IMGSZ = 640
FILTER_MARGINS = (1.0, 1.15)
OUT = ROOT / "results" / "E6_final_measurements" / "detection"


def prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


def greedy_match(gold: list, preds: list) -> tuple[int, int, int]:
    """Greedy one-to-one IoU matching, highest-confidence prediction first."""
    taken: set[int] = set()
    tp = 0
    for p in preds:
        best, best_j = 0.0, -1
        for j, g in enumerate(gold):
            if j in taken:
                continue
            value = iou(g, p)
            if value > best:
                best, best_j = value, j
        if best >= IOU_MATCH:
            taken.add(best_j)
            tp += 1
    return tp, len(preds) - tp, len(gold) - tp


def fixed_threshold(model, split: str, single_cls: bool) -> dict:
    from dohyo import detect_calibration, point_in_ellipse

    counts = {"no_filter": [0, 0, 0], **{f"dohyo_filter_m{m}": [0, 0, 0] for m in FILTER_MARGINS}}
    calib_fallbacks = gold_outside = gold_total = 0
    for image_path in sorted((ROOT / "data/annotations" / split / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gold = gold_boxes(split, image_path.stem, width, height)
        result = model.predict(image, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
        keep = slice(None)
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        if not single_cls:
            keep = classes == 0
        preds = boxes[keep].tolist()  # already sorted by confidence
        found = detect_calibration(image)
        cal = found[0] if found else None
        calib_fallbacks += cal is None

        def inside(box, margin):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            return point_in_ellipse(cx, cy, cal, margin=margin) if cal else inside_dohyo(box, width, height)

        gold_total += len(gold)
        gold_outside += sum(1 for g in gold if not inside(g, 1.0))
        for name, kept in [("no_filter", preds)] + [(f"dohyo_filter_m{m}", [p for p in preds if inside(p, m)]) for m in FILTER_MARGINS]:
            tp, fp, fn = greedy_match(gold, kept)
            counts[name] = [counts[name][0] + tp, counts[name][1] + fp, counts[name][2] + fn]
    out = {name: prf(*c) for name, c in counts.items()}
    out["gold_boxes"] = gold_total
    out["gold_boxes_outside_dohyo_m1.0"] = gold_outside
    out["frames_without_dohyo_fit_used_inscribed_ellipse"] = calib_fallbacks
    return out


def evaluate(name: str, weights: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    modes = [False, True] if name.endswith("coco") else [False]
    report = {"weights": str(weights.relative_to(ROOT))}
    for gold_name, (split, data_yaml) in GOLDS.items():
        for single_cls in modes:
            key = f"{gold_name}{'_class_agnostic' if single_cls else ''}"
            metrics = model.val(data=str(data_yaml), split="test", verbose=False, plots=False, single_cls=single_cls)
            report[key] = {
                "val": {
                    "mAP@0.5": round(float(metrics.box.map50), 4),
                    "mAP@0.5:0.95": round(float(metrics.box.map), 4),
                    "precision_at_maxF1": round(float(metrics.box.mp), 4),
                    "recall_at_maxF1": round(float(metrics.box.mr), 4),
                },
                f"fixed_conf_{CONF}": fixed_threshold(model, split, single_cls),
            }
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    report = {"conf": CONF, "iou_match": IOU_MATCH, "imgsz": IMGSZ, "detectors": {}}
    for name, weights in DETECTORS.items():
        report["detectors"][name] = evaluate(name, weights)
        print(json.dumps({name: report["detectors"][name]}, indent=1), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "detector_eval.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

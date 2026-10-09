"""Does more JP training data change the detector? Seeds, golds and error anatomy.

The JP train split used 8 of the ~25 rounds of its source video. The remaining
rounds were later annotated with SAM 3 and reviewed by hand (candidates_jp), which
more than triples the JP training frames. This script compares detectors trained
without ("before") and with ("after") those rounds, over several seeds, on the two
held-out golds, which never change.

mAP alone hides what changed, so each detector is also scored at a fixed confidence
against the gold boxes, splitting its errors by where they fall:

  * missed: a gold robot with no prediction at IoU >= 0.5;
  * false positive inside the dohyo: a prediction on the arena that matches no gold
    box (e.g. a box on scratch marks);
  * false positive outside the dohyo: a prediction off the arena. The gold labels
    only robots in play, so a detector that also boxes a robot already pushed out
    is charged a false positive here. That is a labeling-convention gap, not a
    hallucination, and it has to be read separately.

Inside/outside uses the ellipse inscribed in the crop, shrunk by the crop margin.
It approximates the dohyo (the crop is clamped at the frame edge on some clips), so
the split is indicative, not a calibrated geometry test.

Needs the detector stack, so run it with the pre-banca environment:

    uv run --project ../pre-banca python data_scaling.py \\
        --before ../../results/training/yolov8s_kanshigan{,_s0,_s1}/weights/best.pt \\
        --after ../../results/training/yolov8s_kanshigan_jpplus{,_s0,_s1}/weights/best.pt \\
        --cases gold_jp:gold_jp25_0020 gold_jp:gold_jp25_0031
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments" / "pre-banca"))

from crop import DEFAULT_MARGIN  # noqa: E402

ANN = ROOT / "data" / "annotations"
CONFIGS = ROOT / "experiments" / "pre-banca" / "configs"
GOLDS = {"gold_br": ("gold_crop", CONFIGS / "kanshigan.yaml"), "gold_jp": ("gold_jp", CONFIGS / "kanshigan_goldjp.yaml")}
OUT_JSON = ROOT / "results" / "E5_jp_data_scaling" / "summary.json"
OUT_FIGURE = ROOT / "results" / "figures" / "jp_data_scaling_cases.png"

CONF = 0.25
IOU_MATCH = 0.5
IMGSZ = 640
DOHYO_RADIUS_SQ = (1 / (1 + DEFAULT_MARGIN)) ** 2
PANEL_WIDTH_PX = 900
COLOR_GOLD = (255, 255, 255)
COLOR_BEFORE = (0, 0, 255)
COLOR_AFTER = (0, 200, 0)

Box = list[float]


def iou(a: Box, b: Box) -> float:
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def gold_boxes(split: str, stem: str, width: int, height: int) -> list[Box]:
    boxes = []
    for line in (ANN / split / "labels" / f"{stem}.txt").read_text().splitlines():
        if not line.strip():
            continue
        _, cx, cy, w, h = map(float, line.split())
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height, (cx + w / 2) * width, (cy + h / 2) * height])
    return boxes


def inside_dohyo(box: Box, width: int, height: int) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return ((cx - width / 2) / (0.5 * width)) ** 2 + ((cy - height / 2) / (0.5 * height)) ** 2 <= DOHYO_RADIUS_SQ


def predict(model, image: np.ndarray) -> tuple[list[Box], list[float]]:
    result = model.predict(image, conf=CONF, imgsz=IMGSZ, verbose=False)[0]
    return result.boxes.xyxy.cpu().numpy().tolist(), result.boxes.conf.cpu().numpy().tolist()


def error_anatomy(model, split: str) -> dict:
    missed = fp_inside = fp_outside = gold_total = 0
    for image_path in sorted((ANN / split / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        height, width = image.shape[:2]
        gold = gold_boxes(split, image_path.stem, width, height)
        predictions, _ = predict(model, image)
        gold_total += len(gold)
        missed += sum(1 for g in gold if max((iou(g, p) for p in predictions), default=0.0) < IOU_MATCH)
        unmatched = [p for p in predictions if max((iou(g, p) for g in gold), default=0.0) < IOU_MATCH]
        fp_inside += sum(1 for p in unmatched if inside_dohyo(p, width, height))
        fp_outside += sum(1 for p in unmatched if not inside_dohyo(p, width, height))
    return {"gold_boxes": gold_total, "missed": missed, "fp_inside_dohyo": fp_inside, "fp_outside_dohyo": fp_outside}


def evaluate(weights: Path) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    report = {"weights": str(weights.relative_to(ROOT))}
    for gold_name, (split, data_yaml) in GOLDS.items():
        metrics = model.val(data=str(data_yaml), split="test", verbose=False, plots=False)
        report[gold_name] = {
            "mAP@0.5": round(float(metrics.box.map50), 4),
            "mAP@0.5:0.95": round(float(metrics.box.map), 4),
            "precision": round(float(metrics.box.mp), 4),
            "recall": round(float(metrics.box.mr), 4),
            **error_anatomy(model, split),
        }
    return report


def aggregate(runs: list[dict]) -> dict:
    summary = {}
    for gold_name in GOLDS:
        summary[gold_name] = {
            key: {
                "mean": round(statistics.mean(run[gold_name][key] for run in runs), 4),
                "std": round(statistics.stdev(run[gold_name][key] for run in runs), 4) if len(runs) > 1 else 0.0,
            }
            for key in runs[0][gold_name]
        }
    return summary


def panel(image: np.ndarray, gold: list[Box], predictions: list[Box], confidences: list[float], title: str, color) -> np.ndarray:
    scale = PANEL_WIDTH_PX / image.shape[1]
    canvas = cv2.resize(image, None, fx=scale, fy=scale)
    for box in gold:
        cv2.rectangle(canvas, (int(box[0] * scale), int(box[1] * scale)), (int(box[2] * scale), int(box[3] * scale)), COLOR_GOLD, 2)
    for box, confidence in zip(predictions, confidences):
        top_left = (int(box[0] * scale), int(box[1] * scale))
        cv2.rectangle(canvas, top_left, (int(box[2] * scale), int(box[3] * scale)), color, 2)
        cv2.putText(canvas, f"{confidence:.2f}", (top_left[0], max(44, top_left[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return canvas


def render_cases(before_weights: Path, after_weights: Path, cases: list[str], out: Path) -> None:
    from ultralytics import YOLO

    before, after = YOLO(str(before_weights)), YOLO(str(after_weights))
    rows = []
    for case in cases:
        gold_name, stem = case.split(":")
        split = GOLDS[gold_name][0]
        image = cv2.imread(str(ANN / split / "images" / f"{stem}.jpg"))
        height, width = image.shape[:2]
        gold = gold_boxes(split, stem, width, height)
        rows.append(np.hstack([
            panel(image, gold, *predict(before, image), f"antes | {stem}", COLOR_BEFORE),
            panel(image, gold, *predict(after, image), f"depois | {stem}", COLOR_AFTER),
        ]))
    widest = max(row.shape[1] for row in rows)
    padded = [cv2.copyMakeBorder(row, 0, 6, 0, widest - row.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)) for row in rows]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(padded))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", type=Path, nargs="+", required=True, help="weights trained without candidates_jp, one per seed")
    ap.add_argument("--after", type=Path, nargs="+", required=True, help="weights trained with candidates_jp, one per seed")
    ap.add_argument("--cases", nargs="*", default=[], help="<gold_br|gold_jp>:<stem> frames for the before/after figure (first seed pair)")
    ap.add_argument("--json", type=Path, default=OUT_JSON)
    ap.add_argument("--figure", type=Path, default=OUT_FIGURE)
    args = ap.parse_args()

    report = {"conf": CONF, "iou_match": IOU_MATCH}
    for name, weights in (("before", args.before), ("after", args.after)):
        runs = [evaluate(w.resolve()) for w in weights]
        report[name] = {"runs": runs, "summary": aggregate(runs)}
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2))
    print(json.dumps({name: report[name]["summary"] for name in ("before", "after")}, indent=2))

    if args.cases:
        render_cases(args.before[0].resolve(), args.after[0].resolve(), args.cases, args.figure)
        print(f"wrote {args.figure}")


if __name__ == "__main__":
    main()

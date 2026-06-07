"""Train the YOLOv8s detector on the SAM-annotated, human-reviewed dataset.

Separate from inference: this consumes the dataset and produces the frozen weights
that stage [4] later loads. Hyperparameters are fixed and versioned here, not
hidden in comments, and the seed is pinned so a run reproduces. Early stopping on
the val mAP plateau guards a small dataset against overfitting.

Usage:
    uv run python train.py --epochs 100
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA_YAML = HERE / "configs" / "kanshigan.yaml"
SEED = 42


def train(model_name: str, epochs: int, imgsz: int, batch: int, patience: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(model_name)
    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        seed=SEED,
        deterministic=True,
        project=str(HERE.parents[1] / "results" / "training"),
        name="yolov8s_kanshigan",
        exist_ok=True,
    )
    summary = {
        "model": model_name,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "seed": SEED,
        "save_dir": str(results.save_dir),
        "best_weights": str(Path(results.save_dir) / "weights" / "best.pt"),
        "metrics": {k: float(v) for k, v in results.results_dict.items()},
    }
    log = Path(results.save_dir) / "train_summary.json"
    log.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["metrics"], indent=2))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patience", type=int, default=20)
    args = ap.parse_args()
    train(args.model, args.epochs, args.imgsz, args.batch, args.patience)


if __name__ == "__main__":
    main()

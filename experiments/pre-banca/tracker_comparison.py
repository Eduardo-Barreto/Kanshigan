"""Controlled tracker comparison: one detector, identical detections, N trackers.

The research question asks how the *tracking* choice affects identity stability, so
the detector must be held fixed. This runs YOLO once over the gold round, caches the
per-frame detections, then replays those same detections through each tracker and
scores every run against the same gold identity (MOTA, IDF1, ID switches). Holding
the detections constant isolates the tracker: any metric gap is the tracker's, not
the detector's.

Motion-only trackers (OC-SORT, ByteTrack) and appearance trackers (DeepOCSORT,
BoT-SORT, with an OSNet ReID model) are compared on the same footing. The gold is
the human-reviewed BR round identity at data/annotations/gold_crop/identity.txt.

Usage:
    uv run python tracker_comparison.py \\
        --video data/processed/clips/br/gold_zb01.mp4 \\
        --weights results/training/yolo26n_kanshigan/weights/best.pt \\
        --gold data/annotations/gold_crop/identity.txt \\
        --out results/E4_tracker_comparison
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from crop import clip_roi
from evaluate import evaluate_tracking
from infer import detect_robots, load_frames, per_frame_calibrations
from tracking import build_tracker, run_tracker

ROOT = Path(__file__).resolve().parents[2]
MOTION_TRACKERS = ("ocsort", "bytetrack")
APPEARANCE_TRACKERS = ("deepocsort", "botsort")
REID_MODEL = "osnet_x0_25_msmt17.pt"


def _build_appearance_tracker(name: str, device: str):
    """Construct a ReID-based tracker via boxmot's zoo, weights auto-resolved."""
    from boxmot.trackers.tracker_zoo import create_tracker, get_tracker_config
    from boxmot.utils import WEIGHTS

    return create_tracker(
        name,
        get_tracker_config(name),
        reid_weights=Path(WEIGHTS) / REID_MODEL,
        device=device,
        half=False,
    )


def _make_tracker(name: str, device: str):
    if name in MOTION_TRACKERS:
        return build_tracker(name)
    return _build_appearance_tracker(name, device)


def cache_detections(weights: Path, frames, calibrations, roi):
    """Run YOLO once; return detections[i] = (bboxes_xywh, confs) per frame."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    return [detect_robots(model, f, cal, roi) for f, cal in zip(frames, calibrations)]


def _write_mot(tracks, path: Path) -> None:
    id_map = {"A": 0, "B": 1}
    rows = []
    for rid, track in tracks.items():
        oid = id_map.get(rid, 0)
        for p in track.points:
            x, y, w, h = p.bbox_xywh_px
            rows.append(f"{p.frame},{oid},{x:.1f},{y:.1f},{w:.1f},{h:.1f}")
    path.write_text("\n".join(rows) + "\n")


def run(video: Path, weights: Path, gold_mot: Path, out_dir: Path, trackers: tuple[str, ...]) -> dict:
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames, fps = load_frames(video)
    calibrations = per_frame_calibrations(frames)
    roi = clip_roi(frames[:: max(1, len(frames) // 12)])
    detections = cache_detections(weights, frames, calibrations, roi)
    n_det = sum(len(b) for b, _ in detections)
    print(f"cached {n_det} detections over {len(frames)} frames; gold = {gold_mot.name}")

    rows = []
    for name in trackers:
        tracker = _make_tracker(name, device)
        started = time.perf_counter()
        tracks = run_tracker(tracker, detections, frames, fps)
        tracker_fps = round(len(frames) / (time.perf_counter() - started), 1)
        pred_mot = out_dir / f"{video.stem}_{name}_tracks.txt"
        _write_mot(tracks, pred_mot)
        metrics = evaluate_tracking(pred_mot, gold_mot)
        kind = "appearance" if name in APPEARANCE_TRACKERS else "motion"
        rows.append({"tracker": name, "kind": kind, **metrics, "tracker_fps": tracker_fps})
        print(f"{name:11s} ({kind:10s})  MOTA={metrics['MOTA']:.3f}  IDF1={metrics['IDF1']:.3f}  IDsw={metrics['ID_switches']}  {tracker_fps} fps")

    report = {
        "video": video.name,
        "detector": weights.parent.parent.name,
        "n_frames": len(frames),
        "n_detections": n_det,
        "gold": gold_mot.name,
        "results": rows,
    }
    (out_dir / "comparison.json").write_text(json.dumps(report, indent=2))
    figures = ROOT / "results" / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plot_comparison(report, figures / "tracker_comparison.png")
    return report


def plot_comparison(report: dict, out: Path) -> None:
    """Two panels: identity accuracy (MOTA/IDF1) and tracker throughput (log FPS).

    Splitting the axes is the point of the figure: accuracy is near-tied across all
    four trackers, while throughput differs by over an order of magnitude, so the
    accuracy-vs-viability trade-off is only visible with both panels side by side.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rows = report["results"]
    names = [r["tracker"] for r in rows]
    mota = [r["MOTA"] for r in rows]
    idf1 = [r["IDF1"] for r in rows]
    idsw = [r["ID_switches"] for r in rows]
    tracker_fps = [r.get("tracker_fps", 0) for r in rows]
    bar_color = ["#3b7dd8" if r["kind"] == "motion" else "#d8763b" for r in rows]
    labels = [f"{n}\n({'aparência' if r['kind'] == 'appearance' else 'movimento'})" for n, r in zip(names, rows)]

    x = np.arange(len(names))
    width = 0.38
    fig, (ax_acc, ax_fps) = plt.subplots(1, 2, figsize=(10, 3.8))

    ax_acc.bar(x - width / 2, mota, width, label="MOTA", color="#3b7dd8")
    ax_acc.bar(x + width / 2, idf1, width, label="IDF1", color="#00a878")
    for i, sw in enumerate(idsw):
        ax_acc.text(x[i], max(mota[i], idf1[i]) + 0.02, f"{sw} ID sw", ha="center", fontsize=8, color="#444")
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels(labels, fontsize=8)
    ax_acc.set_ylim(0, 1.08)
    ax_acc.set_ylabel("pontuação de identidade")
    ax_acc.set_title("Acurácia: MOTA e IDF1")
    ax_acc.legend(loc="lower right")

    ax_fps.bar(x, tracker_fps, width * 1.4, color=bar_color)
    for i, f in enumerate(tracker_fps):
        ax_fps.text(x[i], f * 1.1, f"{f:.0f}", ha="center", fontsize=8, color="#444")
    ax_fps.set_yscale("log")
    ax_fps.set_xticks(x)
    ax_fps.set_xticklabels(labels, fontsize=8)
    ax_fps.set_ylabel("FPS do rastreador (escala log)")
    ax_fps.set_title("Viabilidade: throughput (detecção fixa)")

    fig.suptitle(f"Comparação de rastreadores sobre detecções idênticas ({report['detector']})")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, default=ROOT / "data/processed/clips/br/gold_zb01.mp4")
    ap.add_argument("--weights", type=Path, default=ROOT / "results/training/yolo26n_kanshigan/weights/best.pt")
    ap.add_argument("--gold", type=Path, default=ROOT / "data/annotations/gold_crop/identity.txt")
    ap.add_argument("--out", type=Path, default=ROOT / "results/E4_tracker_comparison")
    ap.add_argument("--trackers", nargs="+", default=list(MOTION_TRACKERS + APPEARANCE_TRACKERS))
    args = ap.parse_args()
    run(args.video, args.weights, args.gold, args.out, tuple(args.trackers))


if __name__ == "__main__":
    main()

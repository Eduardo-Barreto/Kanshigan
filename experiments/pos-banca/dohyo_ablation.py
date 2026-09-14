"""Ablation of the arena-detection stage, without weights and without labels.

Stage [2] picks the dohyo by fitting ellipses to bright contours and scoring each
candidate on size, centrality and a wider-than-tall preference. The module asserts
that this is "far more robust on handheld amateur footage than taking the largest
white blob, where background highlights win", and the log records that the naive
heuristic did fail during development, but the claim was never measured. This
script measures it.

No ground truth is needed, because the dohyo is fixed in the world and its diameter
is known. Two consequences are observable from the fits alone:

  * a correct fit is stable. Under a still camera the centre and the pixel scale
    barely move between frames, whereas a fit that latches onto a background
    highlight jumps frame to frame. Centre jitter and the coefficient of variation
    of cm_per_px therefore rank fit quality with no annotation at all.
  * a correct fit is plausible. Ellipses that are too small, too large or too
    elongated to be an obliquely viewed 154 cm platform are wrong by construction.

Jitter conflates a moving camera with an unstable fit, so it is only comparable
between the two methods on the same clip, never across clips. That comparison is
the point here: both methods see identical frames.

Usage:
    uv run python dohyo_ablation.py --clips clips.json --csv out.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pre-banca"))

from dohyo import MIN_AREA_RATIO, WHITE_THRESHOLD, detect_calibration  # noqa: E402
from schema import DOHYO_DIAMETER_CM, Calibration  # noqa: E402

SAMPLE_FPS = 5.0
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

# Two fits agree when their centres sit within this fraction of the major axis.
AGREE_FRACTION = 0.10


def naive_calibration(frame_bgr: np.ndarray) -> Calibration | None:
    """The baseline the scored fit replaced: largest bright blob, no plausibility test.

    Deliberately kept here rather than in the pipeline: it is the refuted option,
    not a supported mode. Preprocessing matches dohyo.detect_calibration exactly so
    the only variable is candidate selection.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    _, white = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best, best_area = None, 0.0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= best_area:
            continue
        hull = cv2.convexHull(contour)
        if len(hull) < 5:
            continue
        best, best_area = cv2.fitEllipse(hull), area

    if best is None:
        return None
    (cx, cy), (axis_w, axis_h), angle = best
    if axis_w <= 0 or axis_h <= 0:
        return None
    return Calibration(
        center_x_px=cx, center_y_px=cy, axis_w_px=axis_w,
        axis_h_px=axis_h, angle_deg=angle,
        cm_per_px=DOHYO_DIAMETER_CM / max(axis_w, axis_h),
    )


@dataclass(frozen=True)
class StageAudit:
    clip: str
    category: str
    method: str
    n_sampled: int
    detection_rate: float
    median_score: float | None
    center_jitter_norm: float
    scale_cv: float
    median_axis_px: float
    implausible_area_rate: float


def _mad(values: np.ndarray) -> float:
    """Median absolute deviation: robust to the occasional wild fit."""
    return float(np.median(np.abs(values - np.median(values))))


def _summarize(cals: list[Calibration | None], scores: list[float], frame_area: float,
               clip: str, category: str, method: str) -> StageAudit:
    found = [c for c in cals if c is not None]
    n = len(cals)
    if not found:
        return StageAudit(clip, category, method, n, 0.0, None, 0.0, 0.0, 0.0, 0.0)

    cx = np.array([c.center_x_px for c in found])
    cy = np.array([c.center_y_px for c in found])
    major = np.array([max(c.axis_w_px, c.axis_h_px) for c in found])
    scale = np.array([c.cm_per_px for c in found])
    minor = np.array([min(c.axis_w_px, c.axis_h_px) for c in found])
    area_ratio = math.pi * major * minor / 4 / frame_area
    med_major = float(np.median(major))

    return StageAudit(
        clip=clip,
        category=category,
        method=method,
        n_sampled=n,
        detection_rate=round(len(found) / n, 4),
        median_score=round(float(np.median(scores)), 4) if scores else None,
        center_jitter_norm=round(float(np.hypot(_mad(cx), _mad(cy)) / med_major), 4),
        scale_cv=round(float(np.std(scale) / np.mean(scale)), 4),
        median_axis_px=round(med_major, 1),
        implausible_area_rate=round(float(np.mean((area_ratio < MIN_AREA_RATIO) | (area_ratio > 0.85))), 4),
    )


def audit_clip(path: Path, category: str, start_s: float = 0.0, end_s: float | None = None,
               label: str | None = None) -> tuple[StageAudit, StageAudit, float]:
    """Run both fits over the same sampled frames of one clip."""
    label = label or path.stem
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / SAMPLE_FPS))
    first = int(start_s * fps)
    last = min(n_frames - 1, int(end_s * fps)) if end_s else n_frames - 1

    scored: list[Calibration | None] = []
    naive: list[Calibration | None] = []
    scores: list[float] = []
    frame_area = 1.0
    agree = both = 0

    for idx in range(first, last + 1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame_area = float(h * w)
        hit = detect_calibration(frame)
        s_cal = hit[0] if hit else None
        if hit:
            scores.append(hit[1])
        n_cal = naive_calibration(frame)
        scored.append(s_cal)
        naive.append(n_cal)
        if s_cal and n_cal:
            both += 1
            axis = max(s_cal.axis_w_px, s_cal.axis_h_px)
            if math.hypot(s_cal.center_x_px - n_cal.center_x_px,
                          s_cal.center_y_px - n_cal.center_y_px) <= AGREE_FRACTION * axis:
                agree += 1
    cap.release()

    return (
        _summarize(scored, scores, frame_area, label, category, "scored"),
        _summarize(naive, [], frame_area, label, category, "naive"),
        round(agree / both, 4) if both else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True,
                    help="JSON list of {path, category, start_s?, end_s?}")
    ap.add_argument("--csv", type=Path, required=True)
    args = ap.parse_args()

    spec = json.loads(args.clips.read_text())
    rows, agreements = [], []
    for entry in spec:
        path = Path(entry["path"])
        s_audit, n_audit, agree = audit_clip(
            path, entry["category"], entry.get("start_s", 0.0), entry.get("end_s"),
            entry.get("label"))
        rows += [s_audit, n_audit]
        agreements.append({"clip": s_audit.clip, "category": entry["category"], "agreement": agree})
        print(f"{s_audit.clip:<18} {entry['category']:<8} "
              f"scored det={s_audit.detection_rate:.2f} jitter={s_audit.center_jitter_norm:.3f} "
              f"cv={s_audit.scale_cv:.3f} | naive det={n_audit.detection_rate:.2f} "
              f"jitter={n_audit.center_jitter_norm:.3f} cv={n_audit.scale_cv:.3f} | "
              f"concord={agree:.2f}", flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])) + ["agreement"])
        writer.writeheader()
        by_clip = {a["clip"]: a["agreement"] for a in agreements}
        for row in rows:
            writer.writerow({**asdict(row), "agreement": by_clip.get(row.clip)})
    print(f"\n-> {args.csv}")


if __name__ == "__main__":
    main()

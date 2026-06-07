"""Classical-vision dohyo detection and pixel-to-centimeter calibration.

Stage [2] and [3] of the inference pipeline. The dohyo is the large platform
bounded by a bright rim (the tawara). Thresholding luminance and fitting ellipses
to bright contours yields many candidates; the dohyo is the one that is large,
central and wider-than-tall (an obliquely viewed circle). A score selects it,
which is far more robust on handheld amateur footage than taking the largest white
blob, where background highlights win. The known 154 cm diameter then sets scale.

No learning here: the rim is high contrast and the geometry is strong, so a scored
ellipse fit is more robust and cheaper than a model.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from schema import DOHYO_DIAMETER_CM, Calibration

WHITE_THRESHOLD = 180
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
MIN_AREA_RATIO = 0.04
MAX_AREA_RATIO = 0.85
MAX_ASPECT = 5.0
MATCH_MIN_SCORE = 0.30
MATCH_MIN_AREA_RATIO = 0.10


def _score_ellipse(ellipse, height: int, width: int) -> float:
    """Higher is more dohyo-like. Returns -1 to reject implausible ellipses.

    Combines relative size, centrality and a wider-than-tall preference, since an
    obliquely viewed circular platform images wider than tall.
    """
    (cx, cy), (axis_w, axis_h), angle = ellipse
    if axis_w <= 0 or axis_h <= 0:
        return -1.0

    frame_area = height * width
    area_ratio = (math.pi * axis_w * axis_h / 4) / frame_area
    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return -1.0
    if max(axis_w, axis_h) / min(axis_w, axis_h) > MAX_ASPECT:
        return -1.0

    rad = math.radians(angle)
    horiz = abs(axis_w * math.cos(rad)) + abs(axis_h * math.sin(rad))
    vert = abs(axis_w * math.sin(rad)) + abs(axis_h * math.cos(rad))
    wider = min(horiz / (vert + 1e-6), 3.0) / 3.0
    centrality = 1 - math.hypot(cx - width / 2, cy - height / 2) / math.hypot(width / 2, height / 2)
    return 0.5 * min(area_ratio, 0.6) / 0.6 + 0.3 * centrality + 0.2 * wider


def detect_calibration(frame_bgr: np.ndarray) -> tuple[Calibration, float] | None:
    """Best dohyo ellipse on one frame, with its score.

    Returns None when nothing plausible is found, so callers fall back to the last
    valid calibration. The score lets the segmenter tell match frames from intro
    and sponsor cards.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    _, white = cv2.threshold(gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, MORPH_KERNEL, iterations=3)
    contours, _ = cv2.findContours(white, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    best, best_score = None, -1.0
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA_RATIO * height * width:
            continue
        hull = cv2.convexHull(contour)
        if len(hull) < 5:
            continue
        ellipse = cv2.fitEllipse(hull)
        score = _score_ellipse(ellipse, height, width)
        if score > best_score:
            best, best_score = ellipse, score

    if best is None:
        return None
    (cx, cy), (axis_w, axis_h), angle = best
    cm_per_px = DOHYO_DIAMETER_CM / max(axis_w, axis_h)
    return (
        Calibration(
            center_x_px=cx,
            center_y_px=cy,
            axis_w_px=axis_w,
            axis_h_px=axis_h,
            angle_deg=angle,
            cm_per_px=cm_per_px,
        ),
        best_score,
    )


def has_match(frame_bgr: np.ndarray) -> bool:
    """Whether a frame shows the dohyo (match footage), not an intro/sponsor card."""
    found = detect_calibration(frame_bgr)
    if found is None:
        return False
    cal, score = found
    area_ratio = (math.pi * cal.axis_w_px * cal.axis_h_px / 4) / (frame_bgr.shape[0] * frame_bgr.shape[1])
    return score >= MATCH_MIN_SCORE and area_ratio >= MATCH_MIN_AREA_RATIO


def point_in_ellipse(x_px: float, y_px: float, cal: Calibration, margin: float = 1.0) -> bool:
    """Whether a pixel falls inside the dohyo ellipse.

    margin scales the axes: < 1 shrinks the ring (stricter ring-out), > 1 grows it.
    Serves both the detection geometric filter and the ring-out rule.
    """
    rad = math.radians(cal.angle_deg)
    dx, dy = x_px - cal.center_x_px, y_px - cal.center_y_px
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    u = dx * cos_a + dy * sin_a
    v = -dx * sin_a + dy * cos_a
    a = cal.axis_w_px / 2 * margin
    b = cal.axis_h_px / 2 * margin
    return (u / a) ** 2 + (v / b) ** 2 <= 1.0


def stable_calibration(frames_bgr: list[np.ndarray]) -> Calibration:
    """Median calibration over frames, robust to per-frame detection noise.

    Aggregating geometry across frames removes single-frame ellipse jitter. With a
    moving handheld camera this is a coarse approximation; per-frame calibration is
    available via detect_calibration for finer work.
    """
    found = [c for c in (detect_calibration(f) for f in frames_bgr) if c is not None]
    if not found:
        raise RuntimeError("dohyo not detected in any frame")
    cals = [c for c, _ in found]
    return Calibration(
        center_x_px=float(np.median([c.center_x_px for c in cals])),
        center_y_px=float(np.median([c.center_y_px for c in cals])),
        axis_w_px=float(np.median([c.axis_w_px for c in cals])),
        axis_h_px=float(np.median([c.axis_h_px for c in cals])),
        angle_deg=float(np.median([c.angle_deg for c in cals])),
        cm_per_px=float(np.median([c.cm_per_px for c in cals])),
    )

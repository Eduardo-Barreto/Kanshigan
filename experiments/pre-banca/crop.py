"""Dohyo region-of-interest cropping for detection and annotation.

Robots occupy a small fraction of the full frame, so feeding the whole frame to a
640px detector shrinks them below what survives motion blur, and the background
(crowd, hands, mat) invites false positives. Cropping to the dohyo before detection
zooms the robots in roughly threefold and removes the background. The crop is a
fixed per-clip rectangle (the camera moves little within one short round), computed
from the median dohyo ellipse with margin, so boxes map back to native coordinates
by a single offset.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from dohyo import stable_calibration
from schema import Calibration

DEFAULT_MARGIN = 0.20


def _ellipse_half_extents(cal: Calibration) -> tuple[float, float]:
    """Axis-aligned half-width and half-height of the rotated dohyo ellipse."""
    a = cal.axis_w_px / 2
    b = cal.axis_h_px / 2
    ang = math.radians(cal.angle_deg)
    hx = math.hypot(a * math.cos(ang), b * math.sin(ang))
    hy = math.hypot(a * math.sin(ang), b * math.cos(ang))
    return hx, hy


def clip_roi(frames_sample: list[np.ndarray], margin: float = DEFAULT_MARGIN) -> tuple[int, int, int, int]:
    """Fixed crop rectangle (x0, y0, w, h) covering the dohyo plus margin."""
    cal = stable_calibration(frames_sample)
    h, w = frames_sample[0].shape[:2]
    hx, hy = _ellipse_half_extents(cal)
    hx *= 1 + margin
    hy *= 1 + margin
    x0 = max(0, int(cal.center_x_px - hx))
    y0 = max(0, int(cal.center_y_px - hy))
    x1 = min(w, int(cal.center_x_px + hx))
    y1 = min(h, int(cal.center_y_px + hy))
    return x0, y0, x1 - x0, y1 - y0


def crop_frame(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, w, h = roi
    return frame[y0 : y0 + h, x0 : x0 + w]


def box_crop_to_native(box_xywh: tuple[float, float, float, float], roi: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """A bbox measured in crop pixels to native pixels (offset only)."""
    x, y, w, h = box_xywh
    x0, y0, _, _ = roi
    return (x + x0, y + y0, w, h)


def box_native_to_crop_yolo(box_xywh_px, roi: tuple[int, int, int, int]) -> tuple[float, float, float, float]:
    """A native-pixel bbox to YOLO-normalized coords inside the crop."""
    x, y, w, h = box_xywh_px
    x0, y0, cw, ch = roi
    cx = (x + w / 2 - x0) / cw
    cy = (y + h / 2 - y0) / ch
    return (cx, cy, w / cw, h / ch)

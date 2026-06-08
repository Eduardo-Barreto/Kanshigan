"""Shared data structures for the Kanshigan inference pipeline.

These types flow across dohyo, tracking, metrics, events and evaluation, so they
live in one place. Everything downstream of detection speaks centimeters in the
dohyo reference frame; pixel coordinates stay confined to detection and overlay.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Calibration:
    """Maps pixels to centimeters using the detected dohyo ellipse.

    Stores the ellipse exactly as cv2.fitEllipse returns it: center, the two full
    axis lengths (axis_w along the rotated first axis, axis_h along the second) and
    the rotation. The official 3 kg dohyo is 154 cm across, and under an oblique
    camera the unforeshortened (major) axis carries the true diameter, so the scale
    comes from it. The single isotropic scale is an approximation under
    perspective; homography rectification is left as future work.
    """

    center_x_px: float
    center_y_px: float
    axis_w_px: float
    axis_h_px: float
    angle_deg: float
    cm_per_px: float

    @property
    def radius_cm(self) -> float:
        return DOHYO_DIAMETER_CM / 2

    def to_cm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Pixel point to centimeters, origin at the dohyo center, y up."""
        return (
            (x_px - self.center_x_px) * self.cm_per_px,
            (self.center_y_px - y_px) * self.cm_per_px,
        )


@dataclass
class TrackPoint:
    frame: int
    t_s: float
    bbox_xywh_px: tuple[float, float, float, float]

    @property
    def center_px(self) -> tuple[float, float]:
        x, y, w, h = self.bbox_xywh_px
        return (x + w / 2, y + h / 2)


@dataclass
class Track:
    robot_id: str
    points: list[TrackPoint] = field(default_factory=list)

    @property
    def frames(self) -> list[int]:
        return [p.frame for p in self.points]


@dataclass(frozen=True)
class Event:
    kind: str
    t_ms: float
    frame: int
    robot_id: str | None = None
    note: str | None = None


DOHYO_DIAMETER_CM = 154.0
ROBOT_A = "A"
ROBOT_B = "B"

"""Kinematic metric extraction from tracks. Stage [6] of the pipeline.

Tracks arrive as pixel trajectories; detections are first projected to the
dohyo-centered centimeter frame, one calibration per frame. Because the dohyo is
fixed in the world, expressing a robot relative to the dohyo center cancels the
handheld camera's motion: the resulting coordinates are world-stable even when the
camera pans. Velocity and acceleration then come from a Savitzky-Golay filter,
which differentiates and denoises in one pass over a short window, keeping
sub-second events resolvable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.signal import savgol_filter

from schema import Calibration, Track

DEFAULT_SMOOTH_WINDOW = 5
POLY_ORDER = 2

Calibrate = Callable[[int], Calibration]


@dataclass
class Kinematics:
    robot_id: str
    frames: np.ndarray
    t_s: np.ndarray
    x_cm: np.ndarray
    y_cm: np.ndarray
    vx_cms: np.ndarray
    vy_cms: np.ndarray
    speed_cms: np.ndarray
    ax_cms2: np.ndarray
    ay_cms2: np.ndarray
    accel_cms2: np.ndarray

    @property
    def path_length_cm(self) -> float:
        return float(np.sum(np.hypot(np.diff(self.x_cm), np.diff(self.y_cm))))

    @property
    def radius_cm(self) -> np.ndarray:
        return np.hypot(self.x_cm, self.y_cm)


def project_track_to_cm(track: Track, calibrate: Calibrate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Each track point to dohyo-centered cm using that frame's calibration."""
    frames, xs, ys = [], [], []
    for point in track.points:
        cal = calibrate(point.frame)
        cx, cy = point.center_px
        x_cm, y_cm = cal.to_cm(cx, cy)
        frames.append(point.frame)
        xs.append(x_cm)
        ys.append(y_cm)
    return np.array(frames), np.array(xs), np.array(ys)


def _resample_uniform(frames: np.ndarray, x_cm: np.ndarray, y_cm: np.ndarray, fps: float):
    """Fill detection gaps by linear interpolation onto a uniform frame grid.

    Uniform spacing is what makes the Savitzky-Golay derivative well defined; short
    occlusion gaps become interpolated points rather than uneven sampling.
    """
    grid = np.arange(frames.min(), frames.max() + 1)
    return grid, grid / fps, np.interp(grid, frames, x_cm), np.interp(grid, frames, y_cm)


def _fit_window(n_points: int, requested: int) -> int:
    window = min(requested, n_points)
    if window % 2 == 0:
        window -= 1
    return max(window, POLY_ORDER + 1)


def compute_kinematics(
    robot_id: str,
    frames: np.ndarray,
    x_cm: np.ndarray,
    y_cm: np.ndarray,
    fps: float,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
) -> Kinematics:
    frames, t_s, x_cm, y_cm = _resample_uniform(frames, x_cm, y_cm, fps)
    dt = 1.0 / fps

    if len(frames) <= POLY_ORDER + 1:
        vx, vy = np.gradient(x_cm, dt), np.gradient(y_cm, dt)
        ax, ay = np.gradient(vx, dt), np.gradient(vy, dt)
    else:
        window = _fit_window(len(frames), smooth_window)
        x_cm = savgol_filter(x_cm, window, POLY_ORDER)
        y_cm = savgol_filter(y_cm, window, POLY_ORDER)
        vx = savgol_filter(x_cm, window, POLY_ORDER, deriv=1, delta=dt)
        vy = savgol_filter(y_cm, window, POLY_ORDER, deriv=1, delta=dt)
        ax = savgol_filter(x_cm, window, POLY_ORDER, deriv=2, delta=dt)
        ay = savgol_filter(y_cm, window, POLY_ORDER, deriv=2, delta=dt)

    return Kinematics(
        robot_id=robot_id,
        frames=frames,
        t_s=t_s,
        x_cm=x_cm,
        y_cm=y_cm,
        vx_cms=vx,
        vy_cms=vy,
        speed_cms=np.hypot(vx, vy),
        ax_cms2=ax,
        ay_cms2=ay,
        accel_cms2=np.hypot(ax, ay),
    )


def kinematics_from_track(track: Track, calibrate: Calibrate, fps: float, smooth_window: int = DEFAULT_SMOOTH_WINDOW) -> Kinematics:
    frames, x_cm, y_cm = project_track_to_cm(track, calibrate)
    return compute_kinematics(track.robot_id, frames, x_cm, y_cm, fps, smooth_window)


def pairwise_distance_cm(a: Kinematics, b: Kinematics) -> tuple[np.ndarray, np.ndarray]:
    """Center-to-center distance on the frames both robots are present."""
    shared = np.intersect1d(a.frames, b.frames)
    ia = np.searchsorted(a.frames, shared)
    ib = np.searchsorted(b.frames, shared)
    dist = np.hypot(a.x_cm[ia] - b.x_cm[ib], a.y_cm[ia] - b.y_cm[ib])
    return shared, dist


def spatial_heatmap(k: Kinematics, radius_cm: float, bins: int = 24) -> np.ndarray:
    edges = np.linspace(-radius_cm, radius_cm, bins + 1)
    hist, _, _ = np.histogram2d(k.x_cm, k.y_cm, bins=[edges, edges])
    return hist

"""Tracking stage [5]: associate per-frame detections into robot trajectories.

OC-SORT (motion-only, observation-centric) is the primary tracker: it was built
for the non-linear motion that collisions and spins produce, and runs far faster
than real time given ready detections. ByteTrack is supported for the comparative
experiment. Both come from boxmot behind one adapter, so the pipeline swaps
trackers by name.

The integer ids a tracker emits are arbitrary. The study fixes a convention:
robot A is the one further left at the first frame both robots are tracked. The
relabeling that enforces it is pure and unit-tested, separate from the tracker.
"""

from __future__ import annotations

import numpy as np

from schema import ROBOT_A, ROBOT_B, Track, TrackPoint

SUPPORTED_TRACKERS = ("ocsort", "bytetrack")


def build_tracker(name: str = "ocsort", min_conf: float = 0.25):
    if name not in SUPPORTED_TRACKERS:
        raise ValueError(f"tracker must be one of {SUPPORTED_TRACKERS}, got {name!r}")
    if name == "ocsort":
        from boxmot.trackers.ocsort.ocsort import OcSort

        return OcSort(min_conf=min_conf, delta_t=3, inertia=0.2)

    from boxmot.trackers.bytetrack.bytetrack import ByteTrack

    return ByteTrack(min_conf=min_conf, match_thresh=0.8)


def detections_to_array(bboxes_xywh, confs, cls: int = 0) -> np.ndarray:
    """Pack detections as boxmot expects: rows of [x1, y1, x2, y2, conf, cls]."""
    if not bboxes_xywh:
        return np.empty((0, 6), dtype=np.float32)
    rows = []
    for (x, y, w, h), conf in zip(bboxes_xywh, confs):
        rows.append([x, y, x + w, y + h, conf, cls])
    return np.asarray(rows, dtype=np.float32)


def assign_ab(raw_tracks: dict[int, Track]) -> dict[str, Track]:
    """Reduce tracker ids to the two robots and relabel them A and B.

    Keeps the two longest tracks (robustness to id fragmentation), then names the
    leftmost-at-first-shared-frame robot A. With fewer than two tracks, returns
    whatever exists under A (and B if present).
    """
    if not raw_tracks:
        return {}
    ordered = sorted(raw_tracks.values(), key=lambda t: len(t.points), reverse=True)
    top = ordered[:2]
    if len(top) == 1:
        return {ROBOT_A: Track(robot_id=ROBOT_A, points=top[0].points)}

    first, second = top
    left, right = _order_by_first_x(first, second)
    return {
        ROBOT_A: Track(robot_id=ROBOT_A, points=left.points),
        ROBOT_B: Track(robot_id=ROBOT_B, points=right.points),
    }


def _order_by_first_x(t1: Track, t2: Track) -> tuple[Track, Track]:
    shared = sorted(set(t1.frames) & set(t2.frames))
    ref_frame = shared[0] if shared else min(t1.frames[0], t2.frames[0])
    x1 = _center_x_at(t1, ref_frame)
    x2 = _center_x_at(t2, ref_frame)
    return (t1, t2) if x1 <= x2 else (t2, t1)


def _center_x_at(track: Track, frame: int) -> float:
    nearest = min(track.points, key=lambda p: abs(p.frame - frame))
    return nearest.center_px[0]


def run_tracker(tracker, detections_per_frame, frames_bgr, fps: float) -> dict[str, Track]:
    """Feed detections frame by frame, collect trajectories, relabel to A/B.

    detections_per_frame[i] is (bboxes_xywh, confs) for frame i; frames_bgr[i] is
    the image the tracker needs for its internal bookkeeping.
    """
    raw: dict[int, Track] = {}
    for frame_idx, ((bboxes, confs), image) in enumerate(zip(detections_per_frame, frames_bgr)):
        dets = detections_to_array(bboxes, confs)
        tracks = tracker.update(dets, image)
        for row in tracks:
            x1, y1, x2, y2, tid = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            point = TrackPoint(frame=frame_idx, t_s=frame_idx / fps, bbox_xywh_px=(x1, y1, x2 - x1, y2 - y1))
            raw.setdefault(tid, Track(robot_id=str(tid))).points.append(point)
    return assign_ab(raw)

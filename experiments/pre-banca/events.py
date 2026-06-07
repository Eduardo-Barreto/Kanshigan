"""Deterministic event detection. Stage [7] of the pipeline.

Events are read off the cm-space kinematics with explicit, inspectable rules
rather than a learned classifier: the gold set is too small to train an event
model, and a method whose decisions can be traced defends better. Working in the
dohyo-centered frame makes ring-out a radius test, invariant to camera motion.
Thresholds live in a config so they are versioned and reportable both calibrated
and at literature defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metrics import Kinematics, pairwise_distance_cm
from schema import DOHYO_DIAMETER_CM, Event


@dataclass(frozen=True)
class EventConfig:
    """Thresholds for the kinematic event rules.

    Defaults are literature-style starting points; calibrated values from the gold
    set are stored in configs/events.yaml and loaded over these.
    """

    v_min_start_cms: float = 15.0
    dv_contact_cms: float = 40.0
    d_contact_cm: float = 30.0
    ringout_margin: float = 1.0


def _t_ms(frame: int, k: Kinematics) -> float:
    idx = int(np.searchsorted(k.frames, frame))
    idx = min(idx, len(k.t_s) - 1)
    return float(k.t_s[idx] * 1000)


def _round_start(a: Kinematics, b: Kinematics, cfg: EventConfig) -> Event | None:
    shared = np.intersect1d(a.frames, b.frames)
    if shared.size == 0:
        return None
    ia = np.searchsorted(a.frames, shared)
    ib = np.searchsorted(b.frames, shared)
    both_moving = np.minimum(a.speed_cms[ia], b.speed_cms[ib]) > cfg.v_min_start_cms
    hits = np.flatnonzero(both_moving)
    if hits.size == 0:
        return None
    frame = int(shared[hits[0]])
    return Event(kind="round_start", t_ms=_t_ms(frame, a), frame=frame)


def _first_contact(a: Kinematics, b: Kinematics, cfg: EventConfig) -> Event | None:
    shared, dist = pairwise_distance_cm(a, b)
    if shared.size == 0:
        return None
    ia = np.searchsorted(a.frames, shared)
    ib = np.searchsorted(b.frames, shared)
    dv_a = np.abs(np.diff(a.speed_cms[ia], prepend=a.speed_cms[ia][0]))
    dv_b = np.abs(np.diff(b.speed_cms[ib], prepend=b.speed_cms[ib][0]))
    contact = (dv_a > cfg.dv_contact_cms) & (dv_b > cfg.dv_contact_cms) & (dist < cfg.d_contact_cm)
    hits = np.flatnonzero(contact)
    if hits.size == 0:
        return None
    frame = int(shared[hits[0]])
    return Event(kind="first_contact", t_ms=_t_ms(frame, a), frame=frame)


def _ring_out(k: Kinematics, radius_cm: float, cfg: EventConfig) -> Event | None:
    outside = k.radius_cm > radius_cm * cfg.ringout_margin
    hits = np.flatnonzero(outside)
    if hits.size == 0:
        return None
    frame = int(k.frames[hits[0]])
    return Event(kind="ring_out", t_ms=_t_ms(frame, k), frame=frame, robot_id=k.robot_id)


def detect_events(a: Kinematics, b: Kinematics, cfg: EventConfig = EventConfig()) -> list[Event]:
    radius_cm = DOHYO_DIAMETER_CM / 2
    events: list[Event] = []

    start = _round_start(a, b, cfg)
    if start:
        events.append(start)

    contact = _first_contact(a, b, cfg)
    if contact:
        events.append(contact)

    rings = [e for e in (_ring_out(a, radius_cm, cfg), _ring_out(b, radius_cm, cfg)) if e]
    first_ring = min(rings, key=lambda e: e.frame) if rings else None

    if first_ring:
        events.append(first_ring)
        winner = b.robot_id if first_ring.robot_id == a.robot_id else a.robot_id
        end_frame, note = first_ring.frame, "ring_out"
    else:
        winner, end_frame, note = None, int(max(a.frames[-1], b.frames[-1])), "timeout_manual_review"

    events.append(Event(kind="round_end", t_ms=_t_ms(end_frame, a), frame=end_frame, note=note))
    events.append(Event(kind="winner", t_ms=_t_ms(end_frame, a), frame=end_frame, robot_id=winner, note=note))
    return sorted(events, key=lambda e: e.frame)

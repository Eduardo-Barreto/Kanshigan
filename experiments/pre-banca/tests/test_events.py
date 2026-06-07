import numpy as np

from events import EventConfig, detect_events
from metrics import compute_kinematics
from schema import DOHYO_DIAMETER_CM

FPS = 30.0
RADIUS = DOHYO_DIAMETER_CM / 2


def _still(robot_id, x, y, n=40):
    frames = np.arange(n)
    return compute_kinematics(robot_id, frames, np.full(n, float(x)), np.full(n, float(y)), FPS)


class TestRoundStart:
    def test_detects_first_frame_a_robot_charges(self):
        frames = np.arange(40)
        # both still until frame 10, then one robot charges
        x_a = np.where(frames < 10, -20.0, -20.0 + 3.0 * (frames - 10))
        a = compute_kinematics("A", frames, x_a, np.zeros(40), FPS)
        b = compute_kinematics("B", frames, np.full(40, 20.0), np.zeros(40), FPS)
        events = detect_events(a, b)
        start = next(e for e in events if e.kind == "round_start")
        assert 9 <= start.frame <= 14


class TestRingOut:
    def test_flags_robot_leaving_the_dohyo_and_names_the_other_winner(self):
        frames = np.arange(40)
        # B drifts past the rim, A stays inside
        b_x = np.linspace(0, RADIUS + 30, 40)
        a = _still("A", 0, 0)
        b = compute_kinematics("B", frames, b_x, np.zeros(40), FPS)
        events = detect_events(a, b)
        ring = next(e for e in events if e.kind == "ring_out")
        winner = next(e for e in events if e.kind == "winner")
        assert ring.robot_id == "B"
        assert winner.robot_id == "A"

    def test_reports_timeout_when_nobody_leaves(self):
        a = _still("A", -10, 0)
        b = _still("B", 10, 0)
        events = detect_events(a, b)
        winner = next(e for e in events if e.kind == "winner")
        assert winner.robot_id is None
        assert winner.note == "timeout_manual_review"


class TestEventConfig:
    def test_respects_a_shrunk_ringout_margin(self):
        frames = np.arange(40)
        b_x = np.linspace(0, RADIUS - 5, 40)  # stays just inside the true rim
        a = _still("A", 0, 0)
        b = compute_kinematics("B", frames, b_x, np.zeros(40), FPS)
        loose = detect_events(a, b, EventConfig(ringout_margin=1.0))
        strict = detect_events(a, b, EventConfig(ringout_margin=0.9))
        assert not any(e.kind == "ring_out" for e in loose)
        assert any(e.kind == "ring_out" for e in strict)

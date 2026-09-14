import numpy as np

from audit import TELEPORT_STEP_CM, audit_report


def _traj(robot_id, frames, x=None, y=None, speed=None):
    frames = list(frames)
    n = len(frames)
    return {
        "robot_id": robot_id,
        "frames": frames,
        "x_cm": list(x if x is not None else np.zeros(n)),
        "y_cm": list(y if y is not None else np.zeros(n)),
        "speed_cms": list(speed if speed is not None else np.zeros(n)),
    }


def _report(trajectories, n_frames=100, events=None, fps=60.0):
    return {
        "fps": fps,
        "n_frames": n_frames,
        "trajectories": trajectories,
        "events": events if events is not None else [],
    }


def _round(start, end):
    return [
        {"kind": "round_start", "frame": start, "t_ms": 0.0},
        {"kind": "round_end", "frame": end, "t_ms": 0.0, "note": "timeout_manual_review"},
    ]


class TestRoundWindow:
    def test_restricts_to_the_round_so_pre_start_frames_are_not_failures(self):
        # both robots tracked only from frame 50 on; the round also starts at 50
        trajs = [_traj("A", range(50, 101)), _traj("B", range(50, 101))]
        report = _report(trajs, n_frames=101, events=_round(50, 100))
        assert audit_report(report, "clip").cardinality_violation_rate == 0.0

    def test_falls_back_to_the_whole_clip_when_no_round_was_detected(self):
        trajs = [_traj("A", range(0, 101)), _traj("B", range(0, 101))]
        audit = audit_report(_report(trajs, n_frames=101), "clip")
        assert (audit.round_start, audit.round_end) == (0, 100)
        assert audit.round_frames == 101


class TestCardinality:
    def test_counts_frames_missing_a_robot_as_violations(self):
        # B drops out for 10 of the 101 round frames
        trajs = [_traj("A", range(101)), _traj("B", list(range(0, 40)) + list(range(50, 101)))]
        audit = audit_report(_report(trajs, n_frames=101, events=_round(0, 100)), "clip")
        assert audit.one_present_rate == round(10 / 101, 4)
        assert audit.cardinality_violation_rate == round(10 / 101, 4)
        assert audit.longest_gap_frames == 10

    def test_a_clip_where_the_detector_found_nothing_is_a_total_violation(self):
        audit = audit_report(_report([], n_frames=101, events=_round(0, 100)), "clip")
        assert audit.cardinality_violation_rate == 1.0
        assert audit.none_present_rate == 1.0


class TestImplausibleMotion:
    def test_flags_a_jump_no_robot_could_make(self):
        x = [0.0, 0.0, TELEPORT_STEP_CM + 1.0]
        trajs = [_traj("A", [0, 1, 2], x=x), _traj("B", [0, 1, 2])]
        audit = audit_report(_report(trajs, n_frames=3, events=_round(0, 2)), "clip")
        assert audit.n_teleports == 1

    def test_does_not_count_a_gap_as_a_jump(self):
        # same displacement, but across a 30-frame dropout: the robot had time to move
        x = [0.0, TELEPORT_STEP_CM + 1.0]
        trajs = [_traj("A", [0, 30], x=x), _traj("B", [0, 30])]
        audit = audit_report(_report(trajs, n_frames=31, events=_round(0, 30)), "clip")
        assert audit.n_teleports == 0


class TestRingOut:
    def test_reports_whether_the_round_resolved_by_ring_out(self):
        trajs = [_traj("A", range(10)), _traj("B", range(10))]
        events = _round(0, 9) + [{"kind": "ring_out", "frame": 9, "t_ms": 0.0, "robot_id": "B"}]
        assert audit_report(_report(trajs, n_frames=10, events=events), "clip").ring_out_detected
        assert not audit_report(_report(trajs, n_frames=10, events=_round(0, 9)), "clip").ring_out_detected

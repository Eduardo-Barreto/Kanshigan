import numpy as np

from metrics import compute_kinematics, pairwise_distance_cm, project_track_to_cm
from schema import Calibration, Track, TrackPoint


def _const(cal: Calibration):
    return lambda _frame: cal


class TestKinematics:
    def test_computes_constant_velocity(self):
        fps = 30.0
        frames = np.arange(0, 30)
        x_cm = 100.0 * frames / fps
        y_cm = np.zeros_like(x_cm)
        k = compute_kinematics("A", frames, x_cm, y_cm, fps)
        assert np.allclose(np.median(k.vx_cms), 100.0, atol=2.0)
        assert np.allclose(np.median(k.speed_cms), 100.0, atol=2.0)

    def test_reports_near_zero_acceleration_for_constant_velocity(self):
        fps = 30.0
        frames = np.arange(0, 30)
        x_cm = 50.0 * frames / fps
        y_cm = np.zeros_like(x_cm)
        k = compute_kinematics("A", frames, x_cm, y_cm, fps)
        assert np.median(np.abs(k.accel_cms2)) < 5.0

    def test_measures_path_length(self):
        fps = 10.0
        frames = np.arange(0, 11)
        x_cm = np.linspace(0, 100, 11)
        y_cm = np.zeros_like(x_cm)
        k = compute_kinematics("A", frames, x_cm, y_cm, fps)
        assert abs(k.path_length_cm - 100.0) < 1.0

    def test_fills_detection_gaps_by_interpolation(self):
        fps = 10.0
        frames = np.array([0, 1, 2, 5, 6])
        x_cm = np.array([0.0, 10.0, 20.0, 50.0, 60.0])
        y_cm = np.zeros_like(x_cm)
        k = compute_kinematics("A", frames, x_cm, y_cm, fps)
        assert len(k.frames) == 7
        assert np.isclose(k.x_cm[3], 30.0, atol=1.0)


class TestProjection:
    def test_projects_pixels_to_dohyo_centered_cm(self):
        cal = Calibration(center_x_px=100, center_y_px=100, axis_w_px=200, axis_h_px=200, angle_deg=0, cm_per_px=0.5)
        track = Track(robot_id="A", points=[TrackPoint(frame=0, t_s=0.0, bbox_xywh_px=(140, 60, 0, 0))])
        frames, x_cm, y_cm = project_track_to_cm(track, _const(cal))
        assert np.isclose(x_cm[0], 20.0)
        assert np.isclose(y_cm[0], 20.0)


class TestPairwiseDistance:
    def test_measures_distance_on_shared_frames(self):
        fps = 10.0
        frames = np.arange(0, 10)
        a = compute_kinematics("A", frames, np.full(10, -10.0), np.zeros(10), fps)
        b = compute_kinematics("B", frames, np.full(10, 10.0), np.zeros(10), fps)
        shared, dist = pairwise_distance_cm(a, b)
        assert np.allclose(dist, 20.0, atol=1.0)

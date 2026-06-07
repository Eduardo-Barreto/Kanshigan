import numpy as np

from schema import Track, TrackPoint
from tracking import assign_ab, detections_to_array


def _track(tid, x_centers):
    points = [
        TrackPoint(frame=i, t_s=i / 30, bbox_xywh_px=(x - 5, 50 - 5, 10, 10))
        for i, x in enumerate(x_centers)
    ]
    return Track(robot_id=str(tid), points=points)


class TestAssignAB:
    def test_names_the_leftmost_robot_a(self):
        left = _track(7, [10, 12, 14])
        right = _track(3, [90, 88, 86])
        result = assign_ab({7: left, 3: right})
        assert result["A"].points[0].center_px[0] < result["B"].points[0].center_px[0]

    def test_keeps_only_the_two_longest_tracks(self):
        long_a = _track(1, list(range(0, 20)))
        long_b = _track(2, list(range(100, 80, -1)))
        fragment = _track(3, [50, 51])
        result = assign_ab({1: long_a, 2: long_b, 3: fragment})
        assert set(result.keys()) == {"A", "B"}

    def test_returns_single_robot_under_a_when_only_one_track(self):
        result = assign_ab({5: _track(5, [40, 41, 42])})
        assert set(result.keys()) == {"A"}


class TestDetectionsArray:
    def test_packs_xywh_confidence_as_xyxy_rows(self):
        arr = detections_to_array([(10, 20, 30, 40)], [0.9])
        assert arr.shape == (1, 6)
        assert np.allclose(arr[0, :4], [10, 20, 40, 60])
        assert arr[0, 4] == np.float32(0.9)

    def test_returns_empty_for_no_detections(self):
        arr = detections_to_array([], [])
        assert arr.shape == (0, 6)

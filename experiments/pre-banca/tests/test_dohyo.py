import math

from dohyo import point_in_ellipse
from schema import Calibration


def _circle(radius=100.0, cm_per_px=1.0):
    return Calibration(center_x_px=200, center_y_px=200, axis_w_px=2 * radius, axis_h_px=2 * radius, angle_deg=0, cm_per_px=cm_per_px)


class TestPointInEllipse:
    def test_accepts_the_center(self):
        assert point_in_ellipse(200, 200, _circle())

    def test_rejects_a_point_beyond_the_rim(self):
        assert not point_in_ellipse(200 + 120, 200, _circle(radius=100))

    def test_accepts_a_point_inside_the_rim(self):
        assert point_in_ellipse(200 + 80, 200, _circle(radius=100))

    def test_margin_shrinks_the_accepted_region(self):
        cal = _circle(radius=100)
        assert point_in_ellipse(200 + 95, 200, cal, margin=1.0)
        assert not point_in_ellipse(200 + 95, 200, cal, margin=0.9)

    def test_respects_rotation(self):
        # 3:1 ellipse rotated 90 deg: long axis becomes vertical
        cal = Calibration(center_x_px=0, center_y_px=0, axis_w_px=300, axis_h_px=100, angle_deg=90, cm_per_px=1.0)
        assert point_in_ellipse(0, 140, cal)       # along rotated long axis
        assert not point_in_ellipse(140, 0, cal)   # along rotated short axis


class TestCalibrationToCm:
    def test_maps_center_to_origin(self):
        cal = _circle(cm_per_px=0.5)
        assert cal.to_cm(200, 200) == (0.0, 0.0)

    def test_flips_y_axis_upward(self):
        cal = _circle(cm_per_px=1.0)
        x_cm, y_cm = cal.to_cm(210, 190)
        assert x_cm == 10.0
        assert y_cm == 10.0

    def test_derives_radius_from_known_diameter(self):
        assert math.isclose(_circle().radius_cm, 77.0)

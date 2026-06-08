import math

from crop import box_native_to_crop_yolo

# roi = (x0, y0, w, h): a 100x100 crop starting at native (50, 30)
ROI = (50, 30, 100, 100)


class TestBoxNativeToCropYolo:
    def test_maps_an_interior_box_to_normalized_crop_coords(self):
        # native box centered at (100, 80), 20x20 -> crop center (50, 50), 0.2x0.2
        cx, cy, w, h = box_native_to_crop_yolo((90, 70, 20, 20), ROI)
        assert math.isclose(cx, 0.5)
        assert math.isclose(cy, 0.5)
        assert math.isclose(w, 0.2)
        assert math.isclose(h, 0.2)

    def test_clamps_a_box_spilling_past_the_left_edge(self):
        # native box from x=40 (10px left of the crop) to x=60: only x>=50 survives
        cx, cy, w, h = box_native_to_crop_yolo((40, 70, 20, 20), ROI)
        assert 0.0 <= cx <= 1.0
        assert math.isclose(w, 0.1)  # half the box was clipped away
        assert math.isclose(cx, 0.05)  # surviving [0,10]px center at 5px -> 0.05

    def test_never_emits_negative_normalized_coords(self):
        # regression for the diario-16 bug: out-of-crop boxes produced negative coords
        for box in [(40, 20, 20, 20), (140, 120, 30, 30), (45, 25, 10, 10)]:
            cx, cy, w, h = box_native_to_crop_yolo(box, ROI)
            assert cx >= 0.0 and cy >= 0.0
            assert w >= 0.0 and h >= 0.0

    def test_caps_a_box_running_past_the_right_edge(self):
        # box extends to native x=160, crop ends at x=150 -> right clamped to 150
        cx, cy, w, h = box_native_to_crop_yolo((130, 70, 40, 20), ROI)
        assert cx + w / 2 <= 1.0 + 1e-9

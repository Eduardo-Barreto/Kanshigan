from schema import Track, TrackPoint


class TestTrackPoint:
    def test_center_is_the_box_midpoint(self):
        point = TrackPoint(frame=0, t_s=0.0, bbox_xywh_px=(10, 20, 30, 40))
        assert point.center_px == (25.0, 40.0)


class TestTrack:
    def test_frames_lists_point_frames_in_order(self):
        track = Track(
            robot_id="A",
            points=[
                TrackPoint(frame=3, t_s=0.1, bbox_xywh_px=(0, 0, 1, 1)),
                TrackPoint(frame=7, t_s=0.2, bbox_xywh_px=(0, 0, 1, 1)),
            ],
        )
        assert track.frames == [3, 7]

    def test_a_new_track_starts_empty(self):
        assert Track(robot_id="B").frames == []

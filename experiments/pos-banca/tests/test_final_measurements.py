import numpy as np
import pytest

from appearance_separability import auc, overlap_coefficient, pair_similarities
from final_tracking import clip_gold_to_frame, iou_matrix_xywh, trackeval_data
from round_durations import boxes_intersect, consensus, first_contact, first_exit, seconds


class TestGoldClipping:
    def test_keeps_box_inside_frame_unchanged(self):
        gold = {6: {0: np.array([10.0, 10.0, 20.0, 20.0])}}
        assert np.allclose(clip_gold_to_frame(gold, 100, 100)[6][0], [10, 10, 20, 20])

    def test_clips_box_crossing_top_edge(self):
        gold = {6: {0: np.array([10.0, -5.0, 20.0, 20.0])}}
        assert np.allclose(clip_gold_to_frame(gold, 100, 100)[6][0], [10, 0, 20, 15])

    def test_drops_box_fully_outside_frame(self):
        gold = {6: {0: np.array([10.0, -50.0, 20.0, 20.0]), 1: np.array([1.0, 1.0, 5.0, 5.0])}}
        assert list(clip_gold_to_frame(gold, 100, 100)[6]) == [1]


class TestIouMatrix:
    def test_identical_boxes_score_one(self):
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert iou_matrix_xywh([box], [box])[0, 0] == pytest.approx(1.0)

    def test_half_overlap(self):
        a, b = np.array([0.0, 0.0, 10.0, 10.0]), np.array([5.0, 0.0, 10.0, 10.0])
        assert iou_matrix_xywh([a], [b])[0, 0] == pytest.approx(50 / 150)

    def test_empty_side_gives_empty_matrix(self):
        assert iou_matrix_xywh([], [np.array([0.0, 0.0, 1.0, 1.0])]).shape == (0, 1)


class TestTrackEvalData:
    def test_scores_only_gold_frames_and_remaps_ids(self):
        gold = {6: {0: np.array([0.0, 0.0, 10.0, 10.0])}, 12: {0: np.array([1.0, 0.0, 10.0, 10.0])}}
        pred = {5: {7: np.array([0.0, 0.0, 10.0, 10.0])}, 6: {7: np.array([0.0, 0.0, 10.0, 10.0])}}
        data = trackeval_data(gold, pred)
        assert data["num_timesteps"] == 2
        assert data["num_tracker_ids"] == 1 and data["tracker_ids"][0].tolist() == [0]
        assert data["num_gt_dets"] == 2 and data["num_tracker_dets"] == 1

    def test_perfect_tracking_scores_hota_one(self):
        trackeval = pytest.importorskip("trackeval")  # noqa: F841
        from final_tracking import trackeval_scores

        gold = {f: {0: np.array([f, 0.0, 10.0, 10.0]), 1: np.array([f + 50.0, 0.0, 10.0, 10.0])} for f in (6, 12, 18)}
        scores = trackeval_scores(gold, gold)
        assert scores["HOTA"] == pytest.approx(1.0) and scores["te_MOTA"] == pytest.approx(1.0)


class TestSeparability:
    def test_auc_is_one_when_every_intra_beats_every_inter(self):
        assert auc(np.array([0.9, 0.8]), np.array([0.1, 0.2])) == 1.0

    def test_auc_is_half_for_identical_distributions(self):
        assert auc(np.array([0.5, 0.5]), np.array([0.5, 0.5])) == 0.5

    def test_overlap_is_one_for_identical_samples(self):
        sample = np.linspace(0, 1, 100)
        assert overlap_coefficient(sample, sample) == pytest.approx(1.0)

    def test_pairs_split_by_identity_and_time_gap(self):
        emb = np.eye(3)
        pairs = pair_similarities(emb, np.array([0, 0, 1]), np.array([0, 60, 0]))
        assert pairs["intra"].size == 1 and pairs["intra_far"].size == 1 and pairs["inter"].size == 2


class TestRoundInstants:
    def test_intersecting_boxes(self):
        assert boxes_intersect((0, 0, 10, 10), (5, 5, 10, 10))
        assert not boxes_intersect((0, 0, 10, 10), (10, 0, 10, 10))

    def test_first_contact_needs_two_boxes(self):
        frames = {0: [(0, 0, 10, 10)], 6: [(0, 0, 10, 10), (50, 0, 10, 10)], 12: [(0, 0, 10, 10), (5, 0, 10, 10)]}
        assert first_contact(frames) == 12

    def test_first_exit_uses_box_center(self):
        frames = {0: [(0, 0, 10, 10)], 6: [(95, 0, 10, 10)]}
        assert first_exit(frames, lambda _f, x, _y: x < 90) == 6

    def test_duration_rejects_reversed_order(self):
        assert seconds(60, 30, 60.0) is None
        assert seconds(30, 90, 60.0) == 1.0

    def test_consensus_keeps_only_rounds_where_sources_agree_on_exit(self):
        agree = {"clip": "a", "pipeline": {"move_to_exit_s": 1.0, "contact_to_exit_s": 0.5}, "labels": {"pipeline_minus_label_exit_s": 0.05, "contact_to_exit_s": 0.4}}
        disagree = {"clip": "b", "pipeline": {"move_to_exit_s": 9.0, "contact_to_exit_s": 9.0}, "labels": {"pipeline_minus_label_exit_s": 2.0, "contact_to_exit_s": 9.0}}
        unlabeled = {"clip": "c", "pipeline": {"move_to_exit_s": 9.0, "contact_to_exit_s": 9.0}}
        summary = consensus([agree, disagree, unlabeled])
        assert summary["clips"] == ["a"] and summary["pipeline_move_to_exit_s"]["median"] == 1.0

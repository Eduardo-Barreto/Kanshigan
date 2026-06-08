import numpy as np

from evaluate import _iou, _load_mot, evaluate_annotator, evaluate_events


class TestIou:
    def test_returns_one_for_identical_boxes(self):
        box = np.array([0.0, 0.0, 10.0, 10.0])
        assert _iou(box, box) == 1.0

    def test_returns_zero_for_disjoint_boxes(self):
        a = np.array([0.0, 0.0, 10.0, 10.0])
        b = np.array([20.0, 20.0, 30.0, 30.0])
        assert _iou(a, b) == 0.0

    def test_computes_half_overlap(self):
        a = np.array([0.0, 0.0, 2.0, 1.0])
        b = np.array([1.0, 0.0, 3.0, 1.0])  # overlap area 1, union 3
        assert np.isclose(_iou(a, b), 1 / 3)


class TestLoadMot:
    def test_parses_frame_id_box_rows(self, tmp_path):
        mot = tmp_path / "tracks.txt"
        mot.write_text("0,0,10,20,5,5\n0,1,30,40,5,5\n1,0,11,21,5,5\n")
        by_frame = _load_mot(mot)
        assert set(by_frame.keys()) == {0, 1}
        assert set(by_frame[0].keys()) == {0, 1}
        assert np.allclose(by_frame[0][0], [10, 20, 5, 5])


class TestEvaluateAnnotator:
    def _write_label(self, path, boxes):
        path.write_text("\n".join(f"0 {cx} {cy} {w} {h}" for cx, cy, w, h in boxes))

    def test_perfect_agreement_scores_one(self, tmp_path):
        gold = tmp_path / "gold"
        pred = tmp_path / "pred"
        gold.mkdir()
        pred.mkdir()
        boxes = [(0.3, 0.3, 0.2, 0.2), (0.7, 0.7, 0.2, 0.2)]
        self._write_label(gold / "f1.txt", boxes)
        self._write_label(pred / "f1.txt", boxes)
        out = evaluate_annotator(pred, gold)
        assert out["precision"] == 1.0
        assert out["recall"] == 1.0
        assert out["f1"] == 1.0
        assert np.isclose(out["mean_iou_matched"], 1.0)

    def test_counts_a_missed_gold_box_as_a_false_negative(self, tmp_path):
        gold = tmp_path / "gold"
        pred = tmp_path / "pred"
        gold.mkdir()
        pred.mkdir()
        self._write_label(gold / "f1.txt", [(0.3, 0.3, 0.2, 0.2), (0.7, 0.7, 0.2, 0.2)])
        self._write_label(pred / "f1.txt", [(0.3, 0.3, 0.2, 0.2)])  # second robot missed
        out = evaluate_annotator(pred, gold)
        assert out["recall"] == 0.5
        assert out["precision"] == 1.0
        assert out["gold_boxes"] == 2

    def test_counts_an_extra_prediction_as_a_false_positive(self, tmp_path):
        gold = tmp_path / "gold"
        pred = tmp_path / "pred"
        gold.mkdir()
        pred.mkdir()
        self._write_label(gold / "f1.txt", [(0.3, 0.3, 0.2, 0.2)])
        self._write_label(pred / "f1.txt", [(0.3, 0.3, 0.2, 0.2), (0.9, 0.9, 0.05, 0.05)])
        out = evaluate_annotator(pred, gold)
        assert out["recall"] == 1.0
        assert out["precision"] == 0.5


class TestEvaluateEvents:
    def test_matches_an_event_within_tolerance(self):
        pred = {"events": [{"kind": "round_start", "t_ms": 500.0}]}
        gold = {"round_start": [520.0]}
        out = evaluate_events(pred, gold, tol_ms=150.0)
        assert out["round_start"]["recall"] == 1.0
        assert out["round_start"]["precision"] == 1.0
        assert out["round_start"]["mean_temporal_error_ms"] == 20.0

    def test_rejects_an_event_beyond_tolerance(self):
        pred = {"events": [{"kind": "round_start", "t_ms": 100.0}]}
        gold = {"round_start": [520.0]}
        out = evaluate_events(pred, gold, tol_ms=150.0)
        assert out["round_start"]["recall"] == 0.0
        assert out["round_start"]["mean_temporal_error_ms"] is None

    def test_reports_zero_recall_when_event_absent_from_prediction(self):
        pred = {"events": []}
        gold = {"ring_out": [1000.0]}
        out = evaluate_events(pred, gold)
        assert out["ring_out"]["recall"] == 0.0
        assert out["ring_out"]["precision"] == 0.0

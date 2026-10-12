"""Detection and tracking metrics."""

import numpy as np
import pytest

from mot.eval.metrics import score_detections, score_tracking

BOX_A = [0, 0, 50, 100]
BOX_B = [200, 0, 250, 100]


class TestDetectionScore:
    def test_perfect_detection(self):
        boxes = np.array([BOX_A, BOX_B], dtype=np.float32)
        score = score_detections(boxes, boxes)
        assert (score.precision, score.recall, score.f1) == (1.0, 1.0, 1.0)

    def test_missed_detection_lowers_recall(self):
        gt = np.array([BOX_A, BOX_B], dtype=np.float32)
        pred = np.array([BOX_A], dtype=np.float32)
        score = score_detections(gt, pred)
        assert score.false_negatives == 1
        assert score.recall == pytest.approx(0.5)
        assert score.precision == pytest.approx(1.0)

    def test_false_positive_lowers_precision(self):
        gt = np.array([BOX_A], dtype=np.float32)
        pred = np.array([BOX_A, BOX_B], dtype=np.float32)
        score = score_detections(gt, pred)
        assert score.false_positives == 1
        assert score.precision == pytest.approx(0.5)

    def test_empty_inputs_do_not_divide_by_zero(self):
        empty = np.zeros((0, 4), dtype=np.float32)
        score = score_detections(empty, empty)
        assert (score.precision, score.recall, score.f1) == (0.0, 0.0, 0.0)


class TestTrackingScore:
    def test_perfect_tracking_scores_one(self):
        frames = {
            f: (np.array([BOX_A], dtype=np.float32), [1]) for f in range(1, 4)
        }
        score = score_tracking(frames, frames)
        assert score.mota == pytest.approx(1.0)
        assert score.motp == pytest.approx(1.0)
        assert score.id_switches == 0

    def test_id_switch_is_counted_once(self):
        gt = {f: (np.array([BOX_A], dtype=np.float32), [1]) for f in range(1, 4)}
        # The predicted id changes at frame 3 for the same ground-truth person.
        pred = {
            1: (np.array([BOX_A], dtype=np.float32), [7]),
            2: (np.array([BOX_A], dtype=np.float32), [7]),
            3: (np.array([BOX_A], dtype=np.float32), [9]),
        }
        score = score_tracking(gt, pred)
        assert score.id_switches == 1
        # 3 ground-truth boxes, all matched, one switch => 1 - 1/3
        assert score.mota == pytest.approx(2 / 3)

    def test_mota_can_go_negative(self):
        """More errors than ground truth is representable, not clamped."""
        gt = {1: (np.array([BOX_A], dtype=np.float32), [1])}
        pred = {
            1: (
                np.array([[500, 500, 550, 600], [600, 600, 650, 700]], dtype=np.float32),
                [1, 2],
            )
        }
        assert score_tracking(gt, pred).mota < 0

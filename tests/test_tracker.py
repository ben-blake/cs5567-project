"""Association and track lifecycle."""

import numpy as np
import pytest

from mot.track.tracker import Tracker, iou_matrix


def unit(vector):
    """L2-normalise, matching what the Siamese net emits."""
    vector = np.asarray(vector, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class TestIoU:
    def test_identical_boxes(self):
        box = np.array([[0, 0, 10, 10]], dtype=np.float32)
        assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)

    def test_disjoint_boxes(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)
        b = np.array([[20, 20, 30, 30]], dtype=np.float32)
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.0)

    def test_known_overlap(self):
        a = np.array([[0, 0, 10, 10]], dtype=np.float32)   # area 100
        b = np.array([[0, 0, 10, 5]], dtype=np.float32)    # area 50, fully inside
        assert iou_matrix(a, b)[0, 0] == pytest.approx(0.5)

    def test_empty_inputs(self):
        assert iou_matrix(np.zeros((0, 4)), np.zeros((3, 4))).shape == (0, 3)


class TestTracking:
    def test_ids_persist_under_smooth_motion(self):
        tracker = Tracker(min_hits=1, appearance_weight=0.0)
        seen = []
        for frame in range(6):
            dets = np.array([[frame * 4, 0, 60 + frame * 4, 120]], dtype=np.float32)
            seen.append([t.track_id for t in tracker.update(dets)])
        assert seen == [[1]] * 6, "a smoothly moving person must keep one id"

    def test_separate_people_get_separate_ids(self):
        tracker = Tracker(min_hits=1, appearance_weight=0.0)
        dets = np.array([[0, 0, 50, 100], [300, 0, 350, 100]], dtype=np.float32)
        assert len(tracker.update(dets)) == 2

    def test_track_survives_brief_occlusion(self):
        """A few missed frames must not restart the id -- the point of max_age."""
        tracker = Tracker(min_hits=1, max_age=10, appearance_weight=0.0)
        box = np.array([[0, 0, 50, 100]], dtype=np.float32)

        tracker.update(box)
        for _ in range(3):
            tracker.update(np.zeros((0, 4), dtype=np.float32))  # occluded

        reappeared = tracker.update(box)
        assert [t.track_id for t in reappeared] == [1]

    def test_track_is_dropped_after_max_age(self):
        tracker = Tracker(min_hits=1, max_age=2, appearance_weight=0.0)
        tracker.update(np.array([[0, 0, 50, 100]], dtype=np.float32))
        for _ in range(5):
            tracker.update(np.zeros((0, 4), dtype=np.float32))
        assert tracker.tracks == []

    def test_min_hits_suppresses_single_frame_noise(self):
        tracker = Tracker(min_hits=3, appearance_weight=0.0)
        box = np.array([[0, 0, 50, 100]], dtype=np.float32)
        assert tracker.update(box) == []        # tentative
        assert tracker.update(box) == []        # still tentative
        assert len(tracker.update(box)) == 1    # confirmed on the third hit

    def test_appearance_separates_crossing_people(self):
        """Overlap alone is ambiguous when two people cross; appearance resolves it."""
        tracker = Tracker(min_hits=1, appearance_weight=0.9, iou_threshold=0.1)
        red, blue = unit([1, 0, 0]), unit([0, 1, 0])

        left = np.array([0, 0, 50, 100], dtype=np.float32)
        right = np.array([60, 0, 110, 100], dtype=np.float32)
        tracker.update(np.stack([left, right]), np.stack([red, blue]))

        # They swap sides. Boxes alone would swap the ids; appearance should not.
        out = tracker.update(np.stack([right, left]), np.stack([blue, red]))
        by_id = {t.track_id: t.box for t in out}
        assert np.allclose(by_id[1], left), "id 1 (red) must follow the red person"
        assert np.allclose(by_id[2], right)

    def test_appearance_gate_blocks_mismatched_identity(self):
        """A visually different person in the same place starts a new track."""
        tracker = Tracker(
            min_hits=1, appearance_weight=0.9, max_appearance_distance=0.3, iou_threshold=0.1
        )
        box = np.array([[0, 0, 50, 100]], dtype=np.float32)

        tracker.update(box, np.stack([unit([1, 0, 0])]))
        tracker.update(box, np.stack([unit([0, 0, 1])]))  # orthogonal => distance 1.0

        assert tracker._next_id == 3, "mismatched appearance must not reuse the id"

    def test_reset(self):
        tracker = Tracker(min_hits=1)
        tracker.update(np.array([[0, 0, 50, 100]], dtype=np.float32))
        tracker.reset()
        assert tracker.tracks == [] and tracker.frame_count == 0

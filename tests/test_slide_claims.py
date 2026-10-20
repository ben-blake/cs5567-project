"""Every number on the slides must match the code.

A deck drifts from the implementation silently: someone changes a default, and
a figure that was true last week is quietly wrong in front of the class. These
tests fail when that happens.

Numbers that come from a measured run rather than from the code (MOTA, recall,
timings) are not checked here; they live in docs/RESULTS.md with the conditions
they were measured under.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from mot.data.augment import DetectionTransform, reid_transform  # noqa: E402
from mot.data.mot16 import AMBIGUOUS_CLASSES, Annotation, Box  # noqa: E402
from mot.detect.faster_rcnn import build_model, parameter_summary  # noqa: E402
from mot.eval.metrics import TrackingScore  # noqa: E402
from mot.reid.siamese import CROP_SIZE, SiameseNetwork  # noqa: E402
from mot.track.tracker import Tracker  # noqa: E402

DECK = (Path(__file__).resolve().parents[1] / "slides" / "midterm.tex").read_text()


class TestReIDClaims:
    def test_crop_size(self):
        assert CROP_SIZE == (128, 64)
        assert r"128\times64" in DECK

    def test_parameter_count(self):
        total = sum(p.numel() for p in SiameseNetwork().parameters())
        assert 1.45e6 < total < 1.55e6, f"deck says 1.5 M, code has {total:,}"
        assert "1.5" in DECK

    def test_embedding_is_128d_and_unit_norm(self):
        net = SiameseNetwork().eval()
        with torch.no_grad():
            embeddings = net.forward_once(torch.randn(4, 3, *CROP_SIZE))
        assert embeddings.shape[1] == 128
        assert torch.allclose(embeddings.norm(dim=1), torch.ones(4), atol=1e-5)
        assert "128-dimensional L2-normalised" in DECK

    def test_four_convolution_blocks(self):
        net = SiameseNetwork()
        blocks = [m for m in net.features if isinstance(m, torch.nn.Sequential)]
        assert len(blocks) == 4
        assert "Four convolution blocks" in DECK


class TestDetectorClaims:
    def test_parameter_budget(self):
        summary = parameter_summary(build_model(pretrained=False, freeze_backbone=True))
        millions = {k: v / 1e6 for k, v in summary.items()}

        assert millions["total"] == pytest.approx(41.3, abs=0.05)
        assert millions["frozen"] == pytest.approx(26.8, abs=0.05)
        assert millions["trainable"] == pytest.approx(14.5, abs=0.05)

        for claim in (r"41.3\,M", r"26.8\,M", r"14.5\,M"):
            assert claim in DECK, f"deck is missing {claim}"

    def test_trainable_share(self):
        summary = parameter_summary(build_model(pretrained=False, freeze_backbone=True))
        share = 100 * summary["trainable"] / summary["total"]
        assert share == pytest.approx(35.1, abs=0.1)
        assert r"35\%" in DECK


class TestDataClaims:
    def test_ignored_classes(self):
        assert {7, 8, 12} <= AMBIGUOUS_CLASSES
        assert not Annotation(1, 1, Box(0, 0, 1, 1), class_id=7).is_target
        assert "Classes 7, 8 and 12" in DECK

    def test_one_indexed_conversion(self, tmp_path):
        from mot.data.mot16 import parse_annotations

        path = tmp_path / "gt.txt"
        path.write_text("1,3,586,447,85,263,1,1,1\n")
        box = parse_annotations(path)[0].box

        assert box.x1 == 585.0 and box.width == 85.0
        assert "1-indexed" in DECK

    def test_no_vertical_flip_or_rotation(self):
        source = inspect.getsource(DetectionTransform) + inspect.getsource(reid_transform)
        assert "Vertical" not in source and "Rotation" not in source
        assert "No vertical flips" in DECK

    def test_random_erasing_present(self):
        assert "RandomErasing" in inspect.getsource(reid_transform)
        assert "Random erasing" in DECK


class TestTrackerClaims:
    def test_reacquisition_ships_disabled(self):
        """Slide 17 presents the shipped config as the one without these."""
        tracker = Tracker()
        assert tracker.reid_rescue_distance == 0.0
        assert tracker.motion_prediction is False

        # Match the claim rather than one phrasing of it, so the deck can be
        # reworded without a test failing over a synonym.
        import re

        assert re.search(r"[Bb]oth features are (switched off|disabled|off)", DECK), (
            "the deck must still say both re-acquisition features are disabled"
        )

    def test_mota_formula_matches_the_slide(self):
        score = TrackingScore(true_positives=10, false_positives=2, false_negatives=5,
                              id_switches=1, matched_iou_sum=8.0)
        assert score.mota == pytest.approx(1 - (5 + 2 + 1) / 15)
        assert r"\mathrm{FN} + \mathrm{FP} + \mathrm{IDSW}" in DECK

    def test_motp_is_mean_iou_of_matches(self):
        score = TrackingScore(true_positives=10, false_positives=0, false_negatives=0,
                              id_switches=0, matched_iou_sum=8.0)
        assert score.motp == pytest.approx(0.8)
        assert "mean IoU of matched boxes" in DECK


def test_the_withdrawn_number_stays_out_of_the_deck():
    """97.2% came from an unstable measure and must not reappear."""
    assert "97.2" not in DECK, "the withdrawn headline is back in the slides"

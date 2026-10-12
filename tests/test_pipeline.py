"""End-to-end pipeline plumbing.

Uses a stub detector with known boxes so the test exercises cropping,
embedding, association, rendering and serialisation without downloading
pretrained weights or needing a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from PIL import Image, ImageDraw  # noqa: E402

from mot.data.mot16 import Sequence  # noqa: E402
from mot.pipeline import (  # noqa: E402
    TrackingPipeline,
    crop_people,
    longest_track,
    results_to_mot_format,
)
from mot.reid.siamese import SiameseNetwork  # noqa: E402
from mot.track.tracker import Tracker  # noqa: E402

FRAMES = 6
SIZE = (320, 240)


class StubDetector(torch.nn.Module):
    """Returns two boxes that drift right, mimicking two walking people."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, images):
        step = self.calls * 6
        self.calls += 1
        boxes = torch.tensor(
            [[10.0 + step, 40.0, 50.0 + step, 150.0], [180.0, 40.0, 220.0, 150.0]]
        )
        return [
            {
                "boxes": boxes,
                "scores": torch.tensor([0.95, 0.88]),
                "labels": torch.tensor([1, 1]),
            }
        ]


@pytest.fixture
def sequence(tmp_path):
    """A minimal MOT16 sequence on disk."""
    root = tmp_path / "MOT16-TEST"
    (root / "img1").mkdir(parents=True)
    (root / "seqinfo.ini").write_text(
        "[Sequence]\n"
        "name=MOT16-TEST\nimDir=img1\nframeRate=10\n"
        f"seqLength={FRAMES}\nimWidth={SIZE[0]}\nimHeight={SIZE[1]}\nimExt=.jpg\n"
    )
    for frame in range(1, FRAMES + 1):
        image = Image.new("RGB", SIZE, (70, 90, 110))
        draw = ImageDraw.Draw(image)
        step = (frame - 1) * 6
        draw.rectangle([10 + step, 40, 50 + step, 150], fill=(220, 60, 60))
        draw.rectangle([180, 40, 220, 150], fill=(60, 90, 220))
        image.save(root / "img1" / f"{frame:06d}.jpg")
    return Sequence(root)


def test_sequence_fixture_is_well_formed(sequence):
    assert sequence.info.seq_length == FRAMES
    assert sequence.frame_path(1).is_file()
    assert not sequence.has_ground_truth  # test split, like MOT16 test data


def test_crop_people_returns_reid_sized_crops():
    image = Image.new("RGB", SIZE, (128, 128, 128))
    boxes = np.array([[10, 10, 60, 120], [100, 20, 150, 130]], dtype=np.float32)
    crops = crop_people(image, boxes)
    assert crops.shape == (2, 3, 128, 64)
    assert 0.0 <= float(crops.min()) and float(crops.max()) <= 1.0


def test_crop_people_survives_degenerate_and_out_of_bounds_boxes():
    image = Image.new("RGB", (100, 100))
    boxes = np.array([[0, 0, 1, 1], [-50, -50, 20, 20], [90, 90, 500, 500]], dtype=np.float32)
    assert crop_people(image, boxes).shape == (3, 3, 128, 64)


def test_crop_people_empty():
    assert crop_people(Image.new("RGB", (64, 64)), np.zeros((0, 4))).shape[0] == 0


class TestPipeline:
    def test_tracks_two_people_with_stable_ids(self, sequence):
        pipeline = TrackingPipeline(
            StubDetector(), reid=None, tracker=Tracker(min_hits=1, appearance_weight=0.0),
            device=torch.device("cpu"),
        )
        results = pipeline.run_sequence(sequence, progress=False)

        assert len(results) == FRAMES
        assert all(len(r) == 2 for r in results), "both people visible in every frame"

        ids_per_frame = {tuple(sorted(r.track_ids)) for r in results}
        assert ids_per_frame == {(1, 2)}, f"ids must stay stable, saw {ids_per_frame}"

    def test_runs_with_reid_enabled(self, sequence):
        """The Re-ID path must work, not just the overlap-only path."""
        pipeline = TrackingPipeline(
            StubDetector(),
            reid=SiameseNetwork().eval(),
            tracker=Tracker(min_hits=1, appearance_weight=0.5),
            device=torch.device("cpu"),
        )
        results = pipeline.run_sequence(sequence, progress=False)
        assert len(results) == FRAMES
        assert all(len(r) == 2 for r in results)

    def test_max_frames_limits_work(self, sequence):
        pipeline = TrackingPipeline(
            StubDetector(), tracker=Tracker(min_hits=1), device=torch.device("cpu")
        )
        assert len(pipeline.run_sequence(sequence, max_frames=3, progress=False)) == 3

    def test_longest_track(self, sequence):
        pipeline = TrackingPipeline(
            StubDetector(), tracker=Tracker(min_hits=1, appearance_weight=0.0),
            device=torch.device("cpu"),
        )
        results = pipeline.run_sequence(sequence, progress=False)
        assert longest_track(results) in (1, 2)
        assert longest_track([]) is None

    def test_mot_format_round_trips_through_the_parser(self, sequence, tmp_path):
        """Our output must be readable by our own MOT16 parser."""
        from mot.data.mot16 import parse_annotations

        pipeline = TrackingPipeline(
            StubDetector(), tracker=Tracker(min_hits=1, appearance_weight=0.0),
            device=torch.device("cpu"),
        )
        results = pipeline.run_sequence(sequence, progress=False)

        path = tmp_path / "out.txt"
        path.write_text(results_to_mot_format(results))
        parsed = parse_annotations(path)

        assert len(parsed) == sum(len(r) for r in results)
        # Round trip: 1-indexed on write, 0-indexed on read, so we get back
        # the coordinates we started with.
        first_written = results[0]
        assert parsed[0].box.as_xyxy()[0] == pytest.approx(
            float(first_written.boxes[0][0]), abs=0.01
        )

    def test_renders_a_video(self, sequence, tmp_path):
        from mot.pipeline import render_results

        pipeline = TrackingPipeline(
            StubDetector(), tracker=Tracker(min_hits=1, appearance_weight=0.0),
            device=torch.device("cpu"),
        )
        results = pipeline.run_sequence(sequence, progress=False)
        output = render_results(sequence, results, tmp_path / "out.mp4", highlight_id=1)

        assert output.is_file() and output.stat().st_size > 0

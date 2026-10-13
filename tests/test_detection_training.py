"""The detector fine-tuning path.

This is Priority 1 after the midterm and had never been executed: the dataset,
the transform, the collate function and the training loop were all written and
never run together. These tests exercise the whole path on a tiny synthetic
sequence so the next phase does not begin with debugging.

Deliberately small: two frames, one epoch, 96x128 images. The point is that
the plumbing holds, not that anything learns.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from mot.data.augment import DetectionTransform  # noqa: E402
from mot.data.datasets import MOT16DetectionDataset, detection_collate  # noqa: E402
from mot.data.mot16 import Sequence  # noqa: E402
from mot.detect.faster_rcnn import build_model, parameter_summary  # noqa: E402
from mot.detect.train import train_detector  # noqa: E402

FRAMES = 2
SIZE = (128, 96)  # width, height


@pytest.fixture
def sequence(tmp_path):
    """A MOT16 sequence with ground truth, small enough to train on."""
    root = tmp_path / "MOT16-TRAIN"
    (root / "img1").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "seqinfo.ini").write_text(
        "[Sequence]\nname=MOT16-TRAIN\nimDir=img1\nframeRate=30\n"
        f"seqLength={FRAMES}\nimWidth={SIZE[0]}\nimHeight={SIZE[1]}\nimExt=.jpg\n"
    )

    rows = []
    for frame in range(1, FRAMES + 1):
        image = Image.new("RGB", SIZE, (60, 80, 100))
        draw = ImageDraw.Draw(image)
        left = 10 + (frame - 1) * 8
        draw.rectangle([left, 20, left + 24, 80], fill=(200, 70, 70))
        image.save(root / "img1" / f"{frame:06d}.jpg")
        # gt.txt is 1-indexed: frame, id, left, top, width, height, conf, class, vis
        rows.append(f"{frame},1,{left + 1},21,24,60,1,1,1")
    (root / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
    return Sequence(root)


class TestDetectionDataset:
    def test_yields_images_and_targets(self, sequence):
        dataset = MOT16DetectionDataset([sequence])
        assert len(dataset) == FRAMES

        image, target = dataset[0]
        assert image.shape[0] == 3, "image must be CHW float"
        assert 0.0 <= float(image.min()) and float(image.max()) <= 1.0
        assert target["boxes"].shape[1] == 4
        assert (target["labels"] == 1).all(), "single person class"

    def test_boxes_are_valid_for_torchvision(self, sequence):
        """Zero-area boxes make the detection loss return NaN."""
        _, target = MOT16DetectionDataset([sequence])[0]
        boxes = target["boxes"]
        assert (boxes[:, 2] > boxes[:, 0]).all()
        assert (boxes[:, 3] > boxes[:, 1]).all()

    def test_min_visibility_filters(self, sequence):
        """A visibility floor above the data drops every frame."""
        assert len(MOT16DetectionDataset([sequence], min_visibility=1.1)) == 0

    def test_transform_keeps_boxes_inside_the_image(self, sequence):
        dataset = MOT16DetectionDataset([sequence], transform=DetectionTransform(train=True))
        for index in range(len(dataset)):
            image, target = dataset[index]
            height, width = image.shape[1:]
            boxes = target["boxes"]
            assert (boxes[:, 0] >= -1).all() and (boxes[:, 2] <= width + 1).all()
            assert (boxes[:, 1] >= -1).all() and (boxes[:, 3] <= height + 1).all()

    def test_horizontal_flip_mirrors_boxes(self):
        """A flip must move the boxes with the pixels, or training learns noise."""
        image = Image.new("RGB", (100, 50))
        target = {"boxes": torch.tensor([[10.0, 5.0, 30.0, 45.0]]),
                  "labels": torch.ones(1, dtype=torch.int64)}

        transform = DetectionTransform(train=True, flip_probability=1.0)
        _, flipped = transform(image, {k: v.clone() for k, v in target.items()})

        # x1' = W - x2, x2' = W - x1
        assert flipped["boxes"][0, 0].item() == pytest.approx(100 - 30)
        assert flipped["boxes"][0, 2].item() == pytest.approx(100 - 10)
        # Width is preserved, and the box stays well-formed.
        assert flipped["boxes"][0, 2] > flipped["boxes"][0, 0]

    def test_collate_keeps_images_as_a_list(self, sequence):
        """Detection models take lists; images differ in size so stacking fails."""
        loader = DataLoader(MOT16DetectionDataset([sequence]), batch_size=2,
                            collate_fn=detection_collate)
        images, targets = next(iter(loader))
        assert isinstance(images, tuple) and len(images) == 2
        assert isinstance(targets, tuple) and "boxes" in targets[0]


class TestTrainDetector:
    @pytest.mark.slow
    def test_one_epoch_runs_and_reduces_nothing_catastrophically(self, sequence):
        """The fine-tuning loop completes and produces finite losses."""
        model = build_model(num_classes=2, pretrained=False, freeze_backbone=True)
        loader = DataLoader(
            MOT16DetectionDataset([sequence], transform=DetectionTransform(train=True)),
            batch_size=1, collate_fn=detection_collate,
        )

        history = train_detector(model, loader, epochs=1, device=torch.device("cpu"),
                                 log_every=0)

        assert len(history) == 1
        assert np.isfinite(history[0]["total"]), "loss must be finite"
        assert history[0]["total"] > 0

    def test_only_the_heads_carry_gradients(self, sequence):
        """Few-shot fine-tuning: the frozen backbone must not learn."""
        model = build_model(num_classes=2, pretrained=False, freeze_backbone=True)
        assert not any(p.requires_grad for p in model.backbone.parameters())

        summary = parameter_summary(model)
        assert summary["frozen"] > 0 and summary["trainable"] > 0
        assert summary["trainable"] + summary["frozen"] == summary["total"]

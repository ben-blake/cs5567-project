"""PyTorch datasets for detector fine-tuning and Re-ID training."""

from __future__ import annotations

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..reid.siamese import CROP_SIZE
from .mot16 import Sequence, group_by_frame, group_by_track


class MOT16DetectionDataset(Dataset):
    """Frames with pedestrian boxes, in torchvision detection format.

    Each item is ``(image, target)`` where image is a float tensor in [0, 1]
    and target has ``boxes`` (xyxy) and ``labels`` (all 1 = person).
    """

    def __init__(
        self,
        sequences: list[Sequence],
        transform=None,
        min_visibility: float = 0.25,
        frame_stride: int = 1,
    ):
        self.transform = transform
        #: (sequence, frame) pairs that contain at least one usable box.
        self.samples: list[tuple[Sequence, int]] = []
        self._boxes: dict[tuple[str, int], list] = {}

        for sequence in sequences:
            by_frame = group_by_frame(sequence.ground_truth(targets_only=True))
            for frame in sorted(by_frame):
                if frame % frame_stride:
                    continue
                visible = [a for a in by_frame[frame] if a.visibility >= min_visibility]
                if not visible:
                    continue  # skip empty frames: no gradient signal
                self.samples.append((sequence, frame))
                self._boxes[(sequence.name, frame)] = visible

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sequence, frame = self.samples[index]
        image = Image.open(sequence.frame_path(frame)).convert("RGB")
        annotations = self._boxes[(sequence.name, frame)]

        boxes = torch.tensor(
            [a.box.clip(image.width, image.height).as_xyxy() for a in annotations],
            dtype=torch.float32,
        ).reshape(-1, 4)

        # Drop degenerate boxes; torchvision's loss returns NaN on zero area.
        keep = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[keep]

        target = {
            "boxes": boxes,
            "labels": torch.ones((len(boxes),), dtype=torch.int64),
            "image_id": torch.tensor([index]),
        }

        if self.transform is not None:
            image, target = self.transform(image, target)
        else:
            image = torch.from_numpy(
                __import__("numpy").array(image)
            ).permute(2, 0, 1).float() / 255.0

        return image, target


def detection_collate(batch):
    """Detection models take lists, not stacked tensors (images vary in size)."""
    return tuple(zip(*batch, strict=True))


class ReIDPairDataset(Dataset):
    """Pairs of person crops labelled same (1) or different (0).

    Works from any ``identity -> list of crop paths`` mapping, so it serves
    both Market-1501 and crops cut from MOT16 ground truth.
    """

    def __init__(
        self,
        identities: dict[int, list[Path]],
        transform=None,
        pairs_per_epoch: int = 10_000,
        positive_ratio: float = 0.5,
        seed: int = 0,
        cameras: dict[Path, int] | None = None,
        prefer_cross_camera: bool = True,
    ):
        # An identity needs two crops to form a positive pair.
        self.identities = {k: v for k, v in identities.items() if len(v) >= 2}
        if len(self.identities) < 2:
            raise ValueError("need at least 2 identities with 2+ crops each")

        self.ids = sorted(self.identities)
        self.transform = transform
        self.pairs_per_epoch = pairs_per_epoch
        self.positive_ratio = positive_ratio
        self.cameras = cameras or {}
        self.prefer_cross_camera = prefer_cross_camera and bool(self.cameras)
        self._rng = random.Random(seed)

    def _positive_pair(self, identity: int) -> tuple[Path, Path]:
        """Two crops of one person, from different cameras where possible.

        Two crops from the same camera seconds apart share pose, lighting and
        background, so matching them is nearly trivial and the network can
        succeed by memorising the background instead of the person. Market-1501
        evaluation treats same-camera matches as junk for exactly this reason
        (Zheng et al., ICCV 2015, Sec. 3), so we train the way it scores.
        """
        crops = self.identities[identity]
        if not self.prefer_cross_camera:
            first, second = self._rng.sample(crops, 2)
            return first, second

        first = self._rng.choice(crops)
        other_camera = [
            path
            for path in crops
            if path is not first
            and self.cameras.get(path, -1) != self.cameras.get(first, -2)
        ]
        if other_camera:
            return first, self._rng.choice(other_camera)

        # Only one camera saw this person; fall back to any other crop.
        remaining = [path for path in crops if path is not first]
        return first, self._rng.choice(remaining)

    def __len__(self) -> int:
        return self.pairs_per_epoch

    def _load(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            return self.transform(image)
        image = image.resize((CROP_SIZE[1], CROP_SIZE[0]))
        array = __import__("numpy").array(image)
        return torch.from_numpy(array).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, index: int):
        same = self._rng.random() < self.positive_ratio

        if same:
            first, second = self._positive_pair(self._rng.choice(self.ids))
        else:
            id_a, id_b = self._rng.sample(self.ids, 2)
            first = self._rng.choice(self.identities[id_a])
            second = self._rng.choice(self.identities[id_b])

        return self._load(first), self._load(second), torch.tensor(1.0 if same else 0.0)


def parse_market1501_filename(name: str) -> tuple[int, int] | None:
    """``0002_c1s1_000451_03.jpg`` -> ``(person_id, camera_id)``.

    Returns None when the name does not parse.
    """
    parts = name.split("_")
    if len(parts) < 2 or not parts[1].startswith("c"):
        return None
    try:
        person_id = int(parts[0])
        # "c1s1" -> camera 1, sequence 1. Take the digits after "c".
        camera_id = int(parts[1][1 : parts[1].index("s")] if "s" in parts[1] else parts[1][1:])
    except ValueError:
        return None
    return person_id, camera_id


def market1501_identities(
    root: str | Path, split: str = "bounding_box_train"
) -> dict[int, list[Path]]:
    """Index Market-1501 crops by person id.

    Filenames look like ``0002_c1s1_000451_03.jpg``.

    Two identity labels are *not* people and are dropped (Zheng et al., ICCV
    2015, Sec. 3): ``-1`` marks **junk** boxes, where the detector overlapped a
    real person by only 20-50%, and ``0000`` marks **distractors** -- false
    alarms on background, with under 20% overlap. Training on either teaches
    the network that background crops are a person worth matching.
    """
    directory = Path(root) / split
    if not directory.is_dir():
        raise FileNotFoundError(f"Market-1501 split not found: {directory}")

    identities: dict[int, list[Path]] = {}
    for path in sorted(directory.glob("*.jpg")):
        parsed = parse_market1501_filename(path.name)
        if parsed is None:
            continue
        person_id, _camera = parsed
        if person_id <= 0:  # -1 junk, 0000 distractor
            continue
        identities.setdefault(person_id, []).append(path)
    return identities


def market1501_cameras(paths: list[Path]) -> dict[Path, int]:
    """Camera id for each Market-1501 crop, for cross-camera pair sampling."""
    cameras = {}
    for path in paths:
        parsed = parse_market1501_filename(path.name)
        if parsed is not None:
            cameras[path] = parsed[1]
    return cameras


def export_mot16_crops(
    sequence: Sequence,
    output_dir: str | Path,
    min_visibility: float = 0.5,
    min_height: int = 64,
    max_per_identity: int = 40,
) -> dict[int, list[Path]]:
    """Cut person crops out of a sequence's ground truth.

    Gives the Siamese network in-domain training data, which matters because
    Market-1501 and MOT16 differ in camera angle and resolution.
    """
    output_dir = Path(output_dir) / sequence.name
    output_dir.mkdir(parents=True, exist_ok=True)

    by_track = group_by_track(
        [a for a in sequence.ground_truth() if a.visibility >= min_visibility]
    )
    exported: dict[int, list[Path]] = {}

    for track_id, track_annotations in by_track.items():
        # Even sampling across the track beats taking the first N frames,
        # which would all look nearly identical.
        step = max(1, len(track_annotations) // max_per_identity)
        for annotation in track_annotations[::step][:max_per_identity]:
            box = annotation.box
            if box.height < min_height:
                continue
            frame_path = sequence.frame_path(annotation.frame)
            if not frame_path.is_file():
                continue

            image = Image.open(frame_path).convert("RGB")
            clipped = box.clip(image.width, image.height)
            if clipped.width < 8 or clipped.height < 16:
                continue

            crop = image.crop(tuple(int(v) for v in clipped.as_xyxy()))
            out = output_dir / f"{track_id:04d}_{annotation.frame:06d}.jpg"
            crop.save(out, quality=92)
            exported.setdefault(track_id, []).append(out)

    return exported

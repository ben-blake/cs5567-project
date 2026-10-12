"""Parsing for the MOT16 benchmark layout.

A MOT16 sequence directory looks like::

    MOT16-02/
        seqinfo.ini
        img1/000001.jpg ...
        gt/gt.txt          (train split only)
        det/det.txt

Both annotation files are CSV with one box per line. Coordinates are
1-indexed in the file; we convert to 0-indexed on read so they can index
numpy arrays directly.
"""

from __future__ import annotations

import configparser
import csv
from dataclasses import dataclass
from pathlib import Path

# Class ids used by MOT16 gt.txt. Only PEDESTRIAN counts as a target; the
# others mark regions where a detector should not be penalised.
CLASS_PEDESTRIAN = 1
CLASS_PERSON_ON_VEHICLE = 2
CLASS_STATIC_PERSON = 7
CLASS_DISTRACTOR = 8
CLASS_REFLECTION = 12

#: Classes that are people but should not be scored as targets.
AMBIGUOUS_CLASSES = frozenset(
    {CLASS_PERSON_ON_VEHICLE, CLASS_STATIC_PERSON, CLASS_DISTRACTOR, CLASS_REFLECTION}
)


@dataclass(frozen=True)
class Box:
    """An axis-aligned box in ``xyxy`` form, 0-indexed, pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @classmethod
    def from_xywh(cls, x: float, y: float, w: float, h: float) -> "Box":
        return cls(x, y, x + w, y + h)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def clip(self, width: int, height: int) -> "Box":
        """Clamp to an image of the given size."""
        return Box(
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )


@dataclass(frozen=True)
class Annotation:
    """One ground-truth or detection row."""

    frame: int
    track_id: int
    box: Box
    confidence: float = 1.0
    class_id: int = CLASS_PEDESTRIAN
    visibility: float = 1.0

    @property
    def is_target(self) -> bool:
        """True when this row is a pedestrian the tracker is scored on."""
        return self.class_id == CLASS_PEDESTRIAN and self.confidence > 0


@dataclass(frozen=True)
class SequenceInfo:
    """Contents of ``seqinfo.ini``."""

    name: str
    im_dir: str
    frame_rate: int
    seq_length: int
    im_width: int
    im_height: int
    im_ext: str


def parse_annotations(path: str | Path) -> list[Annotation]:
    """Read a MOT16 ``gt.txt`` or ``det.txt`` into annotations.

    Columns are ``frame, id, bb_left, bb_top, bb_width, bb_height`` followed by
    ``conf, class, visibility`` in gt files and ``conf, -1, -1, -1`` in det
    files. Short rows are tolerated so the same reader handles both.
    """
    annotations: list[Annotation] = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.reader(handle):
            if not row or row[0].lstrip().startswith("#"):
                continue
            values = [field.strip() for field in row if field.strip() != ""]
            if len(values) < 6:
                continue

            frame = int(float(values[0]))
            track_id = int(float(values[1]))
            left, top, width, height = (float(v) for v in values[2:6])

            confidence = float(values[6]) if len(values) > 6 else 1.0
            class_id = int(float(values[7])) if len(values) > 7 else CLASS_PEDESTRIAN
            visibility = float(values[8]) if len(values) > 8 else 1.0

            # det.txt uses -1 as a filler for the unused class column.
            if class_id < 0:
                class_id = CLASS_PEDESTRIAN

            annotations.append(
                Annotation(
                    frame=frame,
                    track_id=track_id,
                    # Files are 1-indexed; shift so boxes index arrays directly.
                    box=Box.from_xywh(left - 1.0, top - 1.0, width, height),
                    confidence=confidence,
                    class_id=class_id,
                    visibility=visibility,
                )
            )
    return annotations


def parse_seqinfo(path: str | Path) -> SequenceInfo:
    """Read a ``seqinfo.ini``."""
    parser = configparser.ConfigParser()
    parser.read(path)
    section = parser["Sequence"]
    return SequenceInfo(
        name=section.get("name", Path(path).parent.name),
        im_dir=section.get("imDir", "img1"),
        frame_rate=section.getint("frameRate", 30),
        seq_length=section.getint("seqLength", 0),
        im_width=section.getint("imWidth", 0),
        im_height=section.getint("imHeight", 0),
        im_ext=section.get("imExt", ".jpg"),
    )


def group_by_frame(annotations: list[Annotation]) -> dict[int, list[Annotation]]:
    """Bucket annotations by frame number."""
    frames: dict[int, list[Annotation]] = {}
    for annotation in annotations:
        frames.setdefault(annotation.frame, []).append(annotation)
    return frames


def group_by_track(annotations: list[Annotation]) -> dict[int, list[Annotation]]:
    """Bucket annotations by track id, each list sorted by frame."""
    tracks: dict[int, list[Annotation]] = {}
    for annotation in annotations:
        tracks.setdefault(annotation.track_id, []).append(annotation)
    for boxes in tracks.values():
        boxes.sort(key=lambda a: a.frame)
    return tracks


class Sequence:
    """One MOT16 sequence on disk."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"sequence directory not found: {self.root}")
        self.info = parse_seqinfo(self.root / "seqinfo.ini")

    @property
    def name(self) -> str:
        return self.info.name

    @property
    def has_ground_truth(self) -> bool:
        return (self.root / "gt" / "gt.txt").is_file()

    def frame_path(self, frame: int) -> Path:
        """Absolute path to a 1-indexed frame."""
        return self.root / self.info.im_dir / f"{frame:06d}{self.info.im_ext}"

    def frame_paths(self) -> list[Path]:
        image_dir = self.root / self.info.im_dir
        return sorted(image_dir.glob(f"*{self.info.im_ext}"))

    def ground_truth(self, targets_only: bool = True) -> list[Annotation]:
        """Ground-truth boxes, pedestrians only by default."""
        if not self.has_ground_truth:
            raise FileNotFoundError(f"{self.name} has no gt/gt.txt (test split?)")
        annotations = parse_annotations(self.root / "gt" / "gt.txt")
        if targets_only:
            annotations = [a for a in annotations if a.is_target]
        return annotations

    def public_detections(self) -> list[Annotation]:
        """The benchmark's own detections, for comparison against ours."""
        return parse_annotations(self.root / "det" / "det.txt")

    def __repr__(self) -> str:
        return f"<Sequence {self.name} frames={self.info.seq_length}>"


def find_sequences(root: str | Path) -> list[Sequence]:
    """Every sequence directory under ``root`` (e.g. ``MOT16/train``)."""
    root = Path(root)
    return [Sequence(child) for child in sorted(root.iterdir()) if (child / "seqinfo.ini").is_file()]

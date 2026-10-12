"""End-to-end tracking: frames in, tracked video out.

This is the piece the demo runs. Everything it needs is optional except the
detector -- with no Siamese model it falls back to overlap-only tracking,
which is a useful ablation to show side by side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .config import get_device
from .data.mot16 import Sequence
from .detect.faster_rcnn import COCO_PERSON, detect
from .reid.siamese import CROP_SIZE, SiameseNetwork, embed_crops
from .track.tracker import Track, Tracker
from .viz.draw import draw_frame_label, draw_tracks, write_video


@dataclass
class FrameResult:
    """What the tracker produced for one frame."""

    frame: int
    boxes: np.ndarray
    track_ids: list[int]
    scores: list[float] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.track_ids)


def crop_people(image: Image.Image, boxes: np.ndarray) -> torch.Tensor:
    """Cut and resize person crops for the Re-ID network."""
    height, width = CROP_SIZE
    crops = []
    for box in boxes:
        x1, y1, x2, y2 = (int(round(float(v))) for v in box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        if x2 - x1 < 2 or y2 - y1 < 2:
            # Degenerate box: feed grey so indices stay aligned with boxes.
            crops.append(torch.full((3, height, width), 0.5))
            continue
        crop = image.crop((x1, y1, x2, y2)).resize((width, height))
        crops.append(torch.from_numpy(np.array(crop)).permute(2, 0, 1).float() / 255.0)

    if not crops:
        return torch.empty(0, 3, height, width)
    return torch.stack(crops)


class TrackingPipeline:
    """Detector + optional Re-ID + tracker, applied frame by frame."""

    def __init__(
        self,
        detector: torch.nn.Module,
        reid: SiameseNetwork | None = None,
        tracker: Tracker | None = None,
        score_threshold: float = 0.7,
        person_label: int = COCO_PERSON,
        device: torch.device | None = None,
    ):
        self.device = device or get_device()
        self.detector = detector.eval().to(self.device)
        self.reid = reid.eval().to(self.device) if reid is not None else None
        self.score_threshold = score_threshold
        self.person_label = person_label
        self.tracker = tracker or Tracker(
            appearance_weight=0.5 if reid is not None else 0.0
        )

    def process_image(self, image: Image.Image, frame_number: int) -> FrameResult:
        """Detect, embed, associate -- one frame."""
        tensor = torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0

        detections = detect(
            self.detector,
            [tensor],
            score_threshold=self.score_threshold,
            person_label=self.person_label,
            device=self.device,
        )[0]
        boxes = detections["boxes"].numpy()
        scores = detections["scores"].numpy()

        embeddings = None
        if self.reid is not None and len(boxes):
            crops = crop_people(image, boxes)
            embeddings = embed_crops(self.reid, crops, device=self.device).numpy()

        tracks: list[Track] = self.tracker.update(boxes, embeddings)

        # Report the detection score of whichever box each track matched.
        track_scores = []
        for track in tracks:
            if len(boxes):
                nearest = int(np.argmin(np.abs(boxes - track.box).sum(axis=1)))
                track_scores.append(float(scores[nearest]))
            else:
                track_scores.append(0.0)

        return FrameResult(
            frame=frame_number,
            boxes=np.array([t.box for t in tracks], dtype=np.float32).reshape(-1, 4),
            track_ids=[t.track_id for t in tracks],
            scores=track_scores,
        )

    def run_sequence(
        self,
        sequence: Sequence,
        max_frames: int | None = None,
        start_frame: int = 1,
        progress: bool = True,
    ) -> list[FrameResult]:
        """Track a whole MOT16 sequence."""
        self.tracker.reset()

        frames = list(range(start_frame, sequence.info.seq_length + 1))
        if max_frames is not None:
            frames = frames[:max_frames]

        iterator = frames
        if progress:
            try:
                from tqdm.auto import tqdm

                iterator = tqdm(frames, desc=f"tracking {sequence.name}", unit="frame")
            except ImportError:
                pass

        results = []
        for frame in iterator:
            path = sequence.frame_path(frame)
            if not path.is_file():
                break
            results.append(self.process_image(Image.open(path), frame))
        return results


def render_results(
    sequence: Sequence,
    results: list[FrameResult],
    output_path: str | Path,
    highlight_id: int | None = None,
    fps: int | None = None,
) -> Path:
    """Draw results over the source frames and encode an MP4."""
    rendered = []
    for result in results:
        image = Image.open(sequence.frame_path(result.frame)).convert("RGB")
        image = draw_tracks(
            image, result.boxes, result.track_ids, result.scores, highlight_id=highlight_id
        )
        caption = f"frame {result.frame}  |  {len(result)} tracked"
        if highlight_id is not None:
            caption += f"  |  target ID {highlight_id}"
        rendered.append(draw_frame_label(image, caption))

    return write_video(rendered, output_path, fps=fps or sequence.info.frame_rate)


def results_to_mot_format(results: list[FrameResult]) -> str:
    """Serialise to the MOT16 submission format.

    ``frame, id, bb_left, bb_top, bb_width, bb_height, conf, -1, -1, -1``,
    converting back to the 1-indexed coordinates the benchmark expects.
    """
    lines = []
    for result in results:
        for index, (box, track_id) in enumerate(
            zip(result.boxes, result.track_ids, strict=True)
        ):
            x1, y1, x2, y2 = (float(v) for v in box)
            score = result.scores[index] if index < len(result.scores) else 1.0
            lines.append(
                f"{result.frame},{track_id},{x1 + 1:.2f},{y1 + 1:.2f},"
                f"{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f},-1,-1,-1"
            )
    return "\n".join(lines)


def longest_track(results: list[FrameResult]) -> int | None:
    """The id present in the most frames.

    The assignment asks for one target pedestrian; the most-tracked person is
    the natural choice to highlight.
    """
    counts: dict[int, int] = {}
    for result in results:
        for track_id in result.track_ids:
            counts[track_id] = counts.get(track_id, 0) + 1
    return max(counts, key=lambda track_id: counts[track_id]) if counts else None

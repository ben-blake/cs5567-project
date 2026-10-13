"""Online tracking by detection.

Each frame, detections are matched to existing tracks with the Hungarian
algorithm over a cost that blends box overlap (where is it?) with appearance
distance (what does it look like?). Overlap alone fails when people cross or
occlude each other; appearance alone drifts between similar-looking people.
Together they cover each other's failure modes.

A track survives a few frames without a match, so a short occlusion does not
end it and restart a new id.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:  # SciPy ships with Colab; fall back to a greedy match if it is missing.
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover
    linear_sum_assignment = None


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of ``xyxy`` boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = np.asarray(boxes_a, dtype=np.float32)[:, None, :]  # (N, 1, 4)
    b = np.asarray(boxes_b, dtype=np.float32)[None, :, :]  # (1, M, 4)

    left = np.maximum(a[..., 0], b[..., 0])
    top = np.maximum(a[..., 1], b[..., 1])
    right = np.minimum(a[..., 2], b[..., 2])
    bottom = np.minimum(a[..., 3], b[..., 3])

    intersection = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    area_a = np.clip(a[..., 2] - a[..., 0], 0, None) * np.clip(a[..., 3] - a[..., 1], 0, None)
    area_b = np.clip(b[..., 2] - b[..., 0], 0, None) * np.clip(b[..., 3] - b[..., 1], 0, None)

    union = area_a + area_b - intersection
    return np.where(union > 0, intersection / np.maximum(union, 1e-9), 0.0).astype(np.float32)


def _assign(cost: np.ndarray) -> list[tuple[int, int]]:
    """Minimum-cost assignment, Hungarian when SciPy is present."""
    if cost.size == 0:
        return []
    if linear_sum_assignment is not None:
        rows, cols = linear_sum_assignment(cost)
        return list(zip(rows.tolist(), cols.tolist(), strict=True))

    # Greedy fallback: repeatedly take the cheapest remaining pair.
    pairs, used_rows, used_cols = [], set(), set()
    for row, col in np.dstack(np.unravel_index(np.argsort(cost, axis=None), cost.shape))[0]:
        if row not in used_rows and col not in used_cols:
            pairs.append((int(row), int(col)))
            used_rows.add(row)
            used_cols.add(col)
    return pairs


@dataclass
class Track:
    """One tracked person."""

    track_id: int
    box: np.ndarray                      # xyxy, last observed
    #: Recent appearance embeddings; a gallery is steadier than the latest one.
    gallery: deque = field(default_factory=lambda: deque(maxlen=30))
    hits: int = 1                        # successful matches so far
    age: int = 1                         # frames since the track was created
    time_since_update: int = 0           # frames since the last match
    confirmed: bool = False
    history: list[tuple[int, np.ndarray]] = field(default_factory=list)
    #: Pixels per frame, smoothed. Without this the box freezes while the
    #: person keeps walking, so a track becomes unmatchable within a few
    #: frames of being missed.
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))

    def update(self, box: np.ndarray, embedding: np.ndarray | None, frame: int) -> None:
        box = np.asarray(box, dtype=np.float32)

        if self.history:
            last_frame, last_box = self.history[-1]
            elapsed = max(1, frame - last_frame)
            observed = (box - last_box) / elapsed
            # Exponential smoothing: one noisy detection should not swing the
            # estimate, but the track must still follow a turn.
            self.velocity = (
                observed.astype(np.float32)
                if self.hits <= 1
                else (0.6 * self.velocity + 0.4 * observed).astype(np.float32)
            )

        self.box = box
        if embedding is not None:
            self.gallery.append(np.asarray(embedding, dtype=np.float32))
        self.hits += 1
        self.time_since_update = 0
        self.history.append((frame, self.box.copy()))

    def predicted_box(self, max_extrapolation: int = 15) -> np.ndarray:
        """Where this track should be now, given its last observation.

        Extrapolation is capped: beyond a point a constant-velocity guess is
        worse than no guess, and appearance should take over.
        """
        steps = min(self.time_since_update, max_extrapolation)
        if steps == 0:
            return self.box
        return (self.box + self.velocity * steps).astype(np.float32)

    def mark_missed(self) -> None:
        self.time_since_update += 1

    def appearance_distance(self, embeddings: np.ndarray) -> np.ndarray:
        """Cosine distance from each embedding to this track's gallery.

        Takes the *minimum* over the gallery: matching any recent appearance is
        enough, which tolerates pose and lighting changes.
        """
        if not self.gallery or len(embeddings) == 0:
            return np.full(len(embeddings), 0.5, dtype=np.float32)
        gallery = np.stack(self.gallery)                 # (G, D), unit norm
        similarity = np.asarray(embeddings) @ gallery.T  # (N, G)
        return (1.0 - similarity.max(axis=1)).astype(np.float32)


class Tracker:
    """Greedy online tracker with appearance-gated association.

    Args:
        max_age: frames a track survives unmatched before deletion.
        min_hits: matches needed before a track is reported (suppresses
            one-frame false positives).
        iou_threshold: minimum overlap for a match to be allowed.
        appearance_weight: 0 = motion only, 1 = appearance only.
        max_appearance_distance: cosine distance above which a pair is
            rejected outright, whatever the overlap says.
        reid_rescue_distance: cosine distance below which appearance alone can
            justify a match that overlap would have rejected. This is what
            lets a track survive an occlusion it moved during.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
        appearance_weight: float = 0.5,
        max_appearance_distance: float = 0.6,
        reid_rescue_distance: float = 0.25,
    ):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.appearance_weight = appearance_weight
        self.max_appearance_distance = max_appearance_distance
        self.reid_rescue_distance = reid_rescue_distance

        self.tracks: list[Track] = []
        self._next_id = 1
        self.frame_count = 0

    def _cost_matrix(self, detections: np.ndarray, embeddings: np.ndarray | None) -> np.ndarray:
        """Blended motion + appearance cost; infeasible pairs get a huge cost."""
        # Compare against where each track should be now, not where it was last
        # seen. A missed frame should not cost the track its identity.
        track_boxes = np.array([t.predicted_box() for t in self.tracks], dtype=np.float32)
        overlap = iou_matrix(track_boxes, detections)
        motion_cost = 1.0 - overlap

        if embeddings is None or self.appearance_weight == 0:
            cost = motion_cost
            appearance = None
        else:
            appearance = np.stack([t.appearance_distance(embeddings) for t in self.tracks])
            weight = self.appearance_weight
            cost = (1 - weight) * motion_cost + weight * appearance

        poor_overlap = overlap < self.iou_threshold

        if appearance is None:
            forbidden = poor_overlap
        else:
            # Overlap and appearance are alternative evidence, not joint
            # requirements. Gating on overlap alone forbids precisely the
            # re-acquisition after occlusion that re-identification exists to
            # perform, so a confident appearance match rescues a low-overlap
            # pair. The rescue threshold is deliberately stricter than the
            # blend threshold: re-acquiring across the frame on weak evidence
            # trades one error for a worse one.
            forbidden = poor_overlap & (appearance > self.reid_rescue_distance)
            # A pair whose appearance is outright wrong is never allowed,
            # however well the boxes happen to line up.
            forbidden |= appearance > self.max_appearance_distance

        cost = np.where(forbidden, 1e5, cost)
        return cost.astype(np.float32)

    def update(
        self, detections: np.ndarray, embeddings: np.ndarray | None = None
    ) -> list[Track]:
        """Advance one frame.

        Args:
            detections: ``(N, 4)`` xyxy boxes for this frame.
            embeddings: ``(N, D)`` unit-norm appearance vectors, or None to
                track on overlap alone.

        Returns:
            The confirmed tracks visible in this frame.
        """
        self.frame_count += 1
        detections = np.asarray(detections, dtype=np.float32).reshape(-1, 4)

        for track in self.tracks:
            track.age += 1

        unmatched_detections = set(range(len(detections)))
        matched_tracks: set[int] = set()

        if self.tracks and len(detections):
            cost = self._cost_matrix(detections, embeddings)
            for track_idx, det_idx in _assign(cost):
                if cost[track_idx, det_idx] >= 1e5:
                    continue  # gated out; leave both unmatched
                embedding = embeddings[det_idx] if embeddings is not None else None
                self.tracks[track_idx].update(detections[det_idx], embedding, self.frame_count)
                matched_tracks.add(track_idx)
                unmatched_detections.discard(det_idx)

        # Every track either matched this frame or missed it -- never both.
        for index, track in enumerate(self.tracks):
            if index in matched_tracks:
                if track.hits >= self.min_hits:
                    track.confirmed = True
            else:
                track.mark_missed()

        for det_idx in sorted(unmatched_detections):
            embedding = embeddings[det_idx] if embeddings is not None else None
            self._spawn(detections[det_idx], embedding)

        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [t for t in self.tracks if t.confirmed and t.time_since_update == 0]

    def _spawn(self, box: np.ndarray, embedding: np.ndarray | None) -> Track:
        track = Track(track_id=self._next_id, box=np.asarray(box, dtype=np.float32))
        if embedding is not None:
            track.gallery.append(np.asarray(embedding, dtype=np.float32))
        track.history.append((self.frame_count, track.box.copy()))
        # min_hits=1 means report immediately, so confirm on creation.
        track.confirmed = self.min_hits <= 1
        self.tracks.append(track)
        self._next_id += 1
        return track

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1
        self.frame_count = 0

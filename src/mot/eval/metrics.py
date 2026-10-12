"""Tracking metrics.

A compact CLEAR-MOT implementation: enough to put real numbers on a slide and
to tell whether a change helped, without pulling in a benchmark toolkit.

For the final report, cross-check against the official ``motmetrics`` package;
these are intended to be readable, not authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..track.tracker import _assign, iou_matrix


@dataclass
class DetectionScore:
    """Per-frame detection quality, ignoring identity."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class TrackingScore:
    """CLEAR-MOT summary over a sequence."""

    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    matched_iou_sum: float

    @property
    def mota(self) -> float:
        """Multi-Object Tracking Accuracy.

        ``1 - (FN + FP + IDSW) / ground_truth_count``. Can go negative when
        errors outnumber the ground truth -- that is by design, not a bug.
        """
        ground_truth = self.true_positives + self.false_negatives
        if ground_truth == 0:
            return 0.0
        errors = self.false_negatives + self.false_positives + self.id_switches
        return 1.0 - errors / ground_truth

    @property
    def motp(self) -> float:
        """Mean IoU of matched pairs -- how tight the boxes are."""
        return self.matched_iou_sum / self.true_positives if self.true_positives else 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "MOTA": self.mota,
            "MOTP": self.motp,
            "TP": self.true_positives,
            "FP": self.false_positives,
            "FN": self.false_negatives,
            "IDSW": self.id_switches,
        }


def match_frame(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_threshold: float = 0.5
) -> list[tuple[int, int, float]]:
    """Best one-to-one matching between ground truth and predictions.

    Returns ``(gt_index, pred_index, iou)`` for pairs above the threshold.
    """
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return []

    overlap = iou_matrix(gt_boxes, pred_boxes)
    matches = []
    for gt_idx, pred_idx in _assign(1.0 - overlap):
        score = float(overlap[gt_idx, pred_idx])
        if score >= iou_threshold:
            matches.append((gt_idx, pred_idx, score))
    return matches


def score_detections(
    gt_boxes: np.ndarray, pred_boxes: np.ndarray, iou_threshold: float = 0.5
) -> DetectionScore:
    """Precision/recall for one frame."""
    matches = match_frame(gt_boxes, pred_boxes, iou_threshold)
    true_positives = len(matches)
    return DetectionScore(
        true_positives=true_positives,
        false_positives=len(pred_boxes) - true_positives,
        false_negatives=len(gt_boxes) - true_positives,
    )


def score_tracking(
    gt_per_frame: dict[int, tuple[np.ndarray, list[int]]],
    pred_per_frame: dict[int, tuple[np.ndarray, list[int]]],
    iou_threshold: float = 0.5,
) -> TrackingScore:
    """CLEAR-MOT over a sequence.

    Both arguments map ``frame -> (boxes, ids)``. An id switch is counted when
    a ground-truth identity that was previously matched to one predicted id is
    now matched to a different one.
    """
    totals = dict(tp=0, fp=0, fn=0, idsw=0)
    iou_sum = 0.0
    last_pred_for_gt: dict[int, int] = {}

    for frame in sorted(set(gt_per_frame) | set(pred_per_frame)):
        gt_boxes, gt_ids = gt_per_frame.get(frame, (np.zeros((0, 4)), []))
        pred_boxes, pred_ids = pred_per_frame.get(frame, (np.zeros((0, 4)), []))
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 4)
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32).reshape(-1, 4)

        matches = match_frame(gt_boxes, pred_boxes, iou_threshold)

        for gt_idx, pred_idx, score in matches:
            iou_sum += score
            gt_id, pred_id = gt_ids[gt_idx], pred_ids[pred_idx]
            previous = last_pred_for_gt.get(gt_id)
            if previous is not None and previous != pred_id:
                totals["idsw"] += 1
            last_pred_for_gt[gt_id] = pred_id

        totals["tp"] += len(matches)
        totals["fp"] += len(pred_boxes) - len(matches)
        totals["fn"] += len(gt_boxes) - len(matches)

    return TrackingScore(
        true_positives=totals["tp"],
        false_positives=totals["fp"],
        false_negatives=totals["fn"],
        id_switches=totals["idsw"],
        matched_iou_sum=iou_sum,
    )

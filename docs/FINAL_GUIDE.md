# Post-midterm to final

What remains, why, and in what order. Ordered by expected effect on the grade.

## What the final actually requires

1. All source code, excluding model weights (too large for Canvas).
2. A tracking video of at least one target pedestrian on **one MOT16 test**
   sequence, as a video or a link.
3. A PDF report, at least 4 pages, letter, font size 11 or smaller:
   abstract and introduction; methodology; challenges encountered and how they
   were solved; self-evaluation and citations.
4. An individual anonymous questionnaire, submitted separately by each member.

Plus a final presentation, time to be announced.

**Note the test split.** The midterm demo uses MOT16-02, which is a *training*
sequence, because it has ground truth to score against. The final video must be
on a **test** sequence, which has no ground truth. Run both: the test sequence
for the deliverable, a train sequence for the numbers.

## Priority 1: fine-tune the detector (Ben)

Everything else is downstream of this. Recall near 0.40 is what caps MOTA, and
it is what stops re-identification from contributing.

- `train_detector` in `mot/detect/train.py` is written and unused.
- Freeze the backbone, train the heads, as the assignment suggests.
- Use `MOT16DetectionDataset` with the augmentation already in place.
- Hold out one sequence so the improvement is measured, not assumed.

**Success looks like** recall above 0.6 on MOT16-02. Re-measure MOTA and the
ablation afterwards; do not assume either moved.

## Priority 2: re-measure, then revisit re-acquisition (Ben, Sanjana)

Once recall improves, the earlier conclusion may invert.

- Re-run the ablation. Re-ID may start helping.
- Re-enable `motion_prediction=True` and sweep `reid_rescue_distance` from 0 up.
  It is implemented and tested, just disabled.
- Sweep `max_age`, `iou_threshold` and `appearance_weight` together.

**Report whatever happens.** If re-acquisition still does not help after
fine-tuning, that is a finding worth writing up, not a failure.

## Priority 3: improve re-identification (Tina)

- Add Market-1501 alongside the MOT16 crops for identity diversity.
  `market1501_identities` is written; the download is in `download_data.py`.
- Try `reid_transform(gaussian_mask=True)`. The Market-1501 paper reports
  +5.64% mAP from background suppression alone; we have it implemented and
  have never measured it.
- Report rank-1 and mAP, not only pair accuracy. The paper argues mAP is the
  more honest measure when there are many ground truths per query.

## Priority 4: the final video (Sanjana)

- Pick a **test** sequence: MOT16-01, 03, 06, 07, 08, 12 or 14.
- `scripts/run_tracking.py --sequence MOT16-06 --max-frames 600`.
- No ground truth there, so choose the target by eye and highlight it.
- Keep it short. Upload to Drive or YouTube unlisted and link it from the
  report, rather than attaching 50 MB to Canvas.

## Priority 5: the report (all)

Four pages minimum, LaTeX recommended. Map sections to owners:

| Section | Owner | Source material |
| --- | --- | --- |
| Abstract, introduction | Ben | Slides 2 to 3 |
| Methodology: detection | Ben | Slides 5 to 6 |
| Methodology: data | Geethika | Slides 7 to 9 |
| Methodology: re-identification | Tina | Slides 10 to 13 |
| Challenges | all | See below |
| Self-evaluation, citations | Sanjana | `docs/RESULTS.md` |

### Challenges worth writing up

These are real, documented, and already have evidence behind them. They are the
most valuable part of the report because they show judgment rather than
activity.

1. **The course slides specify four incompatible crop sizes.** How we chose
   128x64 and justified it from the Market-1501 paper.
2. **Re-identification did not help, and we found out why** rather than
   assuming. The ablation plus the threshold sweep.
3. **We fixed a real bug and it made tracking worse.** Both re-acquisition
   features ship disabled, with measurements.
4. **A metric that moved without the system changing.** The longest-track
   headline went 97.2% to 73.0% on re-run; we replaced it with a
   ground-truth-anchored measure.
5. **MOT16 annotation traps.** 1-indexed coordinates, and classes 7, 8 and 12
   that look like people but must not be scored.
6. **`.gitignore` silently excluded a source directory**, so the package was
   incomplete in every clone and the linter never saw those files.

### Citations you will need

Girshick et al., Fast R-CNN (ICCV 2015). Ren et al., Faster R-CNN (NIPS 2015).
He et al., Mask R-CNN (ICCV 2017). Zheng et al., Scalable Person
Re-identification: A Benchmark (ICCV 2015), for Market-1501 and the 128x64
normalisation. Milan et al., MOT16: A Benchmark for Multi-Object Tracking
(arXiv 2016). Bernardin and Stiefelhagen, Evaluating Multiple Object Tracking
Performance: the CLEAR MOT Metrics (2008). Bewley et al., Simple Online and
Realtime Tracking (ICIP 2016), for the association approach.

## Optional, only if ahead

- Cross-check our CLEAR-MOT against the `motmetrics` package. Ours is readable
  rather than authoritative, and the report should say which was used.
- Mask R-CNN for pixel-level segmentation. The assignment marks it optional.
- A Kalman filter in place of the linear motion model.

## Suggested order

1. Fine-tune the detector, measure recall.
2. Re-run the ablation and the sweeps on the improved detector.
3. Improve re-identification, measure with rank-1 and mAP.
4. Produce the final video on a test sequence.
5. Write the report from `docs/RESULTS.md`, which already holds the numbers.
6. Rehearse the final presentation.
7. Everyone submits the questionnaire individually.

Steps 1 and 2 are the ones that change the result. Step 5 is the one most often
left too late.

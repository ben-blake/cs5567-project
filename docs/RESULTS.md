# Baseline results

First end-to-end run on real data, measured in Colab on a T4.
Sequence MOT16-02, 600 frames, COCO-pretrained detector with **no fine-tuning**,
Siamese re-identification trained for 8 epochs on MOT16-derived crops.

Reproduce with `notebooks/midterm_demo.ipynb`.

## Headline

| Measure | Value |
| --- | --- |
| Target pedestrian tracked | **583 / 600 frames (97.2%)** |
| Throughput | 3.86 FPS (600 frames in 155 s) |
| Re-ID pair accuracy | 0.535 before training, **0.932** after |
| Detector parameters trained | 14.5 M of 41.3 M (35.1%) |

The rubric asks that one target pedestrian be tracked across a sequence without
being lost. At 97.2% on the longest track, the baseline already meets that
before any detector fine-tuning.

## Full sequence metrics

| Metric | Value |
| --- | --- |
| MOTA | 0.232 |
| MOTP | 0.792 |
| True positives | 7,004 |
| False positives | 2,714 |
| False negatives | 10,829 |
| ID switches | 149 |

MOTP of 0.792 means matched boxes are tight. MOTA is held down almost entirely
by false negatives: the detector finds roughly 40% of the annotated people.

## Ablation: does re-identification help?

Same detector and thresholds, appearance weight set to zero.

| Metric | With Re-ID | Overlap only |
| --- | --- | --- |
| MOTA | 0.232 | 0.232 |
| MOTP | 0.792 | 0.792 |
| ID switches | 149 | 148 |

**Re-identification currently changes nothing.** This is worth presenting
honestly rather than hiding, and it has a clear explanation.

## Detection threshold sweep (150 frames)

| Threshold | MOTA | Recall | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- |
| 0.50 | -0.006 | 0.509 | 1,679 | 1,676 | 80 |
| 0.60 | 0.105 | 0.490 | 1,244 | 1,741 | 72 |
| 0.70 | 0.190 | 0.455 | 871 | 1,860 | 34 |
| 0.80 | **0.268** | 0.400 | 435 | 2,049 | 15 |

## What this tells us

1. **Detection recall is the bottleneck, not association.** False negatives
   (10,829) outnumber ID switches (149) by seventy to one. Lowering the
   threshold raises recall only from 0.40 to 0.51 while false positives
   quadruple, so recall is capped by people the COCO model does not find at
   all: small, distant and heavily occluded pedestrians in a crowded scene.
   This is the empirical case for fine-tuning on MOT16, and it is exactly what
   the assignment asks for.

2. **Re-ID has nothing to disambiguate yet.** At 30 fps a pedestrian moves a
   few pixels per frame, so the IoU gate usually leaves at most one feasible
   match and appearance never breaks a tie. Re-ID should start paying off once
   recall improves and once we test it where overlap is ambiguous: longer gaps,
   crossing paths, and re-entry after occlusion.

3. **Raising the threshold trades recall for precision**, and on this sequence
   MOTA prefers precision. That is a property of the metric, not a free win:
   a threshold of 0.80 throws away a fifth of the recall we already have.

## Investigating the ID fragmentation

The first run produced 175 track ids for 54 real people, which looked like a
bug. It partly was. A synthetic test showed that with an **identical**
appearance embedding, a person missed for five frames already lost their
identity:

| gap (frames) | IoU with the stale track box | outcome |
| --- | --- | --- |
| 2 | 0.538 | id kept |
| 5 | 0.250 | **new id** |
| 10 | 0.000 | **new id** |

Two causes. A track's box stayed frozen at its last observation while the
person kept walking, and the IoU gate was applied even when an appearance
embedding was available, so a perfect appearance match was rejected before it
was consulted. Together they forbade exactly the re-acquisition that
re-identification exists to perform.

### Fixing it made the tracking worse

Both defects are real, and repairing them does reduce fragmentation. It also
makes every metric that matters worse.

| Configuration | IDs | MOTA | IDSW | FN | target % |
| --- | --- | --- | --- | --- | --- |
| **Original (shipped default)** | 175 | **0.232** | **149** | 10,829 | **97.2** |
| Motion prediction only | 206 | 0.233 | 153 | 10,856 | 72.8 |
| Motion + strict rescue | 78 | 0.212 | 432 | 10,735 | 91.0 |
| Motion + rescue | 54 | 0.212 | 421 | 10,715 | 86.2 |

The bounded rescue produces exactly 54 ids for 54 people, which looks like a
perfect result and is not one: ID switches almost triple, because the same
small set of ids is passed around between people. **Unique id count is a bad
proxy for tracking quality.**

### Why the repair backfires

At recall near 0.40 the tracker is working from sparse, noisy detections. A
fragmented track costs one broken identity. A wrong re-acquisition merges two
people and corrupts both. With a re-identification network at 0.932 pair
accuracy rather than 1.0, and 600 frames of a crowded scene, the rescue makes
more mistakes than it prevents.

The conservative original also protects the headline result: the longest track
stays clean at 97.2% precisely because the tracker refuses risky matches.

### What we did about it

Both features are implemented, tested, and **off by default**
(`motion_prediction=False`, `reid_rescue_distance=0.0`), so the shipped
configuration reproduces the best measured result. They are worth revisiting
after the detector is fine-tuned: re-acquisition needs good candidates to
choose between, and at 0.40 recall it does not have them.

## Next

- Fine-tune the detector on MOT16 and re-measure. Recall is the number to watch.
- Re-run the ablation afterwards, and construct a harder case for Re-ID
  (occlusion and re-entry) rather than judging it on easy frames.
- Sweep `max_age`, `iou_threshold` and `appearance_weight` together once
  recall improves.

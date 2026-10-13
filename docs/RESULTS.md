# Results

Measured in Colab on a T4. Sequence MOT16-02, 600 frames, COCO-pretrained
detector with **no fine-tuning**, Siamese re-identification trained for 8 epochs
on crops cut from MOT16 ground truth.

Reproduce with `notebooks/midterm_demo.ipynb`.

---

## Headline

| Measure | Value |
| --- | --- |
| Best coverage of one real person by a single ID | **218 / 230 frames (94.8%)** |
| Median coverage across 44 people visible 100+ frames | 22.4% |
| MOTA / MOTP | 0.232 / 0.792 |
| Re-ID pair accuracy | 0.535 before training, **0.932** after |
| Throughput | 4.0 FPS (600 frames in 149 s) |
| Detector parameters trained | 14.5 M of 41.3 M (35.1%) |

The rubric asks that one target pedestrian be tracked across a sequence without
being lost. Ground-truth person 35 is held by a single predicted ID for 94.8% of
the frames they appear in, before any detector fine-tuning. The median tells the
other half of the story: most people are not tracked well, and that is a
detection problem rather than a tracking one.

## Full sequence

| Metric | Value |
| --- | --- |
| MOTA | 0.232 |
| MOTP | 0.792 |
| True positives | 7,004 |
| False positives | 2,713 |
| False negatives | 10,829 |
| ID switches | 151 |
| Unique predicted IDs | 176 |

MOTP 0.792 means the boxes we do produce are tight. MOTA is dragged down almost
entirely by false negatives: the detector finds roughly 40% of the annotated
people.

## Ablation: does re-identification help?

Identical detector and thresholds, appearance weight set to zero.

| Metric | With Re-ID | Overlap only |
| --- | --- | --- |
| MOTA | 0.232 | 0.232 |
| MOTP | 0.792 | 0.792 |
| ID switches | 151 | 148 |
| Unique IDs | 176 | 176 |

**Re-identification currently changes nothing.** The explanation is below, and
it is worth presenting rather than hiding.

## Detection threshold sweep (150 frames)

| Threshold | MOTA | Recall | FP | FN | IDSW |
| --- | --- | --- | --- | --- | --- |
| 0.50 | -0.006 | 0.509 | 1,679 | 1,676 | 80 |
| 0.60 | 0.105 | 0.490 | 1,244 | 1,741 | 72 |
| 0.70 | 0.190 | 0.455 | 871 | 1,860 | 34 |
| 0.80 | **0.268** | 0.400 | 435 | 2,049 | 15 |

Recall barely moves: from 0.40 to 0.51 across the whole range, while false
positives quadruple. The people we miss are not low-confidence detections being
filtered out. They are people the COCO model does not find at all, which are the
small, distant and heavily occluded ones that a crowded street is full of.

---

## Why the ID fragmentation is not the problem it looks like

This is the part worth understanding properly, because the result is
counterintuitive.

### The symptom

The tracker produced **176 IDs for 54 real people**. On the face of it the
tracker is inventing people.

### There really was a bug

A test with a synthetic walking person, where the appearance embedding was held
**identical** throughout, showed the identity being lost after a short gap:

| Frames missed | IoU with the track's last known box | Result |
| --- | --- | --- |
| 2 | 0.538 | ID kept |
| 5 | 0.250 | **new ID** |
| 10 | 0.000 | **new ID** |

Two causes, both real:

1. **The track's box did not move.** When a detection was missed, the track kept
   the box from the last frame it was seen in, while the person kept walking. A
   second later the box was nowhere near them.
2. **Appearance was never consulted.** The tracker required boxes to overlap
   before it would even look at appearance. So a perfect appearance match was
   thrown away because the boxes had drifted apart, which is exactly the
   situation re-identification exists to handle.

So `max_age=30` was misleading. The track stayed in memory for 30 frames but
stopped being matchable after about 4.

### Fixing it made the tracking worse

| Configuration | IDs | MOTA | IDSW |
| --- | --- | --- | --- |
| **Original (what we ship)** | 176 | **0.232** | **151** |
| Motion prediction only | 206 | 0.233 | 153 |
| Motion + strict re-acquisition | 78 | 0.212 | 432 |
| Motion + re-acquisition | 54 | 0.212 | 421 |

Read the bottom row carefully. **54 IDs for 54 real people** looks like a perfect
result. It is not: ID switches almost tripled. The tracker is not giving each
person their own ID. It is handing the same small set of IDs around between
different people, like a baton.

### Why

A tracker can be wrong in two different ways, and they do not cost the same.

- **Splitting.** One person gets several IDs over time. You lose the thread, and
  you pick up a new one.
- **Merging.** One ID gets attached to several people. You do not lose the
  thread. You keep following it confidently, onto the wrong person, and now two
  people's trajectories are wrong instead of one being interrupted.

The fix made the tracker more willing to say *"you look like someone I lost,
welcome back."* That is the right instinct when you can trust two things: that
most people who disappear really do come back, and that you can reliably tell
people apart. Neither holds here.

- The detector finds about 40% of people. So most of the time a person
  "reappears", they are someone who was never tracked in the first place, not
  someone returning.
- The re-identification network is 93% accurate on pairs, not 100%. Across 600
  frames of a crowded street, 7% wrong is a lot of wrong.

Put together, an eager tracker spends most of its re-acquisitions merging
strangers. Splitting is ugly and cheap. Merging is tidy-looking and expensive.
At this recall, ugly and cheap wins.

### What we did

Both features are implemented and tested, and both are **off by default**
(`motion_prediction=False`, `reid_rescue_distance=0.0`), so the shipped
configuration reproduces the best measured result.

They are worth revisiting after the detector is fine-tuned. Re-acquisition needs
good candidates to choose between, and at 0.40 recall it does not have them.

### The lesson

**Unique ID count is a bad measure of tracking quality.** It was the number that
made the original look broken, and the number that made the broken version look
fixed. ID switches and MOTA both say the opposite.

---

## A number we had to withdraw

An earlier version of this document reported **97.2%** as the headline, taken
from the longest-lived predicted track as a fraction of the clip.

That number is not reproducible. Re-running an identical configuration produced
**73.0%**, while MOTA, MOTP, ID switches and unique ID count all reproduced
exactly. When tracks fragment, which fragment happens to be longest shifts
between runs, and the headline moves with it even though the tracking has not
changed.

It was also the wrong question. A long-lived track is not necessarily a
correctly tracked person; it could be one ID passed between several people.

`mot.eval.metrics.target_coverage` replaces it by anchoring to the ground truth:
for each real person, which single predicted ID covers the most of their frames.
That is the question the rubric asks, and it holds still across runs.

**Do not quote 97.2% anywhere.** The stable number is 94.8%.

---

## What to do next

1. **Fine-tune the detector.** Recall near 0.40 caps everything else.
2. **Re-measure the ablation afterwards.** The conclusion that re-identification
   does not help is a conclusion about this recall level, not a permanent one.
3. **Then re-test re-acquisition**, which is already implemented and disabled.

Details and ownership in [FINAL_GUIDE.md](FINAL_GUIDE.md).

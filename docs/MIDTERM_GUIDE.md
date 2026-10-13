# Midterm presentation guide

Everything needed to deliver the 15-minute midterm. Read this, then rehearse.

## What is actually required

> 15 minute presentation for each group. The topic should cover each group
> members' workload, timeline, current process and understanding on the
> methods. You should be familiar with the aim of the project, models, and try
> to play around with the codes.

The midterm is **the presentation only**. Source code, the tracking video and
the report are *final* deliverables. Do not spend this week on them.

| Requirement | Where it is covered |
| --- | --- |
| Each member's workload | Slide 1, and each person presents their own module |
| Timeline | Slide 19 |
| Current process | Results and ablation slides |
| Understanding of methods | Slides 3 to 13 |
| Aim and models | Slides 2 to 6 |
| Played with the code | Measured results, ablation, threshold sweep |

## The 15 minutes

Four people, roughly 4 minutes each, then questions. Suggested order follows the
data through the pipeline, so each person builds on the last.

| Who | Slides | Covers |
| --- | --- | --- |
| **Ben** | 1 to 6 | Team, problem, why it matters, architecture, Faster R-CNN, association |
| **Geethika** | 7 to 9 | MOT16, the two annotation traps, augmentation |
| **Tina** | 10 to 13 | Re-identification, Siamese design, contrastive loss, the crop-size contradiction |
| **Sanjana** | 14 to 17 | Metrics, results, the ablation, demo |
| **Ben** | 18 to 19 | The bug we found and turned off, timeline |

One person drives the laptop for the whole talk. Swapping machines costs a
minute and invites an adapter problem.

## The three things worth saying

Lead with these. They are what separates a report of activity from a report of
understanding.

1. **The course slides contradict themselves on crop size.** Four different
   sizes, including reference code implying a fifth. We chose 128x64 and
   justified it from the Market-1501 paper, which normalises to exactly that.
2. **Re-identification currently changes nothing, and we know why.** False
   negatives outnumber ID switches seventy to one. Detection recall near 0.40
   is the bottleneck, so association has nothing left to disambiguate.
3. **We found a real bug, fixed it, and the fix made tracking worse.** Both
   re-acquisition features ship disabled. 54 ids for 54 people looks perfect
   and is not.

## Numbers you must be able to defend

| Claim | Value | If asked how |
| --- | --- | --- |
| Target coverage | 94.8% of one real person's frames | `target_coverage` against ground truth |
| MOTA / MOTP | 0.232 / 0.792 | CLEAR-MOT on MOT16-02, 600 frames |
| Recall | about 0.40 | 7,004 TP against 10,829 FN |
| Trainable parameters | 14.5 M of 41.3 M | Backbone frozen, heads trained |
| Re-ID accuracy | 0.535 to 0.932 | 8 epochs, MOT16 crops, held-out pairs |
| Tests | 56, green in CI | `uv run pytest` |

**Do not quote 97.2%.** An earlier version of the deck used it; it came from
the longest-lived track and moved to 73.0% on re-run with nothing else
changing. 94.8% is the stable, ground-truth-anchored number.

## The demo

- `outputs/MOT16-02_target_clip.mp4`, 8 seconds, highlights the tracked person.
- `outputs/MOT16-02_demo.mp4`, the full 600 frames, if there is time.

**Download both to the presenting laptop beforehand.** They live in a Colab
runtime that expires. Do not stream from Drive and do not run Colab live unless
everything else has already gone well.

## Rehearsal

Timed, standing, no interruptions, at least once. Nobody has ever estimated a
15-minute talk correctly in their head. Expect to cut two or three slides.
Budget 90 seconds for the demo and 2 minutes for questions.

## Questions to expect

- *Why Faster R-CNN and not YOLO?* Accuracy on small objects matters more than
  speed for offline tracking, and distant pedestrians are small. We are not
  real-time by design, at 4 FPS.
- *Why a Siamese network instead of classifying identities?* People enter the
  scene constantly; a classifier would need retraining each time.
- *Your MOTA is low.* Yes, and we know the cause: recall near 0.40. Lowering
  the detection threshold raises recall only to 0.51 while false positives
  quadruple, so the misses are people the COCO model never finds. That is the
  case for fine-tuning.
- *Why is re-identification not helping?* It has nothing to disambiguate yet.
  Shown by the ablation and the threshold sweep.
- *Did you try to fix the fragmentation?* Yes, and it made things worse.
  Slide 18 has the numbers.

## Before you walk in

- [ ] Deck PDF downloaded (Actions, Slides workflow, artifact)
- [ ] Both videos downloaded to the presenting laptop
- [ ] Rehearsed once, timed
- [ ] Everyone can present their own module without reading the slides
- [ ] Laptop tested with the room's adapter if possible

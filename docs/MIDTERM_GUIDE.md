# Midterm guide

Everything needed to deliver the 15-minute midterm presentation. Self-contained:
roles, the speaking split, the numbers, the demo, and a rehearsal plan.

---

## What is required

> 15 minute presentation for each group. The topic should cover each group
> members' workload, timeline, current process and understanding on the
> methods. You should be familiar with the aim of the project, models, and try
> to play around with the codes.

The midterm is **the presentation only.** Source code, the tracking video and
the report are *final* deliverables. Do not spend this week on them.

| Requirement | Where it is covered |
| --- | --- |
| Each member's workload | Slide 1, and each person presents their own module |
| Timeline | Slide 19 |
| Current process | Results and ablation slides |
| Understanding of methods | Slides 3 to 13 |
| Aim and models | Slides 2 to 6 |
| Played with the code | Measured results, ablation, threshold sweep |

---

## Who owns what

Each person owns one module end to end: its code, its slides, and its section of
the final report. Owning a module means understanding it well enough to present
it and answer questions, not that you wrote every line.

### Ben Blake, detection and tracking

Owns `src/mot/detect/` (Faster R-CNN, fine-tuning, checkpoints),
`src/mot/track/` (Hungarian association, cost blending, track lifecycle),
`src/mot/pipeline.py`, and the repository and CI.

### Geethika Padamati, data pipeline

Owns `src/mot/data/mot16.py` (ground-truth and detection parsing),
`src/mot/data/datasets.py` (PyTorch datasets, crop export),
`src/mot/data/augment.py`, and dataset acquisition via
`scripts/download_data.py`.

### Tina Nguyen, re-identification

Owns `src/mot/reid/siamese.py` (network, contrastive loss, embeddings),
`src/mot/reid/train.py` (training loop, pair accuracy), and the training runs.

### Sanjana Aileni, evaluation and demo

Owns `src/mot/eval/metrics.py` (MOTA, MOTP, ID switches, target coverage),
`src/mot/viz/draw.py` (overlays, highlighting, video encoding), and the demo
recording and result figures.

---

## The 15 minutes

Four people, roughly 4 minutes each, then questions. The order follows the data
through the pipeline, so each person builds on the last.

| Who | Slides | Covers |
| --- | --- | --- |
| **Ben** | 1 to 6 | Team, problem, why it matters, architecture, Faster R-CNN, association |
| **Geethika** | 7 to 9 | MOT16, the two annotation traps, augmentation |
| **Tina** | 10 to 13 | Re-identification, Siamese design, contrastive loss, the crop-size contradiction |
| **Sanjana** | 14 to 17 | Metrics, results, the ablation, demo |
| **Ben** | 18 to 19 | The bug we found and turned off, timeline |

**One person drives the laptop for the whole talk.** Swapping machines costs a
minute and invites an adapter problem.

---

## The three things worth leading with

These separate a report of activity from a report of understanding, which is
what "understanding on the methods" is asking for.

### 1. The course slides contradict themselves on crop size

Training is specified at 16x16, inference at 24x24, retrieved crops at 30x16,
and the reference `Siamese_Network` code needs roughly 102x102. No single size
satisfies all four. We chose **128x64**, which is the resolution Market-1501
normalises its own images to, and replaced the hardcoded flatten with
`AdaptiveAvgPool2d` so input size is no longer baked into the architecture.
*Tina presents this.*

### 2. Re-identification currently changes nothing, and we know why

False negatives outnumber ID switches seventy to one. Detection recall near 0.40
is the bottleneck, so association has nothing left to disambiguate. Shown by the
ablation and the threshold sweep. *Sanjana presents this.*

### 3. We found a real bug, fixed it, and the fix made tracking worse

176 IDs for 54 real people looked broken, and there genuinely was a bug: a track
stopped being matchable about four frames after a missed detection. Fixing it
produced exactly 54 IDs for 54 people, which looks perfect and is not: ID
switches almost tripled, because the same IDs were being passed between
different people. Both features ship disabled. *Ben presents this.*

The point to land: **splitting a person into several IDs is cheap, merging
several people into one ID is expensive.** At 40% recall, most apparent
reappearances are strangers, so an eager tracker merges rather than recovers.

---

## Numbers you must be able to defend

| Claim | Value | If asked how |
| --- | --- | --- |
| Target coverage | 94.8% of one real person's frames | `target_coverage` against ground truth |
| MOTA / MOTP | 0.232 / 0.792 | CLEAR-MOT on MOT16-02, 600 frames |
| Recall | about 0.40 | 7,004 TP against 10,829 FN |
| Trainable parameters | 14.5 M of 41.3 M | Backbone frozen, heads trained |
| Re-ID accuracy | 0.535 to 0.932 | 8 epochs, MOT16 crops, held-out pairs |
| Tests | 56, green in CI | `uv run pytest` |

**Do not quote 97.2%.** An earlier version of the deck used it. It came from the
longest-lived track and moved to 73.0% on re-run with nothing else changing.
94.8% is the stable, ground-truth-anchored number. See
[RESULTS.md](RESULTS.md#a-number-we-had-to-withdraw).

---

## Figures and the demo

`notebooks/midterm_demo.ipynb` runs top to bottom in about 25 minutes on a T4
and writes everything the deck needs into `outputs/`.

### Getting the figures into the slides

The deck has five figure slots. Until the files are present each one renders as
a dashed placeholder naming the file it wants, so the deck always builds and
missing figures are obvious.

| File | Slide |
| --- | --- |
| `fig1_detection.png` | What the detector actually sees |
| `fig2_reid_training.png` | Training the similarity model |
| `fig3_reid_similarity.png` | Training the similarity model |
| `fig4_coverage.png` | Coverage across every person |
| `fig5_target_strip.png` | Demo |

Download them from Colab, drop them in `slides/figures/`, then:

```bash
git add -f slides/figures/*.png    # PNGs are gitignored by default
git commit -m "docs: add presentation figures"
```

The next CI build produces a deck with the real figures in place.

### The video

Two files, both produced by the same notebook run:

- `demo_target_clip.mp4`, 8 seconds, target in colour and everyone else in
  grey. **This is the one to show.**
- `demo_full.mp4`, the full 600 frames, if there is time.

**Download both to the presenting laptop beforehand.** They live in a Colab
runtime that expires. Do not stream from Drive, and do not run Colab live in the
room unless everything else has already gone well.

Beamer cannot embed video reliably, so play it outside the deck: open the MP4 in
VLC or QuickTime and alt-tab to it at the `Demo` slide, which exists as the cue
to switch.

### Regenerating it

Open `notebooks/midterm_demo.ipynb` in Colab, **Runtime → Change runtime type →
T4 GPU**, then Run all. About 20 minutes, mostly the 1.9 GB MOT16 download. The
tracking cell has `MAX_FRAMES = 600`.

---

## Getting the slides

You do not need LaTeX installed. Every push builds the deck:

**Actions tab → latest "Slides" run → Artifacts → `midterm-slides`.**

---

## Rehearsal

**Timed, standing, no interruptions, at least once.** Nobody has ever estimated
a 15-minute talk correctly in their head. Expect to cut two or three slides.
Budget 90 seconds for the demo and 2 minutes for questions.

Each person should be able to present their module without reading the slides.

---

## Questions to expect

- *Why Faster R-CNN and not YOLO?* Accuracy on small objects matters more than
  speed for offline tracking, and distant pedestrians are small. We are not
  real-time by design, at 4 FPS.
- *Why a Siamese network instead of classifying identities?* People enter the
  scene constantly, and a classifier would need retraining each time. A
  similarity model never does.
- *What happens when two people cross?* Overlap alone is ambiguous, which is why
  the cost blends appearance. We have an ablation.
- *Your MOTA is low.* Yes, and we know why: recall near 0.40. Lowering the
  detection threshold raises recall only to 0.51 while false positives
  quadruple, so the misses are people the COCO model never finds. That is the
  case for fine-tuning.
- *Why is re-identification not helping?* It has nothing to disambiguate yet.
  Shown by the ablation and the threshold sweep.
- *Did you try to fix the fragmentation?* Yes, and it made things worse. Slide
  18 has the numbers.
- *How do you know it works?* MOTA, MOTP, ID switches and target coverage
  against ground truth, plus 56 unit tests in CI.
- *What is your biggest risk?* Detection recall. Everything else is downstream.

---

## Before you walk in

- [ ] Deck PDF downloaded from the Slides artifact
- [ ] Both videos downloaded to the presenting laptop
- [ ] Rehearsed once, timed
- [ ] Everyone can present their own module without reading the slides
- [ ] One person confirmed as driving the laptop
- [ ] Laptop tested with the room's adapter if possible

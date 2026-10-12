# Group 4 — Midterm Planning Meeting

A 45-minute agenda. The goal is that everyone leaves knowing their 4 minutes of
the presentation and what they owe before the next meeting.

---

## Before the meeting

Ask everyone to do these two things (5 minutes of their time):

1. Send Ben their **GitHub username** so they can be added to the repo.
2. Skim [ROLES.md](ROLES.md) and their own module's source file.

---

## 1. Where we are (5 min — Ben)

The honest status: **the pipeline works end to end today.**

- Detection, Re-ID, tracking, metrics, visualization all implemented
- 36 tests passing, CI green, slides build automatically
- The baseline uses the COCO-pretrained detector, so **we can produce a tracked
  video without training anything** — training makes it better, not possible

What is *not* done: nothing is tuned, the detector is not fine-tuned, and the
Re-ID network has only been trained on a smoke test. That is the work ahead.

> Say plainly: the scaffolding is done so that everyone starts from something
> that runs. Owning a module means understanding it well enough to present it
> and answer questions — not that you wrote every line.

## 2. Role assignments (10 min — everyone)

Walk the table. For each person, confirm they accept the role and understand
the module.

| Member | Role | Presents |
| --- | --- | --- |
| Ben | Detection & tracking | Aim, architecture, Faster R-CNN, association |
| Geethika | Data pipeline | MOT16 format, annotation traps, augmentation |
| Tina | Re-identification | Why Re-ID, Siamese design, contrastive loss |
| Sanjana | Evaluation & demo | Metrics, results, demo video |

**Ask each person directly: "does that work for you?"** Swap now rather than
discover a mismatch the night before.

## 3. The 15 minutes (10 min — everyone)

Four people, ~4 minutes each, plus ~30 seconds of handover. The deck is already
split this way — 17 slides in `slides/midterm.tex`.

Decisions to make in the room:

- **Speaking order.** Suggested: Ben (problem + system) → Geethika (data) →
  Tina (Re-ID) → Sanjana (results + demo) → Ben (timeline, 30s).
  This follows the data through the pipeline, so each person builds on the last.
- **Who clicks.** One person drives the laptop for the whole talk. Swapping
  machines mid-presentation wastes a minute and invites an adapter problem.
- **Where the demo goes.** Currently near the end, after results. Alternative is
  opening with it to grab attention. Decide and stick to it.

## 4. The demo (5 min — Sanjana + Ben)

Agree on:

- **Which sequence.** MOT16-02 by default. It has ground truth, so we can show
  real metrics rather than just a video.
- **Clip length.** 30 seconds maximum.
- **What it shows.** Ideally the target surviving an occlusion — that is what
  the Re-ID component is *for* and what the rubric grades.
- **Backup.** The video file is downloaded to a laptop before the talk. We do
  not stream from Drive and we do not run Colab live in the room unless
  everything else has already gone well.

Procedure is in [RUNBOOK.md](RUNBOOK.md).

## 5. Known problems worth presenting (5 min — Tina, Geethika)

Discussing real problems we hit is better presentation material than claiming
everything worked. Two are ready to use:

- **The course slides contradict themselves on Re-ID crop size** (16x16, 24x24,
  30x16, and reference code implying ~102x102). We chose 128x64 and justified it
  from the Market-1501 paper, which normalises to exactly that. *Tina presents
  this.*
- **MOT16 annotations have two traps**: coordinates are 1-indexed, and classes
  7/8/12 are people who must not be scored. Both would silently wreck metrics.
  *Geethika presents this.*

> This is the "understanding on the methods" the assignment asks for. Finding
> and resolving a contradiction demonstrates more than a working demo does.

## 6. Work before next meeting (5 min — everyone)

| Owner | Task | Done when |
| --- | --- | --- |
| Everyone | Send GitHub username | Added to repo |
| Everyone | Read your module, write your ~4 min | You can present it without the slides |
| Sanjana | Run the notebook, produce a demo clip | MP4 downloaded to a laptop |
| Tina | Train Re-ID properly, save the training curve | Figure in `outputs/` |
| Ben | Fine-tune the detector | Checkpoint on Drive |
| Geethika | Verify augmentation on real crops | Before/after figure |

## 7. Second meeting: full rehearsal

**Timed, standing, no interruptions.** Nobody has rehearsed a 15-minute talk
accurately in their head — it always runs long. Then cut.

---

## Questions to expect from the instructor

Have an answer ready for each:

- *Why Faster R-CNN and not YOLO?* Accuracy on small objects matters more than
  speed for offline tracking; distant pedestrians are small. We are not
  real-time by design.
- *Why a Siamese network instead of classifying identities?* New people enter
  the scene constantly; a classifier would need retraining each time. A
  similarity model never does.
- *What happens when two people cross?* Overlap alone is ambiguous, which is
  exactly why the cost blends appearance. We have an ablation showing it.
- *How do you know it works?* MOTA, MOTP and ID switches against ground truth
  on a train sequence, plus 36 unit tests.
- *What is your biggest risk?* Re-ID quality under heavy occlusion. Mitigation
  is the appearance gallery per track plus `max_age` tolerance.

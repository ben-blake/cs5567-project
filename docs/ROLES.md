# Roles and Workload — Group 4

Four roles, each owning a module of the pipeline end to end: its code, its
slice of the presentation, and its section of the final report. Interfaces
between modules are narrow and already defined in code, so work proceeds in
parallel without blocking.

Each member presents for roughly 4 minutes of the 15-minute midterm slot.

---

## Ben Blake — Detection and Tracking

The detector and the association algorithm that turns
per-frame boxes into persistent identities.

**Owns**

- `src/mot/detect/` — Faster R-CNN setup, few-shot fine-tuning, checkpoints
- `src/mot/track/` — Hungarian association, cost blending, track lifecycle
- `src/mot/pipeline.py` — end-to-end integration
- Repository, CI, and code review

**Presents (~4 min)** — the aim of the project, system architecture, how Faster
R-CNN produces proposals and boxes, and how the tracker decides which detection
belongs to which person.

## Geethika Padamati — Data pipeline

Nothing trains until the data is correct, and the annotation format has sharp
edges (1-indexed coordinates, classes that look like people but are not scored).

**Owns**

- `src/mot/data/mot16.py` — ground-truth and detection parsing
- `src/mot/data/datasets.py` — PyTorch datasets, crop export
- `src/mot/data/augment.py` — augmentation for detection and Re-ID
- Dataset acquisition and the `scripts/download_data.py` path

**Presents (~4 min)** — the MOT16 dataset, what each annotation column means,
which boxes are scored and which are ignored, and what augmentation we apply
and why (no vertical flips: pedestrians are always upright).

## Tina Nguyen — Re-identification

The component that makes tracking survive occlusion: deciding whether two crops
from different frames show the same person.

**Owns**

- `src/mot/reid/siamese.py` — network, contrastive loss, embedding utilities
- `src/mot/reid/train.py` — training loop, pair accuracy
- Market-1501 and MOT16-crop training runs

**Presents (~4 min)** — why identity cannot come from detection alone, the
Siamese design and why it beats classifying a fixed identity set, contrastive
loss, training curves, and the crop-size inconsistency in the course slides and
how we resolved it.

## Sanjana Aileni — Evaluation, visualization and demo

Turns a running system into evidence, and owns what the class actually sees.

**Owns**

- `src/mot/eval/metrics.py` — MOTA, MOTP, ID switches, precision/recall
- `src/mot/viz/draw.py` — overlays, target highlighting, video encoding
- Demo recording, result tables and figures for the slides

**Presents (~4 min)** — how multi-object tracking is measured, what each metric
rewards and penalises, our current numbers, and the demo video.

---

## Timeline

| Phase         | Work                                                     | Owner          |
| ------------- | -------------------------------------------------------- | -------------- |
| Done          | Repo, conventions, CI, full pipeline skeleton with tests | Ben            |
| Done          | MOT16 parsing, datasets, augmentation                    | Ben → Geethika |
| Done          | Siamese network, contrastive loss, training loop         | Ben → Tina     |
| Done          | Metrics, visualization, video encoding                   | Ben → Sanjana  |
| Midterm week  | Baseline tracking run on MOT16-02, demo recorded         | Sanjana        |
| Midterm week  | Slides rehearsed, 15 min timed                           | All            |
| After midterm | Detector fine-tuning, Re-ID trained on real data         | Ben, Tina      |
| After midterm | Metric sweep, ablations, final video                     | Sanjana        |
| Final         | Report sections, final presentation                      | All            |

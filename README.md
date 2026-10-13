# cs5567-project

Video object tracking on the MOT16 dataset — a Faster R-CNN detector paired with a Siamese
re-identification model to track a single target pedestrian across frames.

Course project for COMP-SCI 5567 (Deep Learning), University of Missouri–Kansas City.

## Overview

The system performs offline multi-object tracking in two stages:

1. **Detection** — a torchvision Faster R-CNN, pre-trained and few-shot fine-tuned on the
   MOT16 training data, produces bounding boxes per frame.
2. **Re-identification** — a Siamese similarity network compares detected crops across
   frames and assigns each object a stable ID.

The output is a video with bounding boxes and consistent IDs drawn over the source frames.

## Pipeline

```
MOT16 ground truth  ->  parse + augment
                            |
                            v
              Faster R-CNN few-shot fine-tuning
                            |
                            v
              Siamese network training (Re-ID)
                            |
                            v
          detection + Re-ID per frame  ->  tracked video
```

## Repository structure

```
src/mot/        the package
  data/         MOT16 parsing, datasets, augmentation
  detect/       Faster R-CNN, fine-tuning
  reid/         Siamese network, contrastive loss
  track/        Hungarian association, track lifecycle
  eval/         MOTA, MOTP, ID switches, target coverage
  viz/          overlays and video encoding
  pipeline.py   end to end: frames in, tracked video out
notebooks/      Colab demo, runs the whole pipeline
scripts/        dataset download, command-line tracking
slides/         Beamer deck, built by CI
docs/           guides and measured results
tests/          56 tests
```

## Current results

On MOT16-02, 600 frames, with the COCO-pretrained detector and **no
fine-tuning**: one real pedestrian is held by a single ID for **94.8%** of their
frames, MOTA 0.232, MOTP 0.792. Detection recall near 0.40 is the bottleneck.
Full numbers and analysis in [docs/RESULTS.md](docs/RESULTS.md).

## Getting started

Development targets **Google Colab** (4–6 GB VRAM is sufficient for fine-tuning).

Datasets used:

- [MOT16](https://motchallenge.net/data/MOT16/) — training and test sequences
- [Market-1501](https://zheng-lab-anu.github.io/Project/project_reid.html) — Re-ID similarity training
- MOT16-derived crops — cut from MOT16 ground truth by `scripts/download_data.py --mot16-crops`,
  giving in-domain Re-ID training data with no download

```bash
python scripts/download_data.py --mot16          # ~1.9 GB
python scripts/download_data.py --market1501     # ~146 MB, via Google Drive
python scripts/download_data.py --mot16-crops    # no download
```

## Deliverables

- [ ] Source code (model weights excluded — too large for Canvas)
- [ ] Tracking video of at least one target pedestrian on MOT16 test data
- [ ] PDF report (≥ 4 pages, letter, font size ≤ 11)
- [ ] Final presentation

See [docs/reference/OVERVIEW.md](docs/reference/OVERVIEW.md) for the full requirements and grading rubric.

## Documentation

| Document                                                 | Contents                                                                  |
| -------------------------------------------------------- | ------------------------------------------------------------------------- |
| [docs/MIDTERM_GUIDE.md](docs/MIDTERM_GUIDE.md)           | Roles, speaking split, numbers to defend, demo, rehearsal                 |
| [docs/FINAL_GUIDE.md](docs/FINAL_GUIDE.md)               | Work remaining in priority order, plus how to run everything              |
| [docs/RESULTS.md](docs/RESULTS.md)                       | Measured results, the Re-ID ablation, and why the tracker fix backfired   |
| [docs/reference/OVERVIEW.md](docs/reference/OVERVIEW.md) | Project brief, assignment requirements, implementation guidance, rubric   |
| [docs/reference/RCNN.md](docs/reference/RCNN.md)         | Reference deck: R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN, and YOLO  |

## Team

**Group 4**

| Name              | Role                       | Owns                                      | Contact             |
| ----------------- | -------------------------- | ----------------------------------------- | ------------------- |
| Ben Blake         | Detection & tracking       | `detect/`, `track/`, `pipeline.py`, CI    | bebpph@umsystem.edu |
| Geethika Padamati | Data pipeline              | `data/` — parsing, datasets, augmentation | gphxp@umsystem.edu  |
| Tina Nguyen       | Re-identification          | `reid/` — Siamese network, training       | qpnh58@umsystem.edu |
| Sanjana Aileni    | Evaluation & visualization | `eval/`, `viz/`, demo video               | sa85m@umsystem.edu  |

See [docs/MIDTERM_GUIDE.md](docs/MIDTERM_GUIDE.md) for responsibilities and the presentation split.

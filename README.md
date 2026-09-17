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
docs/
  reference/    project brief, requirements, and lecture reference material
```

## Getting started

Development targets **Google Colab** (4–6 GB VRAM is sufficient for fine-tuning).

Datasets used:

- [MOT16](https://motchallenge.net/data/MOT16/) — training and test sequences
- [Market-1501](https://paperswithcode.com/dataset/market-1501) — Re-ID similarity training

## Deliverables

- [ ] Source code (model weights excluded — too large for Canvas)
- [ ] Tracking video of at least one target pedestrian on MOT16 test data
- [ ] PDF report (≥ 4 pages, letter, font size ≤ 11)
- [ ] Final presentation

See [docs/reference/OVERVIEW.md](docs/reference/OVERVIEW.md) for the full requirements and grading rubric.

## Documentation

| Document                                                 | Contents                                                                 |
| -------------------------------------------------------- | ------------------------------------------------------------------------ |
| [docs/reference/OVERVIEW.md](docs/reference/OVERVIEW.md) | Project brief, assignment requirements, implementation guidance, rubric  |
| [docs/reference/RCNN.md](docs/reference/RCNN.md)         | Reference deck: R-CNN → Fast R-CNN → Faster R-CNN → Mask R-CNN, and YOLO |

## Team

**Group 4**

| Name              | Contact             |
| ----------------- | ------------------- |
| Ben Blake         | bebpph@umsystem.edu |
| Geethika Padamati | gphxp@umsystem.edu  |
| Tina Nguyen       | qpnh58@umsystem.edu |
| Sanjana Aileni    | sa85m@umsystem.edu  |

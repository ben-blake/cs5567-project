# Final guide

What remains between the midterm and the final, in priority order, plus how to
run everything.

---

## What the final requires

1. All source code, excluding model weights (too large for Canvas).
2. A tracking video of at least one target pedestrian on **one MOT16 test**
   sequence, as a video or a link.
3. A PDF report, at least 4 pages, letter, font size 11 or smaller:
   abstract and introduction; methodology; challenges encountered and how they
   were solved; self-evaluation and citations.
4. An individual anonymous questionnaire, submitted separately by each member.

Plus a final presentation, time to be announced.

> **The test split matters.** The midterm demo uses MOT16-02, a *training*
> sequence, because it has ground truth to score against. The final video must
> be on a **test** sequence, which has none. Run both: a test sequence for the
> deliverable, a train sequence for the numbers.

---

## Priority 1: fine-tune the detector (Ben)

Everything else is downstream of this. Recall near 0.40 is what caps MOTA and
what stops re-identification from contributing.

- `train_detector` in `src/mot/detect/train.py` is written and unused.
- Freeze the backbone and train the heads, as the assignment suggests.
- Use `MOT16DetectionDataset` with the augmentation already in place.
- Hold out one sequence so the improvement is measured, not assumed.

**Success looks like** recall above 0.6 on MOT16-02. Re-measure MOTA and the
ablation afterwards rather than assuming either moved.

## Priority 2: re-measure, then revisit re-acquisition (Ben, Sanjana)

Once recall improves, the midterm conclusion may invert.

- Re-run the ablation. Re-identification may start helping.
- Re-enable `motion_prediction=True` and sweep `reid_rescue_distance` upward
  from 0. Both are implemented and tested, just disabled. The reasoning is in
  [RESULTS.md](RESULTS.md#why-the-id-fragmentation-is-not-the-problem-it-looks-like).
- Sweep `max_age`, `iou_threshold` and `appearance_weight` together.

**Report whatever happens.** If re-acquisition still does not help after
fine-tuning, that is a finding worth writing up, not a failure.

## Priority 3: improve re-identification (Tina)

- Add Market-1501 alongside the MOT16 crops for identity diversity.
  `market1501_identities` is written, and the download is in
  `scripts/download_data.py`.
- Try `reid_transform(gaussian_mask=True)`. The Market-1501 paper reports
  +5.64% mAP from background suppression alone. It is implemented and has never
  been measured here.
- Report rank-1 and mAP, not only pair accuracy. The paper argues mAP is the
  more honest measure when there are many ground truths per query.

## Priority 4: the final video (Sanjana)

- Pick a **test** sequence: MOT16-01, 03, 06, 07, 08, 12 or 14.
- `python scripts/run_tracking.py --sequence MOT16-06 --max-frames 600`
- No ground truth there, so choose the target by eye and highlight it.
- Upload to Drive or YouTube unlisted and link from the report, rather than
  attaching 50 MB to Canvas.

## Priority 5: the report (all)

Four pages minimum, LaTeX recommended (Overleaf is free).

| Section | Owner | Source |
| --- | --- | --- |
| Abstract, introduction | Ben | Slides 2 to 3 |
| Methodology: detection | Ben | Slides 5 to 6 |
| Methodology: data | Geethika | Slides 7 to 9 |
| Methodology: re-identification | Tina | Slides 10 to 13 |
| Challenges | all | Below |
| Self-evaluation, citations | Sanjana | [RESULTS.md](RESULTS.md) |

### Challenges worth writing up

These are real, documented, and already have evidence behind them. They are the
most valuable part of the report because they show judgment rather than
activity.

1. **The course slides specify four incompatible crop sizes.** How we chose
   128x64 and justified it from the Market-1501 paper.
2. **Re-identification did not help, and we found out why** rather than
   assuming. The ablation plus the threshold sweep.
3. **We fixed a real bug and it made tracking worse.** Splitting is cheap,
   merging is expensive, and at 40% recall an eager tracker merges strangers.
   Both features ship disabled, with measurements.
4. **A metric that moved without the system changing.** The longest-track
   headline went 97.2% to 73.0% on re-run; we replaced it with a
   ground-truth-anchored measure.
5. **MOT16 annotation traps.** 1-indexed coordinates, and classes 7, 8 and 12
   that look like people but must not be scored.
6. **`.gitignore` silently excluded a source directory**, so the package was
   incomplete in every clone and the linter never saw those files.

### Citations

Girshick, Fast R-CNN (ICCV 2015). Ren et al., Faster R-CNN (NIPS 2015). He et
al., Mask R-CNN (ICCV 2017). Zheng et al., Scalable Person Re-identification: A
Benchmark (ICCV 2015), for Market-1501 and the 128x64 normalisation. Milan et
al., MOT16: A Benchmark for Multi-Object Tracking (arXiv 2016). Bernardin and
Stiefelhagen, Evaluating Multiple Object Tracking Performance: the CLEAR MOT
Metrics (2008). Bewley et al., Simple Online and Realtime Tracking (ICIP 2016).

---

## Running things

### The notebook (the main path)

Everything runs in Colab, nothing needs installing locally.

1. <https://colab.research.google.com> → **GitHub** tab → `ben-blake/cs5567-project`
   → `notebooks/midterm_demo.ipynb`
2. **Runtime → Change runtime type → T4 GPU.** Without it the detector is about
   20x slower.
3. **Runtime → Run all.**

| Step | Time on a T4 |
| --- | --- |
| Install | ~1 min |
| MOT16 download | 5 to 10 min (1.9 GB, once per session) |
| Export Re-ID crops | ~2 min |
| Train Siamese network | ~5 min (8 epochs) |
| Track 600 frames | ~2.5 min |

### When something breaks

| Symptom | Cause | Fix |
| --- | --- | --- |
| `No module named mot` | Install cell skipped, or editable install not picked up | Re-run the setup cell; it adds `src/` to `sys.path` |
| CUDA out of memory | Another notebook holds the GPU | Runtime → Disconnect and delete runtime |
| MOT16 download stalls | motchallenge.net is slow | Re-run; it resumes |
| Everything is slow | Running on CPU | Environment cell should say `device cuda` |
| Market-1501 download fails | Google Drive rate limit | Skip it; `--mot16-crops` gives in-domain data with no download |
| Inline figures hang | Large base64 payloads | Set `SHOW_PLOTS = False`; figures still save to `outputs/` |

### Locally

Only needed if you are editing code rather than running experiments.

```bash
git clone https://github.com/ben-blake/cs5567-project.git
cd cs5567-project
uv venv && uv pip install -e ".[dev]"

uv run pytest -q                      # 56 tests
uv run ruff check src tests scripts
```

Both run in CI on every pull request, so this is for faster feedback only.

### From the command line

```bash
python scripts/download_data.py --mot16          # 1.9 GB
python scripts/download_data.py --mot16-crops    # no download
python scripts/run_tracking.py --sequence MOT16-02 --max-frames 600
python scripts/run_tracking.py --sequence MOT16-02 --max-frames 600 --no-reid
```

### Building the slides

No LaTeX needed: **Actions tab → Slides → Artifacts**. Locally, with a TeX
distribution that has `beamer` and `metropolis`, run `make` in `slides/`.

---

## Suggested order

1. Fine-tune the detector, measure recall.
2. Re-run the ablation and sweeps on the improved detector.
3. Improve re-identification, measure with rank-1 and mAP.
4. Produce the final video on a **test** sequence.
5. Write the report from [RESULTS.md](RESULTS.md), which already holds the
   numbers.
6. Rehearse the final presentation.
7. Everyone submits the questionnaire individually.

Steps 1 and 2 change the result. Step 5 is the one most often left too late.

## Optional, only if ahead

- Cross-check our CLEAR-MOT against the `motmetrics` package. Ours is readable
  rather than authoritative, and the report should say which was used.
- Mask R-CNN for pixel-level segmentation. The assignment marks it optional.
- A Kalman filter in place of the linear motion model.

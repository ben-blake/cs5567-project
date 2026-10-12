# Runbook

How to run the project, test it, and produce the demo video.

---

## Running the notebook (the main path)

Everything runs in Colab. Nothing needs installing locally.

1. Open <https://colab.research.google.com> → **GitHub** tab → paste
   `ben-blake/cs5567-project` → open `notebooks/midterm_demo.ipynb`.
2. **Runtime → Change runtime type → T4 GPU.** Do this first; without it the
   detector runs ~20x slower.
3. **Runtime → Run all**, then watch the first few cells.

Expected timings on a T4:

| Step | Time | Notes |
| --- | --- | --- |
| Install | ~1 min | |
| MOT16 download | 5-10 min | ~1.9 GB, once per session |
| Export Re-ID crops | ~2 min | 3 sequences |
| Train Siamese net | ~5 min | 8 epochs |
| Track 150 frames | ~2 min | detection dominates |
| **Total** | **~20 min** | |

### If something breaks

| Symptom | Cause | Fix |
| --- | --- | --- |
| `No module named mot` | Install cell skipped | Re-run cell 2, then **Runtime → Restart session** |
| CUDA out of memory | Another notebook holds the GPU | Runtime → Disconnect and delete runtime, start over |
| MOT16 download stalls | motchallenge.net is slow | Re-run the cell; it resumes rather than restarting |
| Drive mount prompt loops | Pop-ups blocked | Allow pop-ups, or skip the Drive cell (checkpoints go to local disk) |
| Everything is slow | Running on CPU | Check the environment cell says `device cuda` |
| Market-1501 download fails | Google Drive rate limit | Skip it — `--mot16-crops` gives in-domain data with no download |

---

## Making the demo video

The notebook writes `outputs/MOT16-02_demo.mp4` and plays it inline.

**For the presentation, record a longer one.** In the tracking cell set:

```python
SEQUENCE   = "MOT16-02"
MAX_FRAMES = 600        # 600 frames at 30fps = 20 seconds
```

Then run that cell and everything after it. 600 frames takes ~8 minutes on a T4.

### What makes a good demo clip

- **30 seconds maximum.** You have 15 minutes total; the video is evidence, not
  the talk.
- **Pick a stretch where the target survives an occlusion.** That is the whole
  point of the Re-ID component, and it is what the rubric grades. A clip of
  someone walking across an empty plaza proves nothing.
- **Use `highlight_id`** so the target is in colour and everyone else is grey.
  The assignment asks for one target pedestrian, so show one.
- **Record the ablation too** if time allows: the same clip with
  `appearance_weight=0.0` shows IDs switching where the Re-ID version holds.
  Two clips side by side is the most persuasive thing you can show.

### Getting the file out of Colab

The last cell copies everything to Drive at `MyDrive/cs5567/presentation/`.
If Drive is not mounted, use the Files panel in the left sidebar → download.

**Download it before the presentation.** Do not stream from Drive in the room.

---

## Putting the video in the slides

Beamer cannot embed video reliably across PDF viewers. Two options, in order of
preference:

1. **Play it outside the deck.** Open the MP4 in VLC or QuickTime, alt-tab to
   it at the demo slide. Simple and it always works.
2. **Link from a still.** Put a contact sheet of frames on the slide
   (`mot.viz.draw.contact_sheet`) and keep the video one alt-tab away.

There is a `[standout]` "Demo" slide in the deck as the cue to switch.

---

## Running the code locally (optional)

Only needed if you are editing code rather than running experiments.

```bash
git clone https://github.com/ben-blake/cs5567-project.git
cd cs5567-project
uv venv && uv pip install -e ".[dev]"

uv run pytest -q                      # 36 tests, ~6s
uv run ruff check src tests scripts   # lint
```

Both run in CI on every pull request, so you do not strictly need them locally,
but the feedback is faster.

### Tracking from the command line

```bash
python scripts/download_data.py --mot16
python scripts/run_tracking.py --sequence MOT16-02 --max-frames 300
python scripts/run_tracking.py --sequence MOT16-02 --max-frames 300 --no-reid  # ablation
```

---

## Building the slides

You do not need LaTeX installed. Every push builds the deck in CI:

**Actions tab → latest "Slides" run → Artifacts → `midterm-slides`.**

To build locally you need a TeX distribution with `beamer` and `metropolis`:

```bash
cd slides && make
```

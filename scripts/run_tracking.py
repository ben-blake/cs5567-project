#!/usr/bin/env python3
"""Track a MOT16 sequence and write a result video.

    python scripts/run_tracking.py --sequence MOT16-02 --max-frames 300
    python scripts/run_tracking.py --sequence MOT16-02 --no-reid   # ablation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mot.config import PATHS, describe_environment, get_device  # noqa: E402
from mot.data.mot16 import Sequence  # noqa: E402
from mot.detect.faster_rcnn import build_pretrained_person_detector, load_checkpoint  # noqa: E402
from mot.pipeline import (  # noqa: E402
    TrackingPipeline,
    longest_track,
    render_results,
    results_to_mot_format,
)
from mot.reid.siamese import load_checkpoint as load_reid  # noqa: E402
from mot.track.tracker import Tracker  # noqa: E402


def resolve_sequence(name: str) -> Sequence:
    """Find a sequence by name in either split, or treat it as a path."""
    candidate = Path(name)
    if (candidate / "seqinfo.ini").is_file():
        return Sequence(candidate)
    for split in ("train", "test"):
        path = PATHS.mot16 / split / name
        if (path / "seqinfo.ini").is_file():
            return Sequence(path)
    raise SystemExit(f"sequence {name!r} not found under {PATHS.mot16}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sequence", default="MOT16-02", help="sequence name or path")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--detector-checkpoint", type=Path, default=None,
                        help="fine-tuned detector; omit to use the COCO-pretrained model")
    parser.add_argument("--reid-checkpoint", type=Path, default=None)
    parser.add_argument("--no-reid", action="store_true", help="track on box overlap only")
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument("--appearance-weight", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--highlight", type=int, default=None,
                        help="track id to highlight; default is the longest track")
    args = parser.parse_args()

    print(describe_environment())
    sequence = resolve_sequence(args.sequence)
    print(f"sequence: {sequence}")

    if args.detector_checkpoint:
        print(f"detector: fine-tuned {args.detector_checkpoint}")
        detector = load_checkpoint(args.detector_checkpoint)
        person_label = 1  # our fine-tuned head: 0 background, 1 person
    else:
        print("detector: COCO-pretrained (no fine-tuning)")
        detector = build_pretrained_person_detector()
        person_label = 1  # COCO person

    reid = None
    if not args.no_reid and args.reid_checkpoint:
        print(f"re-id: {args.reid_checkpoint}")
        reid = load_reid(args.reid_checkpoint)
    elif args.no_reid:
        print("re-id: disabled (overlap-only ablation)")
    else:
        print("re-id: none supplied; tracking on overlap only")

    pipeline = TrackingPipeline(
        detector,
        reid=reid,
        tracker=Tracker(appearance_weight=0.0 if reid is None else args.appearance_weight),
        score_threshold=args.score_threshold,
        person_label=person_label,
        device=get_device(),
    )

    started = time.time()
    results = pipeline.run_sequence(sequence, max_frames=args.max_frames)
    elapsed = time.time() - started

    frames = len(results)
    boxes = sum(len(r) for r in results)
    print(f"\n{frames} frames in {elapsed:.1f}s ({frames / max(elapsed, 1e-9):.2f} FPS)")
    print(f"{boxes} tracked boxes, {len({i for r in results for i in r.track_ids})} unique ids")

    PATHS.outputs.mkdir(parents=True, exist_ok=True)
    stem = args.output.stem if args.output else f"{sequence.name}_tracked"

    mot_path = PATHS.outputs / f"{stem}.txt"
    mot_path.write_text(results_to_mot_format(results))
    print(f"wrote {mot_path}")

    highlight = args.highlight if args.highlight is not None else longest_track(results)
    video_path = args.output or (PATHS.outputs / f"{stem}.mp4")
    render_results(sequence, results, video_path, highlight_id=highlight)
    print(f"wrote {video_path} (highlighting id {highlight})")


if __name__ == "__main__":
    main()

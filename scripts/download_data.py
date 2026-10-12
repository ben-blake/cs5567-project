#!/usr/bin/env python3
"""Fetch the datasets.

MOT16 is a direct download. Market-1501 is hosted on Google Drive, which
rate-limits, so this prints manual instructions if the automated fetch fails.

A third source needs no download at all: ``--mot16-crops`` cuts person crops
out of MOT16's own ground truth. Those are perfectly in-domain for this
project -- same cameras, same angles, same clutter as the frames we track --
so they are worth exporting even when Market-1501 downloads cleanly.

Usage (Colab):
    !python scripts/download_data.py --mot16
    !python scripts/download_data.py --market1501
    !python scripts/download_data.py --mot16-crops
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mot.config import PATHS  # noqa: E402

MOT16_URL = "https://motchallenge.net/data/MOT16.zip"

#: Original Market-1501, hosted on Google Drive by the authors.
#: Project page: https://zheng-lab-anu.github.io/Project/project_reid.html
MARKET1501_GDRIVE_ID = "0B8-rUzbwVRk0c054eEozWG9COHM"
MARKET1501_DIRNAME = "Market-1501-v15.09.15"


def _report(count: int, block: int, total: int) -> None:
    if total <= 0:
        return
    done = min(count * block, total)
    percent = 100 * done / total
    print(f"\r  {percent:5.1f}%  ({done / 1024**2:,.0f} / {total / 1024**2:,.0f} MB)", end="")


def download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        print(f"  already downloaded: {destination.name}")
        return destination
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, destination, reporthook=_report)
    print()
    return destination


def unzip(archive: Path, target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    print(f"  extracting {archive.name} -> {target}")
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(target)
    return target


def fetch_mot16() -> Path:
    """MOT16 train + test sequences (~1.9 GB)."""
    print("MOT16")
    target = PATHS.mot16
    if (target / "train").is_dir():
        print(f"  already present: {target}")
        return target

    archive = download(MOT16_URL, PATHS.data / "MOT16.zip")
    unzip(archive, PATHS.data)
    archive.unlink(missing_ok=True)  # reclaim ~2 GB of Colab disk

    print(f"  ready: {target}")
    return target


def fetch_market1501() -> Path | None:
    """Original Market-1501 (~146 MB) via gdown."""
    print("Market-1501")
    target = PATHS.market1501
    if target.is_dir():
        print(f"  already present: {target}")
        return target

    try:
        import gdown  # type: ignore[import-not-found]  # optional extra
    except ImportError:
        print("  gdown not installed; run: pip install gdown")
        gdown = None

    if gdown is not None:
        archive = PATHS.data / "Market-1501.zip"
        try:
            gdown.download(id=MARKET1501_GDRIVE_ID, output=str(archive), quiet=False)
            if archive.is_file() and archive.stat().st_size > 1024:
                unzip(archive, PATHS.data)
                archive.unlink(missing_ok=True)
                if target.is_dir():
                    print(f"  ready: {target}")
                    return target
        except Exception as error:  # noqa: BLE001 - Drive fails in many ways
            print(f"  automated download failed: {error}")

    print(
        "\n  Download Market-1501 manually:\n"
        "    1. https://zheng-lab-anu.github.io/Project/project_reid.html\n"
        f"    2. Unzip so that this path exists:\n       {target}\n"
        "    On Colab, uploading the zip to Drive once is more reliable than\n"
        "    re-downloading it every session.\n"
        "  Alternatively, skip it: --mot16-crops gives in-domain training data.\n"
    )
    return None


def export_crops(max_sequences: int | None = None) -> Path:
    """Cut Re-ID training crops from MOT16 ground truth."""
    from mot.data.datasets import export_mot16_crops
    from mot.data.mot16 import find_sequences

    print("MOT16 Re-ID crops")
    train_root = PATHS.mot16 / "train"
    if not train_root.is_dir():
        raise SystemExit(f"MOT16 train split not found at {train_root}; run --mot16 first")

    output = PATHS.data / "mot16_reid_crops"
    sequences = find_sequences(train_root)[:max_sequences]

    total_identities = total_crops = 0
    for sequence in sequences:
        exported = export_mot16_crops(sequence, output)
        crops = sum(len(paths) for paths in exported.values())
        total_identities += len(exported)
        total_crops += crops
        print(f"  {sequence.name}: {len(exported)} identities, {crops} crops")

    print(f"  ready: {output} ({total_identities} identities, {total_crops} crops)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--mot16", action="store_true", help="MOT16 sequences (~1.9 GB)")
    parser.add_argument(
        "--market1501", action="store_true", help="Market-1501 re-id crops (~146 MB)"
    )
    parser.add_argument(
        "--mot16-crops", action="store_true", help="export re-id crops from MOT16 gt"
    )
    parser.add_argument("--all", action="store_true", help="everything")
    parser.add_argument(
        "--max-sequences", type=int, default=None, help="limit sequences when exporting crops"
    )
    args = parser.parse_args()

    if not any([args.mot16, args.market1501, args.mot16_crops, args.all]):
        parser.print_help()
        raise SystemExit(1)

    PATHS.ensure()
    print(f"data directory: {PATHS.data}")
    free = shutil.disk_usage(PATHS.data).free / 1024**3
    print(f"free space: {free:.1f} GB\n")

    if args.mot16 or args.all:
        fetch_mot16()
    if args.market1501 or args.all:
        fetch_market1501()
    if args.mot16_crops or args.all:
        export_crops(args.max_sequences)


if __name__ == "__main__":
    main()

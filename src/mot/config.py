"""Runtime paths and device selection.

Defaults are chosen for Google Colab, which is where this project runs:
data lands on the local SSD (fast, wiped on disconnect) and checkpoints land
on Drive when it is mounted (slow, survives disconnect). Every path can be
overridden with an environment variable so the same code runs locally.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # import only for type checking; torch is heavy at runtime
    import torch


def in_colab() -> bool:
    """True when running inside a Colab runtime."""
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        # find_spec imports the parent package, which raises when the
        # namespace package `google` is absent entirely (any non-Colab env).
        return False


def drive_mounted() -> bool:
    return Path("/content/drive/MyDrive").is_dir()


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


@dataclass(frozen=True)
class Paths:
    """Where data, checkpoints and outputs live."""

    data: Path
    checkpoints: Path
    outputs: Path

    @property
    def mot16(self) -> Path:
        return self.data / "MOT16"

    @property
    def market1501(self) -> Path:
        return self.data / "Market-1501-v15.09.15"

    def ensure(self) -> Paths:
        for path in (self.data, self.checkpoints, self.outputs):
            path.mkdir(parents=True, exist_ok=True)
        return self


def default_paths() -> Paths:
    """Colab-aware defaults, overridable via MOT_DATA / MOT_CKPT / MOT_OUT."""
    if in_colab():
        # Datasets on local disk: re-downloading is faster than reading Drive.
        data = Path("/content/data")
        # Checkpoints on Drive when available so a disconnect is not fatal.
        base = Path("/content/drive/MyDrive/cs5567") if drive_mounted() else Path("/content")
        checkpoints, outputs = base / "checkpoints", base / "outputs"
    else:
        root = Path(__file__).resolve().parents[2]
        data, checkpoints, outputs = root / "data", root / "checkpoints", root / "outputs"

    return Paths(
        data=_env_path("MOT_DATA", data),
        checkpoints=_env_path("MOT_CKPT", checkpoints),
        outputs=_env_path("MOT_OUT", outputs),
    )


PATHS = default_paths()


def get_device(prefer: str | None = None) -> torch.device:
    """Best available torch device: explicit choice, else CUDA, MPS, or CPU."""
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def describe_environment() -> str:
    """One-line environment summary, handy at the top of a notebook."""
    import torch

    device = get_device()
    parts = [f"torch {torch.__version__}", f"device {device}"]
    if device.type == "cuda":
        parts.append(torch.cuda.get_device_name(0))
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        parts.append(f"{total:.1f} GB VRAM")
    if in_colab():
        parts.append("Drive mounted" if drive_mounted() else "Drive NOT mounted")
    free = shutil.disk_usage(PATHS.data.parent if PATHS.data.exists() else "/").free / 1024**3
    parts.append(f"{free:.0f} GB free")
    return " | ".join(parts)

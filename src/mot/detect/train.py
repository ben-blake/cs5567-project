"""Few-shot fine-tuning for the detector."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import get_device
from .faster_rcnn import save_checkpoint, trainable_parameters


def train_detector(
    model: torch.nn.Module,
    loader: DataLoader,
    epochs: int = 3,
    learning_rate: float = 5e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    log_every: int = 20,
) -> list[dict[str, float]]:
    """Fine-tune Faster R-CNN. Returns per-epoch loss history.

    torchvision's detection models return a dict of losses in train mode and
    predictions in eval mode -- so the model must stay in train mode here.
    """
    device = device or get_device()
    model.train().to(device)

    optimizer = torch.optim.SGD(
        trainable_parameters(model),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    # Warmup avoids the loss exploding in the first few steps with a fresh head.
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate, total_steps=epochs * max(1, len(loader)), pct_start=0.3
    )

    history = []
    for epoch in range(epochs):
        totals: dict[str, float] = {}
        batches = 0

        for step, (images, targets) in enumerate(loader):
            images = [image.to(device) for image in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())

            if not torch.isfinite(loss):
                # Usually a degenerate box slipped through; skip rather than
                # poison every weight with NaN.
                print(f"  skipping non-finite loss at step {step}")
                optimizer.zero_grad(set_to_none=True)
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            for name, value in loss_dict.items():
                totals[name] = totals.get(name, 0.0) + float(value)
            totals["total"] = totals.get("total", 0.0) + float(loss)
            batches += 1

            if log_every and step % log_every == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(loader)} loss {float(loss):.4f}")

        epoch_losses = {k: v / max(1, batches) for k, v in totals.items()}
        epoch_losses["epoch"] = epoch + 1
        history.append(epoch_losses)
        print(f"epoch {epoch + 1}/{epochs} mean loss {epoch_losses.get('total', 0):.4f}")

        if checkpoint_path:
            save_checkpoint(model, checkpoint_path, epoch=epoch + 1, history=history)

    return history

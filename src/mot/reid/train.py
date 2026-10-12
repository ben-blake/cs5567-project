"""Training the Siamese re-identification network."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..config import get_device
from .siamese import ContrastiveLoss, SiameseNetwork, save_checkpoint


@torch.no_grad()
def pair_accuracy(
    model: SiameseNetwork,
    loader: DataLoader,
    threshold: float = 0.5,
    device: torch.device | None = None,
    max_batches: int = 20,
) -> float:
    """Fraction of pairs classified correctly at a distance threshold.

    A plain, interpretable number for the slides: 0.5 is chance.
    """
    device = device or get_device()
    model.eval().to(device)

    correct = total = 0
    for index, (a, b, label) in enumerate(loader):
        if index >= max_batches:
            break
        emb_a, emb_b = model(a.to(device), b.to(device))
        distance = torch.nn.functional.pairwise_distance(emb_a, emb_b).cpu()
        predicted_same = (distance < threshold).float()
        correct += int((predicted_same == label).sum())
        total += len(label)

    return correct / total if total else 0.0


def train_reid(
    model: SiameseNetwork,
    loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    margin: float = 1.0,
    device: torch.device | None = None,
    checkpoint_path: str | Path | None = None,
    val_loader: DataLoader | None = None,
    log_every: int = 50,
) -> list[dict[str, float]]:
    """Train with contrastive loss. Returns per-epoch history."""
    device = device or get_device()
    model.to(device)

    criterion = ContrastiveLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        batches = 0

        for step, (a, b, label) in enumerate(loader):
            a, b, label = a.to(device), b.to(device), label.to(device)

            emb_a, emb_b = model(a, b)
            loss = criterion(emb_a, emb_b, label)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running += float(loss)
            batches += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch + 1} step {step}/{len(loader)} loss {float(loss):.4f}")

        scheduler.step()
        entry = {"epoch": epoch + 1, "loss": running / max(1, batches)}
        if val_loader is not None:
            entry["val_accuracy"] = pair_accuracy(model, val_loader, device=device)
        history.append(entry)

        message = f"epoch {epoch + 1}/{epochs} loss {entry['loss']:.4f}"
        if "val_accuracy" in entry:
            message += f" val_acc {entry['val_accuracy']:.3f}"
        print(message)

        if checkpoint_path:
            save_checkpoint(model, checkpoint_path, epoch=epoch + 1, history=history)

    return history

"""Siamese re-identification network.

The network maps a person crop to an L2-normalised embedding; two crops are
the same person when their embeddings are close. Training uses pairs and a
contrastive loss, so it learns "same or different" rather than a fixed set of
identities -- new people can appear mid-video without retraining.

On input size
-------------
The course slides give three different crop sizes (16x16, 24x24, 30x16) and a
reference ``Siamese_Network`` whose ``nn.Linear(128 * 22 * 22, ...)`` actually
implies a ~102x102 input, so no single size satisfies all of them. We use
128x64 (H x W), the standard person-re-id aspect ratio used by Market-1501,
and replace the brittle flatten with ``AdaptiveAvgPool2d``, which fixes the
feature width regardless of input size. The sizes in the slides are also far
too small to resolve clothing detail, which is the signal re-id depends on.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

#: (height, width) of a person crop. Market-1501's native aspect ratio.
CROP_SIZE = (128, 64)

#: Dimensionality of the appearance embedding.
EMBEDDING_DIM = 128


def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    """Conv -> BN -> ReLU -> MaxPool. BN keeps training stable on small batches."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class SiameseNetwork(nn.Module):
    """Twin CNN producing L2-normalised appearance embeddings.

    Both branches share weights -- "Siamese" means one network applied twice,
    not two networks.
    """

    def __init__(self, embedding_dim: int = EMBEDDING_DIM, in_channels: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.features = nn.Sequential(
            _conv_block(in_channels, 32),   # 128x64 -> 64x32
            _conv_block(32, 64),            #  64x32 -> 32x16
            _conv_block(64, 128),           #  32x16 -> 16x8
            _conv_block(128, 256),          #  16x8  -> 8x4
        )
        # Fixes the feature size independently of the input resolution, so the
        # network tolerates crops that are not exactly CROP_SIZE.
        self.pool = nn.AdaptiveAvgPool2d((4, 2))
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim),
        )

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a batch of crops. Returns ``(N, embedding_dim)``, unit norm."""
        x = self.features(x)
        x = self.pool(x)
        x = self.embed(x)
        # Unit norm makes euclidean distance and cosine similarity equivalent,
        # so one threshold works regardless of which the tracker uses.
        return F.normalize(x, p=2, dim=1)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward_once(a), self.forward_once(b)


class ContrastiveLoss(nn.Module):
    """Pulls matching pairs together, pushes others apart past ``margin``.

    label 1 = same person, 0 = different. With unit-norm embeddings the
    distance is in ``[0, 2]``, so a margin near 1.0 is a sensible default.
    """

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, emb_a: torch.Tensor, emb_b: torch.Tensor, label: torch.Tensor
    ) -> torch.Tensor:
        distance = F.pairwise_distance(emb_a, emb_b)
        label = label.float()
        same = label * distance.pow(2)
        different = (1 - label) * F.relu(self.margin - distance).pow(2)
        return 0.5 * (same + different).mean()


@torch.no_grad()
def embed_crops(
    model: SiameseNetwork,
    crops: torch.Tensor,
    device: torch.device | None = None,
    batch_size: int = 128,
) -> torch.Tensor:
    """Embed a stack of crops ``(N, 3, H, W)``, batching to bound memory."""
    if device is not None:
        model = model.to(device)
    model.eval()

    embeddings = []
    for start in range(0, len(crops), batch_size):
        batch = crops[start : start + batch_size]
        if device is not None:
            batch = batch.to(device)
        embeddings.append(model.forward_once(batch).cpu())
    if not embeddings:
        return torch.empty(0, model.embedding_dim)
    return torch.cat(embeddings)


def cosine_similarity_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity ``(len(a), len(b))``.

    Inputs are already unit norm, so this is a plain matrix product.
    """
    if len(a) == 0 or len(b) == 0:
        return torch.empty(len(a), len(b))
    return a @ b.T


def save_checkpoint(model: SiameseNetwork, path: str | Path, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state": model.state_dict(), "embedding_dim": model.embedding_dim, **extra}, path
    )
    return path


def load_checkpoint(path: str | Path) -> SiameseNetwork:
    checkpoint = torch.load(path, map_location="cpu")
    model = SiameseNetwork(embedding_dim=checkpoint.get("embedding_dim", EMBEDDING_DIM))
    model.load_state_dict(checkpoint["model_state"])
    return model

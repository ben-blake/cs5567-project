"""Faster R-CNN person detector.

Starts from torchvision's COCO-pretrained ResNet50-FPN, which already detects
people well, and few-shot fine-tunes it on MOT16. Only the detection heads are
trained by default; the backbone stays frozen so the whole thing fits in
Colab's VRAM budget.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from ..config import get_device

#: Person class in the COCO label set torchvision ships with.
COCO_PERSON = 1

#: Our fine-tuned head predicts background + person.
NUM_CLASSES = 2


def build_model(
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    trainable_backbone_layers: int | None = None,
) -> nn.Module:
    """Faster R-CNN with a fresh box predictor sized to ``num_classes``.

    ``freeze_backbone`` is the few-shot setting from the assignment: the
    pretrained features are kept as-is and only the heads learn. Set it False
    (optionally with ``trainable_backbone_layers``) for full fine-tuning.
    """
    weights = "DEFAULT" if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=0 if freeze_backbone else trainable_backbone_layers,
    )

    # Swap the 91-class COCO predictor for one sized to our label set.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if freeze_backbone:
        for param in model.backbone.parameters():
            param.requires_grad = False

    return model


def build_pretrained_person_detector() -> nn.Module:
    """COCO-pretrained detector used as-is, no fine-tuning.

    This is the baseline: it needs no training at all, so the pipeline can be
    demonstrated end-to-end before any model is trained.
    """
    return fasterrcnn_resnet50_fpn(weights="DEFAULT")


def trainable_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def parameter_summary(model: nn.Module) -> dict[str, int]:
    """Total vs trainable parameter counts, for the slides."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


@torch.no_grad()
def detect(
    model: nn.Module,
    images: list[torch.Tensor],
    score_threshold: float = 0.7,
    person_label: int = COCO_PERSON,
    device: torch.device | None = None,
) -> list[dict[str, torch.Tensor]]:
    """Run detection and keep confident person boxes.

    ``images`` are float tensors in ``[0, 1]``, shape ``(3, H, W)``, any size.
    Returns one dict per image with ``boxes`` (xyxy) and ``scores``.
    """
    device = device or get_device()
    model.eval().to(device)

    outputs = model([image.to(device) for image in images])

    results = []
    for output in outputs:
        keep = (output["labels"] == person_label) & (output["scores"] >= score_threshold)
        results.append(
            {
                "boxes": output["boxes"][keep].cpu(),
                "scores": output["scores"][keep].cpu(),
            }
        )
    return results


def save_checkpoint(model: nn.Module, path: str | Path, **extra) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), **extra}, path)
    return path


def load_checkpoint(path: str | Path, num_classes: int = NUM_CLASSES) -> nn.Module:
    """Rebuild a fine-tuned detector from a checkpoint."""
    checkpoint = torch.load(path, map_location="cpu")
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state"])
    return model

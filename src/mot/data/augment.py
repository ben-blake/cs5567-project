"""Augmentation.

MOT16 gives us a few thousand frames from a handful of scenes, so the risk is
memorising those scenes rather than learning what a pedestrian looks like.
Augmentation is the cheapest defence.

Detection transforms must move the boxes with the image, which is why they
take and return ``(image, target)`` instead of just an image.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from ..reid.siamese import CROP_SIZE


def to_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image.convert("RGB"))).permute(2, 0, 1).float() / 255.0


class DetectionTransform:
    """Horizontal flip plus colour jitter, keeping boxes in sync.

    No vertical flip or rotation: pedestrians are upright in every frame, so
    those would teach the model poses it will never see.
    """

    def __init__(self, train: bool = True, flip_probability: float = 0.5):
        self.train = train
        self.flip_probability = flip_probability
        self.color = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def __call__(self, image: Image.Image, target: dict) -> tuple[torch.Tensor, dict]:
        if self.train:
            if random.random() < self.flip_probability:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                boxes = target["boxes"].clone()
                if len(boxes):
                    width = image.width
                    # Mirror x: new_x1 = W - old_x2, new_x2 = W - old_x1.
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target["boxes"] = boxes
            image = self.color(image)

        return to_tensor(image), target


def reid_transform(train: bool = True) -> T.Compose:
    """Crop pipeline for the Siamese network.

    Random erasing simulates occlusion, which is exactly when re-id has to work
    hardest -- a partly hidden person must still match their earlier appearance.
    """
    height, width = CROP_SIZE

    if not train:
        return T.Compose([T.Resize((height, width)), T.ToTensor()])

    return T.Compose(
        [
            T.Resize((height + 16, width + 8)),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
        ]
    )

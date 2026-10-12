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
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                boxes = target["boxes"].clone()
                if len(boxes):
                    width = image.width
                    # Mirror x: new_x1 = W - old_x2, new_x2 = W - old_x1.
                    boxes[:, [0, 2]] = width - boxes[:, [2, 0]]
                    target["boxes"] = boxes
            image = self.color(image)

        return to_tensor(image), target


class GaussianMask:
    """Down-weight pixels far from the image centre.

    Zheng et al. (ICCV 2015, Sec. 4.3) report +5.64% mAP on Market-1501 from
    this alone: "the prior that pedestrian is roughly located in the center of
    the image is statistically sound". A detector crop is a box around a
    person, so the person occupies the middle and the corners are background.

    This is why we do *not* need a background-stripped dataset variant: we keep
    real backgrounds, which match the MOT16 frames we run on, and suppress them
    with a spatial prior instead.
    """

    def __init__(self, sigma_scale: float = 0.25, strength: float = 1.0):
        self.sigma_scale = sigma_scale
        self.strength = strength
        self._cache: dict[tuple[int, int], torch.Tensor] = {}

    def _mask(self, height: int, width: int) -> torch.Tensor:
        key = (height, width)
        if key not in self._cache:
            ys = torch.linspace(-0.5, 0.5, height).view(-1, 1)
            xs = torch.linspace(-0.5, 0.5, width).view(1, -1)
            sigma = self.sigma_scale
            mask = torch.exp(-(ys**2 + xs**2) / (2 * sigma**2))
            # strength=0 disables the mask, 1 applies it fully.
            self._cache[key] = (1 - self.strength) + self.strength * mask
        return self._cache[key]

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image * self._mask(image.shape[-2], image.shape[-1])


def reid_transform(train: bool = True, gaussian_mask: bool = False) -> T.Compose:
    """Crop pipeline for the Siamese network.

    Random erasing simulates occlusion, which is exactly when re-id has to work
    hardest -- a partly hidden person must still match their earlier appearance.
    """
    height, width = CROP_SIZE
    mask = [GaussianMask()] if gaussian_mask else []

    if not train:
        return T.Compose([T.Resize((height, width)), T.ToTensor(), *mask])

    return T.Compose(
        [
            # Resize larger then random-crop: simulates the detector
            # misalignment that Market-1501's DPM boxes exhibit, and that our
            # Faster R-CNN boxes will too.
            T.Resize((height + 16, width + 8)),
            T.RandomCrop((height, width)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.ToTensor(),
            T.RandomErasing(p=0.4, scale=(0.02, 0.2)),
            *mask,
        ]
    )

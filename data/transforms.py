from __future__ import annotations

import random
from dataclasses import dataclass

from PIL import Image
import torch
from torchvision.transforms import functional as TF


def validate_infrared_image_intensity(image: Image.Image) -> None:
    """Prevent the image-intensity pipeline from destroying quantitative targets."""
    if image.mode in {"I", "F"} or image.mode.startswith("I;16"):
        raise ValueError(
            f"Quantitative infrared image (PIL mode {image.mode}) requires explicit "
            "physical calibration and fixed training-set normalization before "
            "the image-intensity pipeline; 8-bit grayscale conversion would "
            "discard its radiometric values."
        )


@dataclass
class PairedTransform:
    size: int = 256
    crop_size: int = 256
    horizontal_flip: bool = True
    training: bool = True

    def __call__(self, visible: Image.Image, infrared: Image.Image):
        validate_infrared_image_intensity(infrared)
        visible = TF.resize(visible, [self.size, self.size], antialias=True)
        infrared = TF.resize(infrared, [self.size, self.size], antialias=True)
        if self.training and self.crop_size < self.size:
            max_offset = self.size - self.crop_size
            top = random.randint(0, max_offset)
            left = random.randint(0, max_offset)
            visible = TF.crop(visible, top, left, self.crop_size, self.crop_size)
            infrared = TF.crop(infrared, top, left, self.crop_size, self.crop_size)
        if self.training and self.horizontal_flip and random.random() < 0.5:
            visible = TF.hflip(visible)
            infrared = TF.hflip(infrared)
        visible_tensor = TF.to_tensor(visible) * 2.0 - 1.0
        infrared_tensor = TF.to_tensor(infrared) * 2.0 - 1.0
        return visible_tensor, infrared_tensor


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return ((tensor + 1.0) / 2.0).clamp(0.0, 1.0)


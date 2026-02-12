from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PIL import ImageOps
import torch
try:
    from torchvision import transforms as T  # type: ignore
except Exception:  # pragma: no cover
    T = None


FASHIONMNIST_MEAN = (0.2860,)
FASHIONMNIST_STD = (0.3530,)


def build_transforms(
    train: bool,
    image_size: int = 28,
    normalize: bool = True,
    augment: bool = False,
) -> object:
    if T is not None:
        ops = [T.Resize((image_size, image_size)), T.ToTensor()]

        if train and augment:
            ops = [
                T.RandomCrop(image_size, padding=2),
                T.RandomHorizontalFlip(p=0.5),
                *ops,
            ]

        if normalize:
            ops.append(T.Normalize(FASHIONMNIST_MEAN, FASHIONMNIST_STD))

        return T.Compose(ops)

    def _fallback_transform(image):
        image = image.resize((image_size, image_size))
        if train and augment:
            image = ImageOps.expand(image, border=2, fill=0).crop((2, 2, 2 + image_size, 2 + image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).unsqueeze(0)
        if normalize:
            mean = torch.tensor(FASHIONMNIST_MEAN).view(1, 1, 1)
            std = torch.tensor(FASHIONMNIST_STD).view(1, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    return _fallback_transform


def denormalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(FASHIONMNIST_MEAN, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(FASHIONMNIST_STD, device=x.device).view(1, -1, 1, 1)
    return x * std + mean


def default_input_shape(batch_size: int = 4, image_size: int = 28) -> Tuple[int, int, int, int]:
    return (batch_size, 1, image_size, image_size)

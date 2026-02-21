from __future__ import annotations

from typing import Tuple

import torch
from PIL import Image
from torchvision import transforms
from io import BytesIO


IMAGE_SIZE: Tuple[int, int] = (224, 224)


def build_preprocess_transform(train: bool = False) -> transforms.Compose:
    if train:
        return transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=10),
                transforms.CenterCrop(IMAGE_SIZE),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )


def preprocess_pil_image(image: Image.Image) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    transform = build_preprocess_transform(train=False)
    tensor = transform(rgb_image)
    return tensor


def preprocess_bytes(image_bytes: bytes) -> torch.Tensor:
    image = Image.open(BytesIO(image_bytes))
    return preprocess_pil_image(image)

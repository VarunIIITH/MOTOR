"""Per-backbone image transforms for train and evaluation."""

import torch
from torchvision import transforms
from torchvision.models.video import Swin3D_T_Weights

# ImageNet stats (Swin3D, R3D)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# MViTv2 / S3D use uniform stats
UNIFORM_MEAN = [0.45, 0.45, 0.45]
UNIFORM_STD = [0.225, 0.225, 0.225]

_STATS = {
    "swin3d_t": (IMAGENET_MEAN, IMAGENET_STD),
    "r3d_18":   (IMAGENET_MEAN, IMAGENET_STD),
    "mvit_v2_s": (UNIFORM_MEAN, UNIFORM_STD),
    "s3d":       (UNIFORM_MEAN, UNIFORM_STD),
}


def build_transforms(backbone: str, img_size: int = 224, mode: str = "train"):
    """Build transforms for a given backbone and mode."""
    mean, std = _STATS.get(backbone, (IMAGENET_MEAN, IMAGENET_STD))

    if mode == "train":
        return transforms.Compose([
            transforms.Resize(img_size + 32),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(mean, std),
        ])


def build_gaze_transforms(backbone: str, img_size: int = 224, mode: str = "train"):
    """Gaze stream always uses center-crop (no random flip)."""
    mean, std = _STATS.get(backbone, (IMAGENET_MEAN, IMAGENET_STD))
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean, std),
    ])

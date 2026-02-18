"""Backbone encoder wrappers for video classification."""

import torch.nn as nn
from torchvision.models.video import (
    swin3d_t, Swin3D_T_Weights,
    r3d_18, R3D_18_Weights,
    mvit_v2_s, MViT_V2_S_Weights,
    s3d, S3D_Weights,
)

# Feature dimensions for each backbone after pooling
BACKBONE_DIMS = {
    "swin3d_t": 768,
    "r3d_18": 512,
    "mvit_v2_s": 768,
    "s3d": 1024,
}


class Swin3DEncoder(nn.Module):
    """Swin3D-T encoder → (B, 768) pooled features."""

    def __init__(self, weights=Swin3D_T_Weights.DEFAULT):
        super().__init__()
        self.backbone = swin3d_t(weights=weights)
        self.out_dim = self.backbone.head.in_features
        self.backbone.head = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class R3D18Encoder(nn.Module):
    """ResNet3D-18 encoder → (B, 512) pooled features."""

    def __init__(self, weights=R3D_18_Weights.DEFAULT):
        super().__init__()
        self.backbone = r3d_18(weights=weights)
        self.out_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class R3D18Trunk(nn.Module):
    """R3D-18 truncated: returns conv feature map (B, 512, T', H', W')
    for use in cross-attention fusion."""

    def __init__(self, weights=R3D_18_Weights.DEFAULT):
        super().__init__()
        m = r3d_18(weights=weights)
        self.stem = m.stem
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4
        self.out_channels = 512

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


class MViTv2Encoder(nn.Module):
    """MViTv2-S encoder → (B, 768) pooled features."""

    def __init__(self, weights=MViT_V2_S_Weights.DEFAULT):
        super().__init__()
        self.backbone = mvit_v2_s(weights=weights)
        self.out_dim = 768
        self.backbone.head = nn.Identity()

    def forward(self, x):
        return self.backbone(x)


class S3DEncoder(nn.Module):
    """S3D encoder → (B, 1024) pooled features."""

    def __init__(self, weights=S3D_Weights.DEFAULT):
        super().__init__()
        self.backbone = s3d(weights=weights)
        self.out_dim = self.backbone.classifier[1].in_channels
        self.backbone.classifier = nn.Identity()
        self.pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        x = self.backbone(x)
        if x.dim() == 5:
            x = self.pool(x).flatten(1)
        return x


_REGISTRY = {
    "swin3d_t": Swin3DEncoder,
    "r3d_18": R3D18Encoder,
    "mvit_v2_s": MViTv2Encoder,
    "s3d": S3DEncoder,
}


def build_backbone(name: str, freeze: bool = True) -> nn.Module:
    """Build a backbone encoder by name, optionally frozen."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{name}'. Choose from {list(_REGISTRY)}")
    encoder = _REGISTRY[name]()
    if freeze:
        for p in encoder.parameters():
            p.requires_grad = False
    return encoder

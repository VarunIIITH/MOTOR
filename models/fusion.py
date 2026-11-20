"""Fusion models: compose backbone encoders with classification heads."""

import torch
import torch.nn as nn
from .backbones import build_backbone, R3D18Trunk, BACKBONE_DIMS
from .heads import MLPHead, TelemetryHead


class SingleStreamModel(nn.Module):
    """Single backbone (RGB only) → classification."""

    def __init__(self, backbone_name, n_classes, head_hidden=(512, 128), dropout=0.4):
        super().__init__()
        self.encoder = build_backbone(backbone_name, freeze=True)
        self.head = MLPHead(self.encoder.out_dim, n_classes, head_hidden, dropout)

    def forward(self, rgb):
        return self.head(self.encoder(rgb))


class TwoStreamFusion(nn.Module):
    """RGB + Gaze late fusion → classification."""

    def __init__(self, backbone_name, n_classes, head_hidden=(512, 128), dropout=0.4):
        super().__init__()
        self.rgb_enc = build_backbone(backbone_name, freeze=True)
        self.gaze_enc = build_backbone(backbone_name, freeze=True)
        fused_dim = self.rgb_enc.out_dim + self.gaze_enc.out_dim
        self.head = MLPHead(fused_dim, n_classes, head_hidden, dropout)

    def forward(self, rgb, gaze):
        f_rgb = self.rgb_enc(rgb)
        f_gaze = self.gaze_enc(gaze)
        return self.head(torch.cat([f_rgb, f_gaze], dim=1))


class RGBTeleFusion(nn.Module):
    """RGB + Telemetry late fusion → classification."""

    def __init__(self, backbone_name, n_classes, tele_len=128, tele_dim=32,
                 head_hidden=(512, 128), dropout=0.4):
        super().__init__()
        self.rgb_enc = build_backbone(backbone_name, freeze=True)
        self.tele_enc = TelemetryHead(tele_len, tele_dim)
        fused_dim = self.rgb_enc.out_dim + tele_dim
        self.head = MLPHead(fused_dim, n_classes, head_hidden, dropout)

    def forward(self, rgb, tele):
        f_rgb = self.rgb_enc(rgb)
        f_tele = self.tele_enc(tele)
        return self.head(torch.cat([f_rgb, f_tele], dim=1))


class ThreeStreamFusion(nn.Module):
    """RGB + Gaze + Telemetry late fusion → classification."""

    def __init__(self, backbone_name, n_classes, tele_len=128, tele_dim=32,
                 head_hidden=(512, 128), dropout=0.4):
        super().__init__()
        self.rgb_enc = build_backbone(backbone_name, freeze=True)
        self.gaze_enc = build_backbone(backbone_name, freeze=True)
        self.tele_enc = TelemetryHead(tele_len, tele_dim)
        fused_dim = self.rgb_enc.out_dim + self.gaze_enc.out_dim + tele_dim
        self.head = MLPHead(fused_dim, n_classes, head_hidden, dropout)

    def forward(self, rgb, gaze, tele):
        f_rgb = self.rgb_enc(rgb)
        f_gaze = self.gaze_enc(gaze)
        f_tele = self.tele_enc(tele)
        return self.head(torch.cat([f_rgb, f_gaze, f_tele], dim=1))


class CrossAttnFusion(nn.Module):
    """Cross-attention fusion: RGB queries attend to Gaze keys/values.
    Uses R3D-18 feature maps (not pooled vectors)."""

    def __init__(self, n_classes, d_model=256, nhead=4, dropout=0.1,
                 freeze_backbones=False):
        super().__init__()
        self.rgb_trunk = R3D18Trunk()
        self.gaze_trunk = R3D18Trunk()
        ch = self.rgb_trunk.out_channels

        if freeze_backbones:
            for p in list(self.rgb_trunk.parameters()) + list(self.gaze_trunk.parameters()):
                p.requires_grad = False

        # project conv features to d_model
        self.proj_rgb = nn.Conv3d(ch, d_model, 1)
        self.proj_gaze = nn.Conv3d(ch, d_model, 1)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(512, n_classes),
        )

    def _to_tokens(self, x):
        """(B, D, T, H, W) → (B*T, H*W, D)"""
        B, D, T, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, D)
        return x, B, T

    def forward(self, rgb, gaze):
        q, B, T = self._to_tokens(self.proj_rgb(self.rgb_trunk(rgb)))
        kv, _, _ = self._to_tokens(self.proj_gaze(self.gaze_trunk(gaze)))

        attn_out, _ = self.cross_attn(q, kv, kv)
        x = self.norm1(q + attn_out)
        x = self.norm2(x + self.ffn(x))

        # pool: mean over spatial then mean over time
        x = x.mean(dim=1).view(B, T, -1).mean(dim=1)
        return self.classifier(x)


_FUSION_MAP = {
    "single": SingleStreamModel,
    "two_stream": TwoStreamFusion,
    "rgb_tele": RGBTeleFusion,
    "three_stream": ThreeStreamFusion,
    "cross_attn": CrossAttnFusion,
}


def build_model(cfg: dict) -> nn.Module:
    """Build a model from a config dictionary."""
    mcfg = cfg["model"]
    fusion = mcfg["fusion"]
    n_classes = cfg["data"].get("n_classes", 11)
    backbone = mcfg.get("backbone", "swin3d_t")
    hidden = tuple(mcfg.get("head_hidden", [512, 128]))
    dropout = mcfg.get("dropout", 0.4)

    if fusion == "single":
        return SingleStreamModel(backbone, n_classes, hidden, dropout)
    elif fusion == "two_stream":
        return TwoStreamFusion(backbone, n_classes, hidden, dropout)
    elif fusion == "rgb_tele":
        return RGBTeleFusion(
            backbone, n_classes,
            tele_len=mcfg.get("tele_len", 128),
            tele_dim=mcfg.get("tele_dim", 32),
            head_hidden=hidden, dropout=dropout,
        )
    elif fusion == "three_stream":
        return ThreeStreamFusion(
            backbone, n_classes,
            tele_len=mcfg.get("tele_len", 128),
            tele_dim=mcfg.get("tele_dim", 32),
            head_hidden=hidden, dropout=dropout,
        )
    elif fusion == "cross_attn":
        return CrossAttnFusion(
            n_classes,
            d_model=mcfg.get("attn_dim", 256),
            nhead=mcfg.get("attn_heads", 4),
            dropout=mcfg.get("attn_dropout", 0.1),
            freeze_backbones=mcfg.get("freeze_backbones", False),
        )
    else:
        raise ValueError(f"Unknown fusion '{fusion}'. Choose from {list(_FUSION_MAP)}")

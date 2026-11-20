"""Classification and telemetry projection heads."""

import torch.nn as nn


class MLPHead(nn.Module):
    """LayerNorm → Linear → ReLU → Dropout MLP classification head."""

    def __init__(self, in_dim, n_classes, hidden=(512, 128), dropout=0.4):
        super().__init__()
        layers = [nn.LayerNorm(in_dim)]
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TelemetryHead(nn.Module):
    """Projects raw telemetry vector to a compact embedding."""

    def __init__(self, in_dim=128, out_dim=32, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(128, 64),     nn.ReLU(True), nn.Dropout(dropout),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.proj(x)

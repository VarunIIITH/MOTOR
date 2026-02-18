"""Two-stream dataset: RGB video + Telemetry vector."""

import numpy as np
import torch
from torch.utils.data import Dataset
import decord
decord.bridge.set_bridge("torch")

from .transforms import build_transforms
from utils.telemetry import parse_telemetry_vec, pad_or_trunc


def uniform_indices(n, k):
    return np.linspace(0, max(n - 1, 0), k, dtype=np.int64)


class RGBTeleDataset(Dataset):
    """Returns (rgb_clip, tele_vec, label) for RGB+telemetry fusion."""

    def __init__(self, df, clip_len, backbone, tele_len=256, mode="train",
                 img_size=224, path_col="path_video", label_col="label"):
        self.df = df.reset_index(drop=True)
        self.clip_len = clip_len
        self.path_col = path_col
        self.label_col = label_col
        self.tfm = build_transforms(backbone, img_size, mode)

        # pre-parse telemetry vectors
        self.tele = []
        for _, row in self.df.iterrows():
            vec = parse_telemetry_vec(row, tele_len)
            self.tele.append(torch.tensor(vec, dtype=torch.float32))

    def __len__(self):
        return len(self.df)

    def _read_clip(self, path):
        vr = decord.VideoReader(path)
        idxs = uniform_indices(len(vr), self.clip_len)
        clip = vr.get_batch(idxs).permute(0, 3, 1, 2)
        clip = self.tfm(clip)
        if clip.dim() == 4 and clip.shape[0] == self.clip_len:
            clip = clip.permute(1, 0, 2, 3)
        return clip

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = self._read_clip(row[self.path_col])
        tele = self.tele[idx]
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return rgb, tele, label

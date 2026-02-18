"""Two-stream dataset: RGB + Gaze video clips."""

import numpy as np
import torch
from torch.utils.data import Dataset
import decord
decord.bridge.set_bridge("torch")

from .transforms import build_transforms, build_gaze_transforms


def uniform_indices(n, k):
    return np.linspace(0, max(n - 1, 0), k, dtype=np.int64)


class RGBGazeDataset(Dataset):
    """Returns (rgb_clip, gaze_clip, label) for two-stream fusion."""

    def __init__(self, df, clip_len, backbone, mode="train", img_size=224,
                 path_col="path_video", gaze_col="path_gaze", label_col="label"):
        self.df = df.reset_index(drop=True)
        self.clip_len = clip_len
        self.path_col = path_col
        self.gaze_col = gaze_col
        self.label_col = label_col
        self.tfm_rgb = build_transforms(backbone, img_size, mode)
        self.tfm_gaze = build_gaze_transforms(backbone, img_size, mode)

    def __len__(self):
        return len(self.df)

    def _read_clip(self, path, tfm):
        vr = decord.VideoReader(path)
        idxs = uniform_indices(len(vr), self.clip_len)
        clip = vr.get_batch(idxs).permute(0, 3, 1, 2)  # (T, C, H, W)
        clip = torch.stack([tfm(fr / 255.0) for fr in clip.float()])
        return clip.permute(1, 0, 2, 3)  # (C, T, H, W)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = self._read_clip(row[self.path_col], self.tfm_rgb)
        gaze = self._read_clip(row[self.gaze_col], self.tfm_gaze)
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return rgb, gaze, label

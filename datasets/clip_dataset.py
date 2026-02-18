"""Single-stream video clip dataset for RGB-only models."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import decord
decord.bridge.set_bridge("torch")

from .transforms import build_transforms


def uniform_indices(n, k):
    return np.linspace(0, max(n - 1, 0), k, dtype=np.int64)


class ClipDataset(Dataset):
    """Loads single-stream video clips with uniform frame sampling."""

    def __init__(self, df, clip_len, backbone, mode="train", img_size=224,
                 path_col="path_video", label_col="label"):
        self.df = df.reset_index(drop=True)
        self.clip_len = clip_len
        self.path_col = path_col
        self.label_col = label_col
        self.tfm = build_transforms(backbone, img_size, mode)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        vid_path = row[self.path_col]
        label = int(row[self.label_col])

        vr = decord.VideoReader(vid_path)
        T = len(vr)
        if T == 0:
            raise RuntimeError(f"Empty video: {vid_path}")

        idxs = uniform_indices(T, self.clip_len)
        frames = vr.get_batch(idxs).permute(0, 3, 1, 2)  # (T, C, H, W)
        frames = self.tfm(frames)

        # ensure (C, T, H, W) output
        if frames.dim() == 4 and frames.shape[0] == self.clip_len:
            frames = frames.permute(1, 0, 2, 3)

        return frames, torch.tensor(label, dtype=torch.long)

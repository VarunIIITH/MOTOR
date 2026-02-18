"""Three-stream dataset: RGB + Gaze video + Telemetry vector."""

import numpy as np
import torch
from torch.utils.data import Dataset
import decord
decord.bridge.set_bridge("torch")

from .transforms import build_transforms, build_gaze_transforms
from utils.telemetry import parse_telemetry_vec


def uniform_indices(n, k):
    return np.linspace(0, max(n - 1, 0), k, dtype=np.int64)


class RGBGazeTeleDataset(Dataset):
    """Returns (rgb_clip, gaze_clip, tele_vec, label) for three-stream fusion."""

    def __init__(self, df, clip_len, backbone, tele_len=256, mode="train",
                 img_size=224, path_col="path_video", gaze_col="path_gaze",
                 label_col="label"):
        self.df = df.reset_index(drop=True)
        self.clip_len = clip_len
        self.path_col = path_col
        self.gaze_col = gaze_col
        self.label_col = label_col
        self.tfm_rgb = build_transforms(backbone, img_size, mode)
        self.tfm_gaze = build_gaze_transforms(backbone, img_size, mode)

        self.tele = []
        for _, row in self.df.iterrows():
            vec = parse_telemetry_vec(row, tele_len)
            self.tele.append(torch.tensor(vec, dtype=torch.float32))

    def __len__(self):
        return len(self.df)

    def _read_clip(self, path, tfm):
        vr = decord.VideoReader(path)
        idxs = uniform_indices(len(vr), self.clip_len)
        clip = vr.get_batch(idxs).permute(0, 3, 1, 2)
        clip = torch.stack([tfm(fr / 255.0) for fr in clip.float()])
        return clip.permute(1, 0, 2, 3)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        rgb = self._read_clip(row[self.path_col], self.tfm_rgb)
        gaze = self._read_clip(row[self.gaze_col], self.tfm_gaze)
        tele = self.tele[idx]
        label = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return rgb, gaze, tele, label

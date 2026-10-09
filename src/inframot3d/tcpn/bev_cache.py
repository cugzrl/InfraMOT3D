"""BEV 缓存的坐标定义与读取

原始 CenterPoint BEV 为 [512, 120, 256]，高对应 y，宽对应 x，每格 0.8 m
缓存裁剪 y 在 [-48, 40]，x 在 [0, 108]，PCA 到 128 通道
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

PCA_DIM = 128
CELL = 0.8
CROP = (10, 120, 0, 135)
X0 = 0.0
Y0 = -48.0
HEIGHT = CROP[1] - CROP[0]
WIDTH = CROP[3] - CROP[2]


def xy_to_grid(xy):
    """米制坐标转 grid_sample 坐标，align_corners=False"""
    gx = (xy[..., 0] - X0) / (WIDTH * CELL) * 2.0 - 1.0
    gy = (xy[..., 1] - Y0) / (HEIGHT * CELL) * 2.0 - 1.0
    return torch.stack([gx, gy], dim=-1)


def sample_bev(bev, xy):
    """bev [B,C,H,W]，xy [B,N,K,2] 米制，返回 [B,N,K,C]，越界为 0"""
    grid = xy_to_grid(xy)
    out = F.grid_sample(bev, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return out.permute(0, 2, 3, 1)


class BevStore:
    def __init__(self, root):
        self.root = Path(root)
        self._arrays = {}
        self._index = {}

    def frame(self, sequence_id, frame_id):
        if sequence_id not in self._arrays:
            self._arrays[sequence_id] = np.load(self.root / ("%s.npy" % sequence_id), mmap_mode="r")
            ids = json.loads((self.root / ("%s.json" % sequence_id)).read_text())
            self._index[sequence_id] = {f: i for i, f in enumerate(ids)}
        return self._arrays[sequence_id][self._index[sequence_id][frame_id]]

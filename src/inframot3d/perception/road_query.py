"""历史轨迹查询当前 BEV，车道切向决定额外采样位置"""

import math

import torch
import torch.nn.functional as F
from torch import nn

PC_RANGE = (0.0, -56.0, 204.8, 40.0)
ISO_OFFSETS = torch.tensor(
    [(dx, dy) for dx in (-1.6, 0.0, 1.6) for dy in (-1.6, 0.0, 1.6)],
    dtype=torch.float32,
)
LANE_STEPS = torch.tensor([-4.8, -2.4, 2.4, 4.8], dtype=torch.float32)


def xy_to_norm(xy, height, width):
    """米制坐标转到 grid_sample 的 [-1, 1]，x 对应宽度，y 对应高度"""
    x0, y0, x1, y1 = PC_RANGE
    ix = (xy[..., 0] - x0) / (x1 - x0) * width - 0.5
    iy = (xy[..., 1] - y0) / (y1 - y0) * height - 0.5
    x = ix / max(width - 1, 1) * 2.0 - 1.0
    y = iy / max(height - 1, 1) * 2.0 - 1.0
    return torch.stack([x, y], dim=-1)


class TrackBevQuery(nn.Module):
    def __init__(self, bev_channels=512, hidden=128, use_bev=True, use_lane=True):
        super().__init__()
        self.use_bev = bool(use_bev)
        self.use_lane = bool(use_lane) and self.use_bev
        self.state = nn.Sequential(nn.Linear(8, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        if self.use_bev:
            self.proj = nn.Linear(bev_channels, hidden)
            self.attn = nn.MultiheadAttention(hidden, 4, batch_first=True)
        if self.use_lane:
            # 只加在沿车道采样点上，各向同性查询看不到这个参数
            self.lane_token = nn.Parameter(torch.zeros(hidden))
        self.exist = nn.Linear(hidden, 1)
        self.delta = nn.Linear(hidden, 3)
        # 存在性头保留小权重，否则零初始化会挡住 BEV 投影的梯度
        nn.init.normal_(self.exist.weight, std=0.01)
        nn.init.constant_(self.exist.bias, -2.0)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def sample_sites(self, center, tangent, lane_valid):
        """center [N,2] 是运动外推位置，车道切向只增加沿路采样"""
        device = center.device
        base = ISO_OFFSETS.to(device).view(1, -1, 2) + center[:, None, :]
        if not self.use_lane:
            return base
        steps = LANE_STEPS.to(device).view(1, -1, 1)
        direction = tangent.to(device)
        along = center[:, None, :] + steps * direction[:, None, :]
        # 远离车道的查询不增加沿路点，改回中心，避免把空地图当成道路
        valid = lane_valid.to(device).view(-1, 1, 1)
        along = torch.where(valid > 0, along, center[:, None, :])
        return torch.cat([base, along], dim=1)

    def forward(self, state, bev=None, center=None, tangent=None, lane_valid=None):
        """state [N,8]，bev [1,C,H,W]，输出存在性 logit 和 dx dy dyaw"""
        hidden = self.state(state)
        if self.use_bev:
            sites = self.sample_sites(center, tangent, lane_valid)
            height, width = bev.shape[-2:]
            grid = xy_to_norm(sites, height, width).view(1, sites.shape[0], sites.shape[1], 2)
            sampled = F.grid_sample(bev, grid, align_corners=True, padding_mode="zeros")
            tokens = self.proj(sampled[0].permute(1, 2, 0))
            if self.use_lane:
                tokens = tokens.clone()
                tokens[:, 9:] = tokens[:, 9:] + self.lane_token.view(1, 1, -1) * lane_valid.view(-1, 1, 1)
            query = hidden.unsqueeze(1)
            mixed, _ = self.attn(query, tokens, tokens)
            hidden = hidden + mixed.squeeze(1)
        exist = self.exist(hidden).squeeze(-1)
        delta = self.delta(hidden)
        return exist, delta


def wrap_angle(value):
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def pack_state(box, velocity, dt):
    return [
        float(box[0]) / 100.0,
        float(box[1]) / 40.0,
        float(box[2]),
        math.sin(float(box[3])),
        math.cos(float(box[3])),
        float(velocity[0]) / 10.0,
        float(velocity[1]) / 10.0,
        float(dt),
    ]


def motion_center(box, velocity, dt):
    return [float(box[0]) + float(velocity[0]) * float(dt), float(box[1]) + float(velocity[1]) * float(dt)]


def query_loss(exist_logit, delta, exist_target, box_target):
    prob = torch.sigmoid(exist_logit)
    ce = F.binary_cross_entropy_with_logits(exist_logit, exist_target, reduction="none")
    pt = prob * exist_target + (1.0 - prob) * (1.0 - exist_target)
    loss_exist = ((1.0 - pt).pow(2) * ce).mean()
    positive = exist_target > 0.5
    if torch.any(positive):
        loss_box = F.smooth_l1_loss(delta[positive], box_target[positive])
    else:
        loss_box = exist_logit.sum() * 0.0
    return loss_exist + loss_box, loss_exist.detach(), loss_box.detach()

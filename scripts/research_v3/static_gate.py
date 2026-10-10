"""Tiny residual gate for the fixed-view static BEV memory pilot."""

import torch
from torch import nn


class StaticGate(nn.Module):
    def __init__(self, residual_scale=0.2):
        super().__init__()
        self.residual_scale = float(residual_scale)
        self.net = nn.Sequential(nn.Conv2d(6, 16, 3, padding=1), nn.ReLU(),
                                 nn.Conv2d(16, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias, -2.0)

    def forward(self, current, dynamic, static, valid, hit):
        diff = static - current
        signals = torch.cat((current.abs().mean(1, keepdim=True),
                             static.abs().mean(1, keepdim=True),
                             diff.abs().mean(1, keepdim=True),
                             (dynamic - current).abs().mean(1, keepdim=True),
                             hit, valid.float()), dim=1)
        gate = torch.sigmoid(self.net(torch.tanh(signals))) * valid.float() * (1.0 - 0.8 * hit)
        return dynamic + self.residual_scale * gate * diff, gate


def parameter_count(module):
    return sum(p.numel() for p in module.parameters())

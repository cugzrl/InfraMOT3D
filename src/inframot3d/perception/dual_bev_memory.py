"""Causal two-stream BEV memory for fixed roadside sensors.

The input and output use a detector's native BEV channels. State lives on a
smaller grid and is never shared across sequence identifiers.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DualMemoryState:
    static: Tensor
    dynamic: Tensor
    sequence_id: str
    timestamp: float
    steps: int

    def detach(self):
        return DualMemoryState(
            self.static.detach(), self.dynamic.detach(), self.sequence_id,
            self.timestamp, self.steps,
        )


class ConvGRUCell(nn.Module):
    def __init__(self, channels: int, update_bias: float = 0.0):
        super().__init__()
        self.gates = nn.Conv2d(2 * channels, 2 * channels, 3, padding=1)
        self.candidate = nn.Conv2d(2 * channels, channels, 3, padding=1)
        nn.init.constant_(self.gates.bias[:channels], update_bias)

    def forward(self, observed: Tensor, previous: Tensor) -> Tensor:
        update, reset = self.gates(torch.cat((observed, previous), dim=1)).chunk(2, dim=1)
        update = torch.sigmoid(update)
        reset = torch.sigmoid(reset)
        proposal = torch.tanh(self.candidate(torch.cat((observed, reset * previous), dim=1)))
        return (1.0 - update) * previous + update * proposal


class DualBEVMemory(nn.Module):
    """Read old static/dynamic state, enhance BEV, then write current evidence.

    `flow` is a backward-sampling displacement in memory-grid cells. It is
    predicted from current and historical features, not from GT or D0 tracks.
    """

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int = 96,
        memory_stride: int = 2,
        max_dynamic_gap: float = 0.35,
        max_static_gap: float = 30.0,
        max_flow_cells: float = 8.0,
    ):
        super().__init__()
        if memory_stride < 1:
            raise ValueError("memory_stride must be positive")
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.memory_stride = memory_stride
        self.max_dynamic_gap = max_dynamic_gap
        self.max_static_gap = max_static_gap
        self.max_flow_cells = max_flow_cells

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, stride=memory_stride, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
        )
        flow_input = 3 * hidden_channels + 1
        self.flow_net = nn.Sequential(
            nn.Conv2d(flow_input, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, 2, 1),
        )
        nn.init.zeros_(self.flow_net[-1].weight)
        nn.init.zeros_(self.flow_net[-1].bias)

        route_input = 5 * hidden_channels + 2
        self.route_net = nn.Sequential(
            nn.Conv2d(route_input, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 3, 1),
        )
        self.dynamic_head = nn.Sequential(
            nn.Conv2d(3 * hidden_channels + 1, hidden_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        self.static_gru = ConvGRUCell(hidden_channels, update_bias=-2.0)
        self.dynamic_gru = ConvGRUCell(hidden_channels, update_bias=0.0)
        self.readout = nn.Sequential(
            nn.Conv2d(2 * hidden_channels + 3, hidden_channels, 3, padding=1),
            nn.GroupNorm(8, hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, input_channels, 1),
        )
        nn.init.zeros_(self.readout[-1].weight)
        nn.init.zeros_(self.readout[-1].bias)

    @staticmethod
    def _transport(history: Tensor, flow: Tensor) -> Tensor:
        batch, _, height, width = history.shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=history.device, dtype=history.dtype),
            torch.arange(width, device=history.device, dtype=history.dtype),
            indexing="ij",
        )
        x = xx[None] - flow[:, 0]
        y = yy[None] - flow[:, 1]
        grid = torch.stack(((x + 0.5) * (2.0 / width) - 1.0,
                            (y + 0.5) * (2.0 / height) - 1.0), dim=-1)
        return F.grid_sample(history, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def forward(
        self,
        current: Tensor,
        state: Optional[DualMemoryState],
        sequence_id: str,
        timestamp: float,
        lidar_hit: Optional[Tensor] = None,
        enabled: bool = True,
    ):
        if not enabled:
            return current, state, {}
        if current.ndim != 4 or current.shape[1] != self.input_channels:
            raise ValueError("Unexpected BEV feature shape")
        observed = self.encoder(current)
        batch, _, height, width = observed.shape
        valid = (
            state is not None and state.sequence_id == sequence_id
            and state.static.shape == observed.shape
            and 0.0 <= timestamp - state.timestamp <= self.max_static_gap
        )
        if valid:
            static = state.static
            dt = timestamp - state.timestamp
            dynamic_valid = dt <= self.max_dynamic_gap
            dynamic = state.dynamic if dynamic_valid else torch.zeros_like(observed)
            steps = state.steps
        else:
            static = torch.zeros_like(observed)
            dynamic = torch.zeros_like(observed)
            dt = 0.0
            dynamic_valid = False
            steps = 0

        if lidar_hit is None:
            hit = observed.new_zeros((batch, 1, height, width))
        else:
            hit = F.interpolate(lidar_hit.to(observed.dtype), size=(height, width), mode="area")
        delta = observed.new_full((batch, 1, height, width), min(dt / self.max_dynamic_gap, 1.0))
        flow_features = torch.cat((observed, dynamic, (observed - dynamic).abs(), delta), dim=1)
        flow = self.max_flow_cells * torch.tanh(self.flow_net(flow_features))
        transported = self._transport(dynamic, flow) if dynamic_valid else dynamic

        routing_input = torch.cat((
            observed, static, transported, (observed - static).abs(),
            (observed - transported).abs(), hit, delta,
        ), dim=1)
        route_logits = self.route_net(routing_input)
        if not valid:
            route_logits = torch.cat((
                route_logits[:, :2] - 30.0, route_logits[:, 2:3] + 30.0,
            ), dim=1)
        elif not dynamic_valid:
            route_logits = torch.cat((
                route_logits[:, :1], route_logits[:, 1:2] - 30.0, route_logits[:, 2:3],
            ), dim=1)
        route = route_logits.softmax(dim=1)
        read = route[:, :1] * static + route[:, 1:2] * transported + route[:, 2:3] * observed
        residual = self.readout(torch.cat((observed, read, route), dim=1))
        residual = F.interpolate(residual, size=current.shape[-2:], mode="bilinear", align_corners=False)
        enhanced = current + residual

        dynamic_logits = self.dynamic_head(torch.cat((observed, transported, static, hit), dim=1))
        dynamic_prob = dynamic_logits.sigmoid()
        static_write = observed * (1.0 - dynamic_prob)
        dynamic_write = observed * dynamic_prob
        next_static = self.static_gru(static_write, static)
        next_dynamic = self.dynamic_gru(dynamic_write, transported)
        next_state = DualMemoryState(next_static, next_dynamic, sequence_id, float(timestamp), steps + 1)
        aux = {"flow": flow, "route": route, "dynamic_logits": dynamic_logits,
               "memory_valid": valid, "dynamic_valid": dynamic_valid}
        return enhanced, next_state, aux

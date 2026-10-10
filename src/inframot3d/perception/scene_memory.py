"""Causal, detachable scene-level BEV memory for fixed roadside sensors.

``read`` sees only memory written before the current frame. ``write`` is called
after the current frame has been processed. The convenience ``forward`` keeps
the same ordering. State is caller-owned so interleaved sequences cannot share
memory accidentally.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class SceneMemoryState:
    feature: torch.Tensor
    valid: torch.Tensor
    sequence_id: str
    timestamp: float
    updates: int


class StaticGate(nn.Module):
    """Channel-agnostic residual read gate used by the D3 pilot."""

    def __init__(self, residual_scale: float = 0.2):
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


class LongTermSceneMemory(nn.Module):
    """Stateful-by-argument BEV read/write layer; detector weights stay intact.

    By default the memory is at the input channel count and resolution. A
    different ``memory_channels`` or ``memory_size`` creates a learnable adapter
    that must be trained for that backbone before quantitative use.
    """

    def __init__(self, channels: int, memory_channels: Optional[int] = None,
                 memory_size: Optional[Tuple[int, int]] = None,
                 gate: Optional[nn.Module] = None, read_weight: float = 0.0,
                 write_rate: float = 0.05, max_gap: float = 0.3,
                 enabled: bool = True):
        super().__init__()
        if channels <= 0 or (memory_channels is not None and memory_channels <= 0):
            raise ValueError("channel counts must be positive")
        if not 0.0 <= read_weight <= 1.0 or not 0.0 < write_rate <= 1.0:
            raise ValueError("read_weight and write_rate must be in [0,1]")
        self.channels = channels
        self.memory_channels = memory_channels or channels
        self.memory_size = memory_size
        self.gate = gate
        self.read_weight = float(read_weight)
        self.write_rate = float(write_rate)
        self.max_gap = float(max_gap)
        self.enabled = enabled
        self.encode = (nn.Identity() if self.memory_channels == channels else
                       nn.Conv2d(channels, self.memory_channels, 1))
        self.decode = (nn.Identity() if self.memory_channels == channels else
                       nn.Conv2d(self.memory_channels, channels, 1))

    @staticmethod
    def reset() -> None:
        return None

    def _active(self, state, sequence_id, timestamp):
        if state is None or state.sequence_id != sequence_id:
            return None
        dt = float(timestamp) - state.timestamp
        if dt < 0 or dt > self.max_gap:
            return None
        return state

    def _memory_size(self, current):
        return self.memory_size or current.shape[-2:]

    def _encode(self, current):
        feature = self.encode(current)
        size = self._memory_size(current)
        if feature.shape[-2:] != size:
            feature = F.interpolate(feature, size=size, mode="bilinear", align_corners=False)
        return feature

    def read(self, current, dynamic, state, sequence_id, timestamp, hit=None):
        if not self.enabled:
            return current, None, None
        if current.shape != dynamic.shape or current.shape[1] != self.channels:
            raise ValueError("current/dynamic BEV shapes or channel count differ")
        state = self._active(state, sequence_id, timestamp)
        if state is None:
            return dynamic, None, None
        if hit is None:
            hit = current.new_zeros((current.shape[0], 1, *current.shape[-2:]))
        static = self.decode(state.feature)
        if static.shape[-2:] != current.shape[-2:]:
            static = F.interpolate(static, size=current.shape[-2:],
                                   mode="bilinear", align_corners=False)
        valid = F.interpolate(state.valid.float(), size=current.shape[-2:],
                              mode="nearest").bool()
        if self.gate is not None:
            enhanced, weight = self.gate(current, dynamic, static, valid, hit)
        else:
            weight = self.read_weight * valid.float() * (1.0 - hit)
            enhanced = dynamic + weight * (static - current)
        return enhanced, state, weight

    def write(self, current, state, sequence_id, timestamp, hit=None,
              dynamic_mask=None):
        if not self.enabled:
            return None
        state = self._active(state, sequence_id, timestamp)
        feature = self._encode(current).detach()
        size = feature.shape[-2:]
        if hit is None:
            hit = current.new_ones((current.shape[0], 1, *current.shape[-2:]))
        if dynamic_mask is None:
            dynamic_mask = torch.zeros_like(hit)
        observed = F.interpolate(hit.float(), size=size, mode="nearest") > 0
        dynamic = F.interpolate(dynamic_mask.float(), size=size, mode="nearest") > 0
        keep = observed & ~dynamic
        if state is None or state.feature.shape != feature.shape:
            previous = torch.zeros_like(feature)
            valid = torch.zeros_like(keep)
            updates = 0
        else:
            previous, valid, updates = state.feature, state.valid, state.updates
        fresh = keep & ~valid
        old = keep & valid
        next_feature = torch.where(fresh, feature, previous)
        next_feature = torch.where(old,
                                   (1.0 - self.write_rate) * previous + self.write_rate * feature,
                                   next_feature)
        return SceneMemoryState(next_feature, valid | keep, sequence_id,
                                float(timestamp), updates + 1)

    def forward(self, current, state, sequence_id, timestamp, dynamic=None,
                hit=None, dynamic_mask=None):
        if dynamic is None:
            dynamic = current
        enhanced, state, weight = self.read(current, dynamic, state,
                                            sequence_id, timestamp, hit)
        state = self.write(current, state, sequence_id, timestamp, hit,
                           dynamic_mask)
        return enhanced, state, weight

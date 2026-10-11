"""Persistent object queries with local multi-head BEV cross-attention."""

from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inframot3d.perception.persistent_query_feedback import (
    PersistentQueryFeedback, QueryBankState,
)


class AttentiveQueryFeedback(PersistentQueryFeedback):
    """Read a local current-frame BEV patch for each propagated target query.

    The query bank remains the dynamic state. Feedback is local and starts
    conservatively; no history means exactly the original detector feature.
    """

    def __init__(self, bev_channels: int, query_channels: int = 64,
                 num_heads: int = 4, patch_size: int = 5, **kwargs):
        super().__init__(bev_channels, query_channels, **kwargs)
        if patch_size % 2 != 1 or patch_size < 3:
            raise ValueError("patch_size must be odd and at least 3")
        if query_channels % num_heads:
            raise ValueError("query_channels must be divisible by num_heads")
        self.patch_size = patch_size
        offsets = torch.arange(patch_size, dtype=torch.float32) - patch_size // 2
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        self.register_buffer("patch_offsets", torch.stack((xx, yy), dim=-1).reshape(-1, 2))
        self.position_embedding = nn.Parameter(torch.empty(patch_size ** 2, query_channels))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.metadata_encoder = nn.Sequential(
            nn.Linear(4, query_channels), nn.SiLU(),
            nn.Linear(query_channels, query_channels),
        )
        self.query_norm = nn.LayerNorm(query_channels)
        self.key_norm = nn.LayerNorm(query_channels)
        self.cross_attention = nn.MultiheadAttention(
            query_channels, num_heads, dropout=0.0, batch_first=True,
        )
        self.attention_norm = nn.LayerNorm(query_channels)
        self.feed_forward = nn.Sequential(
            nn.Linear(query_channels, 4 * query_channels), nn.SiLU(),
            nn.Linear(4 * query_channels, query_channels),
        )
        self.output_norm = nn.LayerNorm(query_channels)
        self.feedback_logit = nn.Parameter(torch.tensor(-2.0))

    def _sample_patch(self, encoded: Tensor, centers: Tensor) -> Tensor:
        _, _, height, width = encoded.shape
        pixel = torch.stack((
            (centers[:, 0] - self.origin[0]) / self.cell_size[0],
            (centers[:, 1] - self.origin[1]) / self.cell_size[1],
        ), dim=-1)
        pixel = pixel[:, None, :] + self.patch_offsets.to(pixel.dtype)[None]
        grid = torch.stack((
            (pixel[..., 0] + 0.5) * (2.0 / width) - 1.0,
            (pixel[..., 1] + 0.5) * (2.0 / height) - 1.0,
        ), dim=-1)
        patch = F.grid_sample(encoded, grid[None], padding_mode="zeros",
                              align_corners=False)
        return patch[0].permute(1, 2, 0)

    def read(self, bev: Tensor, state: Optional[QueryBankState],
             sequence_id: str, timestamp: float, enabled: bool = True):
        if not enabled:
            return bev, state, {"valid": False, "count": 0}
        if bev.shape[0] != 1 or bev.shape[1] != self.bev_channels:
            raise ValueError("Attention feedback expects one native-channel BEV frame")
        if (state is None or state.sequence_id != sequence_id
                or not 0.0 <= timestamp - state.timestamp <= self.max_gap):
            return bev, self.empty(bev, sequence_id, timestamp), {"valid": False, "count": 0}
        if len(state.ids) == 0:
            return bev, state, {"valid": False, "count": 0}

        dt = max(float(timestamp - state.timestamp), 1e-4)
        propagated = state.boxes[:, :2] + state.velocity * dt
        encoded = self.bev_encoder(bev)
        patch = self._sample_patch(encoded, propagated)
        metadata = torch.stack((
            state.confidence, state.uncertainty / 10.0,
            state.misses.float() / max(self.max_misses, 1),
            state.confidence.new_full(state.confidence.shape, dt / self.max_gap),
        ), dim=1)
        query = self.query_norm(state.embedding + self.metadata_encoder(metadata))
        keys = self.key_norm(patch + self.position_embedding[None])
        attended, weights = self.cross_attention(
            query[:, None], keys, keys, need_weights=True,
        )
        attended = self.attention_norm(query + attended[:, 0])
        attended = self.output_norm(attended + self.feed_forward(attended))
        joint = torch.cat((state.embedding, attended, metadata), dim=1)
        delta = 2.0 * torch.tanh(self.motion_head(joint))
        centers = propagated + delta
        refreshed = self.observation_update(attended, state.embedding)
        existence_logits = self.reliability(joint).flatten()
        reliability = torch.sigmoid(existence_logits) * state.confidence
        feedback_gain = torch.sigmoid(self.feedback_logit)
        feedback = self._splat(refreshed, centers, reliability * feedback_gain, bev.shape)
        residual = self.feedback_proj(feedback)
        enhanced = bev + residual
        attention_prob = weights[:, 0]
        attention_offset = (attention_prob[..., None] * self.patch_offsets[None]).sum(1)
        updated_boxes = torch.cat((centers, state.boxes[:, 2:]), dim=1)
        read_state = QueryBankState(
            refreshed, updated_boxes, state.velocity, state.confidence,
            state.uncertainty, state.labels, state.ids, state.misses,
            sequence_id, state.timestamp, state.next_id,
        )
        return enhanced, read_state, {
            "valid": True, "count": len(state.ids), "predicted_centers": centers,
            "reliability": reliability, "existence_logits": existence_logits,
            "attention_entropy": -(attention_prob.clamp_min(1e-8) * attention_prob.clamp_min(1e-8).log()).sum(-1).mean(),
            "attention_peak": attention_prob.amax(-1).mean(),
            "attention_center": attention_prob[:, self.patch_size ** 2 // 2].mean(),
            "attention_offset_cells": attention_offset.norm(dim=-1).mean(),
            "feedback_gain": feedback_gain,
            "feedback_rms_ratio": residual.square().mean().sqrt() / bev.square().mean().sqrt().clamp_min(1e-6),
        }

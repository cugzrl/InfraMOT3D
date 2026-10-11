"""Causal multi-object query bank with compact target-local memory tokens."""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inframot3d.perception.attentive_query_feedback import AttentiveQueryFeedback
from inframot3d.perception.persistent_query_feedback import QueryBankState


@dataclass
class LongTermQueryState(QueryBankState):
    tokens: Tensor
    hits: Tensor

    def detach(self):
        return LongTermQueryState(
            self.embedding.detach(), self.boxes.detach(), self.velocity.detach(),
            self.confidence.detach(), self.uncertainty.detach(), self.labels.detach(),
            self.ids.detach(), self.misses.detach(), self.sequence_id,
            self.timestamp, self.next_id, self.tokens.detach(), self.hits.detach(),
        )


class TargetMemoryLayer(nn.Module):
    """ChronoTrack-style token-to-observation cross-attention and token mixing."""

    def __init__(self, channels: int, heads: int):
        super().__init__()
        self.memory_norm = nn.LayerNorm(channels)
        self.observation_norm = nn.LayerNorm(channels)
        self.cross = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.self_attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.self_norm = nn.LayerNorm(channels)
        self.ffn_norm = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, 4 * channels), nn.GELU(),
            nn.Linear(4 * channels, channels),
        )
        self.token_indicator = nn.Parameter(torch.zeros(1, 1, channels))

    def forward(self, memory: Tensor, observation: Tensor, position: Tensor):
        old = self.memory_norm(memory)
        current = self.observation_norm(observation + position)
        keys = torch.cat((old, current), dim=1)
        values = torch.cat((old + self.token_indicator, current), dim=1)
        update, _ = self.cross(old, keys, values, need_weights=False)
        memory = memory + update
        normalized = self.self_norm(memory)
        update, _ = self.self_attention(normalized, normalized, normalized,
                                        need_weights=False)
        memory = memory + update
        return memory + self.ffn(self.ffn_norm(memory))


class LongTermQueryFeedback(AttentiveQueryFeedback):
    """Target-local tokens, reliability-routed readout, and online Query Bank."""

    def __init__(self, bev_channels: int, query_channels: int = 64,
                 num_tokens: int = 4, num_heads: int = 4, patch_size: int = 5,
                 **kwargs):
        super().__init__(bev_channels, query_channels, num_heads=num_heads,
                         patch_size=patch_size, **kwargs)
        self.num_tokens = num_tokens
        self.initial_tokens = nn.Parameter(torch.randn(num_tokens, query_channels) * 0.02)
        self.local_position = nn.Sequential(
            nn.Linear(2, query_channels), nn.GELU(),
            nn.Linear(query_channels, query_channels),
        )
        self.memory_layers = nn.ModuleList(
            [TargetMemoryLayer(query_channels, num_heads) for _ in range(2)]
        )
        self.memory_read = nn.MultiheadAttention(query_channels, num_heads,
                                                  batch_first=True)
        self.memory_read_norm = nn.LayerNorm(query_channels)
        self.memory_read_logit = nn.Parameter(torch.tensor(-1.5))
        self.token_refresh = nn.Sequential(
            nn.Linear(2 * query_channels + 5, query_channels), nn.GELU(),
            nn.Linear(query_channels, 1),
        )
        self.route = nn.Sequential(
            nn.Linear(3 * query_channels + 7, query_channels), nn.GELU(),
            nn.Linear(query_channels, 1),
        )
        self.identity_projection = nn.Sequential(
            nn.Linear(2 * query_channels, query_channels),
            nn.LayerNorm(query_channels),
        )
        self.feedback_logit = nn.Parameter(torch.tensor(-1.5))

    def empty(self, bev: Tensor, sequence_id: str, timestamp: float):
        base = super().empty(bev, sequence_id, timestamp)
        return LongTermQueryState(
            base.embedding, base.boxes, base.velocity, base.confidence,
            base.uncertainty, base.labels, base.ids, base.misses,
            sequence_id, float(timestamp), base.next_id,
            bev.new_empty((0, self.num_tokens, self.query_channels)),
            torch.empty((0,), device=bev.device, dtype=torch.long),
        )

    def _sample_local_patch(self, encoded: Tensor, centers: Tensor,
                            heading: Tensor) -> Tensor:
        if centers.shape[0] == 0:
            return encoded.new_empty((0, self.patch_size ** 2, self.query_channels))
        _, _, height, width = encoded.shape
        offsets = self.patch_offsets.to(encoded.dtype)
        cos, sin = heading.cos(), heading.sin()
        x = offsets[None, :, 0] * cos[:, None] - offsets[None, :, 1] * sin[:, None]
        y = offsets[None, :, 0] * sin[:, None] + offsets[None, :, 1] * cos[:, None]
        px = (centers[:, None, 0] - self.origin[0]) / self.cell_size[0] + x
        py = (centers[:, None, 1] - self.origin[1]) / self.cell_size[1] + y
        grid = torch.stack(((px + 0.5) * 2.0 / width - 1.0,
                            (py + 0.5) * 2.0 / height - 1.0), dim=-1)
        sampled = F.grid_sample(encoded, grid[None], padding_mode="zeros",
                                align_corners=False)
        return sampled[0].permute(1, 2, 0)

    def _position(self, dtype: torch.dtype):
        return self.local_position(self.patch_offsets.to(dtype)[None]
                                   / max(self.patch_size // 2, 1))

    def identity_features(self, state: LongTermQueryState) -> Tensor:
        joint = torch.cat((state.embedding, state.tokens.mean(dim=1)), dim=-1)
        return F.normalize(self.identity_projection(joint), dim=-1)

    def token_diversity(self, state: LongTermQueryState) -> Tensor:
        if len(state.ids) == 0:
            return state.embedding.new_zeros(())
        tokens = F.normalize(state.tokens, dim=-1)
        similarity = tokens @ tokens.transpose(1, 2)
        eye = torch.eye(self.num_tokens, device=tokens.device, dtype=tokens.dtype)
        return ((similarity - eye) ** 2).mean()

    def read(self, bev: Tensor, state: Optional[LongTermQueryState],
             sequence_id: str, timestamp: float, enabled: bool = True):
        if not enabled:
            return bev, state, {"valid": False, "count": 0}
        if bev.shape[0] != 1 or bev.shape[1] != self.bev_channels:
            raise ValueError("Long-term feedback expects one native-channel BEV frame")
        if (state is None or state.sequence_id != sequence_id
                or not 0.0 <= timestamp - state.timestamp <= self.max_gap):
            return bev, self.empty(bev, sequence_id, timestamp), {"valid": False, "count": 0}
        if len(state.ids) == 0:
            return bev, state, {"valid": False, "count": 0}

        dt = max(float(timestamp - state.timestamp), 1e-4)
        predicted = state.boxes[:, :2] + state.velocity * dt
        encoded = self.bev_encoder(bev)
        patch = self._sample_local_patch(encoded, predicted, state.boxes[:, 6])
        local = patch + self._position(patch.dtype)
        memory_context, _ = self.memory_read(
            state.embedding[:, None], state.tokens, state.tokens,
            need_weights=False,
        )
        memory_context = (torch.sigmoid(self.memory_read_logit)
                          * self.memory_read_norm(memory_context[:, 0]))
        metadata = torch.stack((
            state.confidence, state.uncertainty / 10.0,
            state.misses.float() / max(self.max_misses, 1),
            state.confidence.new_full(state.confidence.shape, dt / self.max_gap),
        ), dim=1)
        query = self.query_norm(state.embedding + self.metadata_encoder(metadata)
                                + memory_context)
        attended, weights = self.cross_attention(
            query[:, None], self.key_norm(local), self.key_norm(local),
            need_weights=True,
        )
        attended = self.attention_norm(query + attended[:, 0])
        attended = self.output_norm(attended + self.feed_forward(attended))
        joint = torch.cat((state.embedding, attended, metadata), dim=1)
        delta = 2.0 * torch.tanh(self.motion_head(joint))
        centers = predicted + delta
        refreshed = self.observation_update(attended, state.embedding)
        existence_logits = self.reliability(joint).flatten()
        agreement = F.cosine_similarity(attended, memory_context, dim=-1)
        observed_strength = patch.square().mean(dim=(1, 2)).sqrt()
        age = (state.hits.float() / 4.0).clamp(max=1.0)
        route_input = torch.cat((
            state.embedding, attended, memory_context, metadata,
            agreement[:, None], observed_strength[:, None], age[:, None],
        ), dim=-1)
        route = torch.sigmoid(self.route(route_input)).flatten()
        reliability = (torch.sigmoid(existence_logits) * state.confidence
                       * age * route)
        feedback = self._splat(refreshed + memory_context, centers,
                               reliability * torch.sigmoid(self.feedback_logit),
                               bev.shape)
        residual = self.feedback_proj(feedback)
        updated_boxes = torch.cat((centers, state.boxes[:, 2:]), dim=1)
        read_state = LongTermQueryState(
            refreshed, updated_boxes, state.velocity, state.confidence,
            state.uncertainty, state.labels, state.ids, state.misses,
            sequence_id, state.timestamp, state.next_id, state.tokens, state.hits,
        )
        return bev + residual, read_state, {
            "valid": True, "count": len(state.ids), "predicted_centers": centers,
            "reliability": reliability, "existence_logits": existence_logits,
            "route_mean": route.mean(), "agreement_mean": agreement.mean(),
            "feedback_rms_ratio": residual.square().mean().sqrt()
                                  / bev.square().mean().sqrt().clamp_min(1e-6),
            "attention_entropy": -(weights[:, 0].clamp_min(1e-8)
                                   * weights[:, 0].clamp_min(1e-8).log()).sum(-1).mean(),
        }

    def update(self, bev: Tensor, state: LongTermQueryState, boxes: Tensor,
               scores: Tensor, labels: Tensor, timestamp: float):
        base = super().update(bev, state, boxes, scores, labels, timestamp)
        if len(base.ids) == 0:
            empty = self.empty(bev, state.sequence_id, timestamp)
            empty.next_id = base.next_id
            return empty
        index = {int(identity): i for i, identity in enumerate(state.ids.tolist())}
        previous = torch.stack([
            state.tokens[index[int(identity)]] if int(identity) in index
            else self.initial_tokens
            for identity in base.ids.tolist()
        ])
        hits = torch.stack([
            state.hits[index[int(identity)]] + (base.misses[j] == 0).long()
            if int(identity) in index else base.misses.new_tensor(1)
            for j, identity in enumerate(base.ids.tolist())
        ])
        observed = base.misses == 0
        if bool(observed.any()):
            encoded = self.bev_encoder(bev)
            patches = self._sample_local_patch(
                encoded, base.boxes[observed, :2], base.boxes[observed, 6],
            )
            old = previous[observed]
            updated = old
            for layer in self.memory_layers:
                updated = layer(updated, patches, self._position(patches.dtype))
            mean_observation = patches.mean(dim=1)
            meta = torch.stack((
                base.confidence[observed], base.uncertainty[observed] / 10.0,
                hits[observed].float().clamp(max=10) / 10.0,
                base.velocity[observed, 0] / 30.0,
                base.velocity[observed, 1] / 30.0,
            ), dim=-1)
            gate_input = torch.cat((old, mean_observation[:, None].expand_as(old),
                                    meta[:, None].expand(-1, self.num_tokens, -1)), dim=-1)
            gate = torch.sigmoid(self.token_refresh(gate_input))
            merged = old + gate * (updated - old)
            previous = previous.index_copy(0, torch.where(observed)[0], merged)
        return LongTermQueryState(
            base.embedding, base.boxes, base.velocity, base.confidence,
            base.uncertainty, base.labels, base.ids, base.misses,
            base.sequence_id, base.timestamp, base.next_id, previous, hits,
        )

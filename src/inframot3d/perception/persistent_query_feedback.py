"""Online object-query memory with pre-head BEV feedback.

Queries are born from the detector's own boxes. Their identities, appearance,
position, velocity, confidence, and uncertainty persist causally across frames.
No ground truth or future observation enters this module at inference.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class QueryBankState:
    embedding: Tensor
    boxes: Tensor
    velocity: Tensor
    confidence: Tensor
    uncertainty: Tensor
    labels: Tensor
    ids: Tensor
    misses: Tensor
    sequence_id: str
    timestamp: float
    next_id: int

    def detach(self):
        return QueryBankState(
            self.embedding.detach(), self.boxes.detach(), self.velocity.detach(),
            self.confidence.detach(), self.uncertainty.detach(), self.labels.detach(),
            self.ids.detach(), self.misses.detach(), self.sequence_id,
            self.timestamp, self.next_id,
        )


class PersistentQueryFeedback(nn.Module):
    """Read propagated online queries before detection and update afterward."""

    def __init__(self, bev_channels: int, query_channels: int = 64,
                 max_queries: int = 64, birth_score: float = 0.45,
                 max_gap: float = 0.35, max_misses: int = 3,
                 origin=(0.0, -56.0), cell_size=(0.8, 0.8)):
        super().__init__()
        self.bev_channels = bev_channels
        self.query_channels = query_channels
        self.max_queries = max_queries
        self.birth_score = birth_score
        self.max_gap = max_gap
        self.max_misses = max_misses
        self.origin = origin
        self.cell_size = cell_size
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(bev_channels, query_channels, 1),
            nn.GroupNorm(8, query_channels), nn.SiLU(inplace=True),
        )
        self.box_encoder = nn.Sequential(nn.Linear(7, query_channels), nn.LayerNorm(query_channels))
        self.class_embedding = nn.Embedding(32, query_channels)
        self.observation_update = nn.GRUCell(query_channels, query_channels)
        self.detection_update = nn.GRUCell(query_channels, query_channels)
        self.motion_head = nn.Sequential(
            nn.Linear(2 * query_channels + 4, query_channels), nn.SiLU(),
            nn.Linear(query_channels, 2),
        )
        nn.init.zeros_(self.motion_head[-1].weight)
        nn.init.zeros_(self.motion_head[-1].bias)
        self.reliability = nn.Sequential(
            nn.Linear(2 * query_channels + 4, query_channels), nn.SiLU(),
            nn.Linear(query_channels, 1),
        )
        self.feedback_proj = nn.Conv2d(query_channels, bev_channels, 1)
        nn.init.zeros_(self.feedback_proj.weight)
        nn.init.zeros_(self.feedback_proj.bias)

    def empty(self, bev: Tensor, sequence_id: str, timestamp: float):
        device = bev.device
        dtype = bev.dtype
        return QueryBankState(
            bev.new_empty((0, self.query_channels)), bev.new_empty((0, 7)),
            bev.new_empty((0, 2)), bev.new_empty((0,)), bev.new_empty((0,)),
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.long),
            torch.empty((0,), device=device, dtype=torch.long),
            sequence_id, float(timestamp), 0,
        )

    def _sample(self, encoded: Tensor, centers: Tensor):
        if centers.numel() == 0:
            return encoded.new_empty((0, self.query_channels))
        _, _, height, width = encoded.shape
        px = (centers[:, 0] - self.origin[0]) / self.cell_size[0]
        py = (centers[:, 1] - self.origin[1]) / self.cell_size[1]
        grid = torch.stack(((px + 0.5) * 2 / width - 1,
                            (py + 0.5) * 2 / height - 1), dim=-1)
        sampled = F.grid_sample(encoded, grid[None, :, None], padding_mode="zeros", align_corners=False)
        return sampled[0, :, :, 0].transpose(0, 1)

    def _splat(self, embeddings: Tensor, centers: Tensor, weights: Tensor, shape):
        _, _, height, width = shape
        yy, xx = torch.meshgrid(
            torch.arange(height, device=embeddings.device, dtype=embeddings.dtype),
            torch.arange(width, device=embeddings.device, dtype=embeddings.dtype),
            indexing="ij",
        )
        px = (centers[:, 0] - self.origin[0]) / self.cell_size[0]
        py = (centers[:, 1] - self.origin[1]) / self.cell_size[1]
        distance = ((xx[None] - px[:, None, None]) ** 2
                    + (yy[None] - py[:, None, None]) ** 2)
        kernel = torch.exp(-distance / (2.0 * 2.5 ** 2)) * weights[:, None, None]
        gathered = torch.einsum("khw,kc->chw", kernel, embeddings)
        normalizer = kernel.sum(dim=0).clamp_min(1.0)
        return (gathered / normalizer[None])[None]

    def read(self, bev: Tensor, state: Optional[QueryBankState],
             sequence_id: str, timestamp: float, enabled: bool = True):
        if not enabled:
            return bev, state, {"valid": False, "count": 0}
        if bev.shape[0] != 1 or bev.shape[1] != self.bev_channels:
            raise ValueError("Query feedback expects one native-channel BEV frame")
        if (state is None or state.sequence_id != sequence_id
                or not 0.0 <= timestamp - state.timestamp <= self.max_gap):
            return bev, self.empty(bev, sequence_id, timestamp), {"valid": False, "count": 0}
        if len(state.ids) == 0:
            return bev, state, {"valid": False, "count": 0}
        dt = max(float(timestamp - state.timestamp), 1e-4)
        predicted = state.boxes[:, :2] + state.velocity * dt
        encoded = self.bev_encoder(bev)
        observed = self._sample(encoded, predicted)
        metadata = torch.stack((state.confidence, state.uncertainty,
                                state.misses.float() / max(self.max_misses, 1),
                                state.confidence.new_full(state.confidence.shape, dt)), dim=1)
        joint = torch.cat((state.embedding, observed, metadata), dim=1)
        delta = 2.0 * torch.tanh(self.motion_head(joint))
        centers = predicted + delta
        refreshed = self.observation_update(observed, state.embedding)
        existence_logits = self.reliability(joint).flatten()
        reliability = torch.sigmoid(existence_logits) * state.confidence
        feedback = self._splat(refreshed, centers, reliability, bev.shape)
        enhanced = bev + self.feedback_proj(feedback)
        updated_boxes = torch.cat((centers, state.boxes[:, 2:]), dim=1)
        read_state = QueryBankState(
            refreshed, updated_boxes, state.velocity, state.confidence,
            state.uncertainty, state.labels, state.ids, state.misses,
            sequence_id, state.timestamp, state.next_id,
        )
        return enhanced, read_state, {"valid": True, "count": len(state.ids),
                                      "predicted_centers": centers,
                                      "reliability": reliability,
                                      "existence_logits": existence_logits}

    def update(self, bev: Tensor, state: QueryBankState, boxes: Tensor,
               scores: Tensor, labels: Tensor, timestamp: float):
        """Associate current detections and keep unmatched queries briefly."""
        boxes = boxes.detach()[:, :7]
        scores = scores.detach()
        labels = labels.detach().long()
        eligible = torch.where(scores >= 0.1)[0]
        eligible = eligible[torch.argsort(scores[eligible], descending=True)[:self.max_queries * 2]]
        boxes, scores, labels = boxes[eligible], scores[eligible], labels[eligible]
        encoded = self.bev_encoder(bev)
        observations = self._sample(encoded, boxes[:, :2])
        old_count = len(state.ids)
        pairs = []
        if old_count and len(boxes):
            centers = state.boxes[:, :2]
            distances = torch.cdist(centers.float(), boxes[:, :2].float())
            similarities = F.normalize(state.embedding.detach(), dim=1) @ F.normalize(observations.detach(), dim=1).T
            cost = distances + 1.5 * (1.0 - similarities)
            cost = cost + (state.labels[:, None] != labels[None, :]).float() * 1e4
            cost = cost.masked_fill(distances > 6.0, 1e4)
            old_indices, det_indices = linear_sum_assignment(cost.detach().cpu().numpy())
            pairs = [(int(i), int(j)) for i, j in zip(old_indices, det_indices) if float(cost[i, j]) < 8.0]
        match_old = {i: j for i, j in pairs}
        match_det = {j for _, j in pairs}
        entries = []
        dt = max(float(timestamp - state.timestamp), 1e-3)
        for i in range(old_count):
            if i in match_old:
                j = match_old[i]
                feature = observations[j] + self.box_encoder(boxes[j]) + self.class_embedding(labels[j].clamp(0, 31))
                embedding = self.detection_update(feature[None], state.embedding[i:i + 1])[0]
                previous_center = state.boxes[i, :2] - state.velocity[i] * dt
                velocity = 0.6 * state.velocity[i] + 0.4 * (boxes[j, :2] - previous_center) / dt
                velocity = velocity * (30.0 / velocity.norm().clamp_min(30.0))
                innovation = (boxes[j, :2] - state.boxes[i, :2]).norm()
                entries.append((embedding, boxes[j], velocity, scores[j],
                                (0.7 * state.uncertainty[i] + 0.3 * innovation).clamp(0, 10),
                                labels[j], state.ids[i], state.misses[i] * 0))
            elif int(state.misses[i]) < self.max_misses:
                entries.append((state.embedding[i], state.boxes[i], state.velocity[i],
                                state.confidence[i] * 0.7,
                                (state.uncertainty[i] + 0.5).clamp(max=10),
                                state.labels[i], state.ids[i], state.misses[i] + 1))
        next_id = state.next_id
        for j in range(len(boxes)):
            if j in match_det or float(scores[j]) < self.birth_score:
                continue
            feature = observations[j] + self.box_encoder(boxes[j]) + self.class_embedding(labels[j].clamp(0, 31))
            entries.append((feature, boxes[j], boxes[j].new_zeros(2), scores[j],
                            scores[j].new_tensor(1.0 - float(scores[j])), labels[j],
                            labels[j].new_tensor(next_id), labels[j].new_tensor(0)))
            next_id += 1
        entries.sort(key=lambda item: float(item[3]), reverse=True)
        entries = entries[:self.max_queries]
        if not entries:
            empty = self.empty(bev, state.sequence_id, timestamp)
            empty.next_id = next_id
            return empty
        values = [torch.stack([item[k] for item in entries]) for k in range(8)]
        return QueryBankState(*values, state.sequence_id, float(timestamp), next_id)

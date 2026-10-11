"""Current-evidence-aware readout for persistent target memory."""

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F

from inframot3d.perception.long_term_query_feedback import (
    LongTermQueryFeedback, LongTermQueryState,
)


class ReliableQueryFeedback(LongTermQueryFeedback):
    """Use a preliminary current-only detection pass to route history safely."""

    @staticmethod
    def _current_support(centers: Tensor, labels: Tensor,
                         boxes: Tensor, scores: Tensor, pred_labels: Tensor):
        if boxes.numel() == 0:
            return centers.new_zeros(len(centers))
        distances = torch.cdist(centers.detach().float(), boxes[:, :2].detach().float())
        compatible = labels[:, None] == pred_labels[None, :]
        support = (scores.detach()[None] * torch.exp(-distances.square() / 8.0)
                   * compatible.to(scores.dtype))
        return support.amax(dim=1).to(centers.dtype)

    def read(self, bev: Tensor, state: Optional[LongTermQueryState],
             sequence_id: str, timestamp: float, enabled: bool = True,
             current_boxes: Optional[Tensor] = None,
             current_scores: Optional[Tensor] = None,
             current_labels: Optional[Tensor] = None):
        if not enabled:
            return bev, state, {"valid": False, "count": 0}
        if bev.shape[0] != 1 or bev.shape[1] != self.bev_channels:
            raise ValueError("Reliable Query feedback expects one native-channel BEV frame")
        if (state is None or state.sequence_id != sequence_id
                or not 0.0 <= timestamp - state.timestamp <= self.max_gap):
            return bev, self.empty(bev, sequence_id, timestamp), {"valid": False, "count": 0}
        if len(state.ids) == 0:
            return bev, state, {"valid": False, "count": 0}
        if current_boxes is None or current_scores is None or current_labels is None:
            raise ValueError("Reliable feedback requires causal current-only detections")

        dt = max(float(timestamp - state.timestamp), 1e-4)
        predicted = state.boxes[:, :2] + state.velocity * dt
        encoded = self.bev_encoder(bev)
        patch = self._sample_local_patch(encoded, predicted, state.boxes[:, 6])
        local = patch + self._position(patch.dtype)
        memory_context, _ = self.memory_read(
            state.embedding[:, None], state.tokens, state.tokens, need_weights=False,
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
        support = self._current_support(
            centers, state.labels, current_boxes, current_scores, current_labels,
        )
        age = (state.hits.float() / 4.0).clamp(max=1.0)
        route_input = torch.cat((
            state.embedding, attended, memory_context, metadata,
            agreement[:, None], support[:, None], age[:, None],
        ), dim=-1)
        route_logits = self.route(route_input).flatten()
        route = torch.sigmoid(route_logits)
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
        attention = weights[:, 0].clamp_min(1e-8)
        return bev + residual, read_state, {
            "valid": True, "count": len(state.ids), "predicted_centers": centers,
            "reliability": reliability, "existence_logits": existence_logits,
            "route_logits": route_logits, "route_mean": route.mean(),
            "current_support": support, "support_mean": support.mean(),
            "agreement_mean": agreement.mean(),
            "feedback_rms_ratio": residual.square().mean().sqrt()
                                  / bev.square().mean().sqrt().clamp_min(1e-6),
            "attention_entropy": -(attention * attention.log()).sum(-1).mean(),
        }

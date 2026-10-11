"""Causal state and gradient tests for the long-term Query module."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from inframot3d.perception.long_term_query_feedback import LongTermQueryFeedback


def main():
    torch.manual_seed(11)
    query = LongTermQueryFeedback(32, query_channels=32, max_queries=4)
    query.train()
    first = torch.randn(1, 32, 32, 32, requires_grad=True)
    box = torch.tensor([[10.0, -40.0, 0.0, 4.0, 2.0, 2.0, 0.0]])
    score = torch.tensor([0.9])
    label = torch.tensor([1])
    unmodified, state, info = query.read(first, None, "A", 0.0)
    assert torch.equal(unmodified, first) and not info["valid"]
    state = query.update(first, state, box, score, label, 0.0)
    assert state.tokens.shape == (1, 4, 32) and state.hits.tolist() == [1]
    second = torch.randn(1, 32, 32, 32, requires_grad=True)
    enhanced, read_state, info = query.read(second, state, "A", 0.1)
    assert info["valid"] and info["count"] == 1
    assert enhanced.shape == second.shape
    assert 0.0 <= float(info["route_mean"]) <= 1.0
    updated = query.update(second, read_state, box, score, label, 0.1)
    assert updated.ids.tolist() == [0] and updated.hits.tolist() == [2]
    loss = enhanced.square().mean() + updated.tokens.square().mean()
    loss.backward()
    assert first.grad is not None and first.grad.abs().sum() > 0
    assert second.grad is not None and second.grad.abs().sum() > 0
    assert query.memory_layers[0].cross.in_proj_weight.grad is not None
    disabled, _, _ = query.read(second.detach(), updated.detach(), "A", 0.2,
                                enabled=False)
    assert torch.equal(disabled, second.detach())
    _, reset, reset_info = query.read(second.detach(), updated.detach(), "B", 0.2)
    assert not reset_info["valid"] and len(reset.ids) == 0
    print("long-term query causal state and gradient tests passed")


if __name__ == "__main__":
    main()

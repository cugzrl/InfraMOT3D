import unittest

import torch
from torch import nn

from inframot3d.perception.dual_bev_memory import DualBEVMemory
from inframot3d.perception.openpcdet_temporal_adapter import OpenPCDetTemporalAdapter


class Stage(nn.Module):
    def forward(self, batch):
        return batch


class DummyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_2d = Stage()
        self.dense_head = Stage()
        self.module_list = nn.ModuleList((self.backbone_2d, self.dense_head))


class DualMemoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.model = DualBEVMemory(32, 32)
        self.bev = torch.randn(1, 32, 24, 40)

    def test_disabled_is_exact_identity(self):
        output, state, aux = self.model(self.bev, None, "a", 0.0, enabled=False)
        self.assertIs(output, self.bev)
        self.assertIsNone(state)
        self.assertEqual(aux, {})

    def test_sequence_and_dynamic_gap(self):
        _, state, _ = self.model(self.bev, None, "a", 0.0)
        _, continued, aux = self.model(self.bev, state, "a", 0.1)
        self.assertTrue(aux["memory_valid"])
        self.assertTrue(aux["dynamic_valid"])
        self.assertEqual(continued.steps, 2)
        _, _, aux = self.model(self.bev, continued, "a", 0.6)
        self.assertTrue(aux["memory_valid"])
        self.assertFalse(aux["dynamic_valid"])
        _, reset, aux = self.model(self.bev, continued, "b", 0.7)
        self.assertFalse(aux["memory_valid"])
        self.assertEqual(reset.steps, 1)

    def test_bptt_reaches_memory(self):
        with torch.no_grad():
            self.model.readout[-1].weight.normal_(0.0, 1e-3)
        first = self.bev.clone().requires_grad_(True)
        _, state, _ = self.model(first, None, "a", 0.0)
        output, _, _ = self.model(self.bev, state, "a", 0.1)
        output.square().mean().backward()
        self.assertIsNotNone(first.grad)
        self.assertGreater(float(first.grad.abs().sum()), 0.0)

    def test_openpcdet_adapter(self):
        adapter = OpenPCDetTemporalAdapter(DummyDetector(), self.model)
        batch, state, aux = adapter.forward_step(
            {"spatial_features_2d": self.bev}, None, "a", 0.0,
        )
        self.assertEqual(batch["spatial_features_2d"].shape, self.bev.shape)
        self.assertEqual(state.sequence_id, "a")
        self.assertIn("route", aux)


if __name__ == "__main__":
    unittest.main()

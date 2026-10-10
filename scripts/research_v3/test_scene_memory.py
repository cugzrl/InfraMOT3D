"""Contract tests for the detachable fixed-view scene memory."""

import unittest

import torch

from inframot3d.perception.scene_memory import LongTermSceneMemory, StaticGate


class SceneMemoryTests(unittest.TestCase):
    def setUp(self):
        self.current = torch.ones(1, 4, 8, 10)
        self.hit = torch.ones(1, 1, 8, 10)
        self.clear = torch.zeros_like(self.hit)

    def test_disabled_is_exact_identity(self):
        layer = LongTermSceneMemory(4, enabled=False)
        enhanced, state, weight = layer(self.current, None, "a", 1.0,
                                        dynamic=self.current * 2, hit=self.hit,
                                        dynamic_mask=self.clear)
        self.assertTrue(torch.equal(enhanced, self.current))
        self.assertIsNone(state)
        self.assertIsNone(weight)

    def test_read_uses_only_past_and_resets(self):
        layer = LongTermSceneMemory(4, read_weight=0.5)
        state = layer.write(self.current, None, "a", 1.0, self.hit, self.clear)
        observation = self.current * 3
        enhanced, _, _ = layer.read(observation, observation, state, "a", 1.1,
                                    self.clear)
        self.assertTrue(torch.allclose(enhanced, observation * 0.5 + self.current * 0.5))
        other, _, _ = layer.read(observation, observation, state, "b", 1.1,
                                 self.clear)
        self.assertTrue(torch.equal(other, observation))
        stale, _, _ = layer.read(observation, observation, state, "a", 2.0,
                                 self.clear)
        self.assertTrue(torch.equal(stale, observation))
        self.assertIsNone(layer.reset())

    def test_dynamic_cells_do_not_write(self):
        layer = LongTermSceneMemory(4, read_weight=0.5)
        mask = torch.zeros_like(self.hit)
        mask[:, :, :4] = 1
        state = layer.write(self.current, None, "a", 1.0, self.hit, mask)
        self.assertFalse(state.valid[:, :, :4].any())
        self.assertTrue(state.valid[:, :, 4:].all())

    def test_simple_forward_without_optional_evidence(self):
        layer = LongTermSceneMemory(4)
        enhanced, state, weight = layer(self.current, None, "a", 1.0)
        self.assertTrue(torch.equal(enhanced, self.current))
        self.assertTrue(state.valid.all())
        self.assertIsNone(weight)

    def test_adapters_and_gate(self):
        layer = LongTermSceneMemory(4, memory_channels=2, memory_size=(4, 5),
                                    gate=StaticGate())
        state = layer.write(self.current, None, "a", 1.0, self.hit, self.clear)
        self.assertEqual(state.feature.shape, (1, 2, 4, 5))
        enhanced, _, weight = layer.read(self.current, self.current, state, "a",
                                         1.1, self.clear)
        self.assertEqual(enhanced.shape, self.current.shape)
        self.assertEqual(weight.shape, self.hit.shape)


if __name__ == "__main__":
    unittest.main()

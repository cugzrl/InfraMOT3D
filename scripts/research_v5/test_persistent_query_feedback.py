import unittest

import torch

from inframot3d.perception.persistent_query_feedback import PersistentQueryFeedback


class QueryFeedbackTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.module = PersistentQueryFeedback(32, 32, max_queries=8)
        self.bev = torch.randn(1, 32, 24, 40)

    def test_online_birth_propagation_and_reset(self):
        first, state, info = self.module.read(self.bev, None, "scene-a", 0.0)
        self.assertIs(first, self.bev)
        self.assertFalse(info["valid"])
        boxes = torch.tensor([[12.0, -45.0, 0.0, 4.0, 2.0, 1.5, 0.0],
                              [20.0, -43.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        state = self.module.update(self.bev, state, boxes, torch.tensor([0.9, 0.8]),
                                   torch.tensor([1, 1]), 0.0)
        self.assertEqual(state.ids.tolist(), [0, 1])
        with torch.no_grad():
            self.module.feedback_proj.weight.normal_(0.0, 1e-3)
        second, state, info = self.module.read(self.bev, state, "scene-a", 0.1)
        self.assertTrue(info["valid"])
        self.assertGreater(float((second - self.bev).abs().sum()), 0.0)
        moved = boxes.clone()
        moved[0, 0] += 0.5
        state = self.module.update(self.bev, state, moved,
                                   torch.tensor([0.9, 0.8]), torch.tensor([1, 1]), 0.1)
        self.assertEqual(state.ids.tolist(), [0, 1])
        self.assertGreater(float(state.velocity[0, 0]), 0.0)
        reset, state, info = self.module.read(self.bev, state, "scene-b", 0.2)
        self.assertIs(reset, self.bev)
        self.assertEqual(len(state.ids), 0)

    def test_gradient_through_previous_query(self):
        with torch.no_grad():
            self.module.feedback_proj.weight.normal_(0.0, 1e-3)
        first = self.bev.clone().requires_grad_(True)
        _, state, _ = self.module.read(first, None, "a", 0.0)
        box = torch.tensor([[10.0, -45.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        state = self.module.update(first, state, box, torch.tensor([0.9]), torch.tensor([1]), 0.0)
        output, _, _ = self.module.read(self.bev, state, "a", 0.1)
        output.square().mean().backward()
        self.assertGreater(float(first.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()

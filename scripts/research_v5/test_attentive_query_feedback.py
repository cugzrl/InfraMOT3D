import unittest

import torch

from inframot3d.perception.attentive_query_feedback import AttentiveQueryFeedback


class AttentiveFeedbackTest(unittest.TestCase):
    def test_local_attention_and_causal_reset(self):
        torch.manual_seed(13)
        model = AttentiveQueryFeedback(32, 32, num_heads=4, max_queries=8)
        first_bev = torch.randn(1, 32, 24, 40, requires_grad=True)
        first, state, info = model.read(first_bev, None, "road-a", 0.0)
        self.assertIs(first, first_bev)
        self.assertFalse(info["valid"])

        boxes = torch.tensor([[12.0, -45.0, 0.0, 4.0, 2.0, 1.5, 0.0]])
        state = model.update(first_bev, state, boxes, torch.tensor([0.9]),
                             torch.tensor([1]), 0.0)
        with torch.no_grad():
            model.feedback_proj.weight.normal_(0.0, 1e-3)
        second_bev = torch.randn(1, 32, 24, 40)
        second, state, info = model.read(second_bev, state, "road-a", 0.1)
        self.assertTrue(info["valid"])
        self.assertEqual(info["count"], 1)
        self.assertEqual(info["predicted_centers"].shape, (1, 2))
        self.assertGreater(float((second - second_bev).abs().sum()), 0.0)
        self.assertLess(float(info["attention_entropy"]), 3.22)
        second.square().mean().backward()
        self.assertGreater(float(model.cross_attention.in_proj_weight.grad.abs().sum()), 0.0)
        self.assertGreater(float(first_bev.grad.abs().sum()), 0.0)

        reset, empty, info = model.read(second_bev, state, "road-b", 0.2)
        self.assertIs(reset, second_bev)
        self.assertFalse(info["valid"])
        self.assertEqual(len(empty.ids), 0)


if __name__ == "__main__":
    unittest.main()

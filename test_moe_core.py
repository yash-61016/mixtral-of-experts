"""Unit tests for shared Mixtral MoE utilities."""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from moe_core import (
    ExpertUsageTracker,
    FLOPCounter,
    SequenceDataset,
    auxiliary_load_balancing_loss,
    estimate_dense_ffn_flops,
    estimate_moe_ffn_active_flops,
)


class TestFLOPEstimators(unittest.TestCase):
    def test_moe_active_ffn_flops_is_quarter_of_dense_swiglu_for_x_over_8_experts_with_k2(self):
        batch = 2
        seq_len = 16
        d_model = 64
        dense_d_ff = 256
        n_experts = 8
        k = 2
        d_ff_per_expert = dense_d_ff // n_experts

        dense = estimate_dense_ffn_flops(batch, seq_len, d_model, dense_d_ff)
        moe = estimate_moe_ffn_active_flops(
            batch=batch,
            seq_len=seq_len,
            d_model=d_model,
            d_ff_per_expert=d_ff_per_expert,
            k=k,
        )

        # For apples-to-apples SwiGLU dense baseline, active MoE is 1/4 of dense
        # when d_ff_per_expert = dense_d_ff / 8 and k=2.
        self.assertEqual(dense // 4, moe)

    def test_dense_ffn_estimator_matches_swiglu_three_projection_cost(self):
        batch = 2
        seq_len = 16
        d_model = 64
        dense_d_ff = 256

        dense = estimate_dense_ffn_flops(batch, seq_len, d_model, dense_d_ff)
        expected = 6 * batch * seq_len * d_model * dense_d_ff
        self.assertEqual(expected, dense)

    def test_flop_counter_accumulates_monotonically(self):
        counter = FLOPCounter()
        counter.add(100)
        counter.add(50)
        self.assertEqual(counter.total, 150)


class TestAuxiliaryLoss(unittest.TestCase):
    def test_auxiliary_loss_is_positive_for_collapsed_routing(self):
        probs = torch.tensor(
            [
                [[0.99, 0.005, 0.003, 0.002], [0.98, 0.01, 0.005, 0.005]],
                [[0.97, 0.01, 0.01, 0.01], [0.99, 0.003, 0.003, 0.004]],
            ],
            dtype=torch.float32,
        )
        topk = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 2], [0, 3]],
            ],
            dtype=torch.long,
        )

        loss = auxiliary_load_balancing_loss(
            router_probs=probs,
            expert_indices=topk,
            n_experts=4,
        )

        self.assertGreater(loss.item(), 0.0)


class TestUsageTracker(unittest.TestCase):
    def test_usage_tracker_counts_and_entropy(self):
        tracker = ExpertUsageTracker(n_experts=4)
        topk = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 2], [0, 3]],
            ],
            dtype=torch.long,
        )
        tracker.update(topk)

        freqs = tracker.frequencies()
        self.assertEqual(freqs.shape[0], 4)
        self.assertAlmostEqual(freqs.sum().item(), 1.0, places=6)
        self.assertGreater(tracker.entropy().item(), 0.0)

    def test_window_frequencies_reset_after_snapshot(self):
        tracker = ExpertUsageTracker(n_experts=4)
        topk = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[0, 2], [0, 3]],
            ],
            dtype=torch.long,
        )
        tracker.update(topk)

        window = tracker.window_frequencies(reset=True)
        self.assertAlmostEqual(window.sum().item(), 1.0, places=6)

        empty = tracker.window_frequencies(reset=False)
        self.assertAlmostEqual(empty.sum().item(), 0.0, places=6)


class TestDatasetSampling(unittest.TestCase):
    def test_sequence_dataset_sampling_is_reproducible_with_generator(self):
        tokens = torch.arange(0, 512, dtype=torch.long)
        ds = SequenceDataset(tokens, seq_len=16, device=torch.device("cpu"))
        g1 = torch.Generator().manual_seed(123)
        g2 = torch.Generator().manual_seed(123)

        x1, y1 = ds.sample_batch(batch_size=4, generator=g1)
        x2, y2 = ds.sample_batch(batch_size=4, generator=g2)

        self.assertTrue(torch.equal(x1, x2))
        self.assertTrue(torch.equal(y1, y2))


class TestAuxiliaryLossFormula(unittest.TestCase):
    def test_auxiliary_loss_matches_top1_switch_style_formula(self):
        probs = torch.tensor(
            [
                [[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]],
                [[0.2, 0.7, 0.1], [0.1, 0.7, 0.2]],
            ],
            dtype=torch.float32,
        )
        topk = torch.tensor(
            [
                [[0, 1], [0, 1]],
                [[1, 0], [1, 2]],
            ],
            dtype=torch.long,
        )

        loss = auxiliary_load_balancing_loss(
            router_probs=probs,
            expert_indices=topk,
            n_experts=3,
        )

        top1 = topk[..., 0].reshape(-1)
        f = torch.nn.functional.one_hot(top1, num_classes=3).to(torch.float32).mean(dim=0)
        p = probs.reshape(-1, 3).mean(dim=0)
        expected = 3 * torch.sum(f * p)
        self.assertAlmostEqual(loss.item(), expected.item(), places=6)


if __name__ == "__main__":
    unittest.main()

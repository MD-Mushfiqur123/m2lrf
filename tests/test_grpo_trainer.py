"""
Unit tests for M-2LRF Group Relative Policy Optimization (GRPO) Trainer.
"""

import unittest
import torch
import torch.nn as nn

from m2lrf.trainers.grpo_trainer import M2LRFGRPOTrainer, GRPOConfig
from m2lrf.data.synthetic_reasoning import MathRuleVerifier


class SimplePolicyModel(nn.Module):
    """Simple autoregressive language model mock for GRPO testing."""

    def __init__(self, vocab_size: int = 128, hidden_dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        h = self.embed(input_ids)
        return self.linear(h)


class TestGRPOTrainer(unittest.TestCase):
    """Tests GRPO math, advantage normalization, loss functions, and optimization steps."""

    def setUp(self):
        self.vocab_size = 128
        self.hidden_dim = 32
        self.model = SimplePolicyModel(self.vocab_size, self.hidden_dim)
        self.config = GRPOConfig(
            group_size=4,
            clip_eps=0.2,
            kl_coeff=0.04,
            learning_rate=1e-3,
            max_completion_length=16,
        )
        self.trainer = M2LRFGRPOTrainer(
            model=self.model,
            config=self.config,
            device="cpu",
        )

    def test_01_config_defaults(self):
        cfg = GRPOConfig()
        self.assertEqual(cfg.group_size, 4)
        self.assertEqual(cfg.clip_eps, 0.2)
        self.assertEqual(cfg.kl_coeff, 0.04)

    def test_02_group_advantage_normalization(self):
        # 2 prompts, group size 4 = 8 responses
        rewards = torch.tensor([
            1.2, -0.5, 1.2, 0.2,  # Prompt 1 group
            -0.5, -0.5, 1.2, -0.5  # Prompt 2 group
        ])
        adv = self.trainer.compute_group_advantages(rewards)
        self.assertEqual(adv.shape, rewards.shape)

        # Check that within each group of 4, mean is approximately 0 and std is approximately 1
        g1 = adv[:4]
        g2 = adv[4:]
        self.assertAlmostEqual(g1.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(g1.std().item(), 1.0, places=3)
        self.assertAlmostEqual(g2.mean().item(), 0.0, places=4)
        self.assertAlmostEqual(g2.std().item(), 1.0, places=3)

    def test_03_identical_rewards_advantage_stability(self):
        # When all rewards in group are identical, std is 0 -> advantage must not produce NaN
        rewards = torch.tensor([1.0, 1.0, 1.0, 1.0])
        adv = self.trainer.compute_group_advantages(rewards)
        self.assertFalse(torch.isnan(adv).any())
        self.assertTrue(torch.all(torch.abs(adv) < 1e-4))

    def test_04_evaluate_rewards_with_verifier(self):
        prompts = ["Calculate 10 + 5", "Calculate 7 * 3"]
        completions = [
            "<think>10 + 5 = 15</think> Thus, \\boxed{15}",
            "The answer is 20",  # Incorrect for 7 * 3 = 21
        ]
        ground_truths = ["15", "21"]

        rewards = self.trainer.evaluate_rewards(prompts, completions, ground_truths)
        self.assertEqual(rewards.shape, (2,))
        # First has correct answer (1.0) + think formatting (0.2) = 1.2
        self.assertAlmostEqual(rewards[0].item(), 1.2, places=4)
        # Second has incorrect answer = -0.5
        self.assertAlmostEqual(rewards[1].item(), -0.5, places=4)

    def test_05_per_token_log_probs(self):
        batch_size = 2
        seq_len = 8
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        # Mask first 3 positions as prompt (-100)
        labels = input_ids.clone()
        labels[:, :3] = -100

        log_probs = self.trainer.compute_per_token_log_probs(self.model, input_ids, labels)
        self.assertEqual(log_probs.shape, (batch_size, seq_len - 1))
        # First 2 positions in shifted labels were prompt tokens -> must be zero
        self.assertTrue(torch.all(log_probs[:, :2] == 0.0))
        # Later positions must be strictly negative (valid log probs)
        self.assertTrue(torch.all(log_probs[:, 2:] < 0.0))

    def test_06_train_step_batch_optimization(self):
        # Create a synthetic batch of 4 sequences (group of 4 for 1 prompt)
        batch_size = 4
        seq_len = 10
        input_ids = torch.randint(0, self.vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        labels[:, :4] = -100  # First 4 are prompt

        # 2 correct, 2 incorrect responses
        rewards = torch.tensor([1.2, 1.2, -0.5, -0.5])

        initial_param = next(self.model.parameters()).clone()

        metrics = self.trainer.train_step_batch(input_ids, labels, rewards)

        self.assertIn("policy_loss", metrics)
        self.assertIn("total_loss", metrics)
        self.assertIn("mean_reward", metrics)
        self.assertEqual(metrics["step"], 1)

        # Ensure parameters were updated by gradient descent
        updated_param = next(self.model.parameters())
        self.assertFalse(torch.equal(initial_param, updated_param))


if __name__ == "__main__":
    unittest.main()

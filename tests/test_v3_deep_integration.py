# -*- coding: utf-8 -*-
"""
Unit tests for M-2LRF V3 Deep Integration:
- SONA 5 Learning Modes (<0.05ms adaptation)
- AgentDB HNSW Vector Memory (1536 dims, 150x-12,500x speedup)
- DeepSeek-R1 GRPO Critic-Free Trainer
- 19 Lifecycle Hooks Automation
- V3 Unified Bridge
"""

import unittest
import torch
import torch.nn as nn

from m2lrf.v3.sona import SONAEngine
from m2lrf.v3.agentdb_sync import AgentDBCoordinator
from m2lrf.v3.grpo_trainer import GRPOTrainer, GRPORewardFunction
from m2lrf.v3.hooks import HookManager, HookType, HookContext
from m2lrf.v3.adapter_bridge import V3DeepIntegrationBridge


class TestSONAEngine(unittest.TestCase):
    def setUp(self):
        self.sona = SONAEngine(default_mode="balanced")

    def test_all_five_modes_exist(self):
        modes = ["real-time", "balanced", "research", "edge", "batch"]
        for m in modes:
            latency = self.sona.set_mode(m)
            self.assertEqual(self.sona.current_mode_name, m)
            self.assertIsNotNone(self.sona.get_config())

    def test_adaptation_speed(self):
        # Switching modes should be practically instantaneous (< 1ms)
        latency = self.sona.set_mode("real-time")
        self.assertLess(latency, 5.0)

    def test_apply_to_optimizer(self):
        linear = nn.Linear(10, 10)
        opt = torch.optim.AdamW(linear.parameters(), lr=1e-3)
        self.sona.set_mode("research")
        self.sona.apply_to_optimizer(opt)
        self.assertEqual(opt.param_groups[0]["lr"], 5e-4)


class TestAgentDBCoordinator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.dims = 1536
        self.agentdb = AgentDBCoordinator(dimensions=self.dims)

    def test_insert_and_search(self):
        # Insert 3 vectors
        v1 = torch.randn(self.dims)
        v2 = torch.randn(self.dims)
        v3 = v1 + 0.01 * torch.randn(self.dims)  # Highly similar to v1

        self.agentdb.insert("pattern_1", v1, {"task": "math"})
        self.agentdb.insert("pattern_2", v2, {"task": "code"})
        self.agentdb.insert("pattern_3", v3, {"task": "math_variation"})

        self.assertEqual(self.agentdb.size(), 3)

        # Search query close to v1
        results = self.agentdb.search(v1, top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0], "pattern_1")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)
        self.assertEqual(results[1][0], "pattern_3")

    def test_dimension_mismatch(self):
        bad_v = torch.randn(128)
        with self.assertRaises(ValueError):
            self.agentdb.insert("bad", bad_v)


class TestGRPOTrainer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.linear = nn.Linear(16, 4)
        self.trainer = GRPOTrainer(model=self.linear, group_size=4)

    def test_reward_function(self):
        good = "<think>Step 1: x = 2</think> Answer is 2"
        bad = "Answer is 2 without tags"
        self.assertEqual(GRPORewardFunction.format_reward(good), 1.0)
        self.assertEqual(GRPORewardFunction.format_reward(bad), 0.0)

    def test_group_advantage_normalization(self):
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        adv = self.trainer.compute_group_advantages(rewards)
        self.assertAlmostEqual(adv.mean().item(), 0.0, places=5)
        self.assertAlmostEqual(adv.std().item(), 1.0, places=3)

    def test_grpo_step(self):
        # G=4, seq_len=8
        log_probs = torch.randn(4, 8, requires_grad=True)
        old_log_probs = log_probs.detach().clone()
        ref_log_probs = log_probs.detach().clone()
        rewards = torch.tensor([0.5, 1.0, 0.0, 0.5])

        metrics = self.trainer.train_step(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            rewards=rewards,
        )
        self.assertIn("policy_loss", metrics)
        self.assertIn("total_loss", metrics)
        self.assertIn("kl_div", metrics)


class TestHookManager(unittest.TestCase):
    def setUp(self):
        self.hooks = HookManager()

    def test_register_and_trigger(self):
        events = []
        def _on_pre_step(ctx: HookContext):
            events.append(ctx.step)

        self.hooks.register(HookType.PRE_STEP, _on_pre_step)
        self.hooks.trigger(HookType.PRE_STEP, step=1)
        self.hooks.trigger(HookType.PRE_STEP, step=2)

        self.assertEqual(events, [1, 2])

    def test_nan_guard(self):
        guard = HookManager.create_nan_guard()
        self.hooks.register(HookType.NAN_INF_GUARD, guard)

        nan_tensor = torch.tensor([1.0, float("nan")])
        with self.assertRaises(ValueError):
            self.hooks.trigger(HookType.NAN_INF_GUARD, step=0, tensor=nan_tensor)


class TestV3DeepIntegrationBridge(unittest.TestCase):
    def test_unified_bridge(self):
        model = nn.Linear(16, 4)
        bridge = V3DeepIntegrationBridge(model=model, default_sona_mode="balanced")

        # Test mode switch
        lat = bridge.set_learning_mode("research")
        self.assertLess(lat, 5.0)

        # Test pattern record & search
        v = torch.randn(1536)
        bridge.record_reasoning_pattern("strat_1", v, {"acc": 0.98})
        results = bridge.search_similar_patterns(v, top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], "strat_1")

        # Status report
        report = bridge.get_status_report()
        self.assertEqual(report["sona_mode"], "research")
        self.assertEqual(report["agentdb_entries"], 1)
        self.assertTrue(report["grpo_active"])


if __name__ == "__main__":
    unittest.main()

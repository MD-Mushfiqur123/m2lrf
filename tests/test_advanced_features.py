"""
Unit Tests for Advanced M-2LRF Modules (BitsAndBytes, PEFT, Liger-Kernel, Torchtune)
=====================================================================================
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.optimizers.adamw8bit import AdamW8bit
from m2lrf.adapters.dora import M2LRFDoRALinear
from m2lrf.adapters.loha import M2LRFLoHaLinear
from m2lrf.kernels.fast_fused_linear_ce import fused_linear_cross_entropy
from m2lrf.kernels.fast_kl_div import fast_kl_divergence
from m2lrf.utils.memory_tracker import MemoryTracker


class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_adamw8bit_step(self):
        """Verify 8-bit AdamW executes parameter updates correctly."""
        p = nn.Parameter(torch.randn(100, 50))
        opt = AdamW8bit([p], lr=1e-2, weight_decay=0.01)

        init_p = p.data.clone()
        # Fake loss & grad
        loss = (p ** 2).sum()
        loss.backward()

        opt.step()
        opt.zero_grad()

        # Parameter must have updated
        self.assertFalse(torch.allclose(init_p, p.data))
        # Moments must exist in state
        state = opt.state[p]
        self.assertIn("q_m", state)
        self.assertIn("q_v", state)
        self.assertEqual(state["q_m"].dtype, torch.int8)
        self.assertEqual(state["q_v"].dtype, torch.uint8)

    def test_dora_forward(self):
        """Verify DoRA adapter forward pass and shape preservation."""
        layer = M2LRFDoRALinear(in_features=64, out_features=128, rank=8)
        x = torch.randn(2, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 128))

    def test_loha_forward(self):
        """Verify LoHa Hadamard adapter forward pass."""
        layer = M2LRFLoHaLinear(in_features=64, out_features=128, rank=8)
        x = torch.randn(2, 64)
        out = layer(x)
        self.assertEqual(out.shape, (2, 128))

    def test_fused_linear_ce_parity(self):
        """Verify Fused Linear + CrossEntropy matches standard PyTorch F.linear + F.cross_entropy."""
        batch, hidden, vocab = 4, 32, 64
        x = torch.randn(batch, hidden, requires_grad=True)
        w = torch.randn(vocab, hidden, requires_grad=True)
        targets = torch.tensor([5, 10, -100, 25])

        # Ref
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = w.detach().clone().requires_grad_(True)
        ref_logits = F.linear(x_ref, w_ref)
        ref_loss = F.cross_entropy(ref_logits, targets, ignore_index=-100)
        ref_loss.backward()

        # Fused
        x_fast = x.detach().clone().requires_grad_(True)
        w_fast = w.detach().clone().requires_grad_(True)
        fast_loss = fused_linear_cross_entropy(x_fast, w_fast, targets, ignore_index=-100)
        fast_loss.backward()

        self.assertTrue(torch.allclose(ref_loss, fast_loss, atol=1e-5))
        self.assertTrue(torch.allclose(x_ref.grad, x_fast.grad, atol=1e-5))
        self.assertTrue(torch.allclose(w_ref.grad, w_fast.grad, atol=1e-5))

    def test_fast_kl_divergence(self):
        """Verify KL divergence computation."""
        p_logits = torch.randn(4, 10)
        q_logits = torch.randn(4, 10)
        log_p = F.log_softmax(p_logits, dim=-1)
        log_q = F.log_softmax(q_logits, dim=-1)

        kl = fast_kl_divergence(log_p, log_q)
        self.assertGreaterEqual(kl.item(), -1e-5)

    def test_memory_tracker(self):
        """Verify MemoryTracker runs without error."""
        tracker = MemoryTracker()
        tracker.start()
        # Allocate dummy tensor
        a = torch.randn(100, 100)
        summary = tracker.summary(tokens_processed=1000)
        self.assertIn("M-2LRF HARDWARE & MEMORY PROFILE SUMMARY", summary)


if __name__ == "__main__":
    unittest.main()

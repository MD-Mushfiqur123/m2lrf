"""
Unit Tests for M-2LRF Fused High-Performance Kernels (Unsloth-Inspired)
========================================================================
Validates numerical parity between custom fused kernels and standard PyTorch autograd.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.kernels.fast_cross_entropy import fast_cross_entropy_loss, FastCrossEntropyLoss
from m2lrf.kernels.fast_rms_norm import FastRMSNorm, FastRMSNormFunction
from m2lrf.kernels.fast_rope import fast_apply_rotary_pos_emb
from m2lrf.kernels.fast_swiglu import fast_swiglu
from m2lrf.kernels.fast_lora import fast_lora_forward


class TestFastKernels(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_fast_cross_entropy_numerical_parity(self):
        """Verify Fast Cross Entropy produces identical loss and gradient to F.cross_entropy."""
        batch_size, seq_len, vocab_size = 2, 16, 128
        logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        # Mask half of the targets with -100
        targets[0, :8] = -100

        # Reference
        ref_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-100)
        ref_loss.backward()
        ref_grad = logits.grad.clone()

        # Fast Cross Entropy
        logits_fast = logits.detach().clone().requires_grad_(True)
        fast_loss = fast_cross_entropy_loss(logits_fast, targets, ignore_index=-100)
        fast_loss.backward()
        fast_grad = logits_fast.grad.clone()

        self.assertTrue(torch.allclose(ref_loss, fast_loss, atol=1e-5))
        self.assertTrue(torch.allclose(ref_grad, fast_grad, atol=1e-5))

    def test_fast_rms_norm_numerical_parity(self):
        """Verify Fast RMSNorm forward and backward pass parity."""
        hidden_size = 64
        x = torch.randn(4, 16, hidden_size, requires_grad=True)
        weight = torch.ones(hidden_size, requires_grad=True)

        # Reference PyTorch RMSNorm
        x_ref = x.detach().clone().requires_grad_(True)
        w_ref = weight.detach().clone().requires_grad_(True)
        variance = x_ref.pow(2).mean(-1, keepdim=True)
        ref_y = x_ref * torch.rsqrt(variance + 1e-6) * w_ref
        ref_y.sum().backward()

        # Fast RMSNorm
        x_fast = x.detach().clone().requires_grad_(True)
        w_fast = weight.detach().clone().requires_grad_(True)
        fast_y = FastRMSNormFunction.apply(x_fast, w_fast, 1e-6)
        fast_y.sum().backward()

        self.assertTrue(torch.allclose(ref_y, fast_y, atol=1e-5))
        self.assertTrue(torch.allclose(x_ref.grad, x_fast.grad, atol=1e-5))
        self.assertTrue(torch.allclose(w_ref.grad, w_fast.grad, atol=1e-5))

    def test_fast_swiglu_parity(self):
        """Verify SwiGLU forward and backward parity against PyTorch F.silu(gate) * up."""
        n = 128
        gate = torch.randn(4, n, requires_grad=True)
        up = torch.randn(4, n, requires_grad=True)

        # Ref
        gate_ref = gate.detach().clone().requires_grad_(True)
        up_ref = up.detach().clone().requires_grad_(True)
        ref_out = F.silu(gate_ref) * up_ref
        ref_out.sum().backward()

        # Fast
        gate_fast = gate.detach().clone().requires_grad_(True)
        up_fast = up.detach().clone().requires_grad_(True)
        fast_out = fast_swiglu(gate_fast, up_fast)
        fast_out.sum().backward()

        self.assertTrue(torch.allclose(ref_out, fast_out, atol=1e-5))
        self.assertTrue(torch.allclose(gate_ref.grad, gate_fast.grad, atol=1e-5))
        self.assertTrue(torch.allclose(up_ref.grad, up_fast.grad, atol=1e-5))

    def test_fast_lora_parity(self):
        """Verify Fast LoRA forward and backward parity."""
        batch, in_f, out_f, rank = 8, 32, 48, 8
        scaling = 2.0
        x = torch.randn(batch, in_f, requires_grad=True)
        A = torch.randn(rank, in_f, requires_grad=True)
        B = torch.randn(out_f, rank, requires_grad=True)

        # Ref
        x_ref = x.detach().clone().requires_grad_(True)
        A_ref = A.detach().clone().requires_grad_(True)
        B_ref = B.detach().clone().requires_grad_(True)
        ref_out = F.linear(F.linear(x_ref, A_ref), B_ref) * scaling
        ref_out.sum().backward()

        # Fast
        x_fast = x.detach().clone().requires_grad_(True)
        A_fast = A.detach().clone().requires_grad_(True)
        B_fast = B.detach().clone().requires_grad_(True)
        fast_out = fast_lora_forward(x_fast, A_fast, B_fast, scaling)
        fast_out.sum().backward()

        self.assertTrue(torch.allclose(ref_out, fast_out, atol=1e-5))
        self.assertTrue(torch.allclose(x_ref.grad, x_fast.grad, atol=1e-5))
        self.assertTrue(torch.allclose(A_ref.grad, A_fast.grad, atol=1e-5))
        self.assertTrue(torch.allclose(B_ref.grad, B_fast.grad, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

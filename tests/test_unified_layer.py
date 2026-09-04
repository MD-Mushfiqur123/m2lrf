"""
Unit Tests for M-2LRF Grand Unified Linear Layer (M2LRFUnifiedLinear)
=====================================================================
Validates all composable features, bit-widths, rotation schemes, dynamic activations,
scale double quantization, outlier handling, and in-situ weight merge equivalence.
"""

import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import (
    M2LRFUnifiedLinear,
    M2LRF2BitLinear,
    HadamardDualBasisLinear,
    M2LRF4BitLinear,
    M2LRFW2A8Linear
)
from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.hadamard_transform import generate_synthetic_heavy_tailed_weights


class TestM2LRFUnifiedLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 256
        self.out_features = 128
        self.batch_size = 4
        self.seq_len = 16

    def test_unified_layer_initialization_combinations(self):
        """Test instantiation across diverse flag combinations."""
        configs = [
            {"bits": 2, "group_size": None, "use_hadamard": False, "use_w2a8": False, "double_quant": False},
            {"bits": 2, "group_size": 64, "use_hadamard": False, "use_w2a8": False, "double_quant": True},
            {"bits": 2, "group_size": 64, "use_hadamard": True, "use_w2a8": False, "double_quant": False},
            {"bits": 2, "group_size": 64, "use_hadamard": True, "use_w2a8": True, "double_quant": False},
            {"bits": 4, "group_size": 64, "use_hadamard": False, "use_w2a8": False, "double_quant": False, "codec_type": "nf4"},
            {"bits": 4, "group_size": 32, "use_hadamard": False, "use_w2a8": False, "double_quant": False, "codec_type": "lloyd_max"},
            {"bits": 2, "group_size": 64, "sparse_outliers": True, "rank": 32}
        ]

        w_pretrained = torch.randn(self.out_features, self.in_features)

        for cfg in configs:
            cfg_copy = dict(cfg)
            rank = cfg_copy.pop("rank", 16)
            layer = M2LRFUnifiedLinear(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=rank,
                alpha=16.0,
                bias=True,
                **cfg_copy
            )
            layer.initialize_from_pretrained(w_pretrained, loftq_iters=1)

            x = torch.randn(self.batch_size, self.seq_len, self.in_features)
            out = layer(x)

            self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_features))
            self.assertFalse(torch.isnan(out).any(), f"NaN detected for config: {cfg}")
            self.assertGreater(layer.memory_bytes(), 0)
            self.assertGreater(layer.effective_bpp(), 0.0)

    def test_subclass_inheritance_and_polymorphism(self):
        """Verify that specialized classes inherit from M2LRFUnifiedLinear."""
        l_2b = M2LRF2BitLinear(128, 64)
        l_had = HadamardDualBasisLinear(128, 64)
        l_4b = M2LRF4BitLinear(128, 64)
        l_w2a8 = M2LRFW2A8Linear(128, 64)

        for layer in (l_2b, l_had, l_4b, l_w2a8):
            self.assertIsInstance(layer, M2LRFUnifiedLinear)
            self.assertIsInstance(layer, nn.Module)

    def test_gradient_flow_and_lora_training(self):
        """Verify backpropagation updates only LoRA parameters while base buffers remain fixed."""
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            bits=2,
            group_size=64,
            use_hadamard=True,
            rank=16
        )
        w_orig = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_orig)

        # Base buffers must not require grad
        self.assertFalse(layer.packed_weights.requires_grad)
        self.assertFalse(layer.a0.requires_grad)
        self.assertFalse(layer.a1.requires_grad)
        self.assertFalse(layer.signs.requires_grad)

        # Adapters must require grad
        self.assertTrue(layer.lora_A.requires_grad)
        self.assertTrue(layer.lora_B.requires_grad)

        x = torch.randn(2, 8, self.in_features, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertGreater(torch.norm(layer.lora_A.grad).item(), 0.0)
        self.assertGreater(torch.norm(layer.lora_B.grad).item(), 0.0)
        self.assertIsNotNone(x.grad)

    def test_in_situ_weight_merge_and_effective_weight(self):
        """Verify that merge() fuses LoRA updates and preserves forward pass."""
        layer = M2LRFUnifiedLinear(
            in_features=128,
            out_features=64,
            bits=2,
            group_size=32,
            use_hadamard=True,
            rank=16
        )
        w_orig = torch.randn(64, 128)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(2, 4, 128)
        out_before = layer(x)

        w_eff = layer.dequantize_effective_weight()
        self.assertEqual(w_eff.shape, (64, 128))

        # Merge
        layer.merge()
        self.assertTrue(layer.is_merged)
        self.assertEqual(layer.lora_A.abs().sum().item(), 0.0)
        self.assertEqual(layer.lora_B.abs().sum().item(), 0.0)

        out_after = layer(x)
        rel_diff = (torch.norm(out_before - out_after) / torch.norm(out_before)).item()
        self.assertLess(rel_diff, 0.45)

    def test_outlier_suppression_on_heavy_tailed_weights(self):
        """Verify that Hadamard rotation suppresses outlier kurtosis and elevates SQNR on heavy-tailed distributions."""
        w_heavy = generate_synthetic_heavy_tailed_weights(
            out_features=256,
            in_features=256,
            num_outlier_channels=8,
            outlier_multiplier=15.0,
            seed=42
        )

        layer_unrotated = M2LRFUnifiedLinear(256, 256, bits=2, group_size=None, use_hadamard=False, rank=0)
        layer_unrotated.initialize_from_pretrained(w_heavy)
        w_unrot_eff = layer_unrotated.dequantize_effective_weight().float()
        sqnr_unrot = DualBasisQuantizer.calculate_sqnr(w_heavy, w_unrot_eff)

        layer_rotated = M2LRFUnifiedLinear(256, 256, bits=2, group_size=None, use_hadamard=True, rank=0)
        layer_rotated.initialize_from_pretrained(w_heavy)
        w_rot_eff = layer_rotated.dequantize_effective_weight().float()
        sqnr_rot = DualBasisQuantizer.calculate_sqnr(w_heavy, w_rot_eff)

        # Rotated layer should achieve higher SQNR on heavy-tailed distribution
        self.assertGreater(sqnr_rot, sqnr_unrot + 2.0)


if __name__ == "__main__":
    unittest.main()
"""
Unit Tests for 2024-2026 Breakthrough Inventions: PiSSA, KIVI 2-Bit KV-Cache, and QuaRot
========================================================================================
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.adapters.pissa import M2LRFPiSSALinear
from m2lrf.kernels.kivi_kv_cache import KIVIKVCache
from m2lrf.kernels.quarot_transform import QuaRotLinear


class TestInventions2025(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_pissa_svd_residual_initialization(self):
        """Verify PiSSA extracts principal components and preserves output dimensions."""
        in_f, out_f, rank = 64, 128, 8
        layer = M2LRFPiSSALinear(in_features=in_f, out_features=out_f, rank=rank)

        w = torch.randn(out_f, in_f)
        layer.initialize_from_pretrained(w)

        # Ensure adapter A and B are populated with non-zero singular components
        self.assertFalse(torch.all(layer.lora_A == 0))
        self.assertFalse(torch.all(layer.lora_B == 0))

        # Test forward pass
        x = torch.randn(4, in_f)
        out = layer(x)
        self.assertEqual(out.shape, (4, out_f))

    def test_kivi_2bit_kv_cache(self):
        """Verify KIVI 2-bit asymmetric KV Cache compression and dequantization."""
        n_heads, head_dim, seq_len = 4, 64, 16
        cache = KIVIKVCache(n_heads=n_heads, head_dim=head_dim, max_seq_len=128)

        # Generate mock key and value states
        k_states = torch.randn(n_heads, seq_len, head_dim)
        v_states = torch.randn(n_heads, seq_len, head_dim)

        # Update cache
        cache.update(k_states, v_states)
        self.assertEqual(cache.curr_len, seq_len)

        # Dequantize
        k_deq, v_deq = cache.get_dequantized_kv()
        self.assertEqual(k_deq.shape, (n_heads, seq_len, head_dim))
        self.assertEqual(v_deq.shape, (n_heads, seq_len, head_dim))

        # Check that dequantized representations are strongly correlated with original
        cos_k = F.cosine_similarity(k_states.flatten(), k_deq.flatten(), dim=0)
        cos_v = F.cosine_similarity(v_states.flatten(), v_deq.flatten(), dim=0)
        self.assertGreater(cos_k.item(), 0.70)
        self.assertGreater(cos_v.item(), 0.70)

        # Verify bit packing storage footprint (4 values per byte -> 16 bytes for 64 dims)
        self.assertEqual(cache.packed_dim, 16)
        self.assertEqual(cache.packed_keys.dtype, torch.uint8)

    def test_quarot_dual_sided_rotation(self):
        """Verify QuaRot linear layer with orthogonal Hadamard rotation."""
        in_f, out_f = 64, 64
        layer = QuaRotLinear(in_features=in_f, out_features=out_f, block_size=64)
        nn.init.normal_(layer.weight)

        layer.rotate_weights()
        self.assertTrue(layer.rotated)

        x = torch.randn(2, in_f)
        out = layer(x)
        self.assertEqual(out.shape, (2, out_f))


if __name__ == "__main__":
    unittest.main()

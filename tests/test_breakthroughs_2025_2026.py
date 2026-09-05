# -*- coding: utf-8 -*-
"""
Unit tests for 2025-2026 Breakthrough Modules in M-2LRF:
- LoRA-Pro (ICLR 2025 Spotlight Gradient Projection)
- MiLoRA (Minor Singular Subspace Adaptation)
- DuoAttention (Dual Retrieval & Streaming KV Cache)
- PyramidKV (Pyramidal Information Funneling KV Cache)
- SageAttention (INT8 Quantized Attention)
- VPTQ / Residual Vector Quantization (Vector Codebooks)
"""

import unittest
import torch
import torch.nn as nn

from m2lrf.adapters.lora_pro import M2LRFLoRAProLinear, LoRAProGradientProjector
from m2lrf.adapters.milora import M2LRFMiLoRALinear
from m2lrf.kernels.duo_attention import DuoAttentionHeadClassifier, DuoAttentionKVCache
from m2lrf.kernels.pyramid_kv import PyramidKVAllocator, PyramidKVCache
from m2lrf.kernels.sage_attention import SageAttention, sage_attention_forward
from m2lrf.vector_codec import VectorCodebook, ResidualVectorQuantizer


class TestLoRAPro(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_dim = 64
        self.out_dim = 32
        self.r = 8

    def test_lora_pro_forward_and_backward(self):
        layer = M2LRFLoRAProLinear(self.in_dim, self.out_dim, r=self.r, lora_alpha=16.0)
        x = torch.randn(4, self.in_dim, requires_grad=True)
        out = layer(x)
        self.assertEqual(out.shape, (4, self.out_dim))

        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)

    def test_lora_pro_gradient_alignment(self):
        layer = M2LRFLoRAProLinear(self.in_dim, self.out_dim, r=self.r, lora_alpha=16.0)
        full_grad = torch.randn(self.out_dim, self.in_dim)
        layer.align_gradients(full_grad)
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertEqual(layer.lora_A.grad.shape, (self.r, self.in_dim))
        self.assertEqual(layer.lora_B.grad.shape, (self.out_dim, self.r))

    def test_lora_pro_merge(self):
        layer = M2LRFLoRAProLinear(self.in_dim, self.out_dim, r=self.r, lora_alpha=16.0)
        layer.lora_B.data.fill_(0.1)
        layer.lora_A.data.fill_(0.1)
        w_before = layer.weight.clone()
        w_merged = layer.merge()
        self.assertFalse(torch.allclose(w_before, w_merged))
        self.assertTrue(torch.all(layer.lora_A == 0))
        self.assertTrue(torch.all(layer.lora_B == 0))


class TestMiLoRA(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.in_dim = 64
        self.out_dim = 32
        self.r = 8

    def test_milora_minor_svd_init(self):
        layer = M2LRFMiLoRALinear(self.in_dim, self.out_dim, r=self.r, lora_alpha=16.0)
        W_pretrained = torch.randn(self.out_dim, self.in_dim)
        layer.initialize_from_minor_svd(W_pretrained)

        self.assertTrue(layer.initialized_from_minor_svd)
        # Check that adapter + base roughly matches original W
        recon = layer.weight + layer.scaling * (layer.lora_B @ layer.lora_A)
        diff = torch.norm(W_pretrained - recon) / torch.norm(W_pretrained)
        # With scaling=2.0 and B, A from SVD, residual = W - B @ A
        # so weight + B @ A = W
        self.assertLess(diff.item(), 0.6)

    def test_milora_forward_and_merge(self):
        layer = M2LRFMiLoRALinear(self.in_dim, self.out_dim, r=self.r)
        W_pretrained = torch.randn(self.out_dim, self.in_dim)
        layer.initialize_from_minor_svd(W_pretrained)

        x = torch.randn(2, self.in_dim)
        out = layer(x)
        self.assertEqual(out.shape, (2, self.out_dim))

        merged = layer.merge()
        self.assertEqual(merged.shape, (self.out_dim, self.in_dim))


class TestDuoAttention(unittest.TestCase):
    def test_classifier(self):
        clf = DuoAttentionHeadClassifier(retrieval_ratio=0.25, num_heads=16, num_layers=8)
        clf.initialize_default_heuristic()
        self.assertTrue(clf.is_retrieval_head(4, 0))
        self.assertFalse(clf.is_retrieval_head(4, 15))

    def test_kv_cache_eviction(self):
        num_layers = 2
        num_heads = 4
        head_dim = 16
        cache = DuoAttentionKVCache(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            num_sinks=2,
            window_size=8,
            retrieval_ratio=0.0,  # All streaming heads to test eviction
        )

        # Feed 20 tokens
        for _ in range(20):
            k = torch.randn(1, num_heads, 1, head_dim)
            v = torch.randn(1, num_heads, 1, head_dim)
            out_k, out_v = cache.update(0, k, v)

        # Streaming heads should be capped at sinks (2) + window (8) = 10
        self.assertLessEqual(out_k.shape[2], 10)
        self.assertGreater(cache.memory_bytes(), 0)


class TestPyramidKV(unittest.TestCase):
    def test_allocator(self):
        alloc = PyramidKVAllocator(num_layers=8, max_budget=1024, min_budget=128, decay_gamma=1.0)
        budgets = alloc.budgets
        self.assertEqual(len(budgets), 8)
        self.assertGreater(budgets[0], budgets[-1])
        self.assertGreater(alloc.uniform_comparison_savings(), 0.3)

    def test_cache_layer_budget(self):
        cache = PyramidKVCache(num_layers=4, max_budget=32, min_budget=16, num_sinks=2)
        # Feed 40 tokens to layer 3 (min budget ~16)
        k = torch.randn(1, 2, 40, 8)
        v = torch.randn(1, 2, 40, 8)
        out_k, out_v = cache.update(3, k, v)
        self.assertLessEqual(out_k.shape[2], 20)


class TestSageAttention(unittest.TestCase):
    def test_sage_attention_forward(self):
        batch, heads, seq, dim = 2, 4, 16, 32
        q = torch.randn(batch, heads, seq, dim)
        k = torch.randn(batch, heads, seq, dim)
        v = torch.randn(batch, heads, seq, dim)

        out = sage_attention_forward(q, k, v, is_causal=True, smooth_outliers=True)
        self.assertEqual(out.shape, (batch, heads, seq, dim))
        self.assertFalse(torch.isnan(out).any())

    def test_sage_attention_module(self):
        mod = SageAttention(head_dim=32, smooth_outliers=True)
        q = torch.randn(1, 2, 8, 32)
        k = torch.randn(1, 2, 8, 32)
        v = torch.randn(1, 2, 8, 32)
        out = mod(q, k, v)
        self.assertEqual(out.shape, (1, 2, 8, 32))


class TestVectorCodec(unittest.TestCase):
    def test_codebook_and_rvq(self):
        torch.manual_seed(42)
        W = torch.randn(32, 32)
        rvq = ResidualVectorQuantizer(vector_dim=2, num_stages=2, centroids_per_stage=8)
        rvq.fit(W, num_iters=5)

        indices, recon = rvq.quantize(W)
        self.assertEqual(len(indices), 2)
        self.assertEqual(recon.shape, (32, 32))

        # Check reconstruction is non-trivial
        err = torch.norm(W - recon) / torch.norm(W)
        self.assertLess(err.item(), 1.0)

        # Check dequantize matches quantize output
        deq = rvq.dequantize(indices, target_shape=(32, 32))
        self.assertTrue(torch.allclose(recon, deq, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

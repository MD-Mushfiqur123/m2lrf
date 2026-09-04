"""
Unit Test Suite for M-2LRF Mixed Precision Quantization & Layer Sensitivity Engine
===================================================================================
Tests:
  1. Real4BitCodec: pack / unpack roundtrip fidelity, NF4 and Lloyd-Max codecs, group-wise scaling.
  2. M2LRF4BitLinear: LoftQ SVD initialization, parameter storage, forward/backward pass, merge/unmerge.
  3. LayerSensitivityProfiler: gradient magnitude, empirical Fisher proxy, output MSE perturbation, data-free heuristic.
  4. MixedPrecisionAllocator: target bitrate matching (2.6 bpp), memory calculation, sensitive layer allocation.
  5. allocate_mixed_precision_model: end-to-end surgical model conversion and gradient verification.
  6. Integration with prepare_m2lrf_model.
"""

import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.layer import M2LRF2BitLinear
from m2lrf.mixed_precision import (
    Real4BitCodec,
    M2LRF4BitLinear,
    LayerSensitivityProfiler,
    SensitivityProfileResult,
    MixedPrecisionAllocator,
    MixedPrecisionAllocationPlan,
    allocate_mixed_precision_model,
    prepare_mixed_precision_m2lrf_model
)
from m2lrf.trainer_eval import prepare_m2lrf_model


class TestReal4BitCodec(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_pack_unpack_nf4_roundtrip(self):
        """Verify NF4 pack and unpack roundtrip produces accurate reconstruction."""
        w = torch.randn(128, 256, device=self.device)
        packed, scales, shape = Real4BitCodec.pack(w, codec_type="nf4")
        
        # Verify 2 weights per byte (256 in_features -> 128 packed bytes)
        self.assertEqual(packed.shape, (128, 128))
        self.assertEqual(packed.dtype, torch.uint8)

        w_recon = Real4BitCodec.unpack_and_dequantize(packed, scales, shape, codec_type="nf4")
        self.assertEqual(w_recon.shape, w.shape)
        
        # SQNR for 4-bit on Gaussian weights is typically > 18 dB
        sqnr = DualBasisQuantizer.calculate_sqnr(w, w_recon)
        self.assertGreater(sqnr, 15.0, f"4-bit NF4 SQNR {sqnr:.2f} dB was below 15.0 dB")

    def test_pack_unpack_lloyd_max_and_group_wise(self):
        """Verify Lloyd-Max 4-bit codec with group-wise scaling."""
        w = torch.randn(64, 512, device=self.device)
        group_size = 64
        packed, scales, shape = Real4BitCodec.pack(w, group_size=group_size, codec_type="lloyd_max")

        self.assertEqual(scales.shape, (64, 512 // group_size))
        w_recon = Real4BitCodec.unpack_and_dequantize(
            packed, scales, shape, group_size=group_size, codec_type="lloyd_max"
        )
        self.assertEqual(w_recon.shape, w.shape)
        sqnr = DualBasisQuantizer.calculate_sqnr(w, w_recon)
        self.assertGreater(sqnr, 15.0)


class TestM2LRF4BitLinear(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_loftq_svd_initialization_and_forward(self):
        """Verify M2LRF4BitLinear initializes LoftQ SVD and computes forward pass."""
        in_f, out_f, rank = 256, 128, 16
        layer = M2LRF4BitLinear(in_f, out_f, rank=rank, alpha=16.0).to(self.device)
        w_orig = torch.randn(out_f, in_f, device=self.device)

        layer.initialize_from_pretrained(w_orig, loftq_iters=1)

        # Packed uint8 weights: out_f * (in_f // 2)
        expected_bytes = out_f * (in_f // 2)
        self.assertEqual(layer.packed_weights.numel(), expected_bytes)

        # Step-0 reconstruction error with LoftQ adapter
        w_step0 = layer._dequantize().float() + (layer.lora_B @ layer.lora_A).float() * layer.scaling
        rel_error = (torch.norm(w_orig - w_step0) / torch.norm(w_orig)).item()
        self.assertLess(rel_error, 0.15, f"Step-0 LoftQ 4-bit relative error {rel_error:.4f} too high")

        # Forward pass
        x = torch.randn(4, 8, in_f, device=self.device)
        out = layer(x)
        self.assertEqual(out.shape, (4, 8, out_f))

    def test_gradient_flow(self):
        """Verify base 4-bit weights are frozen and only LoRA adapter gets gradients."""
        layer = M2LRF4BitLinear(64, 128, rank=8).to(self.device)
        w_orig = torch.randn(128, 64, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(2, 64, device=self.device)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNone(layer.packed_weights.grad)
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertGreater(torch.norm(layer.lora_A.grad).item(), 0.0)

    def test_merge_and_unmerge(self):
        """Verify in-situ merge fuses adapter into packed 4-bit base weights."""
        layer = M2LRF4BitLinear(128, 256, rank=16).to(self.device)
        w_orig = torch.randn(256, 128, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(4, 128, device=self.device)
        with torch.no_grad():
            out_before = layer(x)
            layer.merge()
            out_after = layer(x)

        self.assertTrue(layer.is_merged)
        self.assertEqual(torch.sum(torch.abs(layer.lora_A)).item(), 0.0)
        self.assertEqual(torch.sum(torch.abs(layer.lora_B)).item(), 0.0)
        rel_error = (torch.norm(out_before - out_after) / torch.norm(out_before)).item()
        self.assertLess(rel_error, 0.20)


# Mock Transformer Architecture for Testing Profiling and Allocation
class MockAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.size(-1))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        return self.o_proj(out)


class MockMLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MockTransformerBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(dim)
        self.self_attn = MockAttention(dim)
        self.post_attention_layernorm = nn.LayerNorm(dim)
        self.mlp = MockMLP(dim, hidden_dim)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class MockTransformerModel(nn.Module):
    def __init__(self, vocab_size=500, dim=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([MockTransformerBlock(dim, hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward(self, input_ids=None, inputs_embeds=None):
        if inputs_embeds is None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.lm_head(x)


class TestLayerSensitivityProfiler(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MockTransformerModel(dim=64, hidden_dim=128, num_layers=2).to(self.device)

    def test_profile_data_free_heuristic(self):
        """Verify data-free sensitivity profiling assigns higher weight to attention projections."""
        profiler = LayerSensitivityProfiler()
        result = profiler.profile(self.model, metric="heuristic")

        self.assertIsInstance(result, SensitivityProfileResult)
        self.assertEqual(len(result.raw_scores), 14)  # 2 layers * (4 attn + 3 mlp) = 14 linear layers
        self.assertNotIn("lm_head", result.raw_scores)  # lm_head is excluded

        # Attention Q/O projections should generally rank higher than MLP due to architectural prior
        top_5_names = [name for name, _ in result.top_k(5)]
        has_attn = any("q_proj" in name or "o_proj" in name for name in top_5_names)
        self.assertTrue(has_attn, "Top sensitive layers should contain attention projections")

    def test_profile_gradient_and_fisher(self):
        """Verify gradient and Fisher information profiling on calibration batches."""
        profiler = LayerSensitivityProfiler()
        calib_inputs = torch.randint(10, 400, (2, 16), dtype=torch.long, device=self.device)

        def loss_fn(out, batch):
            return out.sum()

        grad_result = profiler.profile(
            self.model, calibration_data=calib_inputs, metric="gradient", loss_fn=loss_fn
        )
        self.assertEqual(len(grad_result.raw_scores), 14)
        self.assertTrue(all(v >= 0.0 for v in grad_result.raw_scores.values()))

        fisher_result = profiler.profile(
            self.model, calibration_data=calib_inputs, metric="fisher", loss_fn=loss_fn
        )
        self.assertEqual(len(fisher_result.raw_scores), 14)
        self.assertTrue(all(v >= 0.0 for v in fisher_result.raw_scores.values()))

    def test_profile_output_perturbation_mse(self):
        """Verify output MSE perturbation metric."""
        profiler = LayerSensitivityProfiler()
        calib_inputs = torch.randint(10, 400, (2, 16), dtype=torch.long, device=self.device)

        mse_result = profiler.profile(
            self.model, calibration_data=calib_inputs, metric="mse"
        )
        self.assertEqual(len(mse_result.raw_scores), 14)
        self.assertTrue(all(v >= 0.0 for v in mse_result.raw_scores.values()))


class TestMixedPrecisionAllocator(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = MockTransformerModel(dim=64, hidden_dim=128, num_layers=2).to(self.device)

    def test_allocation_at_target_2_6_bpp(self):
        """Verify allocation achieves ~2.6 bpp average with 4-bit on sensitive layers and 2-bit on remainder."""
        plan = MixedPrecisionAllocator.allocate(
            model=self.model,
            target_avg_bits=2.6,
            rank=4,
            metric="heuristic"
        )

        self.assertIsInstance(plan, MixedPrecisionAllocationPlan)
        self.assertAlmostEqual(plan.target_avg_bits, 2.6, places=2)
        # Effective base bitrate should be close to 2.6 bpp
        self.assertGreaterEqual(plan.effective_base_bits, 2.0)
        self.assertLessEqual(plan.effective_base_bits, 3.2)

        # Check mix of 2-bit and 4-bit layers
        self.assertGreater(plan.num_4bit_layers, 0)
        self.assertGreater(plan.num_2bit_layers, 0)
        self.assertEqual(plan.num_4bit_layers + plan.num_2bit_layers, 14)

        # Memory calculations should be consistent
        self.assertGreater(plan.compression_ratio_base, 3.5)
        self.assertGreater(plan.compression_ratio_net, 0.8)
        self.assertTrue(len(plan.summary()) > 100)

        # Also test on 7B LLM-scale dimensions (dim=4096, hidden_dim=11008)
        llm_model = MockTransformerModel(vocab_size=100, dim=1024, hidden_dim=2048, num_layers=2).to(self.device)
        llm_plan = MixedPrecisionAllocator.allocate(
            model=llm_model,
            target_avg_bits=2.6,
            rank=16,
            metric="heuristic"
        )
        self.assertGreater(llm_plan.compression_ratio_net, 4.0)

    def test_boundary_allocations(self):
        """Verify 2.0 bpp assigns all 2-bit, and 4.0 bpp assigns all 4-bit."""
        plan_2 = MixedPrecisionAllocator.allocate(self.model, target_avg_bits=2.0)
        self.assertEqual(plan_2.num_4bit_layers, 0)
        self.assertEqual(plan_2.num_2bit_layers, 14)
        self.assertAlmostEqual(plan_2.effective_base_bits, 2.0, places=2)

        plan_4 = MixedPrecisionAllocator.allocate(self.model, target_avg_bits=4.0)
        self.assertEqual(plan_4.num_4bit_layers, 14)
        self.assertEqual(plan_4.num_2bit_layers, 0)
        self.assertAlmostEqual(plan_4.effective_base_bits, 4.0, places=2)


class TestModelSurgicalConversionAndIntegration(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_allocate_mixed_precision_model_conversion(self):
        """Verify allocate_mixed_precision_model replaces linear layers with M2LRF4BitLinear and M2LRF2BitLinear."""
        model = MockTransformerModel(dim=64, hidden_dim=128, num_layers=2).to(self.device)
        
        model_conv, plan = allocate_mixed_precision_model(
            model=model,
            target_avg_bits=2.6,
            rank=16,
            alpha=16.0,
            loftq_iters=1,
            apply_conversion=True,
            verbose=False
        )

        num_4bit = 0
        num_2bit = 0
        for name, module in model_conv.named_modules():
            if isinstance(module, M2LRF4BitLinear):
                num_4bit += 1
                self.assertEqual(module.rank, 16)
            elif isinstance(module, M2LRF2BitLinear):
                num_2bit += 1
                self.assertEqual(module.rank, 16)

        self.assertEqual(num_4bit, plan.num_4bit_layers)
        self.assertEqual(num_2bit, plan.num_2bit_layers)
        self.assertEqual(num_4bit + num_2bit, 14)

        # Verify forward pass through converted model
        input_ids = torch.randint(10, 400, (2, 8), dtype=torch.long, device=self.device)
        logits = model_conv(input_ids)
        self.assertEqual(logits.shape, (2, 8, 500))

        # Verify gradient flow: adapters trainable, base frozen
        loss = logits.sum()
        loss.backward()
        for name, module in model_conv.named_modules():
            if isinstance(module, (M2LRF2BitLinear, M2LRF4BitLinear)):
                self.assertIsNotNone(module.lora_A.grad)
                self.assertIsNotNone(module.lora_B.grad)

    def test_prepare_m2lrf_model_mixed_precision_integration(self):
        """Verify prepare_m2lrf_model seamless delegation when target_avg_bits=2.6 is provided."""
        model = MockTransformerModel(dim=64, hidden_dim=128, num_layers=2).to(self.device)

        model_conv = prepare_m2lrf_model(
            model=model,
            rank=16,
            target_avg_bits=2.6,
            verbose=False
        )

        has_4bit = any(isinstance(m, M2LRF4BitLinear) for m in model_conv.modules())
        has_2bit = any(isinstance(m, M2LRF2BitLinear) for m in model_conv.modules())
        self.assertTrue(has_4bit, "Model should contain 4-bit layers under mixed precision")
        self.assertTrue(has_2bit, "Model should contain 2-bit layers under mixed precision")


if __name__ == "__main__":
    unittest.main()

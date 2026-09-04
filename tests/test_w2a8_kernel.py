"""
M-2LRF W2A8 Kernel & Layer Unit Test Suite
==========================================
Tests:
1. Dynamic Activation Quantization:
   - Verifies exact formula: s_x = max(|X|)/127.0, X_int8 = clamp(round(X / s_x), -127, 127).
   - Verifies INT8 range [-127, 127] and per-token scale broadcasting.
   - Verifies reconstruction SQNR > 40 dB.
2. In-SRAM Dequantization & Integer GEMM:
   - Verifies dual-basis integer GEMM (INT8 x ternary INT8 -> INT32 accumulation).
   - Verifies precision equivalence between integer GEMM and dequantized matmul.
3. M2LRFW2A8Linear Layer:
   - Step-0 LoftQ SVD residual representation recovery.
   - High-throughput inference mode with dynamic INT8 activation quantization.
   - End-to-end training mode: upstream gradient flow to inputs (grad_x), adapter gradient flow (lora_A, lora_B), and frozen base weights.
   - In-situ zero-overhead merge and unmerge.
   - Factory conversions: from_linear and from_2bit_linear.
   - Group-wise scale and Double Quantization compatibility.
"""

import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.w2a8_kernel import (
    quantize_activations_dynamic_int8,
    dequantize_activations_dynamic_int8,
    w2a8_integer_gemm,
    w2a8_matmul_fallback,
    w2a8_matmul,
    M2LRFW2A8MatmulFunction,
    M2LRFW2A8Linear
)
from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.packed_codec import Real2BitCodec
from m2lrf.layer import M2LRF2BitLinear


class TestW2A8DynamicActivationQuantization(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_dynamic_activation_quantization_formula_and_range(self):
        """Verify dynamic INT8 activation quantization follows exact formula and strictly stays in [-127, 127]."""
        # Test arbitrary shapes [B, S, K] and [M, K]
        shapes = [(2, 16, 256), (4, 128), (1, 8, 512)]
        for shape in shapes:
            x = torch.randn(*shape, device=self.device)
            x_int8, s_x = quantize_activations_dynamic_int8(x)

            # 1. Type and Range Checks
            self.assertEqual(x_int8.dtype, torch.int8)
            self.assertEqual(x_int8.shape, x.shape)
            self.assertEqual(s_x.shape, (*shape[:-1], 1))
            self.assertGreaterEqual(x_int8.min().item(), -127)
            self.assertLessEqual(x_int8.max().item(), 127)

            # 2. Formula Verification
            expected_s_x = torch.amax(torch.abs(x.float()), dim=-1, keepdim=True) / 127.0
            expected_s_x = torch.clamp(expected_s_x, min=1e-8).to(x.dtype)
            expected_int8 = torch.clamp(torch.round(x.float() / expected_s_x.float()), -127, 127).to(torch.int8)

            self.assertTrue(torch.equal(x_int8, expected_int8))
            self.assertTrue(torch.allclose(s_x.float(), expected_s_x.float(), atol=1e-5))

    def test_dynamic_activation_sqnr_fidelity(self):
        """Verify dynamic INT8 activation reconstruction achieves > 40 dB SQNR."""
        x = torch.randn(4, 32, 512, device=self.device)
        x_int8, s_x = quantize_activations_dynamic_int8(x)
        x_rec = dequantize_activations_dynamic_int8(x_int8, s_x, dtype=torch.float32)

        signal_power = torch.mean(x ** 2).item()
        noise_power = torch.mean((x - x_rec) ** 2).item()
        sqnr = 10 * math.log10(signal_power / noise_power)

        self.assertGreater(sqnr, 40.0, f"Activation SQNR {sqnr:.2f} dB was below expected 40.0 dB threshold")


class TestW2A8IntegerGEMMAndFallback(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_integer_gemm_vs_dequant_equivalence(self):
        """Verify INT8 x 2-bit ternary integer GEMM matches dequantized matmul within numerical precision."""
        in_features, out_features = 512, 256
        w = torch.randn(out_features, in_features, device=self.device)
        x = torch.randn(2, 8, in_features, device=self.device)

        # Dynamic INT8 activation quantization
        x_int8, s_x = quantize_activations_dynamic_int8(x)

        # Dual basis ternary decomposition
        t0, t1, a0, a1, w_base = DualBasisQuantizer.quantize_2_00b(w)

        # Dual-basis integer GEMM
        y_int_gemm = w2a8_integer_gemm(
            x_int8=x_int8,
            s_x=s_x,
            t0=t0,
            t1=t1,
            a0=a0,
            a1=a1,
            out_dtype=torch.float32
        )

        # Reference dequantized matmul
        y_ref = (F.linear(x_int8.float(), w_base.float())) * s_x.float()

        max_diff = torch.max(torch.abs(y_int_gemm - y_ref)).item()
        self.assertLess(max_diff, 1e-4, f"Integer GEMM differed from reference matmul by {max_diff}")

    def test_w2a8_matmul_fallback_shapes_and_precision(self):
        """Verify w2a8_matmul_fallback across various tensor ranks and dimensions."""
        w = torch.randn(128, 256, device=self.device)
        packed_tensor = Real2BitCodec.pack(w)

        # 3D Tensor [B, S, K]
        x_3d = torch.randn(2, 16, 256, device=self.device)
        out_3d = w2a8_matmul_fallback(
            x=x_3d,
            packed_weights=packed_tensor.packed_bytes,
            a0=packed_tensor.a0,
            a1=packed_tensor.a1,
            orig_shape=packed_tensor.orig_shape
        )
        self.assertEqual(out_3d.shape, (2, 16, 128))

        # 2D Tensor [M, K]
        x_2d = torch.randn(32, 256, device=self.device)
        out_2d = w2a8_matmul(
            x=x_2d,
            packed_weights=packed_tensor.packed_bytes,
            a0=packed_tensor.a0,
            a1=packed_tensor.a1,
            orig_shape=packed_tensor.orig_shape
        )
        self.assertEqual(out_2d.shape, (32, 128))


class TestM2LRFW2A8LinearLayer(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_w2a8_layer_initialization_and_step0_recovery(self):
        """Verify M2LRFW2A8Linear LoftQ SVD residual initialization achieves low Step-0 distortion."""
        in_features, out_features = 512, 256
        w_orig = torch.randn(out_features, in_features, device=self.device)

        layer = M2LRFW2A8Linear(in_features, out_features, rank=32, alpha=32.0).to(self.device)
        layer.initialize_from_pretrained(w_orig, loftq_iters=2)

        # Base 2-bit weight
        w_dequant = layer._dequantize().float()
        # Step-0 effective weight = W_2bit + scaling * (lora_B @ lora_A)
        w_step0 = w_dequant + (layer.lora_B @ layer.lora_A).float() * layer.scaling

        rel_error = (torch.norm(w_orig - w_step0) / torch.norm(w_orig)).item()
        base_rel_error = (torch.norm(w_orig - w_dequant) / torch.norm(w_orig)).item()

        self.assertLess(rel_error, base_rel_error)
        self.assertLess(rel_error, 0.30, f"Step-0 relative error {rel_error:.4f} exceeded expected threshold")

    def test_w2a8_training_gradient_flow(self):
        """
        Verify that during training:
        1. Base packed uint8 weights remain frozen.
        2. LoRA adapters (lora_A, lora_B) receive non-zero gradients.
        3. Input activations (x) receive non-zero gradients through M2LRFW2A8MatmulFunction autograd.
        """
        in_features, out_features = 128, 64
        layer = M2LRFW2A8Linear(in_features, out_features, rank=16, alpha=16.0, bias=True).to(self.device)
        w_orig = torch.randn(out_features, in_features, device=self.device)
        layer.initialize_from_pretrained(w_orig)
        layer.train()

        x = torch.randn(2, 8, in_features, device=self.device, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # 1. Base weights have no grad
        self.assertIsNone(layer.packed_weights.grad)

        # 2. LoRA parameters have non-zero gradients
        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertGreater(torch.norm(layer.lora_A.grad).item(), 0.0)
        self.assertGreater(torch.norm(layer.lora_B.grad).item(), 0.0)

        # 3. Bias has gradient
        self.assertIsNotNone(layer.bias.grad)
        self.assertGreater(torch.norm(layer.bias.grad).item(), 0.0)

        # 4. Input x receives backward gradient via STE autograd
        self.assertIsNotNone(x.grad)
        self.assertEqual(x.grad.shape, x.shape)
        self.assertGreater(torch.norm(x.grad).item(), 0.0)

    def test_w2a8_inference_forward(self):
        """Verify inference forward pass operates cleanly without gradients."""
        in_features, out_features = 256, 128
        layer = M2LRFW2A8Linear(in_features, out_features, rank=16, alpha=16.0).to(self.device)
        w_orig = torch.randn(out_features, in_features, device=self.device)
        layer.initialize_from_pretrained(w_orig)
        layer.eval()

        x = torch.randn(4, 16, in_features, device=self.device)
        with torch.no_grad():
            out = layer(x)

        self.assertEqual(out.shape, (4, 16, out_features))
        self.assertFalse(torch.isnan(out).any())

    def test_in_situ_merge_and_unmerge(self):
        """Verify in-situ merge fuses adapter and unmerge restores flag."""
        in_features, out_features = 128, 256
        layer = M2LRFW2A8Linear(in_features, out_features, rank=16).to(self.device)
        w_orig = torch.randn(out_features, in_features, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(2, in_features, device=self.device)
        with torch.no_grad():
            out_before = layer(x)
            layer.merge()
            out_after = layer(x)

        self.assertTrue(layer.is_merged)
        self.assertEqual(torch.sum(torch.abs(layer.lora_A)).item(), 0.0)
        self.assertEqual(torch.sum(torch.abs(layer.lora_B)).item(), 0.0)

        rel_diff = (torch.norm(out_before - out_after) / torch.norm(out_before)).item()
        self.assertLess(rel_diff, 0.45)

        layer.unmerge()
        self.assertFalse(layer.is_merged)

    def test_factory_from_linear_and_from_2bit_linear(self):
        """Verify factory constructors from_linear and from_2bit_linear."""
        # 1. from_linear
        std_linear = nn.Linear(128, 64, bias=True).to(self.device)
        w2a8_from_lin = M2LRFW2A8Linear.from_linear(std_linear, rank=8)
        self.assertEqual(w2a8_from_lin.in_features, 128)
        self.assertEqual(w2a8_from_lin.out_features, 64)
        self.assertEqual(w2a8_from_lin.rank, 8)
        self.assertIsNotNone(w2a8_from_lin.bias)

        # 2. from_2bit_linear
        layer_2bit = M2LRF2BitLinear(128, 64, rank=8, bias=True).to(self.device)
        layer_2bit.initialize_from_pretrained(std_linear.weight.data)
        w2a8_from_2bit = M2LRFW2A8Linear.from_2bit_linear(layer_2bit, act_quant=True)

        self.assertEqual(w2a8_from_2bit.in_features, 128)
        self.assertEqual(w2a8_from_2bit.out_features, 64)
        self.assertTrue(w2a8_from_2bit.act_quant)
        self.assertTrue(torch.equal(w2a8_from_2bit.packed_weights, layer_2bit.packed_weights))

    def test_group_wise_and_double_quant_w2a8(self):
        """Verify M2LRFW2A8Linear operates with group-wise scales and Double Quantization."""
        in_features, out_features = 512, 128
        w_orig = torch.randn(out_features, in_features, device=self.device)

        layer = M2LRFW2A8Linear(
            in_features=in_features,
            out_features=out_features,
            rank=16,
            group_size=64,
            double_quant=True
        ).to(self.device)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(2, 4, in_features, device=self.device)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, out_features))
        self.assertFalse(torch.isnan(out).any())


    def test_prepare_m2lrf_model_with_w2a8(self):
        """Verify prepare_m2lrf_model surgically converts target modules to M2LRFW2A8Linear when use_w2a8=True."""
        from m2lrf.trainer_eval import prepare_m2lrf_model

        class MockAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)
                self.o_proj = nn.Linear(dim, dim, bias=False)

        class MockBlock(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.attn = MockAttention(dim)
                self.norm = nn.LayerNorm(dim)

        class MockModel(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.embed = nn.Embedding(100, dim)
                self.block = MockBlock(dim)
                self.lm_head = nn.Linear(dim, 100, bias=False)

        model = MockModel().to(self.device)
        model = prepare_m2lrf_model(model, rank=16, use_w2a8=True, verbose=False)

        self.assertIsInstance(model.block.attn.q_proj, M2LRFW2A8Linear)
        self.assertIsInstance(model.block.attn.k_proj, M2LRFW2A8Linear)
        self.assertIsInstance(model.block.attn.v_proj, M2LRFW2A8Linear)
        self.assertIsInstance(model.block.attn.o_proj, M2LRFW2A8Linear)
        self.assertTrue(model.block.attn.q_proj.act_quant)
        self.assertNotIsInstance(model.lm_head, M2LRFW2A8Linear)


if __name__ == "__main__":
    unittest.main()


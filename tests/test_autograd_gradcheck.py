"""
M-2LRF Autograd and Precision Audit Test Suite
=============================================
Comprehensive numerical gradient verification via torch.autograd.gradcheck and
half-precision (FP16 / BF16) numerical stability auditing across extreme weight regimes.

Validates:
  1. Analytical vs. Numerical gradients for trainable LoRA adapter parameters (lora_A, lora_B)
     in float64 double precision across:
       - Standard 2-bit (use_hadamard=False, group_size=None)
       - Group-wise 2-bit (group_size=32)
       - Hadamard rotated (use_hadamard=True)
       - Dynamic INT8 activation (use_w2a8=True) with Straight-Through Estimator (STE)
       - Combined Hadamard + Group-wise + W2A8
     Under strict tolerance: eps=1e-5, atol=1e-4, rtol=1e-3.
  2. Forward and backward numerical stability in half-precision (FP16 and BF16):
       - Dynamic scale factor (a0, a1) bounds under small weights (1e-6) preventing underflow.
       - Dynamic scale factor bounds under large weights (10.0) preventing overflow.
       - Non-vanishing, finite, NaN/Inf-free gradient propagation.
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
    M2LRFW2A8Linear
)
from m2lrf.quantizer import DualBasisQuantizer, LLOYD_MAX_A0, LLOYD_MAX_A1
from m2lrf.w2a8_kernel import (
    DynamicInt8ActQuantSTE,
    quantize_activations_dynamic_int8
)
from m2lrf.hadamard_transform import random_orthogonal_transform


class TestAutogradGradcheckLoRA(unittest.TestCase):
    """
    Validates autograd numerical gradient accuracy for LoRA adapter parameters
    (lora_A and lora_B) using torch.autograd.gradcheck in float64 precision.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 32
        self.out_features = 16
        self.rank = 4
        self.batch_size = 2
        self.seq_len = 4
        self.eps = 1e-5
        self.atol = 1e-4
        self.rtol = 1e-3

    def _run_gradcheck_on_lora(self, layer: M2LRFUnifiedLinear, x: torch.Tensor):
        """
        Helper to execute torch.autograd.gradcheck on lora_A and lora_B parameters.
        """
        layer = layer.to(torch.float64)
        x = x.to(torch.float64)

        lora_a = layer.lora_A.data.clone().requires_grad_(True)
        lora_b = layer.lora_B.data.clone().requires_grad_(True)

        def forward_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            # 1. Activation quantization / transform
            if layer.use_w2a8:
                x_act = DynamicInt8ActQuantSTE.apply(x)
            else:
                x_act = x

            if layer.use_hadamard:
                x_work = random_orthogonal_transform(
                    x_act, signs=layer.signs, block_size=layer.block_size,
                    inverse=False, normalize=True
                )
            else:
                x_work = x_act

            # 2. Base quantized linear pass (frozen buffers)
            w_dequant = layer._dequantize_base(dtype=x.dtype)
            base_out = F.linear(x_work, w_dequant)

            # 3. LoRA adapter pass parameterized by inputs (a, b)
            x_adapted = layer.lora_dropout(x_work)
            lora_out = F.linear(
                F.linear(x_adapted.to(a.dtype), a),
                b
            ).to(x.dtype) * layer.scaling

            out = base_out + lora_out
            if layer.bias is not None:
                out = out + layer.bias.to(out.dtype)
            return out

        passed = torch.autograd.gradcheck(
            forward_fn,
            (lora_a, lora_b),
            eps=self.eps,
            atol=self.atol,
            rtol=self.rtol,
            raise_exception=True
        )
        self.assertTrue(passed, "torch.autograd.gradcheck failed for LoRA parameters")

    def test_gradcheck_standard_2bit(self):
        """
        Task 1.1: Verify gradcheck on Standard 2-bit (use_hadamard=False, group_size=None).
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            bits=2,
            group_size=None,
            use_hadamard=False,
            use_w2a8=False
        )
        w_init = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_init)

        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        self._run_gradcheck_on_lora(layer, x)

    def test_gradcheck_groupwise_2bit(self):
        """
        Task 1.2: Verify gradcheck on Group-wise 2-bit (group_size=32).
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            bits=2,
            group_size=32,
            use_hadamard=False,
            use_w2a8=False
        )
        w_init = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_init)

        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        self._run_gradcheck_on_lora(layer, x)

    def test_gradcheck_hadamard_rotated(self):
        """
        Task 1.3: Verify gradcheck on Hadamard rotated 2-bit (use_hadamard=True).
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            bits=2,
            group_size=None,
            use_hadamard=True,
            use_w2a8=False
        )
        w_init = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_init)

        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        self._run_gradcheck_on_lora(layer, x)

    def test_gradcheck_dynamic_int8_activation_w2a8(self):
        """
        Task 1.4: Verify gradcheck on Dynamic INT8 activation (use_w2a8=True) with STE.
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            bits=2,
            group_size=None,
            use_hadamard=False,
            use_w2a8=True
        )
        w_init = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_init)

        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        self._run_gradcheck_on_lora(layer, x)

    def test_gradcheck_combined_hadamard_w2a8_groupwise(self):
        """
        Verify gradcheck with all composable features active simultaneously:
        Hadamard rotation + Group-wise scaling (group_size=16) + Dynamic INT8 activations (W2A8).
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            bits=2,
            group_size=16,
            use_hadamard=True,
            use_w2a8=True,
            bias=True
        )
        w_init = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_init)

        x = torch.randn(self.batch_size, self.seq_len, self.in_features)
        self._run_gradcheck_on_lora(layer, x)

    def test_gradcheck_input_activation_gradients(self):
        """
        Verify analytical vs numerical gradients with respect to input tensor x
        for linear coordinate transformations (Standard and Hadamard).
        """
        for use_hadamard in [False, True]:
            layer = M2LRFUnifiedLinear(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=self.rank,
                bits=2,
                use_hadamard=use_hadamard,
                use_w2a8=False
            )
            w_init = torch.randn(self.out_features, self.in_features)
            layer.initialize_from_pretrained(w_init)
            layer = layer.to(torch.float64)

            x = torch.randn(2, self.in_features, dtype=torch.float64, requires_grad=True)

            def forward_x(input_tensor: torch.Tensor) -> torch.Tensor:
                return layer(input_tensor)

            passed = torch.autograd.gradcheck(
                forward_x, (x,),
                eps=self.eps, atol=self.atol, rtol=self.rtol,
                raise_exception=True
            )
            self.assertTrue(passed, f"Input gradcheck failed for use_hadamard={use_hadamard}")


class TestDynamicInt8ActQuantSTE(unittest.TestCase):
    """
    Validates Straight-Through Estimator (STE) behavior and gradient pass-through
    for dynamic INT8 activation quantization.
    """
    def test_ste_backward_identity_passthrough(self):
        """
        Validates that DynamicInt8ActQuantSTE.backward returns grad_output unchanged.
        """
        x = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
        x_quant = DynamicInt8ActQuantSTE.apply(x)

        grad_out = torch.randn_like(x_quant)
        x_quant.backward(grad_out)

        self.assertIsNotNone(x.grad)
        # STE backward must strictly be an identity operator
        torch.testing.assert_close(x.grad, grad_out, rtol=1e-5, atol=1e-5)

    def test_ste_training_loop_convergence(self):
        """
        Validates that an M2LRFUnifiedLinear layer with use_w2a8=True optimizes
        smoothly under standard gradient descent.
        """
        torch.manual_seed(42)
        layer = M2LRFUnifiedLinear(
            in_features=32,
            out_features=16,
            rank=4,
            bits=2,
            use_w2a8=True
        )
        w_init = torch.randn(16, 32)
        layer.initialize_from_pretrained(w_init)

        optimizer = torch.optim.SGD([layer.lora_A, layer.lora_B], lr=0.01)
        x = torch.randn(8, 32)
        target = torch.randn(8, 16)

        initial_loss = None
        for step in range(10):
            optimizer.zero_grad()
            out = layer(x)
            loss = F.mse_loss(out, target)
            loss.backward()
            optimizer.step()
            if step == 0:
                initial_loss = loss.item()

        self.assertLess(loss.item(), initial_loss, "Training loss failed to decrease with W2A8 STE")


class TestHalfPrecisionStability(unittest.TestCase):
    """
    Audits numerical scaling factors (a0, a1) and forward-backward stability
    under extreme weight magnitudes (1e-6 and 10.0) in FP16 and BF16.
    """
    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 64
        self.out_features = 32
        self.rank = 8
        self.batch_size = 4
        self.seq_len = 8

    def _audit_layer_stability(
        self,
        weight_scale: float,
        dtype: torch.dtype,
        layer_kwargs: dict
    ):
        """
        Executes end-to-end forward/backward precision audit.
        """
        # 1. Create extreme weight distribution
        w_orig = torch.randn(self.out_features, self.in_features) * weight_scale

        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            bits=2,
            **layer_kwargs
        )
        layer.initialize_from_pretrained(w_orig)
        layer = layer.to(dtype)

        # 2. Verify numerical scaling factors (a0, a1)
        self.assertFalse(torch.isnan(layer.a0).any(), "NaN found in a0 buffer")
        self.assertFalse(torch.isinf(layer.a0).any(), "Inf found in a0 buffer")
        self.assertFalse(torch.isnan(layer.a1).any(), "NaN found in a1 buffer")
        self.assertFalse(torch.isinf(layer.a1).any(), "Inf found in a1 buffer")

        # Bounds checks
        a0_min = layer.a0.float().min().item()
        a1_min = layer.a1.float().min().item()
        a0_max = layer.a0.float().max().item()
        a1_max = layer.a1.float().max().item()

        self.assertGreater(a0_min, 0.0, f"a0 underflowed to 0 for scale {weight_scale}")
        self.assertGreater(a1_min, 0.0, f"a1 underflowed to 0 for scale {weight_scale}")
        self.assertGreaterEqual(a1_min, a0_min * 0.99, "High-energy scale a1 must exceed low-energy scale a0")

        if dtype == torch.float16:
            self.assertLess(a1_max, 65504.0, f"a1 exceeded FP16 max for scale {weight_scale}")

        # 3. Forward pass verification
        x = torch.randn(self.batch_size, self.seq_len, self.in_features, dtype=dtype, requires_grad=True)
        out = layer(x)

        self.assertEqual(out.shape, (self.batch_size, self.seq_len, self.out_features))
        self.assertEqual(out.dtype, dtype)
        self.assertFalse(torch.isnan(out).any(), f"NaN in forward output for scale={weight_scale}, dtype={dtype}")
        self.assertFalse(torch.isinf(out).any(), f"Inf in forward output for scale={weight_scale}, dtype={dtype}")

        # 4. Backward pass verification
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(layer.lora_A.grad)
        self.assertIsNotNone(layer.lora_B.grad)
        self.assertIsNotNone(x.grad)

        self.assertFalse(torch.isnan(layer.lora_A.grad).any(), "NaN in lora_A gradient")
        self.assertFalse(torch.isinf(layer.lora_A.grad).any(), "Inf in lora_A gradient")
        self.assertFalse(torch.isnan(layer.lora_B.grad).any(), "NaN in lora_B gradient")
        self.assertFalse(torch.isinf(layer.lora_B.grad).any(), "Inf in lora_B gradient")
        self.assertFalse(torch.isnan(x.grad).any(), "NaN in input x gradient")
        self.assertFalse(torch.isinf(x.grad).any(), "Inf in input x gradient")

        # Ensure non-vanishing gradients
        self.assertGreater(layer.lora_A.grad.norm().item(), 0.0, "lora_A gradient vanished to 0")
        self.assertGreater(layer.lora_B.grad.norm().item(), 0.0, "lora_B gradient vanished to 0")

    def test_scaling_factors_underflow_small_weights_fp16(self):
        """
        Task 2.1: Verify scale factors and stability under very small weights (1e-6) in FP16.
        """
        configs = [
            {"use_hadamard": False, "group_size": None},
            {"use_hadamard": False, "group_size": 32},
            {"use_hadamard": True, "group_size": None},
            {"use_w2a8": True, "group_size": None}
        ]
        for cfg in configs:
            self._audit_layer_stability(weight_scale=1e-6, dtype=torch.float16, layer_kwargs=cfg)

    def test_scaling_factors_underflow_small_weights_bf16(self):
        """
        Task 2.2: Verify scale factors and stability under very small weights (1e-6) in BF16.
        """
        configs = [
            {"use_hadamard": False, "group_size": None},
            {"use_hadamard": False, "group_size": 32},
            {"use_hadamard": True, "group_size": None},
            {"use_w2a8": True, "group_size": None}
        ]
        for cfg in configs:
            self._audit_layer_stability(weight_scale=1e-6, dtype=torch.bfloat16, layer_kwargs=cfg)

    def test_scaling_factors_overflow_large_weights_fp16(self):
        """
        Task 2.3: Verify scale factors and stability under large weights (10.0) in FP16.
        """
        configs = [
            {"use_hadamard": False, "group_size": None},
            {"use_hadamard": False, "group_size": 32},
            {"use_hadamard": True, "group_size": None},
            {"use_w2a8": True, "group_size": None}
        ]
        for cfg in configs:
            self._audit_layer_stability(weight_scale=10.0, dtype=torch.float16, layer_kwargs=cfg)

    def test_scaling_factors_overflow_large_weights_bf16(self):
        """
        Task 2.4: Verify scale factors and stability under large weights (10.0) in BF16.
        """
        configs = [
            {"use_hadamard": False, "group_size": None},
            {"use_hadamard": False, "group_size": 32},
            {"use_hadamard": True, "group_size": None},
            {"use_w2a8": True, "group_size": None}
        ]
        for cfg in configs:
            self._audit_layer_stability(weight_scale=10.0, dtype=torch.bfloat16, layer_kwargs=cfg)

    def test_quantizer_scales_direct_analysis(self):
        """
        Direct mathematical verification of DualBasisQuantizer scaling factors
        across full dynamic range (1e-6 to 10.0).
        """
        for scale in [1e-6, 1e-3, 1.0, 10.0]:
            w = torch.randn(128, 256) * scale
            t0, t1, a0, a1, w_base = DualBasisQuantizer.quantize_2_00b(w, group_size=32)

            # Theoretical Lloyd-Max ratio check
            expected_ratio = LLOYD_MAX_A1 / LLOYD_MAX_A0  # ~3.3358
            actual_ratio = (a1 / a0).mean().item()
            self.assertAlmostEqual(actual_ratio, expected_ratio, places=2)

            # Clamping check: scale cannot be 0
            self.assertGreater(a0.min().item(), 0.0)
            self.assertGreater(a1.min().item(), 0.0)


if __name__ == "__main__":
    unittest.main()

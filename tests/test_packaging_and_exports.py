"""
Unit Tests for Packaging, Public API Exports, and Submodule Integrity
=====================================================================
Validates:
  1. Package metadata (__version__, __author__, __all__).
  2. Public export integrity (zero missing exports, all attributes present).
  3. Seamless submodule importing and cross-module compatibility aliases.
  4. Micro-benchmark and kernel fallback execution.
"""

import unittest
import torch
import torch.nn as nn
import m2lrf
from m2lrf import (
    M2LRFUnifiedLinear,
    M2LRF2BitLinear,
    HadamardDualBasisLinear,
    M2LRF4BitLinear,
    M2LRFW2A8Linear,
    QuantizedLinearWithLoRA,
    RealPacked2BitLinearLoRA,
    Quantized4BitLinearWithLoRA,
    QuantizedW2A8LinearWithLoRA,
    W2A8Linear,
    DynamicW2A8Linear,
    DualBasisQuantizer,
    DoubleQuantizer,
    SparseOutlierBuffer,
    LLOYD_MAX_A0,
    LLOYD_MAX_A1,
    LLOYD_MAX_TAU,
    Real2BitCodec,
    Packed2BitTensor,
    Real4BitCodec,
    NF4_CENTROIDS,
    LLOYD_MAX_4BIT_CENTROIDS,
    LayerSensitivityProfiler,
    SensitivityProfileResult,
    MixedPrecisionAllocator,
    MixedPrecisionAllocationPlan,
    allocate_mixed_precision_model,
    prepare_mixed_precision_m2lrf_model,
    DynamicInt8ActQuantSTE,
    quantize_activations_dynamic_int8,
    dequantize_activations_dynamic_int8,
    w2a8_integer_gemm,
    w2a8_matmul,
    w2a8_matmul_fallback,
    w2a8_triton_matmul,
    is_power_of_two,
    fast_walsh_hadamard_transform,
    block_fast_walsh_hadamard_transform,
    generate_hadamard_matrix,
    generate_random_orthogonal_matrix,
    random_orthogonal_transform,
    calculate_kurtosis,
    rotate_weights_for_quantization,
    analyze_outlier_suppression,
    verify_hadamard_sqnr_gain,
    generate_synthetic_heavy_tailed_weights,
    convert_linear_to_hadamard_dual_basis,
    prepare_m2lrf_model,
    RealTaskEvaluator,
    ConversationTrainer,
    get_model_device,
    DEFAULT_TARGET_MODULES,
    DEFAULT_EXCLUDE_MODULES,
    GSM8K_8SHOT_PROMPT,
    HAS_TRITON,
    m2lrf_triton_matmul,
    m2lrf_matmul_fallback,
    Uniform4BitLinearLoRA,
    run_benchmark_comparison,
)


class TestPackagingAndExports(unittest.TestCase):
    def test_package_metadata(self):
        """Verify package version, author, and docstring."""
        self.assertEqual(m2lrf.__version__, "2.0.0")
        self.assertEqual(m2lrf.__author__, "MD-Mushfiqur Rahim")
        self.assertIsNotNone(m2lrf.__doc__)
        self.assertIn("M-2LRF", m2lrf.__doc__)

    def test_all_exports_present(self):
        """Ensure every symbol declared in __all__ exists and is accessible."""
        self.assertGreater(len(m2lrf.__all__), 50)
        for symbol in m2lrf.__all__:
            self.assertTrue(
                hasattr(m2lrf, symbol),
                f"Exported symbol '{symbol}' is missing from top-level m2lrf package!"
            )

    def test_constants(self):
        """Verify closed-form Lloyd-Max and NF4 centroid constants."""
        self.assertAlmostEqual(LLOYD_MAX_A0, 0.4527786409, places=6)
        self.assertAlmostEqual(LLOYD_MAX_A1, 1.5104181947, places=6)
        self.assertAlmostEqual(LLOYD_MAX_TAU, (LLOYD_MAX_A0 + LLOYD_MAX_A1) / 2.0, places=6)
        self.assertEqual(NF4_CENTROIDS.numel(), 16)
        self.assertEqual(LLOYD_MAX_4BIT_CENTROIDS.numel(), 16)

    def test_layer_aliases(self):
        """Verify backwards-compatibility aliases point to expected classes."""
        self.assertIs(QuantizedLinearWithLoRA, M2LRF2BitLinear)
        self.assertIs(RealPacked2BitLinearLoRA, M2LRF2BitLinear)
        self.assertIs(W2A8Linear, M2LRFW2A8Linear)
        self.assertIs(DynamicW2A8Linear, M2LRFW2A8Linear)
        self.assertIs(QuantizedW2A8LinearWithLoRA, M2LRFW2A8Linear)

    def test_submodule_imports(self):
        """Verify direct importability of all canonical submodules."""
        import m2lrf.unified_layer
        import m2lrf.quantizer
        import m2lrf.packed_codec
        import m2lrf.mixed_precision
        import m2lrf.hadamard_transform
        import m2lrf.w2a8_kernel
        import m2lrf.trainer_eval
        import m2lrf.triton_kernel
        import m2lrf.deep_benchmark
        import m2lrf.layer
        import m2lrf.m2lrf_core_v1

        self.assertIsNotNone(m2lrf.layer.M2LRF2BitLinear)
        self.assertIsNotNone(m2lrf.m2lrf_core_v1.Real2BitCodec)

    def test_m2lrf_matmul_fallback_execution(self):
        """Verify m2lrf_matmul_fallback runs correctly on CPU."""
        torch.manual_seed(42)
        batch, seq, in_f, out_f = 2, 8, 64, 32
        x = torch.randn(batch, seq, in_f, dtype=torch.float32)
        w = torch.randn(out_f, in_f, dtype=torch.float32)
        packed, a0, a1, shape = Real2BitCodec.pack(w)

        out = m2lrf_matmul_fallback(x, packed, a0, a1, shape)
        self.assertEqual(out.shape, (batch, seq, out_f))
        self.assertFalse(torch.isnan(out).any())

    def test_uniform_4bit_baseline_and_micro_benchmark(self):
        """Verify Uniform4BitLinearLoRA and run_benchmark_comparison execute safely."""
        torch.manual_seed(42)
        linear_4b = Uniform4BitLinearLoRA(in_features=64, out_features=32, rank=8)
        w = torch.randn(32, 64)
        linear_4b.initialize_from_fp16(w)
        x = torch.randn(2, 4, 64)
        out = linear_4b(x)
        self.assertEqual(out.shape, (2, 4, 32))
        self.assertFalse(torch.isnan(out).any())

        # Micro-step benchmark
        res = run_benchmark_comparison(in_features=64, out_features=32, batch_size=2, seq_len=4, steps=2, device=torch.device("cpu"))
        self.assertIn("time_m2lrf_2bit_s", res)
        self.assertIn("time_4bit_s", res)


if __name__ == "__main__":
    unittest.main()

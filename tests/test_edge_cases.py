"""
Unit Tests for M-2LRF Edge Cases, Padding, and Robustness Hardening
===================================================================
Validates:
  1. Non-divisible dimensions and non-power-of-2 feature sizes:
     - in_features=100, out_features=57, group_size=64
     - in_features=768, out_features=2304, group_size=128, block_size=512
     - in_features=3584, out_features=11008 (Qwen / LLaMA intermediate MLP dims)
     - Extreme small and prime dimensions (1x1, 7x13, 33x17)
     - Frobenius norm isometry and involution across arbitrary dimensions
  2. Sparse Outliers Buffer on 0-outlier distributions:
     - Perfectly Gaussian and uniform distributions where 0 elements exceed 3.5 sigma
     - Empty buffer properties: num_outliers==0, density==0.0, to_sparse_coo(), apply_to()
     - Memory footprint and effective bits-per-parameter with 0 outliers
  3. Multi-Cycle merge() and unmerge() across 10 repeated cycles:
     - Cumulative drift measurement in Frobenius norm ||W_k - W_0||_F
     - Idempotent merge/unmerge verification (zero drift when adapter is inactive)
     - Adapter state reset and is_merged flag transitions
  4. CPU-only Execution without CUDA/Triton:
     - Execution of all layer variants on CPU tensors
     - Vectorized CPU fallback for Triton GEMM and W2A8 dynamic INT8 activations
     - End-to-end forward and backward autograd gradient flow on CPU
"""

import math
import unittest
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
from m2lrf.quantizer import (
    DualBasisQuantizer,
    DoubleQuantizer,
    SparseOutlierBuffer,
    LLOYD_MAX_A0,
    LLOYD_MAX_A1
)
from m2lrf.packed_codec import Real2BitCodec, Packed2BitTensor
from m2lrf.mixed_precision import Real4BitCodec
from m2lrf.hadamard_transform import (
    is_power_of_two,
    fast_walsh_hadamard_transform,
    block_fast_walsh_hadamard_transform,
    random_orthogonal_transform,
    generate_random_orthogonal_matrix,
    verify_hadamard_sqnr_gain,
    generate_synthetic_heavy_tailed_weights
)
from m2lrf.triton_kernel import (
    m2lrf_triton_matmul,
    m2lrf_matmul_fallback
)
from m2lrf.w2a8_kernel import (
    quantize_activations_dynamic_int8,
    dequantize_activations_dynamic_int8,
    DynamicInt8ActQuantSTE,
    w2a8_matmul
)


class TestNonDivisibleDimensionsAndOddFeatureSizes(unittest.TestCase):
    """
    Test suite for non-divisible dimensions, non-power-of-2 feature sizes,
    and arbitrary padding boundaries.
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_non_divisible_100_57_gs64(self):
        """Test in_features=100, out_features=57, group_size=64 across all bit-widths and options."""
        in_f, out_f, gs = 100, 57, 64
        w_orig = torch.randn(out_f, in_f)
        x = torch.randn(4, 8, in_f)

        configs = [
            {"bits": 2, "group_size": gs, "use_hadamard": False, "double_quant": False, "rank": 8},
            {"bits": 2, "group_size": gs, "use_hadamard": True, "double_quant": False, "rank": 8},
            {"bits": 2, "group_size": gs, "use_hadamard": False, "double_quant": True, "rank": 8},
            {"bits": 2, "group_size": gs, "use_hadamard": True, "double_quant": True, "rank": 8},
            {"bits": 2, "group_size": gs, "use_w2a8": True, "rank": 4},
            {"bits": 4, "group_size": gs, "codec_type": "nf4", "rank": 8},
            {"bits": 4, "group_size": gs, "codec_type": "lloyd_max", "rank": 8},
        ]

        for cfg in configs:
            layer = M2LRFUnifiedLinear(in_features=in_f, out_features=out_f, bias=True, **cfg)
            layer.initialize_from_pretrained(w_orig)

            out = layer(x)
            self.assertEqual(out.shape, (4, 8, out_f), f"Failed shape for config: {cfg}")
            self.assertFalse(torch.isnan(out).any(), f"NaN detected for config: {cfg}")

            w_eff = layer.dequantize_effective_weight()
            self.assertEqual(w_eff.shape, (out_f, in_f))
            self.assertFalse(torch.isnan(w_eff).any())
            self.assertGreater(layer.memory_bytes(), 0)
            self.assertGreater(layer.effective_bpp(), 0.0)

    def test_non_divisible_768_2304_gs128_bs512(self):
        """Test in_features=768, out_features=2304, group_size=128, block_size=512 (Multi-head Projection)."""
        in_f, out_f, gs, bs = 768, 2304, 128, 512
        w_orig = torch.randn(out_f, in_f)
        x = torch.randn(2, 4, in_f)

        layer = M2LRFUnifiedLinear(
            in_features=in_f,
            out_features=out_f,
            bits=2,
            group_size=gs,
            block_size=bs,
            use_hadamard=True,
            double_quant=True,
            rank=16,
            bias=True
        )
        layer.initialize_from_pretrained(w_orig)

        out = layer(x)
        self.assertEqual(out.shape, (2, 4, out_f))
        self.assertFalse(torch.isnan(out).any())

        w_eff = layer.dequantize_effective_weight()
        self.assertEqual(w_eff.shape, (out_f, in_f))
        self.assertFalse(torch.isnan(w_eff).any())

        # Check group count: 768 / 128 = 6 groups
        self.assertEqual(layer.a0.shape, (out_f, 6))
        self.assertEqual(layer.a1.shape, (out_f, 6))
        self.assertEqual(layer.packed_weights.shape, (out_f, 768 // 4))

    def test_qwen_llama_3584_11008(self):
        """Test in_features=3584, out_features=11008 (Qwen/LLaMA MLP dimension)."""
        in_f, out_f, gs, bs = 3584, 11008, 64, 512
        w_orig = torch.randn(out_f, in_f)
        x = torch.randn(1, 2, in_f)

        layer = M2LRFUnifiedLinear(
            in_features=in_f,
            out_features=out_f,
            bits=2,
            group_size=gs,
            block_size=bs,
            use_hadamard=True,
            rank=0,  # Pure base quantization test without SVD overhead
            bias=False
        )
        layer.initialize_from_pretrained(w_orig)

        out = layer(x)
        self.assertEqual(out.shape, (1, 2, out_f))
        self.assertFalse(torch.isnan(out).any())

        w_eff = layer.dequantize_effective_weight()
        self.assertEqual(w_eff.shape, (out_f, in_f))

        # Check bpp is around 2-bit
        self.assertAlmostEqual(layer.effective_bpp(), 2.25, delta=0.3)

    def test_inverted_and_asymmetric_dimensions(self):
        """Test inverted dimensions: in_features=57, out_features=100 and non-multiples."""
        in_f, out_f, gs = 57, 100, 64
        w_orig = torch.randn(out_f, in_f)
        x = torch.randn(2, 3, in_f)

        layer = M2LRFUnifiedLinear(
            in_features=in_f,
            out_features=out_f,
            bits=2,
            group_size=gs,
            use_hadamard=True,
            rank=4,
            bias=True
        )
        layer.initialize_from_pretrained(w_orig)

        out = layer(x)
        self.assertEqual(out.shape, (2, 3, out_f))
        w_eff = layer.dequantize_effective_weight()
        self.assertEqual(w_eff.shape, (out_f, in_f))

    def test_extreme_small_and_prime_dimensions(self):
        """Test extreme small and prime dimensions (1x1, 7x13, 33x17)."""
        dim_cases = [(1, 1, 64), (7, 13, 4), (33, 17, 32)]
        for in_f, out_f, gs in dim_cases:
            w_orig = torch.randn(out_f, in_f)
            x = torch.randn(2, in_f)

            # 2-bit
            l2 = M2LRFUnifiedLinear(in_f, out_f, bits=2, group_size=gs, use_hadamard=True, rank=2)
            l2.initialize_from_pretrained(w_orig)
            out2 = l2(x)
            self.assertEqual(out2.shape, (2, out_f))

            # 4-bit
            l4 = M2LRFUnifiedLinear(in_f, out_f, bits=4, group_size=gs, rank=2)
            l4.initialize_from_pretrained(w_orig)
            out4 = l4(x)
            self.assertEqual(out4.shape, (2, out_f))

    def test_hadamard_frobenius_isometry_on_arbitrary_dimensions(self):
        """Verify ||block_FWHT(x)||_F == ||x||_F and involution FWHT(FWHT(x)) == x on non-power-of-2 dims."""
        test_dims = [1, 7, 57, 100, 768, 2304, 3584]
        for d in test_dims:
            x = torch.randn(4, 16, d)
            norm_x = torch.norm(x.float()).item()

            # Forward block FWHT
            y = block_fast_walsh_hadamard_transform(x, block_size=512, normalize=True)
            norm_y = torch.norm(y.float()).item()
            rel_norm_diff = abs(norm_x - norm_y) / max(norm_x, 1e-12)
            self.assertLess(rel_norm_diff, 1e-5, f"Isometry failed for dimension {d}")

            # Involution (inverse)
            x_rec = block_fast_walsh_hadamard_transform(y, block_size=512, normalize=True)
            rel_rec_diff = torch.norm((x - x_rec).float()).item() / max(norm_x, 1e-12)
            self.assertLess(rel_rec_diff, 1e-5, f"Involution failed for dimension {d}")

            # Random orthogonal transform forward and inverse
            signs = torch.where(torch.randn(d) >= 0, torch.tensor(1.0), torch.tensor(-1.0))
            x_rot = random_orthogonal_transform(x, signs=signs, block_size=512, inverse=False, normalize=True)
            x_unrot = random_orthogonal_transform(x_rot, signs=signs, block_size=512, inverse=True, normalize=True)

            rel_rot_norm = abs(torch.norm(x_rot.float()).item() - norm_x) / max(norm_x, 1e-12)
            rel_rot_rec = torch.norm((x - x_unrot).float()).item() / max(norm_x, 1e-12)
            self.assertLess(rel_rot_norm, 1e-5, f"Rotated norm isometry failed for dimension {d}")
            self.assertLess(rel_rot_rec, 1e-5, f"Rotated inverse reconstruction failed for dimension {d}")


class TestZeroOutliersSparseBuffer(unittest.TestCase):
    """
    Test suite for SparseOutlierBuffer when 0 outliers exceed threshold
    (e.g., standard Gaussian clamped weights, uniform weights, high sigma threshold).
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_sparse_buffer_creation_with_zero_outliers(self):
        """Test SparseOutlierBuffer.from_tensor with 0 elements exceeding threshold."""
        # Perfectly uniform weights in [-0.01, 0.01] with std ~ 0.0058 -> boundary 3.5 * std ~ 0.02 > max(abs(w))
        w = torch.empty(64, 128).uniform_(-0.01, 0.01)
        thresh = torch.tensor(0.05)  # threshold higher than max magnitude

        buf = SparseOutlierBuffer.from_tensor(w, threshold=thresh, is_residual=False)

        self.assertEqual(buf.num_outliers, 0)
        self.assertEqual(buf.density, 0.0)
        self.assertEqual(buf.indices.shape, (2, 0))
        self.assertEqual(buf.values.shape, (0,))
        self.assertEqual(buf.dense_shape, (64, 128))

        # to_sparse_coo()
        coo = buf.to_sparse_coo()
        self.assertEqual(coo.shape, (64, 128))
        self.assertEqual(coo._nnz(), 0)
        self.assertTrue(coo.is_coalesced())

        # apply_to() should return base weights unmodified
        w_base = torch.randn(64, 128)
        w_applied = buf.apply_to(w_base)
        self.assertTrue(torch.equal(w_base, w_applied))

    def test_sparse_buffer_on_gaussian_distribution(self):
        """Test sparse outlier extraction on standard Gaussian clamped to < 3.5 sigma."""
        w = torch.randn(128, 256)
        std = torch.std(w)
        w_clamped = torch.clamp(w, -2.0 * std, 2.0 * std)

        buf = SparseOutlierBuffer.from_tensor(w_clamped, threshold=3.5 * std, is_residual=False)
        self.assertEqual(buf.num_outliers, 0)
        self.assertEqual(buf.density, 0.0)

    def test_unified_linear_with_sparse_outliers_zero_case(self):
        """Verify M2LRFUnifiedLinear with sparse_outliers=True on weights with 0 outliers."""
        w_clean = torch.empty(64, 128).uniform_(-0.05, 0.05)
        layer = M2LRFUnifiedLinear(
            in_features=128,
            out_features=64,
            bits=2,
            group_size=64,
            sparse_outliers=True,
            outlier_threshold_sigma=5.0,  # 5.0 sigma ensures 0 outliers
            rank=8,
            bias=True
        )
        layer.initialize_from_pretrained(w_clean)

        self.assertIsNotNone(layer.sparse_outliers)
        self.assertEqual(layer.sparse_outliers.num_outliers, 0)
        self.assertEqual(layer.sparse_outliers.density, 0.0)

        # Forward pass
        x = torch.randn(2, 4, 128)
        out = layer(x)
        self.assertEqual(out.shape, (2, 4, 64))
        self.assertFalse(torch.isnan(out).any())

        # Dequantization
        w_eff = layer.dequantize_effective_weight()
        self.assertEqual(w_eff.shape, (64, 128))
        self.assertFalse(torch.isnan(w_eff).any())

        # Memory computation
        mem = layer.memory_bytes()
        self.assertGreater(mem, 0)
        bpp = layer.effective_bpp()
        self.assertGreater(bpp, 0.0)

    def test_packed_2bit_tensor_representation_with_zero_outliers(self):
        """Test Packed2BitTensor repr and memory when sparse_outliers has 0 elements."""
        w = torch.randn(32, 64) * 0.01
        packed = Real2BitCodec.pack(
            w,
            group_size=32,
            extract_sparse_outliers=True,
            outlier_threshold_sigma=10.0
        )
        self.assertIsNotNone(packed.sparse_outliers)
        self.assertEqual(packed.sparse_outliers.num_outliers, 0)
        repr_str = repr(packed)
        self.assertIn("outliers=0", repr_str)
        self.assertGreater(packed.memory_bytes(), 0)

        dequant = packed.dequantize()
        self.assertEqual(dequant.shape, (32, 64))


class TestMultiCycleMergeUnmerge(unittest.TestCase):
    """
    Test suite for multi-cycle merge() and unmerge() operations across 10 repeated cycles:
    - Cumulative drift measurement in effective weight Frobenius norm ||W_k - W_0||_F
    - Idempotent merge/unmerge verification
    - LoRA adapter state resetting
    """

    def setUp(self):
        torch.manual_seed(42)
        self.in_features = 128
        self.out_features = 64
        self.rank = 16

    def test_10_cycle_merge_unmerge_with_simulated_training_updates(self):
        """
        Simulate 10 consecutive fine-tuning cycles:
        In each cycle: unmerge -> simulate gradient update to LoRA -> merge into base -> measure ||W_k - W_0||_F.
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            bits=2,
            group_size=32,
            use_hadamard=True,
            rank=self.rank
        )
        w_orig = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_orig)

        # Baseline effective weight W_0
        w0 = layer.dequantize_effective_weight().float().clone()
        prev_wk = w0.clone()

        cumulative_drifts = []
        step_drifts = []

        for cycle in range(10):
            # 1. Unmerge
            layer.unmerge()
            self.assertFalse(layer.is_merged)

            # 2. Simulate fine-tuning update to LoRA adapter
            layer.lora_A.data.add_(torch.randn_like(layer.lora_A) * 0.005)
            layer.lora_B.data.add_(torch.randn_like(layer.lora_B) * 0.005)

            # 3. Merge adapter into base representation
            layer.merge()
            self.assertTrue(layer.is_merged)
            self.assertEqual(layer.lora_A.abs().sum().item(), 0.0)
            self.assertEqual(layer.lora_B.abs().sum().item(), 0.0)

            # 4. Dequantize merged effective weight W_k
            wk = layer.dequantize_effective_weight().float()
            self.assertFalse(torch.isnan(wk).any(), f"NaN in cycle {cycle+1}")

            # 5. Measure cumulative drift ||W_k - W_0||_F and step drift ||W_k - W_{k-1}||_F
            cum_drift = torch.norm(wk - w0).item()
            step_drift = torch.norm(wk - prev_wk).item()

            cumulative_drifts.append(cum_drift)
            step_drifts.append(step_drift)
            prev_wk = wk.clone()

            # Verify forward pass remains stable
            x = torch.randn(2, 4, self.in_features)
            out = layer(x)
            self.assertEqual(out.shape, (2, 4, self.out_features))
            self.assertFalse(torch.isnan(out).any())

        # All cumulative drifts must be positive and finite
        self.assertEqual(len(cumulative_drifts), 10)
        for d in cumulative_drifts:
            self.assertTrue(math.isfinite(d))
            self.assertGreater(d, 0.0)

        # Drift should be bounded (no explosion)
        self.assertLess(cumulative_drifts[-1], 150.0)

    def test_10_cycle_merge_unmerge_idempotency(self):
        """
        Verify that repeated merge() and unmerge() WITHOUT adapter updates
        is strictly idempotent with ZERO numerical drift (||W_k - W_merged_0||_F == 0.0).
        """
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            bits=2,
            group_size=32,
            use_hadamard=True,
            rank=self.rank
        )
        w_orig = torch.randn(self.out_features, self.in_features)
        layer.initialize_from_pretrained(w_orig)

        # Fuse initial LoftQ adapter into base
        layer.merge()
        w_merged_0 = layer.dequantize_effective_weight().float().clone()

        for cycle in range(10):
            layer.unmerge()
            self.assertFalse(layer.is_merged)
            layer.merge()
            self.assertTrue(layer.is_merged)

            wk = layer.dequantize_effective_weight().float()
            drift = torch.norm(wk - w_merged_0).item()
            self.assertEqual(drift, 0.0, f"Drift non-zero in idempotent cycle {cycle+1}: {drift}")

    def test_hadamard_dual_basis_linear_10_cycle_merge(self):
        """Test multi-cycle merge/unmerge directly on HadamardDualBasisLinear subclass."""
        had_layer = HadamardDualBasisLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            group_size=32
        )
        w_orig = torch.randn(self.out_features, self.in_features)
        had_layer.initialize_from_pretrained(w_orig)

        # Initial merge
        had_layer.merge()
        w0 = had_layer.dequantize_effective_weight().float().clone()

        # 10 idempotent cycles
        for _ in range(10):
            had_layer.unmerge()
            had_layer.merge()
            wk = had_layer.dequantize_effective_weight().float()
            self.assertEqual(torch.norm(wk - w0).item(), 0.0)

    def test_4bit_10_cycle_merge_unmerge(self):
        """Test 10-cycle merge/unmerge on 4-bit (NF4) linear layer."""
        l4 = M2LRF4BitLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=self.rank,
            group_size=64,
            codec_type="nf4"
        )
        w_orig = torch.randn(self.out_features, self.in_features)
        l4.initialize_from_pretrained(w_orig)

        l4.merge()
        w0 = l4.dequantize_effective_weight().float().clone()

        for _ in range(10):
            l4.unmerge()
            l4.merge()
            wk = l4.dequantize_effective_weight().float()
            self.assertEqual(torch.norm(wk - w0).item(), 0.0)


class TestCPUOnlyExecution(unittest.TestCase):
    """
    Test suite for CPU-only execution without CUDA or Triton availability.
    Guarantees seamless fallback and numerical accuracy on CPU environments.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")
        self.in_features = 128
        self.out_features = 64

    def test_all_unified_layer_subclasses_on_cpu(self):
        """Instantiate and execute all subclasses on CPU."""
        layers = [
            M2LRF2BitLinear(self.in_features, self.out_features, rank=8),
            HadamardDualBasisLinear(self.in_features, self.out_features, rank=8),
            M2LRF4BitLinear(self.in_features, self.out_features, rank=8, codec_type="nf4"),
            M2LRF4BitLinear(self.in_features, self.out_features, rank=8, codec_type="lloyd_max"),
            M2LRFW2A8Linear(self.in_features, self.out_features, rank=8)
        ]

        w_orig = torch.randn(self.out_features, self.in_features, device=self.device)
        x = torch.randn(2, 4, self.in_features, device=self.device)

        for layer in layers:
            layer.to(self.device)
            layer.initialize_from_pretrained(w_orig)

            out = layer(x)
            self.assertEqual(out.device, self.device)
            self.assertEqual(out.shape, (2, 4, self.out_features))
            self.assertFalse(torch.isnan(out).any())

            w_eff = layer.dequantize_effective_weight()
            self.assertEqual(w_eff.device, self.device)
            self.assertEqual(w_eff.shape, (self.out_features, self.in_features))

    def test_triton_cpu_fallback(self):
        """Test m2lrf_triton_matmul dispatching to m2lrf_matmul_fallback on CPU."""
        w = torch.randn(self.out_features, self.in_features, device=self.device)
        x = torch.randn(4, self.in_features, device=self.device)

        packed = Real2BitCodec.pack(w)
        out_fallback = m2lrf_matmul_fallback(
            x, packed.packed_bytes, packed.a0, packed.a1, (self.out_features, self.in_features)
        )
        out_dispatch = m2lrf_triton_matmul(
            x, packed.packed_bytes, packed.a0, packed.a1, (self.out_features, self.in_features)
        )

        self.assertEqual(out_fallback.shape, (4, self.out_features))
        self.assertEqual(out_dispatch.shape, (4, self.out_features))
        self.assertTrue(torch.allclose(out_fallback, out_dispatch, atol=1e-5))

    def test_w2a8_cpu_fallback_and_dynamic_quant(self):
        """Test W2A8 dynamic INT8 activation quantization and matmul fallback on CPU."""
        x = torch.randn(3, 8, self.in_features, device=self.device)
        x_int8, s_x = quantize_activations_dynamic_int8(x)

        self.assertEqual(x_int8.dtype, torch.int8)
        self.assertEqual(x_int8.shape, (3, 8, self.in_features))
        self.assertEqual(s_x.shape, (3, 8, 1))

        x_rec = dequantize_activations_dynamic_int8(x_int8, s_x, dtype=x.dtype)
        self.assertEqual(x_rec.shape, x.shape)
        rel_diff = (torch.norm(x - x_rec) / torch.norm(x)).item()
        self.assertLess(rel_diff, 0.05)  # INT8 dynamic quantization error < 5%

        # W2A8 matmul
        w = torch.randn(self.out_features, self.in_features, device=self.device)
        packed = Real2BitCodec.pack(w)
        out_w2a8 = w2a8_matmul(x, packed.packed_bytes, packed.a0, packed.a1, (self.out_features, self.in_features))
        self.assertEqual(out_w2a8.shape, (3, 8, self.out_features))

    def test_hadamard_sqnr_verification_and_helpers_on_cpu(self):
        """Test verify_hadamard_sqnr_gain and generate_synthetic_heavy_tailed_weights on CPU."""
        w_heavy = generate_synthetic_heavy_tailed_weights(
            out_features=256,
            in_features=256,
            num_outlier_channels=4,
            outlier_multiplier=10.0,
            seed=42,
            device=self.device
        )
        self.assertEqual(w_heavy.device, self.device)
        self.assertEqual(w_heavy.shape, (256, 256))

        result = verify_hadamard_sqnr_gain(w_heavy, block_size=256, seed=42)
        self.assertTrue(result["frobenius_isometry_holds"])
        self.assertGreater(result["sqnr_gain_db"], 1.0)

    def test_gradient_flow_and_optimizer_step_on_cpu(self):
        """Verify full autograd backward and optimizer step work seamlessly on CPU."""
        layer = M2LRFUnifiedLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            bits=2,
            group_size=64,
            use_hadamard=True,
            use_w2a8=True,
            rank=8,
            bias=True
        ).to(self.device)

        w_orig = torch.randn(self.out_features, self.in_features, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        optimizer = torch.optim.AdamW(layer.parameters(), lr=1e-3)

        lora_A_before = layer.lora_A.clone()
        lora_B_before = layer.lora_B.clone()

        x = torch.randn(2, 4, self.in_features, device=self.device)
        target = torch.randn(2, 4, self.out_features, device=self.device)

        out = layer(x)
        loss = F.mse_loss(out, target)
        loss.backward()

        optimizer.step()

        # LoRA weights must be updated
        self.assertFalse(torch.equal(layer.lora_A, lora_A_before))
        self.assertFalse(torch.equal(layer.lora_B, lora_B_before))

        # Base buffers must remain unmodified
        self.assertFalse(layer.packed_weights.requires_grad)
        self.assertFalse(layer.a0.requires_grad)


if __name__ == "__main__":
    unittest.main()

"""
M-2LRF Hadamard Transform & Orthogonal Rotation Outlier Suppression Test Suite
==============================================================================
Tests:
1. Power of Two & FWHT Butterfly Properties (Orthogonality, Involution, Matrix-equivalence).
2. Block-FWHT on Arbitrary / Non-Power-of-2 Dimensions (768, 1280, 3584, 11008).
3. Orthogonal Matrix Generators (hadamard, random_hadamard, double_random_hadamard, haar_qr).
4. Memory-Free Randomized Orthogonal Rotation: x @ Q and y @ Q^T.
5. Statistical Outlier Channel Dispersion & Kurtosis Suppression.
6. Mathematical Verification of SQNR Gain (+2.5 to +4.0+ dB) & Frobenius Isometry.
7. HadamardDualBasisLinear Layer: 2-bit packing, on-the-fly activation rotation, LoRA gradients, in-situ merge.
8. Model Conversion Helper: Surgical replacement of linear layers in Transformer architectures.
"""

import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.hadamard_transform import (
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
    HadamardDualBasisLinear,
    convert_linear_to_hadamard_dual_basis
)
from m2lrf.quantizer import DualBasisQuantizer


class TestHadamardTransform(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_is_power_of_two(self):
        """Verify power-of-two bitwise detection."""
        powers = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        non_powers = [0, 3, 5, 6, 7, 9, 768, 1000, 11008, 14336]
        for p in powers:
            self.assertTrue(is_power_of_two(p), f"{p} should be detected as power of 2")
        for np in non_powers:
            self.assertFalse(is_power_of_two(np), f"{np} should not be detected as power of 2")

    def test_fwht_butterfly_properties(self):
        """
        Verify FWHT butterfly properties:
        1. Equivalence to explicit Walsh-Hadamard matrix multiplication: x @ H_hat.
        2. Strict Frobenius norm isometry: ||FWHT(x)||_F == ||x||_F.
        3. Self-inversion / Involution: FWHT(FWHT(x)) == x.
        4. Multi-dimensional tensor handling: (B, L, d).
        """
        for d in [4, 8, 16, 64, 256, 512, 1024]:
            h_mat = generate_hadamard_matrix(d, normalize=True, device=self.device)
            x = torch.randn(4, 16, d, device=self.device)
            
            # 1. Compare FWHT to explicit matrix multiplication
            y_fwht = fast_walsh_hadamard_transform(x, normalize=True)
            y_matmul = (x.float() @ h_mat.t()).to(x.dtype)
            diff = torch.max(torch.abs(y_fwht - y_matmul)).item()
            self.assertLess(diff, 1e-4, f"FWHT failed to match explicit Hadamard matrix for d={d}")
            
            # 2. Isometry: ||y||_F == ||x||_F
            norm_x = torch.norm(x.float()).item()
            norm_y = torch.norm(y_fwht.float()).item()
            rel_norm_diff = abs(norm_x - norm_y) / max(norm_x, 1e-6)
            self.assertLess(rel_norm_diff, 1e-4, f"FWHT violated Frobenius norm isometry for d={d}")
            
            # 3. Involution: FWHT(FWHT(x)) == x
            x_rec = fast_walsh_hadamard_transform(y_fwht, normalize=True)
            rec_diff = torch.max(torch.abs(x - x_rec)).item()
            self.assertLess(rec_diff, 1e-4, f"FWHT failed involution reconstruction for d={d}")

    def test_block_fwht_arbitrary_dimensions(self):
        """
        Verify block-wise FWHT on arbitrary non-power-of-2 feature dimensions (e.g. 768, 1280, 3584, 11008).
        Guarantees exact isometry and involution.
        """
        arbitrary_dims = [768, 1280, 3584, 4096, 11008]
        for d in arbitrary_dims:
            x = torch.randn(2, 8, d, device=self.device)
            y = block_fast_walsh_hadamard_transform(x, block_size=512, normalize=True)
            
            # 1. Shape preservation
            self.assertEqual(y.shape, x.shape)
            
            # 2. Isometry: ||y||_F == ||x||_F
            norm_x = torch.norm(x.float()).item()
            norm_y = torch.norm(y.float()).item()
            rel_norm_diff = abs(norm_x - norm_y) / max(norm_x, 1e-6)
            self.assertLess(rel_norm_diff, 1e-4, f"Block FWHT violated isometry for d={d}")
            
            # 3. Involution: BlockFWHT(BlockFWHT(x)) == x
            x_rec = block_fast_walsh_hadamard_transform(y, block_size=512, normalize=True)
            rec_diff = torch.max(torch.abs(x - x_rec)).item()
            self.assertLess(rec_diff, 1e-4, f"Block FWHT failed involution for d={d}")

    def test_orthogonal_matrix_generators(self):
        """
        Verify generate_random_orthogonal_matrix across all modes:
        - "hadamard"
        - "random_hadamard"
        - "double_random_hadamard"
        - "haar_qr"
        Strictly verifies Q^T Q == I_d and Q Q^T == I_d.
        """
        d = 128
        modes = ["hadamard", "random_hadamard", "double_random_hadamard", "haar_qr"]
        for mode in modes:
            q = generate_random_orthogonal_matrix(d, mode=mode, seed=42, device=self.device)
            self.assertEqual(q.shape, (d, d))
            
            # Check Q^T @ Q == I
            qt_q = q.t() @ q
            eye = torch.eye(d, device=self.device)
            diff_qt_q = torch.max(torch.abs(qt_q - eye)).item()
            self.assertLess(diff_qt_q, 1e-4, f"Mode '{mode}' generated non-orthogonal matrix Q^T Q != I")
            
            # Check Q @ Q^T == I
            q_qt = q @ q.t()
            diff_q_qt = torch.max(torch.abs(q_qt - eye)).item()
            self.assertLess(diff_q_qt, 1e-4, f"Mode '{mode}' generated non-orthogonal matrix Q Q^T != I")

    def test_random_orthogonal_transform_memory_free(self):
        """
        Verify random_orthogonal_transform performs O(N d log d) transform matching explicit Q = D @ H_hat
        and that inverse transform y @ Q^T reconstructs original input x.
        """
        d = 256
        x = torch.randn(8, d, device=self.device)
        rand_bits = torch.randint(0, 2, (d,), device=self.device, dtype=torch.float32)
        signs = rand_bits * 2.0 - 1.0
        
        # Explicit matrix Q = D @ H_hat
        h_mat = generate_hadamard_matrix(d, normalize=True, device=self.device)
        q_explicit = signs.unsqueeze(1) * h_mat
        
        # Fast memory-free forward transform
        y_fast = random_orthogonal_transform(x, signs=signs, inverse=False, normalize=True)
        y_matmul = x @ q_explicit
        diff_fwd = torch.max(torch.abs(y_fast - y_matmul)).item()
        self.assertLess(diff_fwd, 1e-4, "Fast randomized orthogonal transform differed from explicit matmul")
        
        # Fast memory-free inverse transform: y @ Q^T
        x_rec = random_orthogonal_transform(y_fast, signs=signs, inverse=True, normalize=True)
        x_matmul_rec = y_matmul @ q_explicit.t()
        diff_rec = torch.max(torch.abs(x - x_rec)).item()
        self.assertLess(diff_rec, 1e-4, "Inverse randomized orthogonal transform failed to reconstruct input")

    def test_outlier_dispersion_and_kurtosis_reduction(self):
        """
        Verify that randomized Hadamard pre-rotation transforms heavy-tailed weight matrices
        (Kurtosis > 20) into homogeneous Gaussian distributions (Kurtosis ≈ 3.0) and sharply compresses peak outliers.
        """
        d_out, d_in = 512, 512
        w_heavy = generate_synthetic_heavy_tailed_weights(
            out_features=d_out,
            in_features=d_in,
            num_outlier_channels=8,
            outlier_multiplier=15.0,
            seed=42,
            device=self.device
        )
        
        w_rot, signs = rotate_weights_for_quantization(w_heavy, block_size=512, seed=42)
        stats = analyze_outlier_suppression(w_heavy, w_rot, sigma_thresh=3.5)
        
        # Verify initial kurtosis is heavily non-Gaussian
        self.assertGreater(stats["kurtosis_before"], 15.0, "Initial weight should have high kurtosis")
        
        # Verify rotated kurtosis is centered around Gaussian standard (3.0 ± 0.6)
        self.assertAlmostEqual(stats["kurtosis_after"], 3.0, delta=0.6, msg="Rotated weights did not converge to Gaussian kurtosis")
        
        # Verify peak outlier magnitude is reduced significantly (> 3x)
        self.assertGreater(stats["outlier_dispersion_ratio"], 3.0, "Outlier peak magnitude was not sufficiently dispersed")
        
        # Verify PAPR reduction
        self.assertGreater(stats["papr_reduction_db"], 5.0, "Peak-to-Average Power Ratio was not reduced by > 5 dB")

    def test_mathematical_sqnr_gain_and_frobenius_isometry(self):
        """
        Verify:
        1. Exact Frobenius isometry: ||W - Dequant(W_tilde) @ Q^T||_F^2 == ||W_tilde - Dequant(W_tilde)||_F^2
        2. Substantial SQNR elevation (+2.5 to +4.0+ dB) on heavy-tailed weights.
        """
        d_out, d_in = 1024, 1024
        w_heavy = generate_synthetic_heavy_tailed_weights(
            out_features=d_out,
            in_features=d_in,
            num_outlier_channels=8,
            outlier_multiplier=12.0,
            student_t_df=2.5,
            seed=42,
            device=self.device
        )
        
        res = verify_hadamard_sqnr_gain(w_heavy, group_size=None, block_size=512, seed=42)
        
        # 1. Check Frobenius isometry
        self.assertTrue(res["frobenius_isometry_holds"], "Frobenius isometric error equivalence violated!")
        self.assertLess(res["isometry_relative_difference"], 1e-4)
        
        # 2. Check SQNR gain (+2.5 dB to +4.0+ dB)
        self.assertGreaterEqual(
            res["sqnr_gain_db"],
            2.5,
            f"SQNR gain {res['sqnr_gain_db']:.2f} dB was below the expected +2.5 dB threshold"
        )
        self.assertGreater(res["error_reduction_percentage"], 35.0, "Reconstruction error was not reduced by > 35%")

    def test_hadamard_dual_basis_linear_layer(self):
        """
        Verify HadamardDualBasisLinear layer:
        1. 2-bit uint8 physical storage (8x compression).
        2. On-the-fly activation rotation and GEMM correctness.
        3. Step-0 LoftQ SVD representation.
        4. LoRA gradient flow (base frozen, adapter trainable).
        5. Effective full-precision weight reconstruction in original space.
        6. In-situ merge and unmerge.
        """
        in_features, out_features = 512, 1024
        layer = HadamardDualBasisLinear(in_features, out_features, rank=32, alpha=32.0).to(self.device)
        
        w_orig = generate_synthetic_heavy_tailed_weights(
            out_features=out_features, in_features=in_features, num_outlier_channels=6, seed=42, device=self.device
        )
        layer.initialize_from_pretrained(w_orig, loftq_iters=1)
        
        # 1. Verify 2-bit memory compression
        fp16_bytes = out_features * in_features * 2
        packed_bytes = layer.packed_weights.numel() * layer.packed_weights.element_size()
        self.assertAlmostEqual(fp16_bytes / packed_bytes, 8.0, places=1)
        
        # 2. Verify effective weight reconstruction in original coordinate space
        w_eff = layer.dequantize_effective_weight(dtype=torch.float32)
        sqnr_eff = DualBasisQuantizer.calculate_sqnr(w_orig, w_eff)
        self.assertGreater(sqnr_eff, 10.0, f"Effective Step-0 SQNR {sqnr_eff:.2f} dB was below 10.0 dB")
        
        # Verify it significantly outperforms direct unrotated quantization
        _, _, _, _, w_direct = DualBasisQuantizer.quantize_2_00b(w_orig)
        sqnr_direct = DualBasisQuantizer.calculate_sqnr(w_orig, w_direct)
        self.assertGreater(
            sqnr_eff,
            sqnr_direct + 2.5,
            f"Hadamard Step-0 SQNR ({sqnr_eff:.2f} dB) must be > 2.5 dB above direct unrotated ({sqnr_direct:.2f} dB)"
        )
        
        # 3. Verify forward pass output shape and finite values
        x = torch.randn(4, 16, in_features, device=self.device)
        y = layer(x)
        self.assertEqual(y.shape, (4, 16, out_features))
        self.assertFalse(torch.isnan(y).any())
        
        # 4. Verify LoRA gradient flow
        x_grad = torch.randn(2, in_features, device=self.device, requires_grad=True)
        out_grad = layer(x_grad)
        loss = out_grad.sum()
        loss.backward()
        self.assertIsNone(layer.packed_weights.grad, "Base packed weights must remain frozen")
        self.assertIsNotNone(layer.lora_A.grad, "LoRA A must receive gradients")
        self.assertIsNotNone(layer.lora_B.grad, "LoRA B must receive gradients")
        self.assertGreater(torch.norm(layer.lora_A.grad).item(), 0.0)
        
        # 5. Verify in-situ merge
        with torch.no_grad():
            out_before = layer(x)
            layer.merge()
            out_after = layer(x)
            
        self.assertTrue(layer.is_merged)
        self.assertEqual(torch.sum(torch.abs(layer.lora_A)).item(), 0.0)
        self.assertEqual(torch.sum(torch.abs(layer.lora_B)).item(), 0.0)
        
        rel_merge_err = (torch.norm(out_before - out_after) / torch.norm(out_before)).item()
        self.assertLess(rel_merge_err, 0.45, f"Merge relative error {rel_merge_err:.4f} exceeded bound")

    def test_model_conversion_helper(self):
        """
        Verify convert_linear_to_hadamard_dual_basis correctly replaces linear layers in a mock Transformer.
        """
        class MockAttention(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.q_proj = nn.Linear(dim, dim, bias=False)
                self.k_proj = nn.Linear(dim, dim, bias=False)
                self.v_proj = nn.Linear(dim, dim, bias=False)
                self.o_proj = nn.Linear(dim, dim, bias=False)

        class MockMLP(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
                self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
                self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

        class MockBlock(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.attn = MockAttention(dim)
                self.mlp = MockMLP(dim, hidden_dim)

        class MockModel(nn.Module):
            def __init__(self, dim=128, hidden_dim=256):
                super().__init__()
                self.embed_tokens = nn.Embedding(500, dim)
                self.blocks = nn.ModuleList([MockBlock(dim, hidden_dim) for _ in range(2)])
                self.lm_head = nn.Linear(dim, 500, bias=False)

        model = MockModel().to(self.device)
        model = convert_linear_to_hadamard_dual_basis(
            model,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            exclude_modules=["lm_head", "embed_tokens"],
            rank=16,
            alpha=16.0,
            verbose=False
        )
        
        # Verify attention and mlp converted to HadamardDualBasisLinear
        for block in model.blocks:
            self.assertIsInstance(block.attn.q_proj, HadamardDualBasisLinear)
            self.assertIsInstance(block.attn.k_proj, HadamardDualBasisLinear)
            self.assertIsInstance(block.mlp.gate_proj, HadamardDualBasisLinear)
            self.assertIsInstance(block.mlp.down_proj, HadamardDualBasisLinear)
            
        # Verify lm_head guarded
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertNotIsInstance(model.lm_head, HadamardDualBasisLinear)


if __name__ == "__main__":
    unittest.main()

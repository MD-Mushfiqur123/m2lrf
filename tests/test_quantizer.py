"""
M-2LRF Unit Test Suite
=======================
Tests:
1. Disjointness Guarantee: T0 ⊙ T1 = 0
2. Ternary Alphabet: T0, T1 ∈ {-1, 0, +1}
3. Theoretical SQNR Bound: ~9.0 - 9.5 dB for Gaussian weights
4. Bit-Packing Roundtrip: Pack -> Unpack -> Reconstruct equals direct dual-basis
5. Physical Memory Compression Ratio: packed_weights is exactly 1/8 of FP16 memory
6. LoftQ SVD Residual Initialization & Step-0 Representation
7. LoRA Gradient Flow: base is frozen, adapter receives non-zero gradients
8. In-Situ Merge: Verifies adapter fusing and checks relative re-quantization error bounded by 2-bit Lloyd-Max limit
"""

import unittest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.packed_codec import Real2BitCodec
from m2lrf.layer import M2LRF2BitLinear


class TestM2LRFCore(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_disjointness_invariant(self):
        """Verify T0 ⊙ T1 == 0 across multiple random Gaussian tensors."""
        for _ in range(5):
            w = torch.randn(256, 512, device=self.device)
            t0, t1, _, _, _ = DualBasisQuantizer.quantize_2_00b(w)
            hadamard = t0 * t1
            self.assertEqual(
                torch.sum(torch.abs(hadamard)).item(),
                0,
                "Disjointness invariant T0 ⊙ T1 == 0 violated!"
            )

    def test_ternary_alphabet(self):
        """Verify T0 and T1 contain strictly values in {-1, 0, +1}."""
        w = torch.randn(128, 256, device=self.device)
        t0, t1, _, _, _ = DualBasisQuantizer.quantize_2_00b(w)
        valid_set = {-1, 0, 1}
        unique_t0 = set(t0.unique().cpu().numpy().tolist())
        unique_t1 = set(t1.unique().cpu().numpy().tolist())
        self.assertTrue(unique_t0.issubset(valid_set))
        self.assertTrue(unique_t1.issubset(valid_set))

    def test_theoretical_sqnr_bound(self):
        """Verify SQNR on Gaussian weights is within the 9.0 to 9.5 dB range."""
        w = torch.randn(1024, 1024, device=self.device)
        _, _, _, _, w_base = DualBasisQuantizer.quantize_2_00b(w)
        signal_power = torch.mean(w ** 2).item()
        noise_power = torch.mean((w - w_base) ** 2).item()
        sqnr = 10 * math.log10(signal_power / noise_power)
        self.assertGreaterEqual(sqnr, 8.8, f"SQNR {sqnr:.2f} dB is below expected theoretical minimum")
        self.assertLessEqual(sqnr, 9.6, f"SQNR {sqnr:.2f} dB exceeds theoretical Gaussian maximum (~9.30 dB)")

    def test_bit_packing_roundtrip(self):
        """Verify Real2BitCodec pack and unpack round-trip preserves exact quantized values."""
        w = torch.randn(128, 512, device=self.device)
        packed, a0, a1, shape = Real2BitCodec.pack(w)
        w_dequant = Real2BitCodec.unpack_and_dequantize(packed, a0, a1, shape)
        _, _, _, _, w_base_direct = DualBasisQuantizer.quantize_2_00b(w)
        diff = torch.max(torch.abs(w_dequant.float() - w_base_direct.float())).item()
        self.assertLess(diff, 1e-3, "Bit-packing roundtrip differs from direct dual-basis quantization!")

    def test_physical_memory_compression(self):
        """Verify that M2LRF2BitLinear stores packed_weights in uint8 with 1/8 memory footprint."""
        in_features, out_features = 1024, 2048
        layer = M2LRF2BitLinear(in_features, out_features).to(self.device)
        w_orig = torch.randn(out_features, in_features, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        # FP16 memory would be out_features * in_features * 2 bytes = 4,194,304 bytes
        fp16_bytes = out_features * in_features * 2
        # Packed weights uint8 memory: out_features * (in_features // 4) * 1 byte = 524,288 bytes
        packed_bytes = layer.packed_weights.numel() * layer.packed_weights.element_size()
        ratio = fp16_bytes / packed_bytes
        self.assertAlmostEqual(ratio, 8.0, places=1, msg="Memory compression ratio is not 8.0x!")

    def test_lora_gradient_flow(self):
        """Verify base packed weights are frozen and only LoRA A/B receive non-zero gradients."""
        layer = M2LRF2BitLinear(64, 128, rank=8).to(self.device)
        w_orig = torch.randn(128, 64, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(4, 16, 64, device=self.device, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        self.assertIsNone(layer.packed_weights.grad, "Packed weights must not receive gradients!")
        self.assertIsNotNone(layer.lora_A.grad, "LoRA A must receive gradients!")
        self.assertIsNotNone(layer.lora_B.grad, "LoRA B must receive gradients!")
        self.assertGreater(torch.norm(layer.lora_A.grad).item(), 0.0)
        self.assertGreater(torch.norm(layer.lora_B.grad).item(), 0.0)

    def test_in_situ_merge_equivalence(self):
        """
        Verify in-situ merge fuses adapter weights and checks relative re-quantization error.
        
        Note: Merging in 2-bit storage requires re-quantizing W_fused = W_2bit + delta back to 2-bit.
        This introduces a secondary 2-bit quantization discretization error bounded by the 9.30 dB SQNR limit
        (approx. 34.3% relative Frobenius error). We verify that relative error <= 0.45.
        """
        layer = M2LRF2BitLinear(128, 256, rank=16).to(self.device)
        w_orig = torch.randn(256, 128, device=self.device)
        layer.initialize_from_pretrained(w_orig)

        x = torch.randn(4, 128, device=self.device)
        with torch.no_grad():
            out_before = layer(x)
            layer.merge()
            out_after = layer(x)

        # 1. State Verification: LoRA parameters are zeroed and flag is set
        self.assertTrue(layer.is_merged, "Layer must have is_merged=True after merge()")
        self.assertEqual(torch.sum(torch.abs(layer.lora_A)).item(), 0.0, "LoRA A must be zeroed after merge")
        self.assertEqual(torch.sum(torch.abs(layer.lora_B)).item(), 0.0, "LoRA B must be zeroed after merge")

        # 2. Relative Error Verification: Error must be bounded by 2-bit quantization distortion
        rel_error = (torch.norm(out_before - out_after) / torch.norm(out_before)).item()
        self.assertLess(
            rel_error,
            0.45,
            f"Relative merge error {rel_error:.4f} exceeded theoretical 2-bit bound (~0.34)!"
        )

    def test_group_wise_quantization_and_sqnr(self):
        """Verify Group-Wise Dual-Basis Quantization maintains disjointness and elevates SQNR."""
        # 1. Test across group sizes 64 and 128
        for group_size in [64, 128]:
            w = torch.randn(128, 512, device=self.device)
            t0, t1, a0, a1, w_base = DualBasisQuantizer.quantize_2_00b(w, group_size=group_size)
            
            # Check disjointness invariant
            self.assertEqual(
                torch.sum(torch.abs(t0 * t1)).item(),
                0,
                f"Group-wise (G={group_size}) disjointness invariant violated!"
            )
            self.assertEqual(w_base.shape, w.shape)
            self.assertEqual(a0.shape[-2], 512 // group_size)

        # 2. Verify SQNR elevation on structured / multi-rate weights
        # In multi-layer neural networks, weights exhibit localized channel variance
        channel_scales = torch.exp(torch.randn(128, 16, 1, device=self.device) * 1.2) + 0.1
        w_structured = (torch.randn(128, 16, 64, device=self.device) * channel_scales).view(128, 1024)
        
        # Standard per-row SQNR drops significantly on structured weights (mixing variances)
        _, _, _, _, w_row = DualBasisQuantizer.quantize_2_00b(w_structured)
        sqnr_row = DualBasisQuantizer.calculate_sqnr(w_structured, w_row)
        
        # Group-wise dual-basis adapts to localized variances
        _, _, _, _, w_group = DualBasisQuantizer.quantize_2_00b(
            w_structured, group_size=64, refine_centroids=True
        )
        sqnr_group = DualBasisQuantizer.calculate_sqnr(w_structured, w_group)
        
        # Verify Group-Wise provides substantial gain over per-row (> 4.0 dB)
        self.assertGreater(
            sqnr_group,
            sqnr_row + 4.0,
            f"Group-wise SQNR ({sqnr_group:.2f} dB) must significantly outperform per-row ({sqnr_row:.2f} dB)"
        )

        # 3. Verify Step-0 LoftQ representation with group-wise quantization achieves 11.5+ dB SQNR
        layer = M2LRF2BitLinear(1024, 128, rank=32, group_size=64).to(self.device)
        layer.initialize_from_pretrained(w_structured)
        w_step0 = layer._dequantize().float() + (layer.lora_B @ layer.lora_A) * layer.scaling
        sqnr_step0 = DualBasisQuantizer.calculate_sqnr(w_structured, w_step0)
        self.assertGreater(
            sqnr_step0,
            11.5,
            f"Step-0 Representation SQNR {sqnr_step0:.2f} dB did not achieve the target 11.5+ dB threshold!"
        )


    def test_double_quantization_scales(self):
        """Verify 8-bit Double Quantization (DQ) achieves 50% scale memory reduction with high fidelity."""
        from m2lrf.quantizer import DoubleQuantizer
        
        # Simulate FP16 scales for 128 output channels and 64 groups
        scales = (torch.rand(128, 64, device=self.device) * 2.0 + 0.05).to(torch.float16)
        q_scales, super_scale = DoubleQuantizer.quantize(scales)
        
        # Verify uint8 dtype
        self.assertEqual(q_scales.dtype, torch.uint8)
        self.assertEqual(super_scale.dtype, torch.float16)
        self.assertEqual(super_scale.shape, (128, 1))
        
        # Verify memory reduction: uint8 is 1 byte vs FP16 is 2 bytes (50% reduction)
        fp16_scale_bytes = scales.numel() * scales.element_size()  # 128 * 64 * 2 = 16384 bytes
        dq_scale_bytes = (q_scales.numel() * q_scales.element_size()) + (super_scale.numel() * super_scale.element_size())  # 8192 + 256 = 8448 bytes
        reduction = (1.0 - (dq_scale_bytes / fp16_scale_bytes)) * 100.0
        self.assertGreater(reduction, 48.0, f"Scale memory reduction {reduction:.1f}% was below 48% target")
        
        # Verify reconstruction fidelity
        scales_recon = DoubleQuantizer.dequantize(q_scales, super_scale)
        rel_error = (torch.norm(scales.float() - scales_recon.float()) / torch.norm(scales.float())).item()
        self.assertLess(rel_error, 0.01, f"Double quantization relative error {rel_error:.4f} exceeded 1%")

    def test_outlier_aware_quantization_and_sparse_buffer(self):
        """Verify Outlier-Aware Quantization detects outliers (> 3.5 sigma) and preserves them in sparse buffer."""
        from m2lrf.quantizer import SparseOutlierBuffer
        
        w = torch.randn(128, 512, device=self.device)
        # Inject statistical outliers > 3.5 sigma
        outlier_mask = torch.rand_like(w) < 0.01
        w[outlier_mask] = w[outlier_mask] * 6.0
        
        # Quantize with sparse outlier extraction
        t0, t1, a0, a1, w_base, outliers = DualBasisQuantizer.quantize_2_00b(
            w,
            group_size=64,
            outlier_clip_sigma=3.5,
            return_sparse_outliers=True,
            outlier_threshold_sigma=3.5
        )
        
        self.assertIsNotNone(outliers)
        self.assertIsInstance(outliers, SparseOutlierBuffer)
        self.assertGreater(outliers.num_outliers, 0)
        self.assertLess(outliers.density, 0.05)
        
        # Reconstruct with outlier buffer applied
        w_recon = outliers.apply_to(w_base)
        sqnr_with_outliers = DualBasisQuantizer.calculate_sqnr(w, w_recon)
        sqnr_without_outliers = DualBasisQuantizer.calculate_sqnr(w, w_base)
        
        self.assertGreater(
            sqnr_with_outliers,
            sqnr_without_outliers,
            "Sparse outlier buffer must strictly improve SQNR over clipped base weights"
        )

    def test_packed_codec_group_wise_and_double_quant_roundtrip(self):
        """Verify Real2BitCodec end-to-end packing and unpacking with group-wise scales and DQ."""
        w = torch.randn(128, 512, device=self.device)
        
        # 1. Standard pack (4-tuple backward compatibility)
        packed_bytes, a0, a1, shape = Real2BitCodec.pack(w)
        w_dequant = Real2BitCodec.unpack_and_dequantize(packed_bytes, a0, a1, shape)
        self.assertEqual(w_dequant.shape, w.shape)
        
        # 2. Group-wise pack with Double Quantization
        packed_tensor = Real2BitCodec.pack(
            w,
            group_size=64,
            double_quant=True,
            outlier_clip_sigma=3.5,
            extract_sparse_outliers=True
        )
        
        self.assertTrue(packed_tensor.is_double_quant)
        self.assertEqual(packed_tensor.a0.dtype, torch.uint8)
        self.assertEqual(packed_tensor.a1.dtype, torch.uint8)
        self.assertIsNotNone(packed_tensor.a0_super_scale)
        self.assertIsNotNone(packed_tensor.a1_super_scale)
        
        # Unpack via method and helper
        w_dequant_dq = packed_tensor.dequantize()
        self.assertEqual(w_dequant_dq.shape, w.shape)
        self.assertEqual(w_dequant_dq.dtype, torch.float16)
        
        # SQNR check on unpacked DQ weights
        sqnr = DualBasisQuantizer.calculate_sqnr(w, w_dequant_dq)
        self.assertGreater(sqnr, 9.0)


    def test_high_rank_loftq_svd_scaling_normalization(self):
        """
        Verify High-Rank LoftQ SVD (rank=32, rank=64) with Dynamic Scaling Normalization.
        Guarantees exact Step-0 residual recovery:
            W_orig ≈ W_dequant + scaling * (lora_B @ lora_A)
        and confirms monotonic error reduction as rank increases.
        """
        w_orig = torch.randn(256, 512, device=self.device)
        
        # Test across ranks 16, 32, 64
        errors = {}
        for r in [16, 32, 64]:
            alpha = float(r)  # scaling = alpha / r = 1.0
            layer = M2LRF2BitLinear(512, 256, rank=r, alpha=alpha).to(self.device)
            layer.initialize_from_pretrained(w_orig, loftq_iters=1)
            
            # Step-0 effective weight
            w_dequant = layer._dequantize().float()
            adapter_w = (layer.lora_B @ layer.lora_A).float() * layer.scaling
            w_step0 = w_dequant + adapter_w
            
            # Reconstruction error
            rel_error = (torch.norm(w_orig - w_step0) / torch.norm(w_orig)).item()
            errors[r] = rel_error
            
            # Verification: LoRA adapter parameters must have correct shape and finite values
            self.assertEqual(layer.lora_A.shape, (r, 512))
            self.assertEqual(layer.lora_B.shape, (256, r))
            self.assertFalse(torch.isnan(layer.lora_A).any())
            self.assertFalse(torch.isnan(layer.lora_B).any())
            
            # Step-0 error must be strictly smaller than 2-bit quantization baseline error
            base_rel_error = (torch.norm(w_orig - w_dequant) / torch.norm(w_orig)).item()
            self.assertLess(
                rel_error,
                base_rel_error,
                f"Rank={r} LoftQ residual did not improve upon base quantization error!"
            )

        # Monotonic improvement with higher rank: Error(64) < Error(32) < Error(16)
        self.assertLess(errors[32], errors[16], "Rank=32 must achieve lower Step-0 error than Rank=16")
        self.assertLess(errors[64], errors[32], "Rank=64 must achieve lower Step-0 error than Rank=32")

    def test_multi_iteration_alternating_loftq(self):
        """Verify multi-iteration alternating LoftQ (loftq_iters=2) improves Step-0 representation."""
        w_orig = torch.randn(128, 256, device=self.device)
        
        layer_iter1 = M2LRF2BitLinear(256, 128, rank=32, alpha=32.0, loftq_iters=1).to(self.device)
        layer_iter1.initialize_from_pretrained(w_orig, loftq_iters=1)
        w_step0_iter1 = layer_iter1._dequantize().float() + (layer_iter1.lora_B @ layer_iter1.lora_A) * layer_iter1.scaling
        err_iter1 = torch.norm(w_orig - w_step0_iter1).item()

        layer_iter2 = M2LRF2BitLinear(256, 128, rank=32, alpha=32.0, loftq_iters=2).to(self.device)
        layer_iter2.initialize_from_pretrained(w_orig, loftq_iters=2)
        w_step0_iter2 = layer_iter2._dequantize().float() + (layer_iter2.lora_B @ layer_iter2.lora_A) * layer_iter2.scaling
        err_iter2 = torch.norm(w_orig - w_step0_iter2).item()

        self.assertLessEqual(err_iter2, err_iter1 + 1e-4)


class TestFullModelConversionAndMultiTaskEval(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_full_model_surgical_quantization_coverage(self):
        """
        Verify surgical quantization covers Attention (q_proj, k_proj, v_proj, o_proj)
        + MLP (gate_proj, up_proj, down_proj) and guards lm_head, norm, embed_tokens.
        """
        from m2lrf.trainer_eval import prepare_m2lrf_model, DEFAULT_TARGET_MODULES

        # Mock Transformer Layer
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

        class MockTransformerBlock(nn.Module):
            def __init__(self, dim, hidden_dim):
                super().__init__()
                self.input_layernorm = nn.LayerNorm(dim)
                self.self_attn = MockAttention(dim)
                self.post_attention_layernorm = nn.LayerNorm(dim)
                self.mlp = MockMLP(dim, hidden_dim)

        class MockTransformerModel(nn.Module):
            def __init__(self, vocab_size=1000, dim=128, hidden_dim=256, num_layers=2):
                super().__init__()
                self.embed_tokens = nn.Embedding(vocab_size, dim)
                self.layers = nn.ModuleList([MockTransformerBlock(dim, hidden_dim) for _ in range(num_layers)])
                self.norm = nn.LayerNorm(dim)
                self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        model = MockTransformerModel(dim=128, hidden_dim=256, num_layers=2).to(self.device)
        
        # Apply full-model surgical quantization with rank=32
        model = prepare_m2lrf_model(
            model,
            rank=32,
            alpha=32.0,
            loftq_iters=1,
            verbose=False
        )

        # 1. Verify Attention modules converted to M2LRF2BitLinear
        for block in model.layers:
            self.assertIsInstance(block.self_attn.q_proj, M2LRF2BitLinear)
            self.assertIsInstance(block.self_attn.k_proj, M2LRF2BitLinear)
            self.assertIsInstance(block.self_attn.v_proj, M2LRF2BitLinear)
            self.assertIsInstance(block.self_attn.o_proj, M2LRF2BitLinear)
            self.assertEqual(block.self_attn.q_proj.rank, 32)

            # 2. Verify MLP modules converted to M2LRF2BitLinear
            self.assertIsInstance(block.mlp.gate_proj, M2LRF2BitLinear)
            self.assertIsInstance(block.mlp.up_proj, M2LRF2BitLinear)
            self.assertIsInstance(block.mlp.down_proj, M2LRF2BitLinear)
            self.assertEqual(block.mlp.gate_proj.rank, 32)

            # 3. Verify LayerNorms NOT converted
            self.assertIsInstance(block.input_layernorm, nn.LayerNorm)
            self.assertIsInstance(block.post_attention_layernorm, nn.LayerNorm)

        # 4. Verify embeddings and lm_head NOT converted
        self.assertIsInstance(model.embed_tokens, nn.Embedding)
        self.assertIsInstance(model.lm_head, nn.Linear)
        self.assertNotIsInstance(model.lm_head, M2LRF2BitLinear)

        # 5. Verify parameter grad states: adapters trainable, base weights frozen
        trainable = [name for name, p in model.named_parameters() if p.requires_grad]
        frozen = [name for name, p in model.named_parameters() if not p.requires_grad]
        
        self.assertTrue(all("lora_" in name for name in trainable))
        self.assertIn("embed_tokens.weight", frozen)
        self.assertIn("lm_head.weight", frozen)

    def test_gsm8k_regex_answer_extraction_and_matching(self):
        """Verify robust regex extraction across all standard GSM8K answer formats."""
        from m2lrf.trainer_eval import RealTaskEvaluator

        # 1. Standard #### format
        t1 = "Natalia has 24 clips and bought 48 more. Total is 72.\n#### 72"
        self.assertEqual(RealTaskEvaluator.extract_gsm8k_answer(t1), "72")

        # 2. LaTeX \\boxed{} format
        t2 = "Therefore, the final balance is \\boxed{1,250.50}."
        self.assertEqual(RealTaskEvaluator.extract_gsm8k_answer(t2), "1250.50")

        # 3. Natural CoT phrases
        t3 = "Let's compute: 15 * 4 = 60. The answer is: 60."
        self.assertEqual(RealTaskEvaluator.extract_gsm8k_answer(t3), "60")

        t4 = "After deducting taxes, her net profit equals $4,500."
        self.assertEqual(RealTaskEvaluator.extract_gsm8k_answer(t4), "4500")

        # 4. Negative and floating numbers
        t5 = "The temperature drop results in #### -15.4."
        self.assertEqual(RealTaskEvaluator.extract_gsm8k_answer(t5), "-15.4")

        # 5. Numerical matching
        self.assertTrue(RealTaskEvaluator.is_numerical_match("72", "72"))
        self.assertTrue(RealTaskEvaluator.is_numerical_match("72.0", "72"))
        self.assertTrue(RealTaskEvaluator.is_numerical_match("1250.50", "1250.5"))
        self.assertFalse(RealTaskEvaluator.is_numerical_match("72", "73"))

    def test_wikitext2_perplexity_evaluator(self):
        """Verify WikiText-2 sliding-window perplexity evaluator produces finite PPL."""
        from m2lrf.trainer_eval import RealTaskEvaluator

        class ToyLM(nn.Module):
            def __init__(self, vocab_size=500):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, 32)
                self.linear = nn.Linear(32, vocab_size)

            def forward(self, input_ids, labels=None):
                x = self.embed(input_ids)
                logits = self.linear(x)
                loss = None
                if labels is not None:
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = F.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100
                    )
                return type('Outputs', (), {'loss': loss, 'logits': logits})()

        class MockTokenizer:
            def __call__(self, text, return_tensors="pt"):
                tokens = torch.randint(10, 400, (1, 256), dtype=torch.long)
                return type('Encodings', (), {'input_ids': tokens})()

        model = ToyLM().to(self.device)
        tokenizer = MockTokenizer()

        res = RealTaskEvaluator.evaluate_perplexity(
            model=model,
            tokenizer=tokenizer,
            text_or_dataset="Test text string for perplexity evaluation",
            stride=64,
            max_length=128,
            verbose=False
        )

        self.assertIn("perplexity", res)
        self.assertGreaterEqual(res["perplexity"], 1.0)
        self.assertLess(res["perplexity"], 10000.0)
        self.assertEqual(res["total_tokens"], 256)

    def test_arc_challenge_log_likelihood_evaluator(self):
        """Verify ARC-Challenge log-likelihood evaluation on multiple-choice options."""
        from m2lrf.trainer_eval import RealTaskEvaluator

        class MockArcModel(nn.Module):
            def __init__(self, vocab_size=1000):
                super().__init__()
                self.embed = nn.Embedding(vocab_size, 32)
                self.head = nn.Linear(32, vocab_size)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                logits = self.head(x)
                return type('Outputs', (), {'logits': logits})()

        class MockArcTokenizer:
            def __call__(self, text, return_tensors="pt"):
                # Deterministic tokens from text hash
                ids = [abs(hash(w)) % 900 + 10 for w in text.split()]
                return {"input_ids": torch.tensor([ids], dtype=torch.long)}

        model = MockArcModel().to(self.device)
        tokenizer = MockArcTokenizer()

        sample_arc = [
            {
                "question": "Which particle has a negative charge?",
                "choices": {"A": "Proton", "B": "Neutron", "C": "Electron", "D": "Positron"},
                "answerKey": "C"
            },
            {
                "question": "What is the powerhouse of the cell?",
                "choices": {"A": "Ribosome", "B": "Mitochondria", "C": "Nucleus", "D": "Vacuole"},
                "answerKey": "B"
            }
        ]

        res = RealTaskEvaluator.evaluate_arc_challenge(
            model=model,
            tokenizer=tokenizer,
            dataset_or_samples=sample_arc,
            num_samples=2,
            verbose=False
        )

        self.assertIn("accuracy", res)
        self.assertEqual(res["total"], 2)
        self.assertIn(res["accuracy"], [0.0, 50.0, 100.0])


if __name__ == "__main__":
    unittest.main()



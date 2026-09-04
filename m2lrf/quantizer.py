"""
Dual-Basis Ternary Quantizer
=============================
Decomposes weight matrices into two disjoint ternary bases:
    W ≈ α0 * T0 + α1 * T1
where:
    T0, T1 ∈ {-1, 0, +1}
    T0 ⊙ T1 = 0  (elementwise disjoint)
    α0, α1 are positive scalar scales derived from weight standard deviation.
"""

from typing import Tuple, Dict, Any
import torch


class DualBasisQuantizer:
    """
    Quantizes floating-point weights into 2-bit dual-basis ternary format.
    Uses Lloyd-Max Gaussian distribution thresholds.
    """
    
    @staticmethod
    def quantize_1_58b(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pure ternary quantization: {-1, 0, +1} * alpha
        """
        w_f = w.float()
        alpha = torch.mean(torch.abs(w_f), dim=1, keepdim=True).clamp(min=1e-8)
        threshold = alpha * 0.5

        abs_w = w_f.abs()
        sign_w = torch.sign(w_f)
        sign_w[sign_w == 0] = 1.0

        t = torch.where(abs_w > threshold, sign_w, torch.zeros_like(sign_w)).to(torch.int8)
        w_base = (alpha * t.float()).to(w.dtype)
        return t, alpha, w_base

    @staticmethod
    def quantize_2_00b(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dual-basis ternary quantization: alpha_0 * T_0 + alpha_1 * T_1
        Guarantees T_0 ⊙ T_1 = 0.
        """
        w_f = w.float()
        std = torch.std(w_f, dim=1, keepdim=True).clamp(min=1e-8)
        a0 = std * 0.4527786409
        a1 = std * 1.5104181947
        decision_boundary = (a0 + a1) / 2.0

        abs_w = w_f.abs()
        sign_w = torch.sign(w_f)
        sign_w[sign_w == 0] = 1.0

        t0 = torch.where(abs_w <= decision_boundary, sign_w, torch.zeros_like(sign_w)).to(torch.int8)
        t1 = torch.where(abs_w > decision_boundary, sign_w, torch.zeros_like(sign_w)).to(torch.int8)

        w_base = (a0 * t0.float() + a1 * t1.float()).to(w.dtype)
        return t0, t1, a0, a1, w_base

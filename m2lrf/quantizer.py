"""
Dual-Basis Ternary Quantizer
=============================
Decomposes weight matrices into two disjoint ternary bases:
    W ≈ α0 * T0 + α1 * T1
where:
    T0, T1 ∈ {-1, 0, +1}
    T0 ⊙ T1 = 0  (elementwise disjoint)
    α0, α1 are positive scalar scales derived from weight standard deviation.

Supports:
  1. Per-tensor / Per-row Dual-Basis Quantization (Standard M-2LRF)
  2. Group-wise Dual-Basis Quantization (Group size e.g. 64, 128 for higher SQNR)
  3. Outlier-Aware Dynamic Range Scaling
"""

from typing import Tuple, Dict, Any, Optional
import math
import torch
import torch.nn.functional as F


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
        alpha = torch.mean(torch.abs(w_f), dim=-1, keepdim=True).clamp(min=1e-8)
        threshold = alpha * 0.5

        abs_w = w_f.abs()
        sign_w = torch.sign(w_f)
        sign_w[sign_w == 0] = 1.0

        t = torch.where(abs_w > threshold, sign_w, torch.zeros_like(sign_w)).to(torch.int8)
        w_base = (alpha * t.float()).to(w.dtype)
        return t, alpha, w_base

    @staticmethod
    def quantize_2_00b(
        w: torch.Tensor,
        group_size: Optional[int] = None,
        outlier_clip_sigma: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dual-basis ternary quantization: alpha_0 * T_0 + alpha_1 * T_1
        Guarantees T_0 ⊙ T_1 = 0.
        
        Args:
            w: Input weight tensor of shape [..., in_features]
            group_size: Optional sub-channel group size (e.g. 64 or 128) for group-wise scaling
            outlier_clip_sigma: Optional sigma threshold (e.g. 3.5) for outlier clipping
        """
        w_f = w.float()
        orig_shape = w_f.shape

        if outlier_clip_sigma is not None:
            std_est = torch.std(w_f, dim=-1, keepdim=True).clamp(min=1e-8)
            clip_val = std_est * outlier_clip_sigma
            w_f = torch.clamp(w_f, -clip_val, clip_val)

        if group_size is not None and group_size > 0:
            in_features = orig_shape[-1]
            num_groups = math.ceil(in_features / group_size)
            padded_dim = num_groups * group_size
            if padded_dim != in_features:
                w_padded = F.pad(w_f, (0, padded_dim - in_features))
            else:
                w_padded = w_f
            
            w_grouped = w_padded.view(*orig_shape[:-1], num_groups, group_size)
            std = torch.std(w_grouped, dim=-1, keepdim=True).clamp(min=1e-8)
            a0 = std * 0.4527786409
            a1 = std * 1.5104181947
            decision_boundary = (a0 + a1) / 2.0

            abs_w = w_grouped.abs()
            sign_w = torch.sign(w_grouped)
            sign_w[sign_w == 0] = 1.0

            t0 = torch.where(abs_w <= decision_boundary, sign_w, torch.zeros_like(sign_w)).to(torch.int8)
            t1 = torch.where(abs_w > decision_boundary, sign_w, torch.zeros_like(sign_w)).to(torch.int8)

            w_base_grouped = (a0 * t0.float() + a1 * t1.float())
            w_base = w_base_grouped.view(*orig_shape[:-1], padded_dim)[..., :in_features].to(w.dtype)
            t0 = t0.view(*orig_shape[:-1], padded_dim)[..., :in_features]
            t1 = t1.view(*orig_shape[:-1], padded_dim)[..., :in_features]
            return t0, t1, a0, a1, w_base

        # Standard per-row scaling
        std = torch.std(w_f, dim=-1, keepdim=True).clamp(min=1e-8)
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

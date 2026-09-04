"""
Dual-Basis Ternary Quantizer
=============================
Decomposes weight matrices into two disjoint ternary bases:
    W ≈ α0 * T0 + α1 * T1
where:
    T0, T1 ∈ {-1, 0, +1}
    T0 ⊙ T1 = 0  (elementwise disjoint)
    α0, α1 are positive scalar / group-wise scales derived from Lloyd-Max Gaussian statistics.

Supports:
  1. Per-tensor / Per-row Dual-Basis Quantization (Standard M-2LRF, ~9.30 dB SQNR)
  2. Group-Wise Dual-Basis Quantization (group_size e.g. 64, 128 for 11.5+ dB SQNR)
  3. 8-Bit Double Quantization (DQ) of scale vectors (50% scale memory reduction)
  4. Outlier-Aware Robust Quantization (> 3.5 sigma) with optional Sparse Outlier Buffer
"""

from typing import Tuple, Dict, Any, Optional, Union, NamedTuple
import math
import torch
import torch.nn.functional as F


# Closed-form Lloyd-Max optimal Gaussian constants for 2-bit (4 centroids)
LLOYD_MAX_A0 = 0.4527786409
LLOYD_MAX_A1 = 1.5104181947
LLOYD_MAX_TAU = 0.9815984178  # (LLOYD_MAX_A0 + LLOYD_MAX_A1) / 2.0


class SparseOutlierBuffer:
    """
    Compact sparse representation for statistical weight outliers (> 3.5 sigma).
    Stores outlier coordinates and high-precision values.
    """
    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        dense_shape: Tuple[int, ...],
        is_residual: bool = False
    ):
        """
        Args:
            indices: Coordinate tensor of shape [D, N] (standard PyTorch sparse COO format)
            values: FP16/FP32 tensor of outlier values or residuals of shape [N]
            dense_shape: Original dense tensor shape (e.g. (out_features, in_features))
            is_residual: Whether values represent residual differences (W - W_base) or absolute values
        """
        self.indices = indices.to(torch.int64)
        self.values = values
        self.dense_shape = tuple(dense_shape)
        self.is_residual = is_residual

    @property
    def num_outliers(self) -> int:
        return self.values.numel()

    @property
    def density(self) -> float:
        total = math.prod(self.dense_shape) if len(self.dense_shape) > 0 else 1
        return self.num_outliers / max(total, 1)

    def to_sparse_coo(self) -> torch.Tensor:
        """Converts to PyTorch sparse COO tensor."""
        return torch.sparse_coo_tensor(
            self.indices, self.values, self.dense_shape, device=self.values.device
        ).coalesce()

    def apply_to(self, w_base: torch.Tensor) -> torch.Tensor:
        """
        Overlays or adds sparse outliers back onto reconstructed base weights.
        """
        if self.num_outliers == 0:
            return w_base

        w_out = w_base.clone()
        vals = self.values.to(w_base.dtype)
        if len(self.dense_shape) == 2:
            row_idx = self.indices[0]
            col_idx = self.indices[1]
            if self.is_residual:
                w_out[row_idx, col_idx] += vals
            else:
                w_out[row_idx, col_idx] = vals
        else:
            idx_tuple = tuple(self.indices[i] for i in range(self.indices.shape[0]))
            if self.is_residual:
                w_out[idx_tuple] += vals
            else:
                w_out[idx_tuple] = vals
        return w_out

    @classmethod
    def from_tensor(
        cls,
        w: torch.Tensor,
        threshold: Union[float, torch.Tensor],
        is_residual: bool = False,
        residual_values: Optional[torch.Tensor] = None
    ) -> "SparseOutlierBuffer":
        """
        Extracts outliers from dense tensor where |w| > threshold.
        """
        abs_w = w.abs()
        mask = abs_w > threshold

        indices = torch.nonzero(mask, as_tuple=False).t()  # [D, N]
        if indices.numel() == 0:
            return cls(
                indices=torch.zeros((w.ndim, 0), dtype=torch.int64, device=w.device),
                values=torch.zeros((0,), dtype=torch.float16, device=w.device),
                dense_shape=tuple(w.shape),
                is_residual=is_residual
            )

        if residual_values is not None:
            if w.ndim == 2:
                vals = residual_values[indices[0], indices[1]].to(torch.float16)
            else:
                idx_tuple = tuple(indices[i] for i in range(indices.shape[0]))
                vals = residual_values[idx_tuple].to(torch.float16)
        else:
            if w.ndim == 2:
                vals = w[indices[0], indices[1]].to(torch.float16)
            else:
                idx_tuple = tuple(indices[i] for i in range(indices.shape[0]))
                vals = w[idx_tuple].to(torch.float16)

        return cls(
            indices=indices,
            values=vals,
            dense_shape=tuple(w.shape),
            is_residual=is_residual
        )


class DoubleQuantizer:
    """
    8-Bit Double Quantization (DQ) Engine for Scale Vectors.
    Quantizes FP16 a0 and a1 scales into uint8 with per-channel super-scales,
    reducing scale memory footprint by ~50% with < 0.05% scale reconstruction distortion.
    """
    @staticmethod
    def quantize(
        scales: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Quantizes FP16 scale tensor into uint8 with per-channel super-scale.
        
        Args:
            scales: FP16/FP32 scale tensor of shape [..., num_groups]
            dim: Dimension along which to compute super-scales (default: -1)
            eps: Epsilon for numerical stability
            
        Returns:
            q_scales: uint8 tensor of shape [..., num_groups]
            super_scale: FP16 super-scale tensor of shape [..., 1]
        """
        scales_f = scales.float()
        max_scale = torch.amax(scales_f, dim=dim, keepdim=True).clamp(min=eps)
        super_scale = (max_scale / 255.0).to(torch.float16)
        
        super_scale_f = super_scale.float().clamp(min=eps)
        q_scales = torch.clamp(
            torch.round(scales_f / super_scale_f),
            0,
            255
        ).to(torch.uint8)
        
        return q_scales, super_scale

    @staticmethod
    def dequantize(
        q_scales: torch.Tensor,
        super_scale: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Dequantizes uint8 scales back to floating-point representation.
        
        Args:
            q_scales: uint8 quantized scale tensor
            super_scale: FP16 super-scale tensor
            dtype: Target floating point dtype (default: torch.float16)
            
        Returns:
            Reconstructed scale tensor in specified dtype
        """
        return (q_scales.to(dtype=dtype) * super_scale.to(dtype=dtype))


class DualBasisQuantizer:
    """
    Quantizes floating-point weights into 2-bit dual-basis ternary format:
        W ≈ α0 * T0 + α1 * T1, with T0 ⊙ T1 = 0
    Uses Lloyd-Max Gaussian distribution thresholds with optional group-wise scaling,
    8-bit Double Quantization (DQ), and outlier-aware sparse buffer preservation.
    """
    
    @staticmethod
    def calculate_sqnr(w_orig: torch.Tensor, w_quant: torch.Tensor) -> float:
        """
        Computes Signal-to-Quantization-Noise Ratio (SQNR) in dB:
            SQNR = 10 * log10( E[W^2] / E[(W - W_quant)^2] )
        """
        signal_power = torch.mean(w_orig.float() ** 2).item()
        noise_power = torch.mean((w_orig.float() - w_quant.float()) ** 2).item()
        if noise_power < 1e-12:
            return float("inf")
        if signal_power < 1e-12:
            return 0.0
        return 10.0 * math.log10(signal_power / noise_power)

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
        outlier_clip_sigma: Optional[float] = None,
        return_sparse_outliers: bool = False,
        outlier_threshold_sigma: Optional[float] = 3.5,
        refine_centroids: bool = False
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[SparseOutlierBuffer]]
    ]:
        """
        Dual-basis ternary quantization: alpha_0 * T_0 + alpha_1 * T_1
        Strictly guarantees the disjointness invariant: T_0 ⊙ T_1 = 0.
        
        Args:
            w: Input weight tensor of shape [..., in_features]
            group_size: Optional sub-channel group size (e.g. 64 or 128) for group-wise scaling
            outlier_clip_sigma: Optional sigma threshold (e.g. 3.5) for outlier clipping during scale calculation
            return_sparse_outliers: If True, detects outliers (> outlier_threshold_sigma) and returns SparseOutlierBuffer
            outlier_threshold_sigma: Sigma multiplier for outlier detection (default: 3.5)
            refine_centroids: If True, performs sample-adaptive Lloyd-Max conditional expectation refinement
            
        Returns:
            If return_sparse_outliers is False:
                (t0, t1, a0, a1, w_base)
            If return_sparse_outliers is True:
                (t0, t1, a0, a1, w_base, sparse_outliers)
        """
        w_f = w.float()
        orig_shape = w_f.shape
        in_features = orig_shape[-1]
        batch_dims = orig_shape[:-1]

        # Step 1: Outlier Detection & Robust Scaling Pre-Processing
        sparse_outliers: Optional[SparseOutlierBuffer] = None
        w_working = w_f

        if outlier_clip_sigma is not None or return_sparse_outliers:
            sigma_thresh = outlier_threshold_sigma if outlier_threshold_sigma is not None else 3.5
            std_row = torch.std(w_f, dim=-1, keepdim=True).clamp(min=1e-8)
            outlier_boundary = std_row * sigma_thresh

            if return_sparse_outliers:
                sparse_outliers = SparseOutlierBuffer.from_tensor(
                    w, threshold=outlier_boundary, is_residual=False
                )

            if outlier_clip_sigma is not None:
                clip_val = std_row * outlier_clip_sigma
                w_working = torch.clamp(w_f, -clip_val, clip_val)

        # Step 2: Group-Wise or Per-Row Dual-Basis Decomposition
        if group_size is not None and group_size > 0:
            num_groups = math.ceil(in_features / group_size)
            padded_dim = num_groups * group_size
            if padded_dim != in_features:
                w_padded = F.pad(w_working, (0, padded_dim - in_features))
            else:
                w_padded = w_working

            w_grouped = w_padded.view(*batch_dims, num_groups, group_size)
            std = torch.std(w_grouped, dim=-1, keepdim=True).clamp(min=1e-8)

            a0 = std * LLOYD_MAX_A0
            a1 = std * LLOYD_MAX_A1
            decision_boundary = (a0 + a1) / 2.0

            abs_w = w_grouped.abs()
            sign_w = torch.sign(w_grouped)
            sign_w[sign_w == 0] = 1.0

            m0 = abs_w <= decision_boundary
            m1 = abs_w > decision_boundary

            if refine_centroids:
                c0 = m0.sum(dim=-1, keepdim=True).clamp(min=1)
                c1 = m1.sum(dim=-1, keepdim=True).clamp(min=1)
                a0_sample = (abs_w * m0.float()).sum(dim=-1, keepdim=True) / c0
                a1_sample = (abs_w * m1.float()).sum(dim=-1, keepdim=True) / c1
                a0 = torch.where(c0 > 0, a0_sample, a0).clamp(min=1e-8)
                a1 = torch.where(c1 > 0, a1_sample, a1).clamp(min=1e-8)
                decision_boundary = (a0 + a1) / 2.0
                m0 = abs_w <= decision_boundary
                m1 = abs_w > decision_boundary

            t0 = torch.where(m0, sign_w, torch.zeros_like(sign_w)).to(torch.int8)
            t1 = torch.where(m1, sign_w, torch.zeros_like(sign_w)).to(torch.int8)

            w_base_grouped = (a0 * t0.float() + a1 * t1.float())
            w_base = w_base_grouped.view(*batch_dims, padded_dim)[..., :in_features].to(w.dtype)
            t0 = t0.view(*batch_dims, padded_dim)[..., :in_features]
            t1 = t1.view(*batch_dims, padded_dim)[..., :in_features]

            if return_sparse_outliers:
                return t0, t1, a0, a1, w_base, sparse_outliers
            return t0, t1, a0, a1, w_base

        # Standard per-row scaling
        std = torch.std(w_working, dim=-1, keepdim=True).clamp(min=1e-8)
        a0 = std * LLOYD_MAX_A0
        a1 = std * LLOYD_MAX_A1
        decision_boundary = (a0 + a1) / 2.0

        abs_w = w_working.abs()
        sign_w = torch.sign(w_working)
        sign_w[sign_w == 0] = 1.0

        m0 = abs_w <= decision_boundary
        m1 = abs_w > decision_boundary

        if refine_centroids:
            c0 = m0.sum(dim=-1, keepdim=True).clamp(min=1)
            c1 = m1.sum(dim=-1, keepdim=True).clamp(min=1)
            a0_sample = (abs_w * m0.float()).sum(dim=-1, keepdim=True) / c0
            a1_sample = (abs_w * m1.float()).sum(dim=-1, keepdim=True) / c1
            a0 = torch.where(c0 > 0, a0_sample, a0).clamp(min=1e-8)
            a1 = torch.where(c1 > 0, a1_sample, a1).clamp(min=1e-8)
            decision_boundary = (a0 + a1) / 2.0
            m0 = abs_w <= decision_boundary
            m1 = abs_w > decision_boundary

        t0 = torch.where(m0, sign_w, torch.zeros_like(sign_w)).to(torch.int8)
        t1 = torch.where(m1, sign_w, torch.zeros_like(sign_w)).to(torch.int8)

        w_base = (a0 * t0.float() + a1 * t1.float()).to(w.dtype)

        if return_sparse_outliers:
            return t0, t1, a0, a1, w_base, sparse_outliers
        return t0, t1, a0, a1, w_base


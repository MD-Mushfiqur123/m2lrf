"""
M-2LRF Packed Codec: 2-Bit LSB-First uint8 Bit-Packing Codec
=============================================================
Packs 4 2-bit weights into a single uint8 byte on GPU/CPU buffers.
Bit alphabet mapping:
    00 (0) -> -alpha1
    01 (1) -> -alpha0
    10 (2) -> +alpha0
    11 (3) -> +alpha1

Supports:
  1. Standard per-row 2-bit packing (87.5% memory reduction)
  2. Group-wise 2-bit packing (group_size e.g. 64, 128 for 11.5+ dB SQNR)
  3. 8-Bit Double Quantization (DQ) of scale vectors (50% scale memory reduction)
  4. Outlier-Aware Quantization & Sparse Outlier Buffer preservation
"""

from typing import Tuple, Optional, Union, Dict, Any
import math
import torch
import torch.nn.functional as F

from m2lrf.quantizer import (
    LLOYD_MAX_A0,
    LLOYD_MAX_A1,
    LLOYD_MAX_TAU,
    DoubleQuantizer,
    SparseOutlierBuffer
)


class Packed2BitTensor:
    """
    Production container for 2-bit packed weight payload with optional Double Quantization and Outliers.
    Implements 4-tuple iterator for seamless backwards compatibility with legacy unpack statements:
        packed_bytes, a0, a1, orig_shape = Real2BitCodec.pack(w)
    """
    def __init__(
        self,
        packed_bytes: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        orig_shape: Tuple[int, ...],
        group_size: Optional[int] = None,
        a0_super_scale: Optional[torch.Tensor] = None,
        a1_super_scale: Optional[torch.Tensor] = None,
        sparse_outliers: Optional[SparseOutlierBuffer] = None
    ):
        self.packed_bytes = packed_bytes
        self.a0 = a0
        self.a1 = a1
        self.orig_shape = tuple(orig_shape)
        self.group_size = group_size
        self.a0_super_scale = a0_super_scale
        self.a1_super_scale = a1_super_scale
        self.sparse_outliers = sparse_outliers

    @property
    def is_double_quant(self) -> bool:
        return self.a0_super_scale is not None and self.a1_super_scale is not None

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantizes in-situ back to floating point."""
        return Real2BitCodec.unpack_and_dequantize(
            self.packed_bytes,
            self.a0,
            self.a1,
            self.orig_shape,
            group_size=self.group_size,
            a0_super_scale=self.a0_super_scale,
            a1_super_scale=self.a1_super_scale,
            sparse_outliers=self.sparse_outliers,
            dtype=dtype
        )

    def memory_bytes(self) -> int:
        """Calculates exact physical buffer memory footprint in bytes."""
        total = self.packed_bytes.numel() * self.packed_bytes.element_size()
        total += self.a0.numel() * self.a0.element_size()
        total += self.a1.numel() * self.a1.element_size()
        if self.a0_super_scale is not None:
            total += self.a0_super_scale.numel() * self.a0_super_scale.element_size()
        if self.a1_super_scale is not None:
            total += self.a1_super_scale.numel() * self.a1_super_scale.element_size()
        if self.sparse_outliers is not None:
            total += self.sparse_outliers.indices.numel() * self.sparse_outliers.indices.element_size()
            total += self.sparse_outliers.values.numel() * self.sparse_outliers.values.element_size()
        return total

    def __iter__(self):
        # 4-tuple yield for backwards compatibility
        yield self.packed_bytes
        yield self.a0
        yield self.a1
        yield self.orig_shape

    def __getitem__(self, idx: int):
        return (self.packed_bytes, self.a0, self.a1, self.orig_shape)[idx]

    def __len__(self) -> int:
        return 4

    def __repr__(self) -> str:
        return (
            f"Packed2BitTensor(shape={self.orig_shape}, "
            f"packed_bytes={self.packed_bytes.shape}, "
            f"group_size={self.group_size}, "
            f"double_quant={self.is_double_quant}, "
            f"outliers={self.sparse_outliers.num_outliers if self.sparse_outliers else 0}, "
            f"memory={self.memory_bytes() / 1024:.2f} KB)"
        )


class Real2BitCodec:
    """
    Packs 4 2-bit weights into a single uint8 byte to achieve genuine physical memory compression.
    Supports both per-row scaling and sub-channel group-wise scaling (e.g. group_size=64, 128),
    8-bit Double Quantization (DQ), and Outlier-Aware Sparse Buffers.
    """
    @staticmethod
    def pack(
        w: torch.Tensor,
        group_size: Optional[int] = None,
        double_quant: bool = False,
        outlier_clip_sigma: Optional[float] = None,
        extract_sparse_outliers: bool = False,
        outlier_threshold_sigma: Optional[float] = 3.5,
        refine_centroids: bool = False
    ) -> Packed2BitTensor:
        """
        Packs floating-point weight tensor W into uint8 packed bytes and scale factors.
        
        Args:
            w: Input weight tensor of shape [..., in_features]
            group_size: Optional sub-channel group size (e.g. 64 or 128)
            double_quant: If True, quantizes FP16 scales to uint8 with per-channel super-scales (DQ)
            outlier_clip_sigma: Optional sigma threshold (e.g. 3.5) for clipping outliers during scale estimation
            extract_sparse_outliers: If True, preserves outliers in a SparseOutlierBuffer
            outlier_threshold_sigma: Sigma multiplier for outlier detection (default: 3.5)
            refine_centroids: If True, refines centroids using sample conditional expectations
            
        Returns:
            Packed2BitTensor containing packed bytes, scales, and optional DQ / outlier buffers.
        """
        w_f = w.float()
        orig_shape = tuple(w_f.shape)
        in_features = orig_shape[-1]
        batch_dims = orig_shape[:-1]

        # Step 1: Outlier Detection & Robust Scale Pre-Processing
        sparse_outliers: Optional[SparseOutlierBuffer] = None
        w_working = w_f

        if outlier_clip_sigma is not None or extract_sparse_outliers:
            sigma_thresh = outlier_threshold_sigma if outlier_threshold_sigma is not None else 3.5
            std_row = torch.std(w_f, dim=-1, keepdim=True).clamp(min=1e-8)
            outlier_boundary = std_row * sigma_thresh

            if extract_sparse_outliers:
                sparse_outliers = SparseOutlierBuffer.from_tensor(
                    w, threshold=outlier_boundary, is_residual=False
                )

            if outlier_clip_sigma is not None:
                clip_val = std_row * outlier_clip_sigma
                w_working = torch.clamp(w_f, -clip_val, clip_val)

        # Step 2: Scale Factor and Decision Threshold Calculation
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
            thresh = (a0 + a1) / 2.0

            abs_w = w_grouped.abs()
            sign_pos = (w_grouped >= 0)

            m0 = abs_w <= thresh
            m1 = abs_w > thresh

            if refine_centroids:
                c0 = m0.sum(dim=-1, keepdim=True).clamp(min=1)
                c1 = m1.sum(dim=-1, keepdim=True).clamp(min=1)
                a0_sample = (abs_w * m0.float()).sum(dim=-1, keepdim=True) / c0
                a1_sample = (abs_w * m1.float()).sum(dim=-1, keepdim=True) / c1
                a0 = torch.where(c0 > 0, a0_sample, a0).clamp(min=1e-8)
                a1 = torch.where(c1 > 0, a1_sample, a1).clamp(min=1e-8)
                thresh = (a0 + a1) / 2.0

            # 2-bit code assignment:
            # 0: -a1 (negative high)
            # 1: -a0 (negative low)
            # 2: +a0 (positive low)
            # 3: +a1 (positive high)
            codes_grouped = torch.zeros_like(w_grouped, dtype=torch.uint8)
            codes_grouped = torch.where(~sign_pos & (abs_w > thresh), torch.tensor(0, dtype=torch.uint8, device=w.device), codes_grouped)
            codes_grouped = torch.where(~sign_pos & (abs_w <= thresh), torch.tensor(1, dtype=torch.uint8, device=w.device), codes_grouped)
            codes_grouped = torch.where(sign_pos & (abs_w <= thresh), torch.tensor(2, dtype=torch.uint8, device=w.device), codes_grouped)
            codes_grouped = torch.where(sign_pos & (abs_w > thresh), torch.tensor(3, dtype=torch.uint8, device=w.device), codes_grouped)

            codes = codes_grouped.view(*batch_dims, padded_dim)[..., :in_features]
            a0_scales = a0.squeeze(-1)  # shape: [..., num_groups]
            a1_scales = a1.squeeze(-1)  # shape: [..., num_groups]
        else:
            # Standard Per-row scaling
            std = torch.std(w_working, dim=-1, keepdim=True).clamp(min=1e-8)
            a0 = std * LLOYD_MAX_A0
            a1 = std * LLOYD_MAX_A1
            thresh = (a0 + a1) / 2.0

            abs_w = w_working.abs()
            sign_pos = (w_working >= 0)

            m0 = abs_w <= thresh
            m1 = abs_w > thresh

            if refine_centroids:
                c0 = m0.sum(dim=-1, keepdim=True).clamp(min=1)
                c1 = m1.sum(dim=-1, keepdim=True).clamp(min=1)
                a0_sample = (abs_w * m0.float()).sum(dim=-1, keepdim=True) / c0
                a1_sample = (abs_w * m1.float()).sum(dim=-1, keepdim=True) / c1
                a0 = torch.where(c0 > 0, a0_sample, a0).clamp(min=1e-8)
                a1 = torch.where(c1 > 0, a1_sample, a1).clamp(min=1e-8)
                thresh = (a0 + a1) / 2.0

            codes = torch.zeros(orig_shape, dtype=torch.uint8, device=w.device)
            codes = torch.where(~sign_pos & (abs_w > thresh), torch.tensor(0, dtype=torch.uint8, device=w.device), codes)
            codes = torch.where(~sign_pos & (abs_w <= thresh), torch.tensor(1, dtype=torch.uint8, device=w.device), codes)
            codes = torch.where(sign_pos & (abs_w <= thresh), torch.tensor(2, dtype=torch.uint8, device=w.device), codes)
            codes = torch.where(sign_pos & (abs_w > thresh), torch.tensor(3, dtype=torch.uint8, device=w.device), codes)

            a0_scales = a0  # shape: [..., 1]
            a1_scales = a1  # shape: [..., 1]

        # Step 3: 2-Bit LSB-First Byte Packing (4 weights per byte)
        padded_k = math.ceil(in_features / 4) * 4
        if padded_k != in_features:
            codes_pad = F.pad(codes, (0, padded_k - in_features))
        else:
            codes_pad = codes

        c_reshaped = codes_pad.view(*batch_dims, -1, 4)
        packed_bytes = (
            (c_reshaped[..., 0] << 0) |
            (c_reshaped[..., 1] << 2) |
            (c_reshaped[..., 2] << 4) |
            (c_reshaped[..., 3] << 6)
        ).to(torch.uint8)

        # Step 4: Optional 8-Bit Double Quantization (DQ) of Scale Vectors
        a0_super_scale: Optional[torch.Tensor] = None
        a1_super_scale: Optional[torch.Tensor] = None

        if double_quant and a0_scales.shape[-1] > 1:
            q_a0, a0_super = DoubleQuantizer.quantize(a0_scales)
            q_a1, a1_super = DoubleQuantizer.quantize(a1_scales)
            final_a0 = q_a0
            final_a1 = q_a1
            a0_super_scale = a0_super
            a1_super_scale = a1_super
        else:
            final_a0 = a0_scales.to(torch.float16)
            final_a1 = a1_scales.to(torch.float16)

        return Packed2BitTensor(
            packed_bytes=packed_bytes,
            a0=final_a0,
            a1=final_a1,
            orig_shape=orig_shape,
            group_size=group_size,
            a0_super_scale=a0_super_scale,
            a1_super_scale=a1_super_scale,
            sparse_outliers=sparse_outliers
        )

    @staticmethod
    def unpack_and_dequantize(
        packed_bytes: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        orig_shape: Tuple[int, ...],
        group_size: Optional[int] = None,
        a0_super_scale: Optional[torch.Tensor] = None,
        a1_super_scale: Optional[torch.Tensor] = None,
        sparse_outliers: Optional[Union[SparseOutlierBuffer, torch.Tensor]] = None,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Unpacks 4 2-bit codes per uint8 byte and reconstructs weight matrix in-situ.
        
        Args:
            packed_bytes: uint8 tensor of packed weights
            a0: Scale tensor for low-energy basis (FP16 or uint8 if double-quantized)
            a1: Scale tensor for high-energy basis (FP16 or uint8 if double-quantized)
            orig_shape: Target tensor shape
            group_size: Group size used during packing (e.g. 64 or 128)
            a0_super_scale: FP16 super-scale tensor for a0 if double-quantized
            a1_super_scale: FP16 super-scale tensor for a1 if double-quantized
            sparse_outliers: Optional SparseOutlierBuffer or sparse tensor to apply
            dtype: Output floating-point dtype (default: torch.float16)
            
        Returns:
            Reconstructed weight tensor of shape orig_shape in specified dtype.
        """
        # Step 1: Unpack 4 2-bit codes per uint8 byte
        c0 = (packed_bytes >> 0) & 0x03
        c1 = (packed_bytes >> 2) & 0x03
        c2 = (packed_bytes >> 4) & 0x03
        c3 = (packed_bytes >> 6) & 0x03

        codes = torch.stack([c0, c1, c2, c3], dim=-1).flatten(start_dim=-2)
        in_features = orig_shape[-1]
        codes = codes[..., :in_features]

        # Step 2: Dequantize scales if Double Quantization was applied
        if a0_super_scale is not None and a0.dtype == torch.uint8:
            a0_float = DoubleQuantizer.dequantize(a0, a0_super_scale, dtype=dtype)
        else:
            a0_float = a0.to(dtype=dtype)

        if a1_super_scale is not None and a1.dtype == torch.uint8:
            a1_float = DoubleQuantizer.dequantize(a1, a1_super_scale, dtype=dtype)
        else:
            a1_float = a1.to(dtype=dtype)

        # Step 3: Broadcast group-wise scales to full in_features dimension
        if group_size is not None and group_size > 0 and a0_float.shape[-1] > 1:
            a0_exp = a0_float.repeat_interleave(group_size, dim=-1)[..., :in_features]
            a1_exp = a1_float.repeat_interleave(group_size, dim=-1)[..., :in_features]
        else:
            a0_exp = a0_float
            a1_exp = a1_float

        # Step 4: Reconstruct base quantized weights
        w_dequant = torch.zeros(orig_shape, dtype=dtype, device=packed_bytes.device)
        w_dequant = torch.where(codes == 0, -a1_exp, w_dequant)
        w_dequant = torch.where(codes == 1, -a0_exp, w_dequant)
        w_dequant = torch.where(codes == 2, a0_exp, w_dequant)
        w_dequant = torch.where(codes == 3, a1_exp, w_dequant)

        # Step 5: Overlay sparse outliers if present
        if sparse_outliers is not None:
            if isinstance(sparse_outliers, SparseOutlierBuffer):
                w_dequant = sparse_outliers.apply_to(w_dequant)
            elif isinstance(sparse_outliers, torch.Tensor):
                if sparse_outliers.is_sparse:
                    w_dequant = w_dequant + sparse_outliers.to_dense().to(dtype)
                else:
                    w_dequant = w_dequant + sparse_outliers.to(dtype)

        return w_dequant

    @classmethod
    def unpack_tensor(
        cls,
        packed_tensor: Packed2BitTensor,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """Convenience helper to unpack a Packed2BitTensor instance."""
        return packed_tensor.dequantize(dtype=dtype)


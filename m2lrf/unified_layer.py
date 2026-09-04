import math
from typing import Optional, Tuple, Dict, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.quantizer import (
    DualBasisQuantizer,
    DoubleQuantizer,
    SparseOutlierBuffer,
    LLOYD_MAX_A0,
    LLOYD_MAX_A1,
    LLOYD_MAX_TAU
)
from m2lrf.packed_codec import Real2BitCodec, Packed2BitTensor
from m2lrf.hadamard_transform import (
    random_orthogonal_transform,
    rotate_weights_for_quantization,
    is_power_of_two
)
from m2lrf.mixed_precision import Real4BitCodec
from m2lrf.w2a8_kernel import (
    DynamicInt8ActQuantSTE,
    quantize_activations_dynamic_int8
)


class M2LRFUnifiedLinear(nn.Module):
    """
    Canonical Unified Linear Layer for M-2LRF.

    All quantization paradigms, rotation schemes, and adapter configurations
    compose seamlessly within a single execution graph.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 2,
        group_size: Optional[int] = 64,
        use_hadamard: bool = False,
        use_w2a8: bool = False,
        double_quant: bool = False,
        sparse_outliers: bool = False,
        rank: int = 16,
        alpha: float = 16.0,
        loftq_iters: int = 1,
        bias: bool = False,
        lora_dropout: float = 0.0,
        block_size: Optional[int] = 512,
        codec_type: str = "nf4",
        outlier_threshold_sigma: float = 3.5
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bits = int(bits)
        if self.bits not in (2, 4):
            raise ValueError(f"M2LRFUnifiedLinear supports bits=2 or bits=4, got bits={bits}")

        self.group_size = group_size
        self.use_hadamard = bool(use_hadamard)
        self.use_w2a8 = bool(use_w2a8)
        self.double_quant = bool(double_quant)
        self.sparse_outliers_enabled = bool(sparse_outliers)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 1.0
        self.loftq_iters = max(1, int(loftq_iters))
        self.block_size = block_size
        self.codec_type = codec_type.lower()
        self.outlier_threshold_sigma = float(outlier_threshold_sigma)

        self.orig_shape = (self.out_features, self.in_features)

        # 1. Base Quantized Weight Storage
        if self.bits == 2:
            self.packed_k = math.ceil(in_features / 4)
            self.register_buffer("packed_weights", torch.zeros(out_features, self.packed_k, dtype=torch.uint8))
            
            num_groups = math.ceil(in_features / group_size) if (group_size is not None and group_size > 0 and group_size < in_features) else 1
            scale_dtype = torch.uint8 if (self.double_quant and num_groups > 1) else torch.float16
            self.register_buffer("a0", torch.zeros(out_features, num_groups, dtype=scale_dtype))
            self.register_buffer("a1", torch.zeros(out_features, num_groups, dtype=scale_dtype))

            if self.double_quant and num_groups > 1:
                self.register_buffer("a0_super_scale", torch.zeros(out_features, 1, dtype=torch.float16))
                self.register_buffer("a1_super_scale", torch.zeros(out_features, 1, dtype=torch.float16))
            else:
                self.register_buffer("a0_super_scale", None)
                self.register_buffer("a1_super_scale", None)
        else:  # bits == 4
            self.packed_k = math.ceil(in_features / 2)
            self.register_buffer("packed_weights", torch.zeros(out_features, self.packed_k, dtype=torch.uint8))
            num_groups = math.ceil(in_features / group_size) if (group_size is not None and group_size > 0 and group_size < in_features) else 1
            self.register_buffer("scales", torch.zeros(out_features, num_groups, dtype=torch.float16))
            self.register_buffer("a0", None)
            self.register_buffer("a1", None)
            self.register_buffer("a0_super_scale", None)
            self.register_buffer("a1_super_scale", None)

        # 2. Orthogonal Hadamard Sign Vector
        if self.use_hadamard:
            self.register_buffer("signs", torch.ones(in_features, dtype=torch.float16))
        else:
            self.register_buffer("signs", None)

        # 3. Sparse Outlier Buffer (optional)
        self.sparse_outliers: Optional[SparseOutlierBuffer] = None

        # 4. Trainable LoRA Adapter (LoftQ Residual SVD)
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, dtype=torch.float32))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        # 5. LoRA Dropout
        if lora_dropout > 0.0 and self.rank > 0:
            self.lora_dropout = nn.Dropout(p=float(lora_dropout))
        else:
            self.lora_dropout = nn.Identity()

        # 6. Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self.is_merged = False

    @torch.no_grad()
    def initialize_from_pretrained(
        self,
        weight: torch.Tensor,
        signs: Optional[torch.Tensor] = None,
        loftq_iters: Optional[int] = None,
        niter: int = 4
    ):
        """
        Initializes base quantized weights and high-rank LoftQ SVD adapters
        from a full-precision pre-trained weight tensor.
        """
        w_f = weight.float()
        iters = int(loftq_iters) if loftq_iters is not None else self.loftq_iters
        iters = max(1, iters)

        # Step 1: Orthogonal Hadamard Rotation
        if self.use_hadamard:
            if signs is not None:
                self.signs.copy_(signs.to(dtype=self.signs.dtype, device=self.signs.device))
            else:
                signs_init = torch.where(
                    torch.randn(self.in_features, device=weight.device) >= 0,
                    torch.tensor(1.0, dtype=torch.float16, device=weight.device),
                    torch.tensor(-1.0, dtype=torch.float16, device=weight.device)
                )
                self.signs.copy_(signs_init)

            w_target = random_orthogonal_transform(
                w_f, signs=self.signs, block_size=self.block_size, inverse=False, normalize=True
            )
        else:
            w_target = w_f.clone()

        w_base = w_target.clone()

        # Step 2: Alternating LoftQ Optimization Loop
        for iter_idx in range(iters):
            if self.bits == 2:
                packed_tensor = Real2BitCodec.pack(
                    w_base,
                    group_size=self.group_size,
                    double_quant=self.double_quant,
                    extract_sparse_outliers=self.sparse_outliers_enabled,
                    outlier_threshold_sigma=self.outlier_threshold_sigma
                )
                w_dequant = packed_tensor.dequantize(dtype=torch.float32)
                if self.sparse_outliers_enabled:
                    self.sparse_outliers = packed_tensor.sparse_outliers
            else:  # bits == 4
                packed_bytes, scales, _ = Real4BitCodec.pack(
                    w_base,
                    group_size=self.group_size,
                    codec_type=self.codec_type
                )
                w_dequant = Real4BitCodec.unpack_and_dequantize(
                    packed_bytes, scales, self.orig_shape, group_size=self.group_size,
                    codec_type=self.codec_type, dtype=torch.float32
                )

            if self.rank <= 0 or self.lora_A is None or self.lora_B is None:
                break

            residual = w_target - w_dequant
            scale = self.scaling if self.scaling > 0 else 1.0
            norm_factor = 1.0 / math.sqrt(scale)
            max_possible_rank = min(self.out_features, self.in_features)
            q_dim = min(self.rank, max_possible_rank)

            svd_success = False
            try:
                if q_dim < max_possible_rank:
                    u, s, v = torch.svd_lowrank(residual, q=q_dim, niter=niter)
                else:
                    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
                    v = vh.t()

                s_clamped = s[:q_dim].clamp(min=1e-12)
                sqrt_s = torch.diag(torch.sqrt(s_clamped) * norm_factor)

                self.lora_B.zero_()
                self.lora_A.zero_()

                self.lora_B.data[:, :q_dim].copy_(u[:, :q_dim] @ sqrt_s)
                self.lora_A.data[:q_dim, :].copy_(sqrt_s @ v[:, :q_dim].t())
                svd_success = True
            except Exception:
                svd_success = False

            if not svd_success:
                try:
                    u, s, vh = torch.linalg.svd(residual, full_matrices=False)
                    r = min(self.rank, len(s))
                    s_clamped = s[:r].clamp(min=1e-12)
                    sqrt_s = torch.diag(torch.sqrt(s_clamped) * norm_factor)
                    self.lora_B.zero_()
                    self.lora_A.zero_()
                    self.lora_B.data[:, :r].copy_(u[:, :r] @ sqrt_s)
                    self.lora_A.data[:r, :].copy_(sqrt_s @ vh[:r, :])
                    svd_success = True
                except Exception:
                    nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
                    nn.init.zeros_(self.lora_B)

            if iters > 1 and iter_idx < iters - 1 and svd_success:
                adapter_recon = (self.lora_B.data @ self.lora_A.data) * scale
                w_base = w_target - adapter_recon

        # Step 3: Finalize Buffer Storage
        if self.bits == 2:
            self.packed_weights.copy_(packed_tensor.packed_bytes)
            self.a0.copy_(packed_tensor.a0)
            self.a1.copy_(packed_tensor.a1)
            if self.double_quant and packed_tensor.is_double_quant:
                if self.a0_super_scale is not None and packed_tensor.a0_super_scale is not None:
                    self.a0_super_scale.copy_(packed_tensor.a0_super_scale)
                if self.a1_super_scale is not None and packed_tensor.a1_super_scale is not None:
                    self.a1_super_scale.copy_(packed_tensor.a1_super_scale)
        else:  # bits == 4
            self.packed_weights.copy_(packed_bytes)
            self.scales.copy_(scales)

    initialize_from_weights = initialize_from_pretrained
    init_from_pretrained = initialize_from_pretrained

    def _dequantize_base(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Dequantizes base weights into floating point tensor in working coordinate space."""
        if self.bits == 2:
            return Real2BitCodec.unpack_and_dequantize(
                self.packed_weights,
                self.a0,
                self.a1,
                self.orig_shape,
                group_size=self.group_size,
                a0_super_scale=self.a0_super_scale,
                a1_super_scale=self.a1_super_scale,
                sparse_outliers=self.sparse_outliers,
                dtype=dtype
            )
        else:  # bits == 4
            return Real4BitCodec.unpack_and_dequantize(
                self.packed_weights,
                self.scales,
                self.orig_shape,
                group_size=self.group_size,
                codec_type=self.codec_type,
                dtype=dtype
            )

    _dequantize = _dequantize_base
    dequantize = _dequantize_base

    def dequantize_effective_weight(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """
        Reconstructs the full-precision effective weight matrix in the ORIGINAL (unrotated) feature space:
            W_eff = (W_dequant + scaling * (lora_B @ lora_A)) @ Q^T  (if hadamard)
        """
        w_dequant = self._dequantize_base(dtype=torch.float32)
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None and not self.is_merged:
            adapter = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            w_eff = w_dequant + adapter
        else:
            w_eff = w_dequant

        if self.use_hadamard:
            w_orig_eff = random_orthogonal_transform(
                w_eff, signs=self.signs, block_size=self.block_size, inverse=True, normalize=True
            )
            return w_orig_eff.to(dtype=dtype)
        return w_eff.to(dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Unified Forward Pass:
          1. Dynamic INT8 Activation Quantization (if use_w2a8)
          2. Fast Walsh-Hadamard Activation Rotation (if use_hadamard): X_rot = X @ Q
          3. Base GEMM: X_rot @ W_dequant^T
          4. LoRA Adapter GEMM: scaling * (X_rot @ lora_A^T @ lora_B^T)
          5. Bias addition
        """
        if self.use_w2a8:
            x_act = DynamicInt8ActQuantSTE.apply(x)
        else:
            x_act = x

        if self.use_hadamard:
            x_work = random_orthogonal_transform(
                x_act, signs=self.signs, block_size=self.block_size, inverse=False, normalize=True
            )
        else:
            x_work = x_act

        w_dequant = self._dequantize_base(dtype=x.dtype)
        base_out = F.linear(x_work, w_dequant)

        if self.is_merged or self.rank <= 0 or self.lora_A is None or self.lora_B is None:
            out = base_out
        else:
            x_adapted = self.lora_dropout(x_work)
            lora_out = F.linear(
                F.linear(x_adapted.float(), self.lora_A),
                self.lora_B
            ).to(x.dtype) * self.scaling
            out = base_out + lora_out

        if self.bias is not None:
            out = out + self.bias.to(out.dtype)

        return out

    @torch.no_grad()
    def merge(self):
        """
        Fuses the trained LoRA adapter permanently into the packed base weights (Zero-Overhead).
        """
        if not self.is_merged and self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            w_working = self._dequantize_base(dtype=torch.float32) + delta.float()
            
            if self.use_hadamard:
                w_orig = random_orthogonal_transform(
                    w_working, signs=self.signs, block_size=self.block_size, inverse=True, normalize=True
                )
                self.initialize_from_pretrained(w_orig, signs=self.signs, loftq_iters=1)
            else:
                self.initialize_from_pretrained(w_working, loftq_iters=1)

            self.lora_A.zero_()
            self.lora_B.zero_()
            self.is_merged = True

    @torch.no_grad()
    def unmerge(self):
        """Marks layer as unmerged."""
        self.is_merged = False

    @property
    def trainable_parameters(self) -> int:
        """Returns total trainable adapter parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_base_parameters(self) -> int:
        """Returns equivalent full-precision parameter count."""
        return self.in_features * self.out_features

    def memory_bytes(self) -> int:
        """Calculates exact physical memory footprint in bytes."""
        total = self.packed_weights.numel() * self.packed_weights.element_size()
        if self.bits == 2:
            total += self.a0.numel() * self.a0.element_size()
            total += self.a1.numel() * self.a1.element_size()
            if self.a0_super_scale is not None:
                total += self.a0_super_scale.numel() * self.a0_super_scale.element_size()
            if self.a1_super_scale is not None:
                total += self.a1_super_scale.numel() * self.a1_super_scale.element_size()
        else:
            total += self.scales.numel() * self.scales.element_size()

        if self.signs is not None:
            total += self.signs.numel() * self.signs.element_size()
        if self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            total += self.lora_A.numel() * self.lora_A.element_size()
            total += self.lora_B.numel() * self.lora_B.element_size()
        if self.bias is not None:
            total += self.bias.numel() * self.bias.element_size()
        return total

    def effective_bpp(self) -> float:
        """Computes effective bits-per-parameter for base storage."""
        base_bytes = self.packed_weights.numel() * self.packed_weights.element_size()
        if self.bits == 2:
            base_bytes += self.a0.numel() * self.a0.element_size()
            base_bytes += self.a1.numel() * self.a1.element_size()
            if self.a0_super_scale is not None:
                base_bytes += self.a0_super_scale.numel() * self.a0_super_scale.element_size()
            if self.a1_super_scale is not None:
                base_bytes += self.a1_super_scale.numel() * self.a1_super_scale.element_size()
        else:
            base_bytes += self.scales.numel() * self.scales.element_size()
        return (base_bytes * 8.0) / max(self.total_base_parameters, 1)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"use_hadamard={self.use_hadamard}, use_w2a8={self.use_w2a8}, "
            f"double_quant={self.double_quant}, rank={self.rank}, "
            f"alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"effective_bpp={self.effective_bpp():.2f}, merged={self.is_merged}"
        )


class M2LRF2BitLinear(M2LRFUnifiedLinear):
    """Canonical 2-Bit Dual-Basis Linear Layer."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        lora_dropout: float = 0.0,
        loftq_iters: int = 1,
        group_size: Optional[int] = None,
        double_quant: bool = False,
        sparse_outliers: bool = False
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bits=2,
            group_size=group_size,
            use_hadamard=False,
            use_w2a8=False,
            double_quant=double_quant,
            sparse_outliers=sparse_outliers,
            rank=rank,
            alpha=alpha,
            loftq_iters=loftq_iters,
            bias=bias,
            lora_dropout=lora_dropout
        )


class HadamardDualBasisLinear(M2LRFUnifiedLinear):
    """Canonical 2-Bit Hadamard Dual-Basis Linear Layer with Outlier Dispersion."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        lora_dropout: float = 0.0,
        loftq_iters: int = 1,
        group_size: Optional[int] = None,
        block_size: Optional[int] = 512,
        use_fast_transform: bool = True
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bits=2,
            group_size=group_size,
            use_hadamard=True,
            use_w2a8=False,
            double_quant=False,
            sparse_outliers=False,
            rank=rank,
            alpha=alpha,
            loftq_iters=loftq_iters,
            bias=bias,
            lora_dropout=lora_dropout,
            block_size=block_size
        )

    def rotate_activations(self, x: torch.Tensor) -> torch.Tensor:
        return random_orthogonal_transform(
            x, signs=self.signs, block_size=self.block_size, inverse=False, normalize=True
        )

    def de_rotate_activations(self, y: torch.Tensor) -> torch.Tensor:
        return random_orthogonal_transform(
            y, signs=self.signs, block_size=self.block_size, inverse=True, normalize=True
        )


class M2LRF4BitLinear(M2LRFUnifiedLinear):
    """Canonical 4-Bit Linear Layer (NF4 / Lloyd-Max)."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        lora_dropout: float = 0.0,
        loftq_iters: int = 1,
        group_size: Optional[int] = 64,
        codec_type: str = "nf4"
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bits=4,
            group_size=group_size,
            use_hadamard=False,
            use_w2a8=False,
            double_quant=False,
            sparse_outliers=False,
            rank=rank,
            alpha=alpha,
            loftq_iters=loftq_iters,
            bias=bias,
            lora_dropout=lora_dropout,
            codec_type=codec_type
        )


class M2LRFW2A8Linear(M2LRFUnifiedLinear):
    """Canonical 2-Bit Weight x INT8 Dynamic Activation Linear Layer."""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        lora_dropout: float = 0.0,
        loftq_iters: int = 1,
        group_size: Optional[int] = None,
        use_hadamard: bool = False
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            bits=2,
            group_size=group_size,
            use_hadamard=use_hadamard,
            use_w2a8=True,
            double_quant=False,
            sparse_outliers=False,
            rank=rank,
            alpha=alpha,
            loftq_iters=loftq_iters,
            bias=bias,
            lora_dropout=lora_dropout
        )


QuantizedLinearWithLoRA = M2LRF2BitLinear
RealPacked2BitLinearLoRA = M2LRF2BitLinear
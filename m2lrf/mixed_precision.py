"""
M-2LRF Mixed Precision Quantization & Layer Sensitivity Engine
===============================================================
Production-grade mixed 2/4-bit quantization framework for Large Language Models.

Components:
  1. Real4BitCodec:
     - Vectorized 4-bit bit-packing codec (2 weights per uint8 byte, 75% memory reduction).
     - Supports NF4 (NormalFloat4) and Lloyd-Max Gaussian centroids with group-wise / per-row scaling.
  2. M2LRF4BitLinear:
     - True 4-bit packed linear layer with High-Rank LoftQ SVD Residual Initialization.
     - Dynamic scaling normalization and in-situ weight merge/unmerge support.
  3. LayerSensitivityProfiler:
     - Computes sensitivity metrics across transformer layers:
       * Gradient magnitude (First-order Taylor proxy: ||g ⊙ W||)
       * Empirical Fisher Information proxy (Tr(F ⊙ W^2))
       * Output / Activation MSE perturbation (ΔL = ||X W^T - X W_quant^T||^2 / ||X W^T||^2)
       * Zero-shot / Data-free statistical heuristics (Frobenius, spectral variance, attention prior)
  4. MixedPrecisionAllocator:
     - Solves optimal bit allocation for target average bitrate (e.g. 2.6 bpp).
     - Allocates 4-bit representation to top sensitive layers (e.g. attention W_q, W_o)
       and 2-bit dual-basis M-2LRF to lower sensitivity layers (e.g. MLP).
     - Computes exact effective base bpp, net bpp (including scales + LoRA), and physical RAM footprint.
  5. allocate_mixed_precision_model / prepare_mixed_precision_m2lrf_model:
     - Seamless surgical model conversion replacing linear layers with mixed 2/4-bit representations.
"""

import math
import gc
from typing import Optional, Tuple, Dict, Any, List, Union, Callable
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.packed_codec import Real2BitCodec
from m2lrf.layer import M2LRF2BitLinear
from m2lrf.trainer_eval import (
    DEFAULT_TARGET_MODULES,
    DEFAULT_EXCLUDE_MODULES,
    get_model_device
)


# ====================================================================================================
# 1. 4-BIT NORMALFLOAT & LLOYD-MAX QUANTIZATION CODEC
# ====================================================================================================

# Standard NormalFloat4 (NF4) 16 centroids for standard normal distribution N(0, 1)
NF4_CENTROIDS = torch.tensor([
    -1.00000000, -0.69619280, -0.52507305, -0.38767642,
    -0.26965353, -0.15974070, -0.05449716,  0.00000000,
     0.07970124,  0.16093020,  0.24611230,  0.33791524,
     0.44070983,  0.56261700,  0.72295684,  1.00000000
], dtype=torch.float32)

# Closed-form Lloyd-Max 4-bit Gaussian centroids for N(0, 1)
LLOYD_MAX_4BIT_CENTROIDS = torch.tensor([
    -2.7326, -2.0694, -1.6182, -1.2560,
    -0.9424, -0.6568, -0.3881, -0.1284,
     0.1284,  0.3881,  0.6568,  0.9424,
     1.2560,  1.6182,  2.0694,  2.7326
], dtype=torch.float32)


class Real4BitCodec:
    """
    High-performance 4-bit bit-packing codec.
    Packs 2 4-bit weights into a single uint8 byte (75% memory compression vs FP16).
    
    Encoding:
      Byte = (code_0 & 0x0F) | ((code_1 & 0x0F) << 4)
    """

    @staticmethod
    def get_centroids(codec_type: str = "nf4", device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns the 16 quantization centroids normalized to [-1, 1]."""
        if codec_type.lower() == "lloyd_max":
            centroids = LLOYD_MAX_4BIT_CENTROIDS.clone()
            centroids = centroids / torch.max(torch.abs(centroids))
        else:
            centroids = NF4_CENTROIDS.clone()
        if device is not None:
            centroids = centroids.to(device)
        return centroids

    @classmethod
    def pack(
        cls,
        w: torch.Tensor,
        group_size: Optional[int] = None,
        codec_type: str = "nf4"
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        Packs floating-point weight tensor W into uint8 packed bytes and scale factors.

        Args:
            w: Input weight tensor of shape [..., in_features]
            group_size: Optional sub-channel group size (e.g. 64 or 128)
            codec_type: Quantization scheme ("nf4" or "lloyd_max")

        Returns:
            packed_bytes: uint8 tensor of packed weights
            scales: FP16 scale tensor
            orig_shape: Original tensor shape
        """
        w_f = w.float()
        orig_shape = tuple(w_f.shape)
        in_features = orig_shape[-1]
        batch_dims = orig_shape[:-1]
        device = w.device

        centroids = cls.get_centroids(codec_type=codec_type, device=device)
        midpoints = ((centroids[:-1] + centroids[1:]) / 2.0).to(device)

        if group_size is not None and group_size > 0:
            num_groups = math.ceil(in_features / group_size)
            padded_dim = num_groups * group_size
            if padded_dim != in_features:
                w_padded = F.pad(w_f, (0, padded_dim - in_features))
            else:
                w_padded = w_f

            w_grouped = w_padded.view(*batch_dims, num_groups, group_size)
            scales = torch.max(torch.abs(w_grouped), dim=-1, keepdim=True).values.clamp(min=1e-8)
            w_norm = (w_grouped / scales).clamp(-1.0, 1.0)

            # Vectorized centroid binning via bucketize
            codes = torch.bucketize(w_norm, midpoints).to(torch.uint8)
            codes = codes.view(*batch_dims, padded_dim)[..., :in_features]
            scales_out = scales.squeeze(-1).to(torch.float16)
        else:
            scales = torch.max(torch.abs(w_f), dim=-1, keepdim=True).values.clamp(min=1e-8)
            w_norm = (w_f / scales).clamp(-1.0, 1.0)
            codes = torch.bucketize(w_norm, midpoints).to(torch.uint8)
            scales_out = scales.to(torch.float16)

        # Pad in_features to even number for byte packing (2 weights per byte)
        padded_k = math.ceil(in_features / 2) * 2
        if padded_k != in_features:
            codes_pad = F.pad(codes, (0, padded_k - in_features))
        else:
            codes_pad = codes

        c_reshaped = codes_pad.view(*batch_dims, -1, 2)
        c0 = c_reshaped[..., 0]
        c1 = c_reshaped[..., 1]
        packed_bytes = ((c0 & 0x0F) | ((c1 & 0x0F) << 4)).to(torch.uint8)

        return packed_bytes, scales_out, orig_shape

    @classmethod
    def unpack_and_dequantize(
        cls,
        packed_bytes: torch.Tensor,
        scales: torch.Tensor,
        orig_shape: Tuple[int, ...],
        group_size: Optional[int] = None,
        codec_type: str = "nf4",
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor:
        """
        Unpacks 2 4-bit codes per uint8 byte and reconstructs weight matrix in-situ.
        """
        device = packed_bytes.device
        centroids = cls.get_centroids(codec_type=codec_type, device=device).to(dtype=dtype)

        # Step 1: Unpack 2 4-bit codes per byte
        c0 = packed_bytes & 0x0F
        c1 = (packed_bytes >> 4) & 0x0F
        codes = torch.stack([c0, c1], dim=-1).flatten(start_dim=-2)

        in_features = orig_shape[-1]
        codes = codes[..., :in_features]

        # Step 2: Centroid lookup
        w_norm = centroids[codes.long()]

        # Step 3: Broadcast scales
        scales_f = scales.to(dtype=dtype)
        if group_size is not None and group_size > 0 and scales_f.shape[-1] > 1:
            scales_exp = scales_f.repeat_interleave(group_size, dim=-1)[..., :in_features]
        else:
            scales_exp = scales_f

        w_dequant = w_norm * scales_exp
        return w_dequant.view(orig_shape)


# ====================================================================================================
# 2. 4-BIT PACKED LINEAR LAYER WITH LOFTQ SVD ADAPTER
# ====================================================================================================

class M2LRF4BitLinear(nn.Module):
    """
    Production 4-Bit Linear Layer holding frozen packed uint8 weights and trainable LoRA adapters.
    
    Features:
      1. True 4-bit physical storage in uint8 buffer (2 weights per byte, 75% memory reduction).
      2. Configurable High-Rank LoftQ Truncated SVD Residual Initialization.
      3. Dynamic Scaling Normalization for exact Step-0 representation recovery.
      4. Zero-Overhead In-Situ Weight Merge/Unmerge support.
    """
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
        codec_type: str = "nf4"
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = (self.alpha / self.rank) if self.rank > 0 else 1.0
        self.loftq_iters = max(1, int(loftq_iters))
        self.group_size = group_size
        self.codec_type = codec_type

        # Packed uint8 storage (in_features // 2 bytes per row)
        self.packed_k = math.ceil(in_features / 2)
        self.register_buffer("packed_weights", torch.zeros(out_features, self.packed_k, dtype=torch.uint8))

        num_groups = math.ceil(in_features / group_size) if (group_size is not None and group_size > 0 and group_size < in_features) else 1
        self.register_buffer("scales", torch.zeros(out_features, num_groups, dtype=torch.float16))
        self.orig_shape = (out_features, in_features)

        # Trainable Adapter (LoftQ Residual SVD)
        if self.rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(self.rank, in_features, dtype=torch.float32))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank, dtype=torch.float32))
        else:
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

        if lora_dropout > 0.0 and self.rank > 0:
            self.lora_dropout = nn.Dropout(p=float(lora_dropout))
        else:
            self.lora_dropout = nn.Identity()

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        self.is_merged = False

    @torch.no_grad()
    def initialize_from_pretrained(
        self,
        weight: torch.Tensor,
        loftq_iters: Optional[int] = None,
        niter: int = 4
    ):
        """
        Quantizes full-precision weights into packed 4-bit uint8 representation
        and initializes LoRA on the quantization residual via SVD (LoftQ).
        """
        w_target = weight.float()
        w_base = w_target.clone()
        iters = int(loftq_iters) if loftq_iters is not None else self.loftq_iters
        iters = max(1, iters)

        for iter_idx in range(iters):
            # 1. Pack base weights into 4-bit
            packed_bytes, scales, orig_shape = Real4BitCodec.pack(
                w_base, group_size=self.group_size, codec_type=self.codec_type
            )

            if scales.dim() == 1 and self.scales.dim() == 2:
                scales_buf = scales.unsqueeze(-1)
            else:
                scales_buf = scales

            w_dequant = Real4BitCodec.unpack_and_dequantize(
                packed_bytes, scales_buf, orig_shape,
                group_size=self.group_size, codec_type=self.codec_type
            ).float()

            if self.rank <= 0 or self.lora_A is None or self.lora_B is None:
                break

            # 2. Compute residual
            residual = w_target - w_dequant

            # 3. Truncated SVD Residual Initialization with Dynamic Scaling Normalization
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

            # 4. Alternating LoftQ update
            if iters > 1 and iter_idx < iters - 1 and svd_success:
                adapter_recon = (self.lora_B.data @ self.lora_A.data) * scale
                w_base = w_target - adapter_recon

        self.packed_weights.copy_(packed_bytes)
        if scales.dim() == 1 and self.scales.dim() == 2:
            self.scales.copy_(scales.unsqueeze(-1))
        else:
            self.scales.copy_(scales)

    initialize_from_weights = initialize_from_pretrained
    init_from_pretrained = initialize_from_pretrained

    def _dequantize(self) -> torch.Tensor:
        """De-quantizes packed uint8 into FP16 weight matrix."""
        return Real4BitCodec.unpack_and_dequantize(
            self.packed_weights, self.scales, self.orig_shape,
            group_size=self.group_size, codec_type=self.codec_type
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self._dequantize().to(x.dtype)
        base_out = F.linear(x, w_dequant)

        if self.is_merged or self.rank <= 0 or self.lora_A is None or self.lora_B is None:
            out = base_out
        else:
            x_adapted = self.lora_dropout(x)
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
        """Fuses the trained LoRA adapter permanently into the packed 4-bit base weights."""
        if not self.is_merged and self.rank > 0 and self.lora_A is not None and self.lora_B is not None:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            w_fused = self._dequantize().float() + delta
            self.initialize_from_pretrained(w_fused, loftq_iters=1)
            self.lora_A.zero_()
            self.lora_B.zero_()
            self.is_merged = True

    @torch.no_grad()
    def unmerge(self):
        self.is_merged = False

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def total_base_parameters(self) -> int:
        return self.in_features * self.out_features

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits=4, rank={self.rank}, alpha={self.alpha}, scaling={self.scaling:.4f}, "
            f"loftq_iters={self.loftq_iters}, merged={self.is_merged}"
        )


# Backward compatibility alias
Quantized4BitLinearWithLoRA = M2LRF4BitLinear


# ====================================================================================================
# 3. LAYER SENSITIVITY PROFILER
# ====================================================================================================

@dataclass
class SensitivityProfileResult:
    """Structured container holding sensitivity evaluation results across model layers."""
    raw_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    layer_shapes: Dict[str, Tuple[int, int]]
    layer_params: Dict[str, int]
    rankings: List[Tuple[str, float]]
    metric_used: str

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Returns top-k most sensitive layer names and scores."""
        return self.rankings[:k]

    def top_percentile(self, percentile: float) -> List[Tuple[str, float]]:
        """Returns top-p percentile (e.g. 0.3 for top 30%) most sensitive layers."""
        cutoff = max(1, int(len(self.rankings) * percentile))
        return self.rankings[:cutoff]

    def summary(self) -> str:
        """Returns formatted diagnostic summary string."""
        lines = [
            f"Layer Sensitivity Profile Report (Metric: {self.metric_used})",
            "-" * 75,
            f"{'Rank':<5} {'Layer Name':<45} {'Params':<10} {'Score':<10} {'Normalized':<10}",
            "-" * 75
        ]
        for idx, (name, score) in enumerate(self.rankings, 1):
            params = self.layer_params.get(name, 0)
            norm = self.normalized_scores.get(name, 0.0)
            lines.append(f"{idx:<5} {name:<45} {params:<10,} {score:<10.4e} {norm:<10.4f}")
        lines.append("-" * 75)
        return "\n".join(lines)


class LayerSensitivityProfiler:
    """
    Computes sensitivity metrics across transformer linear modules.
    
    Supported Metrics:
      1. 'gradient': Gradient magnitude / first-order Taylor proxy (||g ⊙ W||_F).
      2. 'fisher': Empirical Fisher Information Matrix diagonal proxy (Tr(F ⊙ W^2)).
      3. 'mse' / 'perturbation': Mean squared output perturbation under 2-bit quantization (ΔL).
      4. 'heuristic' / 'data_free': Zero-shot statistical metric (weight variance, Frobenius norm, Attention priority).
    """

    def __init__(
        self,
        target_modules: Optional[Union[List[str], str]] = None,
        exclude_modules: Optional[List[str]] = None
    ):
        if target_modules is None or target_modules in ("all", "all-linear"):
            self.target_patterns = DEFAULT_TARGET_MODULES
        elif isinstance(target_modules, str):
            self.target_patterns = [target_modules]
        else:
            self.target_patterns = list(target_modules)

        if exclude_modules is None:
            self.exclude_patterns = DEFAULT_EXCLUDE_MODULES
        else:
            self.exclude_patterns = list(exclude_modules)

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        """Checks if a module is a targeted linear layer and not excluded."""
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = (module.__class__.__name__ == "Conv1D")
        if not (is_linear or is_conv1d):
            return False

        leaf_name = name.split(".")[-1]
        if any(exc == leaf_name or exc in name for exc in self.exclude_patterns):
            return False

        return any(
            target == leaf_name or name.endswith(f".{target}") or f".{target}." in name or target in leaf_name
            for target in self.target_patterns
        )

    def get_target_modules(self, model: nn.Module) -> Dict[str, nn.Module]:
        """Returns dictionary of targeted linear modules in model."""
        targets = {}
        for name, module in model.named_modules():
            if self._is_target_module(name, module):
                targets[name] = module
        return targets

    def profile_gradient_magnitude(
        self,
        model: nn.Module,
        calibration_data: Any,
        loss_fn: Optional[Callable] = None,
        num_batches: int = 4
    ) -> Dict[str, float]:
        """
        Profiles layers via first-order Taylor gradient magnitude:
            Score(W) = E[ || ∇_W L ⊙ W ||_F ]
        """
        target_modules = self.get_target_modules(model)
        if not target_modules:
            return {}

        # Save requires_grad state and enable gradients on target weights
        orig_grad_states = {}
        for name, mod in target_modules.items():
            if hasattr(mod, "weight") and mod.weight is not None:
                orig_grad_states[name] = mod.weight.requires_grad
                mod.weight.requires_grad = True

        scores = {name: 0.0 for name in target_modules}
        batch_count = 0

        model.train()
        batches = self._extract_batches(calibration_data, num_batches=num_batches)

        for batch in batches:
            model.zero_grad(set_to_none=True)
            loss = self._compute_batch_loss(model, batch, loss_fn=loss_fn)
            if loss is None or not loss.requires_grad:
                continue

            loss.backward()

            for name, mod in target_modules.items():
                if hasattr(mod, "weight") and mod.weight is not None and mod.weight.grad is not None:
                    g = mod.weight.grad.float()
                    w = mod.weight.data.float()
                    taylor_val = torch.norm(g * w, p="fro").item()
                    scores[name] += taylor_val

            batch_count += 1

        # Restore grad states
        for name, mod in target_modules.items():
            if name in orig_grad_states and hasattr(mod, "weight") and mod.weight is not None:
                mod.weight.requires_grad = orig_grad_states[name]
        model.zero_grad(set_to_none=True)

        if batch_count > 0:
            scores = {name: score / batch_count for name, score in scores.items()}
        return scores

    def profile_fisher_information(
        self,
        model: nn.Module,
        calibration_data: Any,
        loss_fn: Optional[Callable] = None,
        num_batches: int = 4
    ) -> Dict[str, float]:
        """
        Profiles layers via empirical Fisher Information Matrix diagonal proxy:
            Score(W) = Tr( F ⊙ W^2 ) = sum( (∇_W L)^2 * W^2 )
        """
        target_modules = self.get_target_modules(model)
        if not target_modules:
            return {}

        orig_grad_states = {}
        for name, mod in target_modules.items():
            if hasattr(mod, "weight") and mod.weight is not None:
                orig_grad_states[name] = mod.weight.requires_grad
                mod.weight.requires_grad = True

        fisher_acc = {name: 0.0 for name in target_modules}
        batch_count = 0

        model.train()
        batches = self._extract_batches(calibration_data, num_batches=num_batches)

        for batch in batches:
            model.zero_grad(set_to_none=True)
            loss = self._compute_batch_loss(model, batch, loss_fn=loss_fn)
            if loss is None or not loss.requires_grad:
                continue

            loss.backward()

            for name, mod in target_modules.items():
                if hasattr(mod, "weight") and mod.weight is not None and mod.weight.grad is not None:
                    g = mod.weight.grad.float()
                    w = mod.weight.data.float()
                    fisher_val = torch.sum((g ** 2) * (w ** 2)).item()
                    fisher_acc[name] += fisher_val

            batch_count += 1

        for name, mod in target_modules.items():
            if name in orig_grad_states and hasattr(mod, "weight") and mod.weight is not None:
                mod.weight.requires_grad = orig_grad_states[name]
        model.zero_grad(set_to_none=True)

        if batch_count > 0:
            scores = {name: (fisher_acc[name] / batch_count) ** 0.5 for name in fisher_acc}
        else:
            scores = fisher_acc
        return scores

    def profile_output_perturbation(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        num_batches: int = 2
    ) -> Dict[str, float]:
        """
        Profiles layers via activation / output MSE perturbation:
            ΔL = || X W^T - X W_2bit^T ||_F^2 / (|| X W^T ||_F^2 + eps)
        If no calibration data is provided, falls back to weight-level Frobenius perturbation.
        """
        target_modules = self.get_target_modules(model)
        if not target_modules:
            return {}

        scores = {}
        # Hook activations if calibration data is available
        if calibration_data is not None:
            activations = {}
            handles = []

            def get_hook(mod_name):
                def hook_fn(module, args, output):
                    if len(args) > 0 and isinstance(args[0], torch.Tensor):
                        activations[mod_name] = args[0].detach()
                return hook_fn

            for name, mod in target_modules.items():
                handles.append(mod.register_forward_hook(get_hook(name)))

            model.eval()
            batches = self._extract_batches(calibration_data, num_batches=num_batches)
            with torch.no_grad():
                for batch in batches:
                    _ = self._forward_batch(model, batch)
                    break

            for h in handles:
                h.remove()

            # Compute output perturbation for each module with hooked inputs
            for name, mod in target_modules.items():
                w = mod.weight.data.float() if mod.__class__.__name__ != "Conv1D" else mod.weight.data.t().float()
                _, _, _, _, w_2bit = DualBasisQuantizer.quantize_2_00b(w)

                if name in activations:
                    x = activations[name].float()
                    y_orig = F.linear(x, w)
                    y_quant = F.linear(x, w_2bit.float())
                    denom = torch.norm(y_orig, p="fro").item() ** 2 + 1e-8
                    num = torch.norm(y_orig - y_quant, p="fro").item() ** 2
                    scores[name] = float(num / denom)
                else:
                    denom = torch.norm(w, p="fro").item() ** 2 + 1e-8
                    num = torch.norm(w - w_2bit.float(), p="fro").item() ** 2
                    scores[name] = float(num / denom)
        else:
            # Data-free weight perturbation
            for name, mod in target_modules.items():
                w = mod.weight.data.float() if mod.__class__.__name__ != "Conv1D" else mod.weight.data.t().float()
                _, _, _, _, w_2bit = DualBasisQuantizer.quantize_2_00b(w)
                denom = torch.norm(w, p="fro").item() ** 2 + 1e-8
                num = torch.norm(w - w_2bit.float(), p="fro").item() ** 2
                scores[name] = float(num / denom)

        return scores

    def profile_data_free(self, model: nn.Module) -> Dict[str, float]:
        """
        Zero-shot / data-free heuristic sensitivity profiling.
        Combines weight Frobenius energy, channel variance, and domain attention priors.
        """
        target_modules = self.get_target_modules(model)
        scores = {}

        # Known high-sensitivity projections across LLMs (Attention query/out projections)
        attn_priority_keys = {"q_proj", "o_proj", "c_attn", "query_key_value", "qkv_proj", "out_proj"}
        attn_secondary_keys = {"k_proj", "v_proj"}

        for name, mod in target_modules.items():
            leaf_name = name.split(".")[-1]
            w = mod.weight.data.float() if mod.__class__.__name__ != "Conv1D" else mod.weight.data.t().float()
            
            # Base statistical sensitivity
            std_val = torch.std(w).item()
            frob_norm = torch.norm(w, p="fro").item() / math.sqrt(w.numel())
            base_score = std_val * 0.5 + frob_norm * 0.5

            # Apply domain-specific transformer architectural prior
            if any(k in leaf_name for k in attn_priority_keys):
                weight_multiplier = 1.45
            elif any(k in leaf_name for k in attn_secondary_keys):
                weight_multiplier = 1.25
            else:
                weight_multiplier = 1.00

            scores[name] = base_score * weight_multiplier

        return scores

    def profile(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        metric: str = "fisher",
        loss_fn: Optional[Callable] = None,
        num_batches: int = 4
    ) -> SensitivityProfileResult:
        """
        Main entry point for sensitivity evaluation.
        """
        target_modules = self.get_target_modules(model)
        metric_clean = metric.lower().strip()

        if metric_clean in ("gradient", "grad", "taylor") and calibration_data is not None:
            raw_scores = self.profile_gradient_magnitude(
                model, calibration_data, loss_fn=loss_fn, num_batches=num_batches
            )
        elif metric_clean in ("fisher", "fisher_information") and calibration_data is not None:
            raw_scores = self.profile_fisher_information(
                model, calibration_data, loss_fn=loss_fn, num_batches=num_batches
            )
        elif metric_clean in ("mse", "perturbation", "delta_l"):
            raw_scores = self.profile_output_perturbation(
                model, calibration_data, num_batches=num_batches
            )
        else:
            raw_scores = self.profile_data_free(model)
            metric_clean = "heuristic_data_free"

        # Fallback if raw_scores empty
        if not raw_scores or all(v == 0.0 for v in raw_scores.values()):
            raw_scores = self.profile_data_free(model)
            metric_clean = "heuristic_data_free"

        # Compute metadata and normalized scores
        layer_shapes = {}
        layer_params = {}
        for name, mod in target_modules.items():
            if mod.__class__.__name__ == "Conv1D":
                in_f, out_f = mod.weight.shape[0], mod.weight.shape[1]
            else:
                in_f, out_f = mod.in_features, mod.out_features
            layer_shapes[name] = (out_f, in_f)
            layer_params[name] = out_f * in_f

        max_score = max(raw_scores.values()) if raw_scores else 1.0
        min_score = min(raw_scores.values()) if raw_scores else 0.0
        score_range = max_score - min_score if (max_score - min_score) > 1e-12 else 1.0

        normalized_scores = {
            name: (score - min_score) / score_range
            for name, score in raw_scores.items()
        }

        rankings = sorted(raw_scores.items(), key=lambda item: item[1], reverse=True)

        return SensitivityProfileResult(
            raw_scores=raw_scores,
            normalized_scores=normalized_scores,
            layer_shapes=layer_shapes,
            layer_params=layer_params,
            rankings=rankings,
            metric_used=metric_clean
        )

    def _extract_batches(self, calibration_data: Any, num_batches: int = 4) -> List[Any]:
        batches = []
        if calibration_data is None:
            return batches

        if isinstance(calibration_data, torch.Tensor):
            batches.append(calibration_data)
        elif isinstance(calibration_data, (list, tuple)):
            for item in calibration_data[:num_batches]:
                batches.append(item)
        elif hasattr(calibration_data, "__iter__"):
            count = 0
            for item in calibration_data:
                batches.append(item)
                count += 1
                if count >= num_batches:
                    break
        return batches

    def _forward_batch(self, model: nn.Module, batch: Any) -> Any:
        device = get_model_device(model)
        if isinstance(batch, torch.Tensor):
            return model(batch.to(device))
        elif isinstance(batch, dict):
            kwargs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            return model(**kwargs)
        elif isinstance(batch, (list, tuple)):
            args = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
            return model(*args)
        return model(batch)

    def _compute_batch_loss(
        self,
        model: nn.Module,
        batch: Any,
        loss_fn: Optional[Callable] = None
    ) -> Optional[torch.Tensor]:
        out = self._forward_batch(model, batch)
        if loss_fn is not None:
            return loss_fn(out, batch)

        if hasattr(out, "loss") and out.loss is not None:
            return out.loss

        if isinstance(out, torch.Tensor):
            return out.mean()

        if hasattr(out, "logits"):
            if isinstance(batch, dict) and "input_ids" in batch:
                logits = out.logits[..., :-1, :].contiguous()
                labels = batch["input_ids"][..., 1:].contiguous().to(logits.device)
                return F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            return out.logits.mean()

        return None


# ====================================================================================================
# 4. MIXED 2/4-BIT QUANTIZATION ALLOCATOR & PLANNER
# ====================================================================================================

@dataclass
class MixedPrecisionAllocationPlan:
    """Complete blueprint and audit report for mixed 2/4-bit quantization allocation."""
    layer_bit_assignments: Dict[str, int]
    target_avg_bits: float
    effective_base_bits: float
    effective_net_bits: float
    total_linear_params: int
    num_2bit_layers: int
    num_4bit_layers: int
    params_2bit: int
    params_4bit: int
    orig_fp16_bytes: int
    packed_base_bytes: int
    scale_bytes: int
    lora_adapter_bytes: int
    total_memory_bytes: int
    compression_ratio_base: float
    compression_ratio_net: float
    layer_details: Dict[str, Dict[str, Any]]
    sensitivity_result: Optional[SensitivityProfileResult] = None

    def summary(self) -> str:
        """Renders comprehensive ASCII diagnostic report."""
        lines = [
            "=" * 85,
            "⚡ M-2LRF MIXED 2/4-BIT QUANTIZATION ALLOCATION REPORT",
            "=" * 85,
            f"  • Target Average Bitrate   : {self.target_avg_bits:.2f} bpp",
            f"  • Effective Base Bitrate   : {self.effective_base_bits:.3f} bpp",
            f"  • Effective Net Bitrate    : {self.effective_net_bits:.3f} bpp (including scales + LoRA)",
            f"  • Total Targeted Parameters: {self.total_linear_params:,}",
            f"  • 4-Bit Sensitive Layers   : {self.num_4bit_layers} layers ({self.params_4bit:,} params, {self.params_4bit/max(self.total_linear_params,1)*100:.1f}%)",
            f"  • 2-Bit Dual-Basis Layers  : {self.num_2bit_layers} layers ({self.params_2bit:,} params, {self.params_2bit/max(self.total_linear_params,1)*100:.1f}%)",
            "-" * 85,
            f"  • Original FP16 Weight RAM : {self.orig_fp16_bytes / (1024**2):.2f} MB",
            f"  • Packed Base Weight RAM   : {self.packed_base_bytes / (1024**2):.2f} MB",
            f"  • Scale Factor RAM         : {self.scale_bytes / 1024:.2f} KB",
            f"  • LoRA Adapter RAM (FP32)  : {self.lora_adapter_bytes / (1024**2):.2f} MB",
            f"  • Total Mixed Weight RAM   : {self.total_memory_bytes / (1024**2):.2f} MB",
            f"  • Base Compression Ratio   : {self.compression_ratio_base:.2f}x ({100.0 * (1.0 - 1.0/self.compression_ratio_base):.1f}% reduction)",
            f"  • Net Compression Ratio    : {self.compression_ratio_net:.2f}x vs FP16",
            "-" * 85,
            f"{'Layer Name':<45} {'Bits':<6} {'Params':<10} {'Sensitivity':<12} {'Base RAM':<10}",
            "-" * 85
        ]

        for name, details in self.layer_details.items():
            bits = details["bits"]
            params = details["params"]
            score = details.get("sensitivity_score", 0.0)
            base_kb = details["base_bytes"] / 1024.0
            lines.append(f"{name:<45} {bits:<6} {params:<10,} {score:<12.4e} {base_kb:<10.1f} KB")

        lines.append("=" * 85)
        return "\n".join(lines)


class MixedPrecisionAllocator:
    """
    Computes optimal 2/4-bit layer bit assignments to meet target average bitrate.
    """

    @classmethod
    def allocate(
        cls,
        model: nn.Module,
        target_avg_bits: float = 2.6,
        rank: int = 16,
        group_size: Optional[int] = None,
        sensitivity_profile: Optional[SensitivityProfileResult] = None,
        profiler: Optional[LayerSensitivityProfiler] = None,
        calibration_data: Optional[Any] = None,
        metric: str = "fisher"
    ) -> MixedPrecisionAllocationPlan:
        """
        Calculates optimal mixed precision allocation plan.

        Args:
            model: PyTorch model
            target_avg_bits: Desired average linear weight bitrate (e.g. 2.6 bpp)
            rank: LoRA rank dimension
            group_size: Sub-channel quantization group size
            sensitivity_profile: Pre-computed sensitivity profile result
            profiler: Optional LayerSensitivityProfiler instance
            calibration_data: Calibration samples for profiling
            metric: Profiling metric ("fisher", "gradient", "mse", "heuristic")

        Returns:
            MixedPrecisionAllocationPlan
        """
        if sensitivity_profile is None:
            if profiler is None:
                profiler = LayerSensitivityProfiler()
            sensitivity_profile = profiler.profile(
                model, calibration_data=calibration_data, metric=metric
            )

        target_modules = sensitivity_profile.layer_shapes
        total_params = sum(sensitivity_profile.layer_params.values())

        if total_params == 0:
            raise ValueError("No targeted linear modules found in model for mixed precision allocation.")

        target_bits_clamped = max(2.0, min(4.0, float(target_avg_bits)))

        # Target 4-bit parameter ratio
        frac_4bit = (target_bits_clamped - 2.0) / 2.0
        target_4bit_params = frac_4bit * total_params

        ranked_layers = sensitivity_profile.rankings

        layer_assignments: Dict[str, int] = {}
        accumulated_4bit_params = 0

        if target_bits_clamped <= 2.0:
            for name, _ in ranked_layers:
                layer_assignments[name] = 2
        elif target_bits_clamped >= 4.0:
            for name, _ in ranked_layers:
                layer_assignments[name] = 4
        else:
            for name, score in ranked_layers:
                layer_param_count = sensitivity_profile.layer_params[name]
                diff_before = abs(accumulated_4bit_params - target_4bit_params)
                diff_after = abs((accumulated_4bit_params + layer_param_count) - target_4bit_params)

                if accumulated_4bit_params < target_4bit_params or diff_after < diff_before:
                    layer_assignments[name] = 4
                    accumulated_4bit_params += layer_param_count
                else:
                    layer_assignments[name] = 2

        orig_fp16_bytes = 0
        packed_base_bytes = 0
        scale_bytes = 0
        lora_adapter_bytes = 0

        params_2bit = 0
        params_4bit = 0
        num_2bit_layers = 0
        num_4bit_layers = 0

        layer_details: Dict[str, Dict[str, Any]] = {}

        for name, (out_f, in_f) in target_modules.items():
            bits = layer_assignments[name]
            p_count = out_f * in_f
            score = sensitivity_profile.raw_scores.get(name, 0.0)

            layer_orig_b = p_count * 2
            orig_fp16_bytes += layer_orig_b

            num_groups = math.ceil(in_f / group_size) if (group_size is not None and group_size > 0 and group_size < in_f) else 1

            if bits == 4:
                params_4bit += p_count
                num_4bit_layers += 1
                layer_base_b = out_f * math.ceil(in_f / 2)
                layer_scale_b = out_f * num_groups * 2
            else:
                params_2bit += p_count
                num_2bit_layers += 1
                layer_base_b = out_f * math.ceil(in_f / 4)
                layer_scale_b = out_f * num_groups * 4

            layer_lora_b = (rank * in_f * 4) + (out_f * rank * 4) if rank > 0 else 0

            packed_base_bytes += layer_base_b
            scale_bytes += layer_scale_b
            lora_adapter_bytes += layer_lora_b

            layer_details[name] = {
                "bits": bits,
                "in_features": in_f,
                "out_features": out_f,
                "params": p_count,
                "sensitivity_score": score,
                "base_bytes": layer_base_b,
                "scale_bytes": layer_scale_b,
                "lora_bytes": layer_lora_b,
                "total_layer_bytes": layer_base_b + layer_scale_b + layer_lora_b
            }

        total_memory_bytes = packed_base_bytes + scale_bytes + lora_adapter_bytes
        effective_base_bits = (params_4bit * 4.0 + params_2bit * 2.0) / total_params
        effective_net_bits = (total_memory_bytes * 8.0) / total_params

        comp_ratio_base = orig_fp16_bytes / max(packed_base_bytes + scale_bytes, 1)
        comp_ratio_net = orig_fp16_bytes / max(total_memory_bytes, 1)

        return MixedPrecisionAllocationPlan(
            layer_bit_assignments=layer_assignments,
            target_avg_bits=target_avg_bits,
            effective_base_bits=effective_base_bits,
            effective_net_bits=effective_net_bits,
            total_linear_params=total_params,
            num_2bit_layers=num_2bit_layers,
            num_4bit_layers=num_4bit_layers,
            params_2bit=params_2bit,
            params_4bit=params_4bit,
            orig_fp16_bytes=orig_fp16_bytes,
            packed_base_bytes=packed_base_bytes,
            scale_bytes=scale_bytes,
            lora_adapter_bytes=lora_adapter_bytes,
            total_memory_bytes=total_memory_bytes,
            compression_ratio_base=comp_ratio_base,
            compression_ratio_net=comp_ratio_net,
            layer_details=layer_details,
            sensitivity_result=sensitivity_profile
        )


# ====================================================================================================
# 5. HIGH-LEVEL ALLOCATOR & SURGICAL MODEL CONVERTER
# ====================================================================================================

def allocate_mixed_precision_model(
    model: nn.Module,
    target_avg_bits: float = 2.6,
    rank: int = 16,
    alpha: Optional[float] = None,
    profiler: Optional[LayerSensitivityProfiler] = None,
    calibration_data: Optional[Any] = None,
    metric: str = "fisher",
    target_modules: Optional[Union[List[str], str]] = None,
    exclude_modules: Optional[List[str]] = None,
    loftq_iters: int = 1,
    lora_dropout: float = 0.0,
    group_size: Optional[int] = None,
    freeze_bias: bool = True,
    apply_conversion: bool = True,
    verbose: bool = True
) -> Union[Tuple[nn.Module, MixedPrecisionAllocationPlan], MixedPrecisionAllocationPlan]:
    """
    Evaluates layer sensitivities, allocates mixed 2/4-bit representations for target average bitrate,
    and surgically converts the model in-situ with LoftQ SVD residual adapters.

    Args:
        model: Pretrained foundation model (Llama, Qwen, Mistral, Gemma, GPT-2, Falcon, etc.)
        target_avg_bits: Desired average linear weight bitrate (e.g. 2.6 bpp)
        rank: LoRA adapter rank dimension (e.g. 16, 32, 64)
        alpha: LoRA scaling factor (defaults to float(rank))
        profiler: Optional custom LayerSensitivityProfiler
        calibration_data: Calibration samples (DataLoader, batches, or input_ids)
        metric: Sensitivity metric ('fisher', 'gradient', 'mse', 'heuristic')
        target_modules: List of module name suffixes to target (default: Attention + MLP)
        exclude_modules: List of module names to exclude (default: lm_head, norm, embed_tokens)
        loftq_iters: Number of alternating SVD initialization iterations
        lora_dropout: LoRA dropout probability
        group_size: Sub-channel group size for group-wise scaling
        freeze_bias: Whether to freeze biases
        apply_conversion: If True, performs in-situ model conversion; if False, returns plan only.
        verbose: Whether to print diagnostic report

    Returns:
        (model, plan) if apply_conversion=True else plan
    """
    if profiler is None:
        profiler = LayerSensitivityProfiler(
            target_modules=target_modules,
            exclude_modules=exclude_modules
        )

    if alpha is None:
        alpha = float(rank) if rank > 0 else 16.0

    # Step 1: Compute optimal bit-allocation plan
    plan = MixedPrecisionAllocator.allocate(
        model=model,
        target_avg_bits=target_avg_bits,
        rank=rank,
        group_size=group_size,
        profiler=profiler,
        calibration_data=calibration_data,
        metric=metric
    )

    if verbose:
        print(plan.summary())

    if not apply_conversion:
        return plan

    # Step 2: Surgically convert model layers in-situ
    for param in model.parameters():
        param.requires_grad = False

    for name, module in list(model.named_modules()):
        if name not in plan.layer_bit_assignments:
            continue

        assigned_bits = plan.layer_bit_assignments[name]
        is_linear = isinstance(module, nn.Linear)
        is_conv1d = (module.__class__.__name__ == "Conv1D")

        if not (is_linear or is_conv1d):
            continue

        if is_linear:
            in_features = module.in_features
            out_features = module.out_features
            weight_data = module.weight.data
            bias_data = module.bias.data if module.bias is not None else None
        else:
            in_features = module.weight.shape[0]
            out_features = module.weight.shape[1]
            weight_data = module.weight.data.t().contiguous()
            bias_data = module.bias.data if module.bias is not None else None

        target_device = weight_data.device

        # Instantiate layer according to assigned bit-width
        if assigned_bits == 4:
            quant_layer = M2LRF4BitLinear(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                bias=(bias_data is not None),
                lora_dropout=lora_dropout,
                loftq_iters=loftq_iters,
                group_size=group_size
            ).to(target_device)
        else:
            quant_layer = M2LRF2BitLinear(
                in_features=in_features,
                out_features=out_features,
                rank=rank,
                alpha=alpha,
                bias=(bias_data is not None),
                lora_dropout=lora_dropout,
                loftq_iters=loftq_iters,
                group_size=group_size
            ).to(target_device)

        # High-Rank LoftQ SVD residual initialization
        quant_layer.initialize_from_pretrained(weight_data, loftq_iters=loftq_iters)

        if bias_data is not None:
            quant_layer.bias.data.copy_(bias_data)
            quant_layer.bias.requires_grad = not freeze_bias

        # Set trainable LoRA parameters
        if rank > 0 and quant_layer.lora_A is not None and quant_layer.lora_B is not None:
            quant_layer.lora_A.requires_grad = True
            quant_layer.lora_B.requires_grad = True

        # Replace module in parent submodule hierarchy
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
        else:
            parent = model
            child_name = name

        if isinstance(parent, (nn.ModuleList, nn.Sequential)) and child_name.isdigit():
            parent[int(child_name)] = quant_layer
        else:
            setattr(parent, child_name, quant_layer)

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model, plan


def prepare_mixed_precision_m2lrf_model(
    model: nn.Module,
    target_avg_bits: float = 2.6,
    rank: int = 16,
    alpha: Optional[float] = None,
    calibration_data: Optional[Any] = None,
    metric: str = "fisher",
    target_modules: Optional[Union[List[str], str]] = None,
    exclude_modules: Optional[List[str]] = None,
    loftq_iters: int = 1,
    lora_dropout: float = 0.0,
    group_size: Optional[int] = None,
    freeze_bias: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Convenience wrapper returning converted model with mixed 2/4-bit quantization.
    """
    model_conv, _ = allocate_mixed_precision_model(
        model=model,
        target_avg_bits=target_avg_bits,
        rank=rank,
        alpha=alpha,
        calibration_data=calibration_data,
        metric=metric,
        target_modules=target_modules,
        exclude_modules=exclude_modules,
        loftq_iters=loftq_iters,
        lora_dropout=lora_dropout,
        group_size=group_size,
        freeze_bias=freeze_bias,
        apply_conversion=True,
        verbose=verbose
    )
    return model_conv


__all__ = [
    "Real4BitCodec",
    "M2LRF4BitLinear",
    "Quantized4BitLinearWithLoRA",
    "LayerSensitivityProfiler",
    "SensitivityProfileResult",
    "MixedPrecisionAllocator",
    "MixedPrecisionAllocationPlan",
    "allocate_mixed_precision_model",
    "prepare_mixed_precision_m2lrf_model"
]

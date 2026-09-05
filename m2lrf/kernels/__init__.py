"""
M-2LRF High-Performance GPU Kernels (Unsloth-Inspired)
======================================================
"""

from m2lrf.kernels.fast_cross_entropy import (
    fast_cross_entropy_loss,
    FastCrossEntropyLoss,
    FastCrossEntropyFunction
)
from m2lrf.kernels.fast_rms_norm import (
    FastRMSNorm,
    FastRMSNormFunction
)
from m2lrf.kernels.fast_rope import (
    fast_apply_rotary_pos_emb
)
from m2lrf.kernels.fast_swiglu import (
    fast_swiglu,
    FastSwiGLUFunction
)
from m2lrf.kernels.fast_lora import (
    fast_lora_forward,
    FastLoRAFunction
)

__all__ = [
    "fast_cross_entropy_loss",
    "FastCrossEntropyLoss",
    "FastCrossEntropyFunction",
    "FastRMSNorm",
    "FastRMSNormFunction",
    "fast_apply_rotary_pos_emb",
    "fast_swiglu",
    "FastSwiGLUFunction",
    "fast_lora_forward",
    "FastLoRAFunction"
]

"""
M-2LRF Canonical Layer Module
==============================
Exposes the flagship unified linear layer M2LRFUnifiedLinear alongside backwards-compatible
specializations M2LRF2BitLinear, HadamardDualBasisLinear, M2LRF4BitLinear, and M2LRFW2A8Linear.
"""

from m2lrf.unified_layer import (
    M2LRFUnifiedLinear,
    M2LRF2BitLinear,
    HadamardDualBasisLinear,
    M2LRF4BitLinear,
    M2LRFW2A8Linear,
    QuantizedLinearWithLoRA,
    RealPacked2BitLinearLoRA
)

__all__ = [
    "M2LRFUnifiedLinear",
    "M2LRF2BitLinear",
    "HadamardDualBasisLinear",
    "M2LRF4BitLinear",
    "M2LRFW2A8Linear",
    "QuantizedLinearWithLoRA",
    "RealPacked2BitLinearLoRA"
]
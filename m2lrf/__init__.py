"""
M-2LRF: Multi-Rate Low-Rank Factorization & 2-Bit Dual-Basis Engine
===================================================================
Production Package:
  - M2LRF2BitLinear: True 2-bit packed Linear layer with LoftQ SVD residual initialization.
  - QuantizedLinearWithLoRA: Backwards-compatible alias to M2LRF2BitLinear.
  - DualBasisQuantizer: Closed-form Lloyd-Max dual-basis ternary quantizer (T0 ⊙ T1 = 0).
  - Real2BitCodec: 4 weights per uint8 byte bit-packing codec.
  - prepare_m2lrf_model: Universal model converter for transformer architectures.
"""

from m2lrf.quantizer import DualBasisQuantizer
from m2lrf.packed_codec import Real2BitCodec
from m2lrf.layer import M2LRF2BitLinear, QuantizedLinearWithLoRA
from m2lrf.trainer_eval import prepare_m2lrf_model

__version__ = "1.0.0"
__author__ = "MD-Mushfiqur Rahim"

__all__ = [
    "M2LRF2BitLinear",
    "QuantizedLinearWithLoRA",
    "DualBasisQuantizer",
    "Real2BitCodec",
    "prepare_m2lrf_model"
]

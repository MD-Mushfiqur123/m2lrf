"""
M-2LRF Core v1: Canonical Core Implementation
==============================================
Exports:
  - M2LRF2BitLinear: Production 2-bit packed linear layer with LoftQ SVD.
  - Real2BitCodec: High-performance bit-packing codec.
"""

from m2lrf.packed_codec import Real2BitCodec
from m2lrf.layer import M2LRF2BitLinear

__all__ = [
    "Real2BitCodec",
    "M2LRF2BitLinear"
]

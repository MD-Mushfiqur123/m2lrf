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

from m2lrf.quantizer import DualBasisQuantizer, DoubleQuantizer, SparseOutlierBuffer
from m2lrf.packed_codec import Real2BitCodec, Packed2BitTensor
from m2lrf.layer import M2LRF2BitLinear, QuantizedLinearWithLoRA, RealPacked2BitLinearLoRA
from m2lrf.mixed_precision import (
    Real4BitCodec,
    M2LRF4BitLinear,
    Quantized4BitLinearWithLoRA,
    LayerSensitivityProfiler,
    SensitivityProfileResult,
    MixedPrecisionAllocator,
    MixedPrecisionAllocationPlan,
    allocate_mixed_precision_model,
    prepare_mixed_precision_m2lrf_model
)
from m2lrf.w2a8_kernel import (
    M2LRFW2A8Linear,
    W2A8Linear,
    DynamicW2A8Linear,
    quantize_activations_dynamic_int8,
    dequantize_activations_dynamic_int8,
    w2a8_integer_gemm,
    w2a8_matmul,
    w2a8_matmul_fallback,
    w2a8_triton_matmul
)
from m2lrf.hadamard_transform import (
    is_power_of_two,
    fast_walsh_hadamard_transform,
    block_fast_walsh_hadamard_transform,
    generate_hadamard_matrix,
    generate_random_orthogonal_matrix,
    random_orthogonal_transform,
    calculate_kurtosis,
    rotate_weights_for_quantization,
    analyze_outlier_suppression,
    verify_hadamard_sqnr_gain,
    generate_synthetic_heavy_tailed_weights,
    HadamardDualBasisLinear,
    convert_linear_to_hadamard_dual_basis
)
from m2lrf.trainer_eval import (
    prepare_m2lrf_model,
    RealTaskEvaluator,
    ConversationTrainer,
    get_model_device,
    DEFAULT_TARGET_MODULES,
    DEFAULT_EXCLUDE_MODULES,
    GSM8K_8SHOT_PROMPT
)

__version__ = "1.0.0"
__author__ = "MD-Mushfiqur Rahim"

__all__ = [
    "M2LRF2BitLinear",
    "QuantizedLinearWithLoRA",
    "RealPacked2BitLinearLoRA",
    "Real4BitCodec",
    "M2LRF4BitLinear",
    "Quantized4BitLinearWithLoRA",
    "LayerSensitivityProfiler",
    "SensitivityProfileResult",
    "MixedPrecisionAllocator",
    "MixedPrecisionAllocationPlan",
    "allocate_mixed_precision_model",
    "prepare_mixed_precision_m2lrf_model",
    "M2LRFW2A8Linear",
    "W2A8Linear",
    "DynamicW2A8Linear",
    "quantize_activations_dynamic_int8",
    "dequantize_activations_dynamic_int8",
    "w2a8_integer_gemm",
    "w2a8_matmul",
    "w2a8_matmul_fallback",
    "w2a8_triton_matmul",
    "DualBasisQuantizer",
    "DoubleQuantizer",
    "SparseOutlierBuffer",
    "Real2BitCodec",
    "Packed2BitTensor",
    "is_power_of_two",
    "fast_walsh_hadamard_transform",
    "block_fast_walsh_hadamard_transform",
    "generate_hadamard_matrix",
    "generate_random_orthogonal_matrix",
    "random_orthogonal_transform",
    "calculate_kurtosis",
    "rotate_weights_for_quantization",
    "analyze_outlier_suppression",
    "verify_hadamard_sqnr_gain",
    "generate_synthetic_heavy_tailed_weights",
    "HadamardDualBasisLinear",
    "convert_linear_to_hadamard_dual_basis",
    "prepare_m2lrf_model",
    "RealTaskEvaluator",
    "ConversationTrainer",
    "get_model_device",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_EXCLUDE_MODULES",
    "GSM8K_8SHOT_PROMPT"
]




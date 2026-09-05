"""
M-2LRF: Multi-Rate Low-Rank Factorization & 2-Bit Dual-Basis Engine
===================================================================
Production Package:
  - M2LRFUnifiedLinear: The Grand Canonical Unified Composable Linear Layer (2/4-bit, Hadamard FWHT, W2A8, Group Scaling, Double Quantization, Sparse Outliers, High-Rank LoftQ).
  - M2LRF2BitLinear: True 2-bit packed Linear layer with LoftQ SVD residual initialization.
  - HadamardDualBasisLinear: 2-bit FWHT pre-rotated linear layer for outlier channel dispersion.
  - M2LRF4BitLinear: 4-bit packed NF4 / Lloyd-Max linear layer with LoftQ.
  - M2LRFW2A8Linear: 2-bit weight x dynamic INT8 activation linear layer.
  - DualBasisQuantizer: Closed-form Lloyd-Max dual-basis ternary quantizer (T0 ⊙ T1 = 0).
  - Real2BitCodec: 4 weights per uint8 byte bit-packing codec.
  - Real4BitCodec: 2 weights per uint8 byte bit-packing codec.
  - prepare_m2lrf_model: Universal model converter for transformer architectures.
"""

from m2lrf.unified_layer import (
    M2LRFUnifiedLinear,
    M2LRF2BitLinear,
    HadamardDualBasisLinear,
    M2LRF4BitLinear,
    M2LRFW2A8Linear,
    QuantizedLinearWithLoRA,
    RealPacked2BitLinearLoRA,
    W2A8Linear,
    DynamicW2A8Linear,
    QuantizedW2A8LinearWithLoRA
)
from m2lrf.quantizer import (
    DualBasisQuantizer,
    DoubleQuantizer,
    SparseOutlierBuffer,
    LLOYD_MAX_A0,
    LLOYD_MAX_A1,
    LLOYD_MAX_TAU
)
from m2lrf.packed_codec import (
    Real2BitCodec,
    Packed2BitTensor
)
from m2lrf.mixed_precision import (
    Real4BitCodec,
    Quantized4BitLinearWithLoRA,
    LayerSensitivityProfiler,
    SensitivityProfileResult,
    MixedPrecisionAllocator,
    MixedPrecisionAllocationPlan,
    allocate_mixed_precision_model,
    prepare_mixed_precision_m2lrf_model,
    NF4_CENTROIDS,
    LLOYD_MAX_4BIT_CENTROIDS
)
from m2lrf.w2a8_kernel import (
    DynamicInt8ActQuantSTE,
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
from m2lrf.triton_kernel import (
    HAS_TRITON,
    m2lrf_triton_matmul,
    m2lrf_matmul_fallback
)
from m2lrf.deep_benchmark import (
    Uniform4BitLinearLoRA,
    run_benchmark_comparison
)

# Unsloth-Inspired Fast Kernels
from m2lrf.kernels import (
    fast_cross_entropy_loss,
    FastCrossEntropyLoss,
    FastRMSNorm,
    fast_apply_rotary_pos_emb,
    fast_swiglu,
    fast_lora_forward,
    fused_linear_cross_entropy,
    fast_kl_divergence
)

# Unsloth-Inspired Model Loaders & Patchers
from m2lrf.models import (
    FastM2LRFModel,
    BaseArchitecturePatcher,
    LlamaPatcher,
    QwenPatcher,
    MistralPatcher
)

# Axolotl-Inspired Data & Packing
from m2lrf.data import (
    SequencePacker,
    CompletionOnlyDataCollator,
    get_formatter,
    AlpacaFormatter,
    ChatMLFormatter,
    Llama3Formatter,
    DPOFormatter
)

# Axolotl-Inspired Trainers
from m2lrf.trainers import (
    M2LRFSFTTrainer,
    M2LRFDPOTrainer,
    M2LRFORPOTrainer
)

# Axolotl-Inspired Declarative Config
from m2lrf.config import (
    M2LRFConfig,
    QuantConfig,
    DatasetConfig,
    TrainingArgsConfig
)

# Multi-Format Production Exporters
from m2lrf.export import (
    export_to_huggingface,
    export_to_gguf
)

# BitsAndBytes-Inspired 8-Bit Optimizers
from m2lrf.optimizers import AdamW8bit

# PEFT-Inspired Advanced Adapters
from m2lrf.adapters import (
    M2LRFDoRALinear,
    M2LRFLoHaLinear
)

# Torchtune-Inspired Hardware Profilers
from m2lrf.utils import MemoryTracker

__version__ = "2.0.0"
__author__ = "MD-Mushfiqur Rahim"

__all__ = [
    # Fast Models & Enterprise Entry Points
    "FastM2LRFModel",
    "FastCrossEntropyLoss",
    "fast_cross_entropy_loss",
    "fused_linear_cross_entropy",
    "fast_kl_divergence",
    "FastRMSNorm",
    "fast_apply_rotary_pos_emb",
    "fast_swiglu",
    "fast_lora_forward",
    "AdamW8bit",
    "M2LRFDoRALinear",
    "M2LRFLoHaLinear",
    "MemoryTracker",
    "SequencePacker",
    "CompletionOnlyDataCollator",
    "M2LRFSFTTrainer",
    "M2LRFDPOTrainer",
    "M2LRFORPOTrainer",
    "M2LRFConfig",
    "export_to_huggingface",
    "export_to_gguf",
    # Canonical Unified & Specialized Layers
    "M2LRFUnifiedLinear",
    "M2LRF2BitLinear",
    "HadamardDualBasisLinear",
    "M2LRF4BitLinear",
    "M2LRFW2A8Linear",
    "QuantizedLinearWithLoRA",
    "RealPacked2BitLinearLoRA",
    "Quantized4BitLinearWithLoRA",
    "QuantizedW2A8LinearWithLoRA",
    "W2A8Linear",
    "DynamicW2A8Linear",
    # Quantizers, Codecs & Constants
    "DualBasisQuantizer",
    "DoubleQuantizer",
    "SparseOutlierBuffer",
    "LLOYD_MAX_A0",
    "LLOYD_MAX_A1",
    "LLOYD_MAX_TAU",
    "Real2BitCodec",
    "Packed2BitTensor",
    "Real4BitCodec",
    "NF4_CENTROIDS",
    "LLOYD_MAX_4BIT_CENTROIDS",
    # Sensitivity Profiling & Mixed Precision
    "LayerSensitivityProfiler",
    "SensitivityProfileResult",
    "MixedPrecisionAllocator",
    "MixedPrecisionAllocationPlan",
    "allocate_mixed_precision_model",
    "prepare_mixed_precision_m2lrf_model",
    # W2A8 Kernel & Dynamic Activation Quantization
    "DynamicInt8ActQuantSTE",
    "quantize_activations_dynamic_int8",
    "dequantize_activations_dynamic_int8",
    "w2a8_integer_gemm",
    "w2a8_matmul",
    "w2a8_matmul_fallback",
    "w2a8_triton_matmul",
    # Fast Walsh-Hadamard Transform & Outlier Suppression
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
    "convert_linear_to_hadamard_dual_basis",
    # Model Conversion, Training & Evaluation
    "prepare_m2lrf_model",
    "RealTaskEvaluator",
    "ConversationTrainer",
    "get_model_device",
    "DEFAULT_TARGET_MODULES",
    "DEFAULT_EXCLUDE_MODULES",
    "GSM8K_8SHOT_PROMPT",
    # Triton Kernel & Micro-Benchmarks
    "HAS_TRITON",
    "m2lrf_triton_matmul",
    "m2lrf_matmul_fallback",
    "Uniform4BitLinearLoRA",
    "run_benchmark_comparison"
]

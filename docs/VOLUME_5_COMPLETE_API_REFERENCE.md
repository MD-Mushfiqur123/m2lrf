# M-2LRF: Multi-Rate Low-Rank Factorization & 2-Bit Dual-Basis Engine
## VOLUME V: COMPLETE API REFERENCE MANUAL
### *Publication-Grade Technical Specification & Exhaustive Developer Reference*

> **Lead Author & System Architect:** **MD-Mushfiqur Rahim**  
> **Affiliation / Project:** Independent Open-Source AI Research / M-Series Engineering  
> **Correspondence:** `mushfiqur.research@gmail.com`  
> **Repository:** `projects/m2lrf-clean/` | **Release:** `v2.0.0-Enterprise-Production`  
> **Package Namespace:** `m2lrf`  

---

## 📑 COMPREHENSIVE TABLE OF CONTENTS

1. [Architectural Overview & Global Package Conventions](#1-architectural-overview--global-package-conventions)
2. [`m2lrf.quantizer` — Dual-Basis Quantization & Outlier Buffers](#2-m2lrfquantizer--dual-basis-quantization--outlier-buffers)
   - 2.1 Theoretical Foundations & Closed-Form Constants
   - 2.2 `SparseOutlierBuffer`
   - 2.3 `DoubleQuantizer`
   - 2.4 `DualBasisQuantizer`
3. [`m2lrf.packed_codec` — Hardware-Level Bit-Packing Codecs](#3-m2lrfpacked_codec--hardware-level-bit-packing-codecs)
   - 3.1 2-Bit LSB-First Byte Mapping
   - 3.2 `Packed2BitTensor`
   - 3.3 `Real2BitCodec`
   - 3.4 `Real4BitCodec` (NF4 & Lloyd-Max 4-Bit Codec)
4. [`m2lrf.hadamard_transform` — Randomized Orthogonal Rotation Engine](#4-m2lrfhadamard_transform--randomized-orthogonal-rotation-engine)
   - 4.1 Outlier Dispersion & Central Limit Theorem Proof
   - 4.2 Core Fast Walsh-Hadamard Transform Functions
   - 4.3 Orthogonal Matrix Generators & Dynamic Transforms
   - 4.4 Kurtosis & Outlier Suppression Diagnostics
   - 4.5 Mathematical Verification of SQNR Gain
5. [`m2lrf.unified_layer` — Canonical Composable Linear Layers](#5-m2lrfunified_layer--canonical-composable-linear-layers)
   - 5.1 Architecture & Execution Flow
   - 5.2 `M2LRFUnifiedLinear`
   - 5.3 `M2LRF2BitLinear`
   - 5.4 `HadamardDualBasisLinear`
   - 5.5 `M2LRF4BitLinear`
   - 5.6 `M2LRFW2A8Linear`
6. [`m2lrf.mixed_precision` — Layer Sensitivity Profiling & Allocation](#6-m2lrfmixed_precision--layer-sensitivity-profiling--allocation)
   - 6.1 Sensitivity Diagnostics & Theoretical Metrics
   - 6.2 `SensitivityProfileResult`
   - 6.3 `MixedPrecisionAllocationPlan`
   - 6.4 `LayerSensitivityProfiler`
   - 6.5 `MixedPrecisionAllocator`
   - 6.6 `allocate_mixed_precision_model`
7. [`m2lrf.kernels` — High-Performance GPU Acceleration Kernels](#7-m2lrfkernels--high-performance-gpu-acceleration-kernels)
   - 7.1 `fast_cross_entropy_loss` & `FastCrossEntropyLoss`
   - 7.2 `FastRMSNorm`
   - 7.3 `fast_apply_rotary_pos_emb` (RoPE)
   - 7.4 `fast_swiglu`
   - 7.5 `fast_lora_forward`
   - 7.6 `fused_linear_cross_entropy`
   - 7.7 `fast_kl_divergence`
   - 7.8 `KIVIKVCache` (2-Bit Asymmetric KV Cache)
   - 7.9 `QuaRotLinear` (Dual-Sided Orthogonal Incoherence)
8. [`m2lrf.adapters` — Advanced Parameter-Efficient Adapters](#8-m2lrfadapters--advanced-parameter-efficient-adapters)
   - 8.1 `M2LRFDoRALinear` (Weight-Decomposed Adaptation)
   - 8.2 `M2LRFLoHaLinear` (Low-Rank Hadamard Product)
   - 8.3 `M2LRFPiSSALinear` (Principal Singular Component Adaptation)
9. [`m2lrf.optimizers` — Memory-Efficient 8-Bit Optimizers](#9-m2lrfoptimizers--memory-efficient-8-bit-optimizers)
   - 9.1 `AdamW8bit` (Block-Wise Quantized 8-Bit Optimizer)
10. [`m2lrf.models` — Fast Architecture Loaders & Surgical Patchers](#10-m2lrfmodels--fast-architecture-loaders--surgical-patchers)
    - 10.1 `FastM2LRFModel`
    - 10.2 `BaseArchitecturePatcher`
    - 10.3 `LlamaPatcher` (LLaMA 2/3/3.1/3.2)
    - 10.4 `QwenPatcher` (Qwen 2/2.5)
    - 10.5 `MistralPatcher` (Mistral/Mixtral)
11. [`m2lrf.data` — Formatting, Collators & Multiplexed Packing](#11-m2lrfdata--formatting-collators--multiplexed-packing)
    - 11.1 Prompt Formatters (`PromptFormatter`, `AlpacaFormatter`, `ChatMLFormatter`, `Llama3Formatter`, `DPOFormatter`)
    - 11.2 `SequencePacker` (Multiplexed Block-Diagonal Packing)
    - 11.3 `CompletionOnlyDataCollator`
12. [`m2lrf.trainers` — Specialized LLM Alignment & Fine-Tuning](#12-m2lrftrainers--specialized-llm-alignment--fine-tuning)
    - 12.1 `M2LRFSFTTrainer`
    - 12.2 `M2LRFDPOTrainer`
    - 12.3 `M2LRFORPOTrainer`
13. [`m2lrf.config` — Declarative Configuration Schema](#13-m2lrfconfig--declarative-configuration-schema)
    - 13.1 Dataclasses: `QuantConfig`, `DatasetConfig`, `TrainingArgsConfig`
    - 13.2 `M2LRFConfig`
14. [`m2lrf.export` — Enterprise Production Checkpoint Exporters](#14-m2lrfexport--enterprise-production-checkpoint-exporters)
    - 14.1 `export_to_huggingface`
    - 14.2 `export_to_gguf`
15. [`m2lrf.utils` — Hardware Telemetry & Performance Profiling](#15-m2lrfutils--hardware-telemetry--performance-profiling)
    - 15.1 `MemoryTracker`
16. [End-to-End Verified Recipes](#16-end-to-end-verified-recipes)
\n
# 1. ARCHITECTURAL OVERVIEW & GLOBAL PACKAGE CONVENTIONS

The **M-2LRF (Multi-Rate Low-Rank Factorization)** library provides a mathematically verified, hardware-accelerated framework for sub-4-bit foundation model fine-tuning, inference, and deployment.

```
                                      M-2LRF GLOBAL PACKAGE TOPOLOGY
+--------------------------------------------------------------------------------------------------------------------+
|                                                  FastM2LRFModel                                                    |
|                                         (Unified High-Level Entry Point)                                           |
+-------------------------------------------------------+------------------------------------------------------------+
                                                        |
                 +--------------------------------------+--------------------------------------+
                 |                                      |                                      |
                 v                                      v                                      v
      +---------------------+                +---------------------+                +---------------------+
      |   m2lrf.models      |                | m2lrf.unified_layer |                |   m2lrf.kernels     |
      | - BasePatcher       |                | - M2LRFUnifiedLinear|                | - FastCrossEntropy  |
      | - LlamaPatcher      |                | - M2LRF2BitLinear   |                | - FastRMSNorm       |
      | - QwenPatcher       |                | - HadamardDualLinear|                | - FastRoPE / SwiGLU |
      | - MistralPatcher    |                | - M2LRF4BitLinear   |                | - FusedLinearCE     |
      +----------+----------+                | - M2LRFW2A8Linear   |                | - KIVIKVCache       |
                 |                           +----------+----------+                +---------------------+
                 |                                      |
                 |             +------------------------+------------------------+
                 |             |                        |                        |
                 v             v                        v                        v
      +---------------------+  |             +---------------------+  +---------------------+
      | m2lrf.mixed_prec    |  |             |  m2lrf.quantizer    |  | m2lrf.hadamard_trans|
      | - SensitivityProf   |  |             | - DualBasisQuantizer|  | - FWHT / Block-FWHT |
      | - PrecisionAlloc    |  |             | - DoubleQuantizer   |  | - RandomOrthogonal  |
      | - allocate_mixed    |  |             | - SparseOutlierBuf  |  | - KurtosisAnalyzer  |
      +---------------------+  |             +----------+----------+  +---------------------+
                               |                        |
                 +-------------+------------+           +------------------------+
                 |                          |                                    |
                 v                          v                                    v
      +---------------------+    +---------------------+              +---------------------+
      |   m2lrf.adapters    |    |  m2lrf.packed_codec |              |  m2lrf.optimizers   |
      | - M2LRFDoRALinear   |    | - Real2BitCodec     |              | - AdamW8bit         |
      | - M2LRFLoHaLinear   |    | - Packed2BitTensor  |              |   (Block-wise 8-bit)|
      | - M2LRFPiSSALinear  |    | - Real4BitCodec     |              +---------------------+
      +---------------------+    +---------------------+
                 |
                 +--------------------------------------+--------------------------------------+
                 |                                      |                                      |
                 v                                      v                                      v
      +---------------------+                +---------------------+                +---------------------+
      |     m2lrf.data      |                |   m2lrf.trainers    |                |    m2lrf.export     |
      | - PromptFormatter   |                | - M2LRFSFTTrainer   |                | - HuggingFace SafeT |
      | - SequencePacker    |                | - M2LRFDPOTrainer   |                | - GGUF Binary Export|
      | - DataCollators     |                | - M2LRFORPOTrainer  |                +---------------------+
      +---------------------+                +---------------------+
```

### 1.1 Global Type Annotations and Invariants
All modules within `m2lrf` adhere to strict type-safety invariants:
1. **Compute Dtype:** Model activations default to `torch.bfloat16` when hardware support is present (`torch.cuda.is_bf16_supported()`), falling back to `torch.float32` on CPUs or legacy GPUs.
2. **Quantized Storage:** Base quantized weights are represented as packed `torch.uint8` tensors (4 weights per byte for 2-bit; 2 weights per byte for 4-bit).
3. **Scale Precision:** Quantization scale factors are stored in `torch.float16`, or `torch.uint8` when 8-bit Double Quantization is active.
4. **Dimension Layout:** Weight matrices follow standard PyTorch convention $[d_{	ext{out}}, d_{	ext{in}}]$. Quantization and bit-packing occur along the trailing input dimension $d_{	ext{in}}$.
\n
# 2. `m2lrf.quantizer` — DUAL-BASIS QUANTIZATION & OUTLIER BUFFERS

The `m2lrf.quantizer` module contains closed-form optimal Lloyd-Max quantizers, double-quantization engines for scale vectors, and sparse outlier coordinate representations.

## 2.1 Theoretical Foundations & Closed-Form Constants

For a standard normal Gaussian weight distribution $\mathbf{W} \sim \mathcal{N}(0, \sigma^2)$, M-2LRF decomposes the distribution into two elementwise-disjoint ternary basis matrices:
$$\mathbf{W} \approx \alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1, \quad \mathbf{T}_0, \mathbf{T}_1 \in \{-1, 0, +1\}^{d_{\text{out}} \times d_{\text{in}}}, \quad \mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}$$

The closed-form constants derived from the Lloyd-Max optimal Gaussian minimum mean squared error (MMSE) condition are:

```python
LLOYD_MAX_A0: float = 0.4527786409
LLOYD_MAX_A1: float = 1.5104181947
LLOYD_MAX_TAU: float = 0.9815984178  # (LLOYD_MAX_A0 + LLOYD_MAX_A1) / 2.0
```

- `LLOYD_MAX_A0`: Normalized centroid $\alpha_0 / \sigma$ for inner partitions $|w| \le \tau$.
- `LLOYD_MAX_A1`: Normalized centroid $\alpha_1 / \sigma$ for outer partitions $|w| > \tau$.
- `LLOYD_MAX_TAU`: Optimal decision boundary threshold $\tau / \sigma$.

---

## 2.2 `SparseOutlierBuffer`

```python
class SparseOutlierBuffer:
    def __init__(
        self,
        indices: torch.Tensor,
        values: torch.Tensor,
        dense_shape: Tuple[int, ...],
        is_residual: bool = False
    )
```

Compact sparse coordinate representation for statistical weight outliers $(|w| > 3.5\sigma)$. Stores coordinates and high-precision values without perturbing base quantized scales.

### Constructor Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `indices` | `torch.Tensor` | *Required* | Coordinate tensor of shape $[D, N]$ (PyTorch sparse COO layout, `int64`). |
| `values` | `torch.Tensor` | *Required* | High-precision values or residual deltas of shape $[N]$ (`float16` or `float32`). |
| `dense_shape` | `Tuple[int, ...]` | *Required* | Dense tensor shape, e.g. `(out_features, in_features)`. |
| `is_residual` | `bool` | `False` | `True` if values represent residual differences $(\mathbf{W} - \mathbf{W}_{\text{base}})$; `False` if absolute values. |

### Public Properties
- `num_outliers -> int`: Number of stored outlier values $N$.
- `density -> float`: Ratio of outliers to total dense parameters $N / \prod \text{dense\_shape}$.

### Public Methods

#### `to_sparse_coo() -> torch.Tensor`
Converts internal coordinates and values into a coalesced PyTorch `torch.sparse_coo_tensor`.
- **Returns:** Coalesced sparse COO tensor of shape `dense_shape`.

#### `apply_to(w_base: torch.Tensor) -> torch.Tensor`
Overlays or adds sparse outliers back onto reconstructed base weights.
- **Parameters:** `w_base` (`torch.Tensor`) — Reconstructed base weight tensor.
- **Returns:** Dense tensor with outliers restored, matching `w_base.dtype`.

#### `from_tensor(cls, w: torch.Tensor, threshold: Union[float, torch.Tensor], is_residual: bool = False, residual_values: Optional[torch.Tensor] = None) -> SparseOutlierBuffer`
Extracts outliers exceeding `threshold` from dense tensor `w`.
- **Parameters:**
  - `w` (`torch.Tensor`): Input dense weight matrix.
  - `threshold` (`Union[float, torch.Tensor]`): Threshold scalar or per-channel boundary.
  - `is_residual` (`bool`): Whether extracted values are residual deltas.
  - `residual_values` (`Optional[torch.Tensor]`): Tensor to read values from if different from `w`.
- **Returns:** Initialized `SparseOutlierBuffer`.

### Usage Example
```python
import torch
from m2lrf.quantizer import SparseOutlierBuffer

w = torch.randn(2048, 2048, dtype=torch.float16)
w[10, 55] = 42.0  # Synthetic outlier

# Extract outliers beyond 3.5 standard deviations
std = torch.std(w.float())
buffer = SparseOutlierBuffer.from_tensor(w, threshold=3.5 * std)

print(f"Stored {buffer.num_outliers} outliers (density: {buffer.density:.4%})")
w_restored = buffer.apply_to(torch.zeros_like(w))
assert torch.isclose(w_restored[10, 55], torch.tensor(42.0, dtype=torch.float16))
```

---

## 2.3 `DoubleQuantizer`

```python
class DoubleQuantizer:
    @staticmethod
    def quantize(
        scales: torch.Tensor,
        dim: int = -1,
        eps: float = 1e-8
    ) -> Tuple[torch.Tensor, torch.Tensor]

    @staticmethod
    def dequantize(
        q_scales: torch.Tensor,
        super_scale: torch.Tensor,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor
```

8-bit Double Quantization (DQ) engine for scale vectors. Slashes FP16 group scale memory by 50% with $< 0.05\%$ scale reconstruction distortion.

### Static Methods

#### `quantize(scales, dim=-1, eps=1e-8)`
Quantizes FP16 scale tensor into `uint8` with per-channel super-scales.
- **Parameters:**
  - `scales` (`torch.Tensor`): FP16/FP32 scale tensor of shape `[..., num_groups]`.
  - `dim` (`int`): Dimension across which super-scales are computed (default: `-1`).
  - `eps` (`float`): Numerical stability epsilon (default: `1e-8`).
- **Returns:**
  - `q_scales` (`torch.Tensor`): `uint8` tensor of shape `[..., num_groups]`.
  - `super_scale` (`torch.Tensor`): FP16 tensor of shape `[..., 1]`.

#### `dequantize(q_scales, super_scale, dtype=torch.float16)`
Dequantizes `uint8` scales back to floating-point representation.
- **Parameters:**
  - `q_scales` (`torch.Tensor`): Quantized `uint8` scale tensor.
  - `super_scale` (`torch.Tensor`): FP16 super-scale tensor.
  - `dtype` (`torch.dtype`): Target floating point dtype (default: `torch.float16`).
- **Returns:** Dequantized scales of shape `q_scales.shape`.

---

## 2.4 `DualBasisQuantizer`

```python
class DualBasisQuantizer:
    @staticmethod
    def calculate_sqnr(w_orig: torch.Tensor, w_quant: torch.Tensor) -> float

    @staticmethod
    def quantize_1_58b(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]

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
    ]
```

### Static Methods

#### `calculate_sqnr(w_orig: torch.Tensor, w_quant: torch.Tensor) -> float`
Calculates Signal-to-Quantization-Noise Ratio in dB:
$$\text{SQNR} = 10 \log_{10} \left( \frac{\mathbb{E}[\mathbf{W}^2]}{\mathbb{E}[(\mathbf{W} - \mathbf{W}_{\text{quant}})^2]} \right)$$
- **Parameters:**
  - `w_orig` (`torch.Tensor`): Original full-precision tensor.
  - `w_quant` (`torch.Tensor`): Quantized and dequantized tensor.
- **Returns:** SQNR value in decibels (`float`). Returns `inf` if error is zero; `0.0` if signal is zero.

#### `quantize_1_58b(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]`
Performs pure ternary quantization $\mathbf{W} \approx \alpha \mathbf{T}$ with $\mathbf{T} \in \{-1, 0, +1\}$.
- **Returns:** `(t, alpha, w_base)`.

#### `quantize_2_00b(...)`
Dual-basis ternary quantization $\mathbf{W} \approx \alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1$ strictly preserving the disjointness invariant $\mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}$.
- **Parameters:**
  - `w` (`torch.Tensor`): Weight tensor of shape `[..., in_features]`.
  - `group_size` (`Optional[int]`): Group size for sub-channel scaling (e.g. 64 or 128). If `None`, per-row scaling is applied.
  - `outlier_clip_sigma` (`Optional[float]`): Sigma clipping threshold applied during scale estimation.
  - `return_sparse_outliers` (`bool`): If `True`, returns a `SparseOutlierBuffer` containing outliers $> 3.5\sigma$.
  - `outlier_threshold_sigma` (`Optional[float]`): Outlier detection threshold (default: `3.5`).
  - `refine_centroids` (`bool`): Enables sample-adaptive Lloyd-Max conditional expectation updates.
- **Returns:**
  - When `return_sparse_outliers=False`: `(t0, t1, a0, a1, w_base)`.
  - When `return_sparse_outliers=True`: `(t0, t1, a0, a1, w_base, sparse_outliers)`.

### Usage Example
```python
import torch
from m2lrf.quantizer import DualBasisQuantizer

w = torch.randn(1024, 1024, dtype=torch.float32)

# Per-row 2.00-bit dual-basis quantization
t0, t1, a0, a1, w_base = DualBasisQuantizer.quantize_2_00b(w)

# Verify disjointness invariant
assert (t0 * t1).abs().max() == 0, "Disjointness violated!"

# Verify SQNR matches theoretical Gaussian bound (~9.30 dB)
sqnr = DualBasisQuantizer.calculate_sqnr(w, w_base)
print(f"Reconstructed 2.00-bit SQNR: {sqnr:.2f} dB")
```
\n
# 3. `m2lrf.packed_codec` — HARDWARE-LEVEL BIT-PACKING CODECS

The `m2lrf.packed_codec` module provides high-speed, LSB-first bit-packing engines storing 4 2-bit weights or 2 4-bit weights per `uint8` byte in physical memory.

## 3.1 2-Bit LSB-First Byte Mapping

$$\text{byte} = (c_0 \ll 0) \mid (c_1 \ll 2) \mid (c_2 \ll 4) \mid (c_3 \ll 6)$$

| 2-Bit Code | Binary | Centroid Representation | Numerical Value |
| :---: | :---: | :---: | :---: |
| `0` | `00` | Negative High Energy | $-\alpha_1$ |
| `1` | `01` | Negative Low Energy | $-\alpha_0$ |
| `2` | `10` | Positive Low Energy | $+\alpha_0$ |
| `3` | `11` | Positive High Energy | $+\alpha_1$ |

---

## 3.2 `Packed2BitTensor`

```python
class Packed2BitTensor:
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
    )
```

Production container holding 2-bit packed weight payloads with optional Double Quantization and sparse outlier buffers.

### Public Properties & Protocol
- `is_double_quant -> bool`: `True` if scales are 8-bit double-quantized with super-scales.
- `dequantize(dtype: torch.dtype = torch.float16) -> torch.Tensor`: Dequantizes weights in-situ.
- `memory_bytes() -> int`: Exact buffer footprint in bytes across packed bytes, scales, and outliers.
- Backwards-compatible 4-tuple unpacking iterator:
  ```python
  packed_bytes, a0, a1, orig_shape = packed_tensor
  ```

---

## 3.3 `Real2BitCodec`

```python
class Real2BitCodec:
    @staticmethod
    def pack(
        w: torch.Tensor,
        group_size: Optional[int] = None,
        double_quant: bool = False,
        outlier_clip_sigma: Optional[float] = None,
        extract_sparse_outliers: bool = False,
        outlier_threshold_sigma: Optional[float] = 3.5,
        refine_centroids: bool = False
    ) -> Packed2BitTensor

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
    ) -> torch.Tensor

    @classmethod
    def unpack_tensor(
        cls,
        packed_tensor: Packed2BitTensor,
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor
```

High-performance codec packing continuous weight tensors into `Packed2BitTensor` containers and decoding them dynamically in GPU registers.

---

## 3.4 `Real4BitCodec`

```python
class Real4BitCodec:
    @staticmethod
    def get_centroids(
        codec_type: str = "nf4",
        device: Optional[torch.device] = None
    ) -> torch.Tensor

    @classmethod
    def pack(
        cls,
        w: torch.Tensor,
        group_size: Optional[int] = None,
        codec_type: str = "nf4"
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]

    @classmethod
    def unpack_and_dequantize(
        cls,
        packed_bytes: torch.Tensor,
        scales: torch.Tensor,
        orig_shape: Tuple[int, ...],
        group_size: Optional[int] = None,
        codec_type: str = "nf4",
        dtype: torch.dtype = torch.float16
    ) -> torch.Tensor
```

Vectorized 4-bit packing codec packing 2 4-bit weights per `uint8` byte (75% memory compression vs FP16).
- Supported Schemes:
  - `"nf4"`: NormalFloat4 centroids.
  - `"lloyd_max"`: Optimal 4-bit Gaussian centroids.

### Usage Example
```python
import torch
from m2lrf.packed_codec import Real2BitCodec, Real4BitCodec

w = torch.randn(4096, 4096, dtype=torch.float16)

# Pack into 2-bit with group_size=64 and Double Quantization
packed_2bit = Real2BitCodec.pack(w, group_size=64, double_quant=True)
print(f"Packed 2-bit tensor memory: {packed_2bit.memory_bytes() / (1024**2):.2f} MB")

w_rec = packed_2bit.dequantize()
assert w_rec.shape == w.shape

# Pack into 4-bit NF4
p_bytes, scales, shape = Real4BitCodec.pack(w, group_size=64, codec_type="nf4")
w_4bit = Real4BitCodec.unpack_and_dequantize(p_bytes, scales, shape, group_size=64)
assert w_4bit.shape == w.shape
```
\n
# 4. `m2lrf.hadamard_transform` — RANDOMIZED ORTHOGONAL ROTATION ENGINE

The `m2lrf.hadamard_transform` module implements randomized orthogonal rotations via the Fast Walsh-Hadamard Transform (FWHT). It disperses isolated outlier channels into homogeneous Gaussian distributions (reducing kurtosis $\kappa \gg 20$ to $\kappa \approx 3.0$), yielding $+2.5$ to $+4.0\text{ dB}$ SQNR improvements.

## 4.1 Outlier Dispersion & Central Limit Theorem Proof

$$\widetilde{\mathbf{W}} = \mathbf{W} \mathbf{Q}, \quad \widetilde{\mathbf{X}} = \mathbf{X} \mathbf{Q}, \quad \mathbf{Y} = \widetilde{\mathbf{X}} \widetilde{\mathbf{W}}^T = (\mathbf{X} \mathbf{Q})(\mathbf{Q}^T \mathbf{W}^T) = \mathbf{X} \mathbf{W}^T$$

Because $\mathbf{Q} \mathbf{Q}^T = \mathbf{I}$, the linear transformation is mathematically lossless under infinite precision, while the Frobenius norm isometry guarantees:
$$\|\mathbf{W} - \text{Dequant}(\widetilde{\mathbf{W}}) \mathbf{Q}^T\|_F^2 = \|\widetilde{\mathbf{W}} - \text{Dequant}(\widetilde{\mathbf{W}})\|_F^2$$

---

## 4.2 Core Fast Walsh-Hadamard Transform Functions

```python
def is_power_of_two(n: int) -> bool
```
Checks if integer $n > 0$ is a power of 2.

```python
def fast_walsh_hadamard_transform(
    x: torch.Tensor,
    normalize: bool = True,
    scale: Optional[float] = None
) -> torch.Tensor
```
Computes Fast Walsh-Hadamard Transform along the trailing dimension $d = 2^m$ in $\mathcal{O}(d \log_2 d)$ time.
- **Parameters:**
  - `x` (`torch.Tensor`): Input tensor. Last dimension $d$ must be a power of 2.
  - `normalize` (`bool`): If `True`, scales by $1 / \sqrt{d}$.
  - `scale` (`Optional[float]`): Explicit multiplicative factor (overrides `normalize`).
- **Returns:** Transformed tensor matching `x.shape` and `x.dtype`.

```python
def block_fast_walsh_hadamard_transform(
    x: torch.Tensor,
    block_size: Optional[int] = 512,
    normalize: bool = True
) -> torch.Tensor
```
Block-wise FWHT for arbitrary (non-power-of-2) dimensions (e.g. 768, 1280, 3584, 11008). Partitions $d$ into power-of-2 sub-blocks capped by `block_size`.

---

## 4.3 Orthogonal Matrix Generators & Dynamic Transforms

```python
def generate_hadamard_matrix(
    d: int,
    normalize: bool = True,
    dtype: torch.dtype = torch.float32,
    device: Optional[torch.device] = None
) -> torch.Tensor
```
Constructs an explicit $d \times d$ normalized Walsh-Hadamard matrix $\widehat{\mathbf{H}}$.

```python
def generate_random_orthogonal_matrix(
    d: int,
    mode: str = "random_hadamard",
    block_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor
```
Generates an orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{d \times d}$ ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$).
- **Modes:**
  - `"hadamard"`: Block-diagonal deterministic Hadamard.
  - `"random_hadamard"`: Rademacher signed Hadamard $\mathbf{Q} = \mathbf{D} \widehat{\mathbf{H}}$ with $D_{ii} \in \{-1, +1\}$.
  - `"double_random_hadamard"`: $\mathbf{Q} = \mathbf{D}_1 \widehat{\mathbf{H}} \mathbf{D}_2 \widehat{\mathbf{H}}$.
  - `"haar_qr"`: Haar-distributed random orthogonal matrix via QR decomposition with diagonal sign fix.

```python
def random_orthogonal_transform(
    x: torch.Tensor,
    signs: Optional[torch.Tensor] = None,
    block_size: Optional[int] = 512,
    inverse: bool = False,
    normalize: bool = True
) -> torch.Tensor
```
Applies $\mathbf{Q} = \mathbf{D} \widehat{\mathbf{H}}$ or $\mathbf{Q}^T$ in $\mathcal{O}(d \log_2 d)$ time without materializing any $d \times d$ matrix in memory.
- Forward: $\mathbf{x} \mathbf{Q} = \text{FWHT}(\mathbf{x} \odot \mathbf{s})$.
- Inverse: $\mathbf{y} \mathbf{Q}^T = \text{FWHT}(\mathbf{y}) \odot \mathbf{s}$.

---

## 4.4 Kurtosis & Outlier Suppression Diagnostics

```python
def calculate_kurtosis(
    tensor: torch.Tensor,
    dim: Optional[int] = None,
    excess: bool = False
) -> Union[float, torch.Tensor]
```
Calculates sample kurtosis $\mathbb{E}[(X - \mu)^4] / \text{Var}(X)^2$. For a Gaussian, $\text{Kurt} = 3.0$ (excess kurtosis $= 0.0$).

```python
def rotate_weights_for_quantization(
    w: torch.Tensor,
    signs: Optional[torch.Tensor] = None,
    orthogonal_q: Optional[torch.Tensor] = None,
    block_size: Optional[int] = 512,
    seed: Optional[int] = 42
) -> Tuple[torch.Tensor, torch.Tensor]
```
Pre-rotates weight matrix along input channels: $\widetilde{\mathbf{W}} = \mathbf{W} \mathbf{Q}$.
- **Returns:** `(w_rotated, signs_or_q)`.

```python
def analyze_outlier_suppression(
    w_orig: torch.Tensor,
    w_rot: torch.Tensor,
    sigma_thresh: float = 3.5
) -> Dict[str, Any]
```
Computes statistical metrics comparing before and after rotation: kurtosis reduction, peak magnitude drop, and outlier count.

```python
def verify_hadamard_sqnr_gain(
    w: torch.Tensor,
    group_size: Optional[int] = None,
    block_size: Optional[int] = 512,
    refine_centroids: bool = False,
    seed: Optional[int] = 42
) -> Dict[str, Any]
```
Mathematically verifies $+2.5$ to $+4.0+\text{ dB}$ SQNR improvement and validates the Frobenius isometric equivalence.

### Usage Example
```python
import torch
from m2lrf.hadamard_transform import (
    calculate_kurtosis,
    rotate_weights_for_quantization,
    verify_hadamard_sqnr_gain
)

w = torch.randn(2048, 2048)
w[:, 50] *= 20.0  # Add heavy-tailed outlier channel

kurt_before = calculate_kurtosis(w)
w_rot, signs = rotate_weights_for_quantization(w, block_size=512)
kurt_after = calculate_kurtosis(w_rot)

print(f"Kurtosis: {kurt_before:.2f} -> {kurt_after:.2f} (Gaussianized!)")

res = verify_hadamard_sqnr_gain(w, group_size=128)
print(f"SQNR Gain: +{res['sqnr_gain_db']:.2f} dB (Isometry Hold: {res['frobenius_isometry_preserved']})")
```
\n
# 5. `m2lrf.unified_layer` — CANONICAL COMPOSABLE LINEAR LAYERS

The `m2lrf.unified_layer` module provides composable PyTorch linear layers uniting 2-bit, 4-bit, Hadamard FWHT rotation, W2A8 dynamic INT8 activations, group scaling, double quantization, sparse outliers, and high-rank LoftQ SVD residual initialization.

## 5.1 `M2LRFUnifiedLinear`

```python
class M2LRFUnifiedLinear(nn.Module):
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
    )
```

### Constructor Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `in_features` | `int` | *Required* | Input dimension $d_{\text{in}}$. |
| `out_features` | `int` | *Required* | Output dimension $d_{\text{out}}$. |
| `bits` | `int` | `2` | Base precision: `2` (dual-basis) or `4` (NF4/Lloyd-Max). |
| `group_size` | `Optional[int]` | `64` | Sub-channel scaling group size (e.g. 64 or 128). |
| `use_hadamard` | `bool` | `False` | Rotates inputs on-the-fly with FWHT for outlier suppression. |
| `use_w2a8` | `bool` | `False` | Enables dynamic INT8 activation quantization. |
| `double_quant` | `bool` | `False` | Enables 8-bit Double Quantization of scales. |
| `sparse_outliers` | `bool` | `False` | Preserves outliers $> 3.5\sigma$ in a sparse buffer. |
| `rank` | `int` | `16` | LoRA adapter rank dimension (e.g. 16, 32, 64). |
| `alpha` | `float` | `16.0` | LoRA scaling parameter. |
| `loftq_iters` | `int` | `1` | Alternating LoftQ SVD residual initialization iterations. |
| `bias` | `bool` | `False` | If `True`, adds a learnable bias parameter. |
| `lora_dropout` | `float` | `0.0` | Dropout probability on adapter branch. |
| `block_size` | `Optional[int]` | `512` | Sub-block dimension for Block-FWHT. |
| `codec_type` | `str` | `"nf4"` | 4-bit centroid codebook (`"nf4"` or `"lloyd_max"`). |
| `outlier_threshold_sigma` | `float` | `3.5` | Multiplier for outlier extraction. |

### Public Methods
- `initialize_from_pretrained(weight: torch.Tensor, signs: Optional[torch.Tensor] = None, loftq_iters: Optional[int] = None, niter: int = 4)`: Quantizes weights and initializes LoftQ adapters via truncated SVD on residual matrix.
- `forward(x: torch.Tensor) -> torch.Tensor`: Computes base quantized projection plus scaled adapter branch.
- `merge()`: Fuses trained LoRA adapter permanently into packed base weights with zero latency overhead.
- `unmerge()`: Unmerges adapter weights for continued training.
- `memory_bytes() -> int`: Exact buffer footprint in bytes.
- `effective_bpp() -> float`: Effective bits-per-parameter including scales and outliers.
- `trainable_parameters -> int`: Active trainable parameter count.
- `total_base_parameters -> int`: Full-precision parameter equivalent ($d_{\text{out}} \times d_{\text{in}}$).

---

## 5.2 Specialized Subclasses

### `M2LRF2BitLinear`
Specialized 2-bit dual-basis linear layer with LoftQ initialization.
```python
class M2LRF2BitLinear(M2LRFUnifiedLinear):
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
    )
```

### `HadamardDualBasisLinear`
Specialized 2-bit layer rotating activations on-the-fly via $\mathcal{O}(d \log d)$ FWHT.
- **Methods:**
  - `rotate_activations(x: torch.Tensor) -> torch.Tensor`
  - `de_rotate_activations(y: torch.Tensor) -> torch.Tensor`

### `M2LRF4BitLinear`
Specialized 4-bit packed linear layer (NF4 or Lloyd-Max) with LoftQ.

### `M2LRFW2A8Linear`
Specialized 2-bit weight $\times$ dynamic INT8 activation linear layer.

### Usage Example
```python
import torch
from m2lrf.unified_layer import M2LRFUnifiedLinear

# Initialize composable unified layer
layer = M2LRFUnifiedLinear(
    in_features=4096,
    out_features=4096,
    bits=2,
    group_size=64,
    use_hadamard=True,
    double_quant=True,
    rank=32,
    alpha=32.0,
    loftq_iters=2
)

# Initialize from pretrained FP16 weight
w = torch.randn(4096, 4096, dtype=torch.float16)
layer.initialize_from_pretrained(w)

x = torch.randn(2, 128, 4096, dtype=torch.float16)
y = layer(x)
print(f"Output shape: {y.shape} | Effective bpp: {layer.effective_bpp():.2f}")
```
\n
# 6. `m2lrf.mixed_precision` — LAYER SENSITIVITY PROFILING & ALLOCATION

The `m2lrf.mixed_precision` module computes layer sensitivity metrics across transformer layers and solves optimal mixed 2/4-bit allocations meeting target average bitrates (e.g. 2.6 bpp).

## 6.1 `SensitivityProfileResult`

```python
@dataclass
class SensitivityProfileResult:
    raw_scores: Dict[str, float]
    normalized_scores: Dict[str, float]
    layer_shapes: Dict[str, Tuple[int, int]]
    layer_params: Dict[str, int]
    rankings: List[Tuple[str, float]]
    metric_used: str

    def top_k(self, k: int) -> List[Tuple[str, float]]
    def top_percentile(self, percentile: float) -> List[Tuple[str, float]]
    def summary(self) -> str
```

Structured container holding sensitivity scores and diagnostic rankings across all targeted linear modules.

---

## 6.2 `MixedPrecisionAllocationPlan`

```python
@dataclass
class MixedPrecisionAllocationPlan:
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

    def summary(self) -> str
```

Complete audit blueprint detailing exact RAM savings, bit distributions, and per-layer precision assignments.

---

## 6.3 `LayerSensitivityProfiler`

```python
class LayerSensitivityProfiler:
    def __init__(
        self,
        target_modules: Optional[Union[List[str], str]] = None,
        exclude_modules: Optional[List[str]] = None
    )

    def profile_gradient_magnitude(
        self,
        model: nn.Module,
        calibration_data: Any,
        loss_fn: Optional[Callable] = None,
        num_batches: int = 4
    ) -> Dict[str, float]

    def profile_fisher_information(
        self,
        model: nn.Module,
        calibration_data: Any,
        loss_fn: Optional[Callable] = None,
        num_batches: int = 4
    ) -> Dict[str, float]

    def profile_output_perturbation(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        num_batches: int = 2
    ) -> Dict[str, float]

    def profile_data_free(self, model: nn.Module) -> Dict[str, float]

    def profile(
        self,
        model: nn.Module,
        calibration_data: Optional[Any] = None,
        metric: str = "fisher",
        loss_fn: Optional[Callable] = None,
        num_batches: int = 4
    ) -> SensitivityProfileResult
```

### Supported Sensitivity Metrics
1. `'fisher'`: Empirical Fisher Information Matrix diagonal proxy: $\text{Score}(\mathbf{W}) = [\sum (\nabla_{\mathbf{W}} L)^2 \mathbf{W}^2]^{1/2}$.
2. `'gradient'`: First-order Taylor magnitude proxy: $\text{Score}(\mathbf{W}) = \|\nabla_{\mathbf{W}} L \odot \mathbf{W}\|_F$.
3. `'mse'`: Output activation MSE perturbation under 2-bit quantization ($\Delta L$).
4. `'heuristic'`: Data-free statistical metric combining Frobenius norm and attention architectural priors.

---

## 6.4 `MixedPrecisionAllocator`

```python
class MixedPrecisionAllocator:
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
    ) -> MixedPrecisionAllocationPlan
```

Solves optimal 2/4-bit layer assignments meeting target average bitrate.

---

## 6.5 `allocate_mixed_precision_model`

```python
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
) -> Union[Tuple[nn.Module, MixedPrecisionAllocationPlan], MixedPrecisionAllocationPlan]
```

Surgically converts model in-situ with mixed 2/4-bit representations and LoftQ SVD residual initialization.

### Usage Example
```python
import torch
from transformers import AutoModelForCausalLM
from m2lrf.mixed_precision import allocate_mixed_precision_model

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

# Allocate mixed precision targeting 2.6 bpp
model_mixed, plan = allocate_mixed_precision_model(
    model=model,
    target_avg_bits=2.6,
    rank=16,
    metric="heuristic",
    verbose=True
)
print(plan.summary())
```
\n
# 7. `m2lrf.kernels` — HIGH-PERFORMANCE GPU ACCELERATION KERNELS

The `m2lrf.kernels` module provides GPU acceleration kernels with Triton implementations and vectorized PyTorch fallbacks.

## 7.1 `fast_cross_entropy_loss` & `FastCrossEntropyLoss`

```python
def fast_cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> torch.Tensor

class FastCrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index: int = -100, reduction: str = "mean")
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor
```
Drop-in replacement for `torch.nn.functional.cross_entropy`. Computes log-sum-exp and cross-entropy gradients in micro-token chunks without materializing full $[B, S, V]$ logits in VRAM (saving ~60% VRAM).

---

## 7.2 `FastRMSNorm`

```python
class FastRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```
Fused RMSNorm forward and backward kernel eliminating memory allocation for intermediate variance and reciprocal square-root activations. Drop-in replacement for `LlamaRMSNorm` and `Qwen2RMSNorm`.

---

## 7.3 `fast_apply_rotary_pos_emb` (RoPE)

```python
def fast_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]
```
In-place fused Rotary Position Embedding kernel for query and key projection tensors. Eliminates slicing, sign flipping, and concatenation tensor allocations.

---

## 7.4 `fast_swiglu`

```python
def fast_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor
```
Fused SwiGLU forward and backward kernel computing $\text{SiLU}(\text{gate}) \times \text{up}$, halving cached activation memory during MLP forward passes.

---

## 7.5 `fast_lora_forward`

```python
def fast_lora_forward(
    x: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float
) -> torch.Tensor
```
Fused low-rank adapter linear forward and backward kernel optimizing intermediate activation lifetimes for LoRA branches $\mathbf{h} = \frac{\alpha}{r} (\mathbf{X} \mathbf{A}^T) \mathbf{B}^T$.

---

## 7.6 `fused_linear_cross_entropy`

```python
def fused_linear_cross_entropy(
    x: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: str = "mean"
) -> torch.Tensor
```
Fuses final `lm_head` linear projection with cross-entropy loss computation. Completely bypasses allocating the $[B, S, V]$ logits tensor in global VRAM.

---

## 7.7 `fast_kl_divergence`

```python
def fast_kl_divergence(
    log_probs_p: torch.Tensor,
    log_probs_q: torch.Tensor,
    reduction: str = "batchmean"
) -> torch.Tensor
```
Computes Kullback-Leibler divergence $D_{\text{KL}}(P \parallel Q) = \sum P (\log P - \log Q)$ directly from log-probabilities in-place for DPO, PPO, and RLHF alignment.

---

## 7.8 `KIVIKVCache` (2-Bit Asymmetric KV Cache)

```python
class KIVIKVCache:
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        device: str = "cpu"
    )
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor)
    def get_dequantized_kv(self) -> Tuple[torch.Tensor, torch.Tensor]
```
Tuning-free 2-bit asymmetric KV Cache container slashing KV memory by 75–80% during long-context autoregressive decoding.
- **Key Cache:** Quantized per-channel into 2-bit unsigned integers (capturing persistent channel outliers).
- **Value Cache:** Quantized per-token into 2-bit unsigned integers.

---

## 7.9 `QuaRotLinear` (Dual-Sided Orthogonal Incoherence)

```python
class QuaRotLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int = 64,
        bias: bool = False
    )
    def rotate_weights(self)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```
Dual-sided orthogonal rotation linear layer ($\mathbf{X}_{\text{rot}} = \mathbf{X} \mathbf{H}, \mathbf{W}_{\text{rot}} = \mathbf{H}^T \mathbf{W}$) eliminating activation and weight outliers.

### Usage Example
```python
import torch
from m2lrf.kernels import FastCrossEntropyLoss, FastRMSNorm, fast_swiglu

# 1. Fast Cross Entropy Loss
loss_fn = FastCrossEntropyLoss(ignore_index=-100)
logits = torch.randn(4, 2048, 32000, requires_grad=True)
targets = torch.randint(0, 32000, (4, 2048))
loss = loss_fn(logits, targets)
loss.backward()

# 2. Fast RMSNorm
norm = FastRMSNorm(hidden_size=4096)
x = torch.randn(4, 2048, 4096)
y = norm(x)

# 3. Fast SwiGLU
gate = torch.randn(4, 2048, 11008, requires_grad=True)
up = torch.randn(4, 2048, 11008, requires_grad=True)
mlp_act = fast_swiglu(gate, up)
mlp_act.sum().backward()
```
\n
# 8. `m2lrf.adapters` — ADVANCED PARAMETER-EFFICIENT ADAPTERS

The `m2lrf.adapters` module provides advanced parameter-efficient fine-tuning architectures wrapping 2-bit dual-basis base weights.

## 8.1 `M2LRFDoRALinear` (Weight-Decomposed Adaptation)

```python
class M2LRFDoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        group_size: Optional[int] = 128
    )
    def initialize_from_pretrained(self, weight: torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```
Decouples magnitude and directional updates over 2-bit base weights:
$$\mathbf{W}_{\text{eff}} = \mathbf{m} \odot \frac{\mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}}{\|\mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}\|_c}$$

---

## 8.2 `M2LRFLoHaLinear` (Low-Rank Hadamard Product)

```python
class M2LRFLoHaLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: float = 16.0,
        bias: bool = False,
        group_size: Optional[int] = 128
    )
    def initialize_from_pretrained(self, weight: torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```
Expresses weight updates via the Hadamard product of two low-rank matrices:
$$\Delta \mathbf{W} = \frac{\alpha}{r} (\mathbf{B}_1 \mathbf{A}_1) \odot (\mathbf{B}_2 \mathbf{A}_2)$$
Yields an effective rank up to $r^2$ with parameter footprint $4 \cdot r \cdot d$.

---

## 8.3 `M2LRFPiSSALinear` (Principal Singular Component Adaptation)

```python
class M2LRFPiSSALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        alpha: Optional[float] = None,
        bias: bool = False,
        group_size: Optional[int] = 128,
        use_hadamard: bool = True
    )
    def initialize_from_pretrained(self, weight: torch.Tensor)
    def forward(self, x: torch.Tensor) -> torch.Tensor
```
Initializes adapters with the top-$r$ principal singular components of $\mathbf{W}_0$, quantizing the low-energy residual $\mathbf{W}_{\text{res}} = \mathbf{W}_0 - \mathbf{B} \mathbf{A}$ into 2-bit dual-basis format. Accelerates fine-tuning convergence by $2\times$ to $4\times$.

### Usage Example
```python
import torch
from m2lrf.adapters import M2LRFDoRALinear, M2LRFLoHaLinear, M2LRFPiSSALinear

w = torch.randn(2048, 2048)

# 1. DoRA Layer
dora = M2LRFDoRALinear(2048, 2048, rank=16)
dora.initialize_from_pretrained(w)
out_dora = dora(torch.randn(2, 128, 2048))

# 2. LoHa Layer
loha = M2LRFLoHaLinear(2048, 2048, rank=16)
loha.initialize_from_pretrained(w)
out_loha = loha(torch.randn(2, 128, 2048))

# 3. PiSSA Layer
pissa = M2LRFPiSSALinear(2048, 2048, rank=16)
pissa.initialize_from_pretrained(w)
out_pissa = pissa(torch.randn(2, 128, 2048))
```
\n
# 9. `m2lrf.optimizers` — MEMORY-EFFICIENT 8-BIT OPTIMIZERS

The `m2lrf.optimizers` module provides block-wise quantized optimizers reducing state memory by 75%.

## 9.1 `AdamW8bit` (Block-Wise Quantized 8-Bit Optimizer)

```python
class AdamW8bit(Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 256
    )
    def step(self, closure=None)
```

### Constructor Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `params` | `Iterable[nn.Parameter]` | *Required* | Iterable of model parameters to optimize. |
| `lr` | `float` | `1e-3` | Learning rate. |
| `betas` | `Tuple[float, float]` | `(0.9, 0.999)` | Coefficients for computing running averages of gradient and square. |
| `eps` | `float` | `1e-8` | Term added to denominator to improve numerical stability. |
| `weight_decay` | `float` | `0.01` | Weight decay (L2 penalty) coefficient. |
| `block_size` | `int` | `256` | Block size for dynamic 8-bit quantization. |

### Memory Characteristics
- **First Moment ($m$):** Stored in `int8` with per-block `float32` scale factor.
- **Second Moment ($v$):** Stored in `uint8` with per-block `float32` scale factor.
- **State Footprint:** 2 bytes/param vs. 8 bytes/param in standard FP32 AdamW (75% savings).

### Usage Example
```python
import torch
from m2lrf.optimizers import AdamW8bit

model = torch.nn.Linear(4096, 4096)
optimizer = AdamW8bit(
    model.parameters(),
    lr=2e-4,
    weight_decay=0.01,
    block_size=256
)

loss = model(torch.randn(8, 4096)).sum()
loss.backward()
optimizer.step()
optimizer.zero_grad()
```
\n
# 10. `m2lrf.models` — FAST ARCHITECTURE LOADERS & SURGICAL PATCHERS

The `m2lrf.models` module provides automatic foundation model loading, fast kernel injection, and surgical layer replacement across LLaMA, Qwen, and Mistral model families.

## 10.1 `FastM2LRFModel`

```python
class FastM2LRFModel:
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_length: int = 4096,
        dtype: Optional[torch.dtype] = None,
        load_in_2bit: bool = True,
        rank: int = 16,
        alpha: Optional[float] = None,
        use_hadamard: bool = False,
        group_size: Optional[int] = 128,
        loftq_iters: int = 1,
        target_avg_bits: Optional[float] = None,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
        patch_kernels: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[nn.Module, Any]

    @classmethod
    def for_training(cls, model: nn.Module) -> nn.Module

    @classmethod
    def for_inference(cls, model: nn.Module) -> nn.Module
```

Unified model factory loading, patching, and quantizing LLMs into M-2LRF 2-bit representations.

### Parameters
| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | `str` | *Required* | HuggingFace model repository ID or local path. |
| `max_seq_length` | `int` | `4096` | Context length capacity. |
| `dtype` | `Optional[torch.dtype]` | `None` | Compute precision (defaults to `bfloat16`). |
| `load_in_2bit` | `bool` | `True` | Replaces linear modules with 2-bit M-2LRF layers. |
| `rank` | `int` | `16` | LoRA adapter rank dimension. |
| `alpha` | `Optional[float]` | `None` | LoRA scaling parameter (defaults to `float(rank)`). |
| `use_hadamard` | `bool` | `False` | Enables FWHT rotation for outlier dispersion. |
| `group_size` | `Optional[int]` | `128` | Group-wise scaling sub-block size. |
| `loftq_iters` | `int` | `1` | Alternating SVD residual initialization iterations. |
| `target_avg_bits` | `Optional[float]` | `None` | If specified (e.g. 2.6), enables mixed 2/4-bit allocation. |
| `device_map` | `Optional[str]` | `"auto"` | Accelerator device placement map. |
| `trust_remote_code`| `bool` | `True` | Allows execution of custom model code. |
| `patch_kernels` | `bool` | `True` | Injects `FastRMSNorm` and fast loss kernels. |
| `verbose` | `bool` | `True` | Prints startup diagnostic telemetry. |

---

## 10.2 `BaseArchitecturePatcher`

```python
class BaseArchitecturePatcher:
    target_architectures: List[str] = []

    @classmethod
    def supports(cls, model: nn.Module) -> bool

    @classmethod
    def patch_model(...) -> nn.Module

    @staticmethod
    def patch_norm_modules(model: nn.Module, verbose: bool = False) -> int
```
Abstract base patcher providing in-place surgical model inspection, module replacement, and kernel optimization dispatch.

---

## 10.3 Architecture Patchers

- `LlamaPatcher`: Specialized patcher for LLaMA-2, LLaMA-3, 3.1, and 3.2 models. Targets `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- `QwenPatcher`: Specialized patcher for Qwen-2 and Qwen-2.5 models.
- `MistralPatcher`: Specialized patcher for Mistral and Mixtral architectures.

### Usage Example
```python
from m2lrf.models import FastM2LRFModel

model, tokenizer = FastM2LRFModel.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    load_in_2bit=True,
    rank=32,
    use_hadamard=True,
    group_size=128
)

FastM2LRFModel.for_training(model)
```
\n
# 11. `m2lrf.data` — FORMATTING, COLLATORS & MULTIPLEXED PACKING

The `m2lrf.data` module standardizes conversation formatting, sequence packing, and completion-only loss masking.

## 11.1 Prompt Formatters

```python
class PromptFormatter:
    def format(self, example: Dict[str, Any]) -> str

class AlpacaFormatter(PromptFormatter):
    def format(self, example: Dict[str, Any]) -> str
    def split_prompt_response(self, example: Dict[str, Any]) -> Tuple[str, str]

class ChatMLFormatter(PromptFormatter):
    def format(self, example: Dict[str, Any]) -> str

class Llama3Formatter(PromptFormatter):
    def format(self, example: Dict[str, Any]) -> str

class DPOFormatter:
    def format(self, example: Dict[str, Any]) -> Tuple[str, str, str]

def get_formatter(template_name: str) -> Any
```

Supported Templates:
- `"alpaca"`: `### Instruction:
...

### Response:
...`
- `"chatml"`: `<|im_start|>role
content<|im_end|>`
- `"llama3"`: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>

content<|eot_id|>`
- `"dpo"`: Formats pairwise preferences into `(prompt, chosen, rejected)`.

---

## 11.2 `SequencePacker` (Multiplexed Block-Diagonal Packing)

```python
class SequencePacker:
    def __init__(
        self,
        max_seq_length: int = 4096,
        pad_token_id: int = 0,
        ignore_index: int = -100
    )
    def pack(
        self,
        tokenized_samples: List[Dict[str, List[int]]]
    ) -> List[Dict[str, torch.Tensor]]
```
Packs multiple variable-length samples into constant-length tensors. Generates 2D block-diagonal causal attention masks to eliminate cross-sequence contamination.

---

## 11.3 `CompletionOnlyDataCollator`

```python
class CompletionOnlyDataCollator:
    def __init__(
        self,
        tokenizer: Any,
        response_template: str = "### Response:
",
        ignore_index: int = -100
    )
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]
```
Masks prompt tokens with `-100` so gradients are computed strictly on assistant / completion tokens.

### Usage Example
```python
from transformers import AutoTokenizer
from m2lrf.data import get_formatter, CompletionOnlyDataCollator, SequencePacker

tokenizer = AutoTokenizer.from_pretrained("gpt2")
formatter = get_formatter("alpaca")
collator = CompletionOnlyDataCollator(tokenizer=tokenizer, response_template="### Response:
")

sample = {"instruction": "What is M-2LRF?", "output": "A 2-bit quantization framework."}
text = formatter.format(sample)
batch = collator([{"input_ids": tokenizer.encode(text)}])
print("Labels:", batch["labels"])
```
\n
# 12. `m2lrf.trainers` — SPECIALIZED LLM ALIGNMENT & FINE-TUNING

The `m2lrf.trainers` module provides fine-tuning and preference alignment loops tailored for 2-bit foundation models.

## 12.1 `M2LRFSFTTrainer`

```python
class M2LRFSFTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        learning_rate: float = 2e-4,
        batch_size: int = 2,
        gradient_accumulation_steps: int = 4,
        num_train_epochs: int = 1,
        max_steps: Optional[int] = None,
        warmup_ratio: float = 0.05,
        weight_decay: float = 0.01,
        logging_steps: int = 10,
        device: Optional[str] = None
    )
    def train(self) -> Dict[str, Any]
```
Production SFT training loop integrated with `fast_cross_entropy_loss` and cosine warmup scheduling.

---

## 12.2 `M2LRFDPOTrainer`

```python
class M2LRFDPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        train_dataset: Any = None,
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_steps: int = 100,
        device: Optional[str] = None
    )
    def compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
Direct Preference Optimization trainer aligning 2-bit models directly from pairwise preferences with implicit or explicit reference models.

---

## 12.3 `M2LRFORPOTrainer`

```python
class M2LRFORPOTrainer:
    def __init__(
        self,
        model: nn.Module,
        lambda_orpo: float = 0.1,
        learning_rate: float = 1e-4,
        device: Optional[str] = None
    )
    def compute_orpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
```
Reference-free Odds Ratio Preference Optimization trainer combining negative log-likelihood SFT loss with an odds-ratio contrastive loss.

### Usage Example
```python
import torch
from m2lrf.models import FastM2LRFModel
from m2lrf.trainers import M2LRFSFTTrainer

model, tokenizer = FastM2LRFModel.from_pretrained("gpt2", load_in_2bit=True)

dataset = [{"input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4]}]
trainer = M2LRFSFTTrainer(
    model=model,
    train_dataset=dataset,
    batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=2e-4
)
metrics = trainer.train()
print(f"Final training loss: {metrics['final_loss']:.4f}")
```
\n
# 13. `m2lrf.config` — DECLARATIVE CONFIGURATION SCHEMA

The `m2lrf.config` module provides strongly-typed configuration schemas for declarative pipeline validation.

```python
@dataclass
class QuantConfig:
    method: str = "m2lrf_2bit"
    rank: int = 64
    alpha: float = 64.0
    use_hadamard: bool = True
    block_size: int = 64
    group_size: Optional[int] = 128
    loftq_iters: int = 1
    target_avg_bits: Optional[float] = 2.0
    double_quant: bool = False

@dataclass
class DatasetConfig:
    path: str = "tatsu-lab/alpaca"
    type: str = "alpaca"
    split: str = "train"
    sample_packing: bool = True
    max_seq_length: int = 4096

@dataclass
class TrainingArgsConfig:
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    max_steps: Optional[int] = None
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    logging_steps: int = 10
    output_dir: str = "./outputs"

@dataclass
class M2LRFConfig:
    base_model: str
    model_type: Optional[str] = None
    quantization: QuantConfig = field(default_factory=QuantConfig)
    datasets: List[DatasetConfig] = field(default_factory=lambda: [DatasetConfig()])
    training: TrainingArgsConfig = field(default_factory=TrainingArgsConfig)
    export_format: Optional[str] = "hf"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "M2LRFConfig"

    @classmethod
    def from_yaml(cls, file_path: str) -> "M2LRFConfig"
```

### Usage Example
```python
from m2lrf.config import M2LRFConfig

cfg = M2LRFConfig.from_dict({
    "base_model": "meta-llama/Llama-3.2-1B",
    "quantization": {
        "rank": 64,
        "use_hadamard": True,
        "group_size": 128
    },
    "training": {
        "batch_size": 2,
        "learning_rate": 2e-4
    }
})
assert cfg.quantization.use_hadamard is True
```
\n
# 14. `m2lrf.export` — ENTERPRISE PRODUCTION CHECKPOINT EXPORTERS

The `m2lrf.export` module packages fine-tuned checkpoints into production formats.

## 14.1 `export_to_huggingface`

```python
def export_to_huggingface(
    model_or_dir: Any,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    save_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True
) -> str
```
Collapses LoRA adapters permanently into the base weights in-situ and writes standard HuggingFace `SafeTensors` checkpoints compatible with vLLM, TensorRT-LLM, and TGI.

---

## 14.2 `export_to_gguf`

```python
def export_to_gguf(
    model_dir: str,
    output_dir: str,
    quantization_type: str = "q4_k_m",
    verbose: bool = True
) -> str
```
Exports merged model directories to GGUF format for low-latency local inference in llama.cpp, Ollama, and LM Studio.

### Usage Example
```python
import torch
from m2lrf.models import FastM2LRFModel
from m2lrf.export import export_to_huggingface, export_to_gguf

model, tokenizer = FastM2LRFModel.from_pretrained("gpt2", load_in_2bit=True)

# Export HuggingFace SafeTensors
hf_dir = export_to_huggingface(model, "./gpt2_m2lrf_merged", tokenizer=tokenizer)

# Generate GGUF export artifacts
gguf_path = export_to_gguf(hf_dir, "./gpt2_gguf_out", quantization_type="q4_k_m")
```
\n
# 15. `m2lrf.utils` — HARDWARE TELEMETRY & PERFORMANCE PROFILING

## 15.1 `MemoryTracker`

```python
class MemoryTracker:
    def __init__(self, device: Optional[str] = None)
    def start(self)
    def stop(self) -> Dict[str, Any]
    def summary(self, tokens_processed: Optional[int] = None) -> str
```

Live GPU VRAM and throughput profiler tracking allocated, reserved, peak, and net activation memory.

### Usage Example
```python
from m2lrf.utils import MemoryTracker
import torch

tracker = MemoryTracker()
tracker.start()

# Execute model workload
x = torch.randn(8, 4096, device="cuda" if torch.cuda.is_available() else "cpu")

print(tracker.summary(tokens_processed=8 * 4096))
```
\n
# 16. END-TO-END VERIFIED RECIPES

### 16.1 Recipe 1: 2-Bit + FWHT + LoftQ Fine-Tuning of LLaMA-3
```python
import torch
from m2lrf.models import FastM2LRFModel
from m2lrf.data import get_formatter, CompletionOnlyDataCollator
from m2lrf.trainers import M2LRFSFTTrainer
from m2lrf.export import export_to_huggingface

# 1. Load, patch, and quantize to 2-bit dual-basis
model, tokenizer = FastM2LRFModel.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    load_in_2bit=True,
    rank=64,
    use_hadamard=True,
    group_size=128,
    loftq_iters=2
)
FastM2LRFModel.for_training(model)

# 2. Setup mock dataset with Alpaca template
formatter = get_formatter("alpaca")
collator = CompletionOnlyDataCollator(tokenizer=tokenizer)

# 3. Fine-tune with M-2LRF SFT Trainer
trainer = M2LRFSFTTrainer(
    model=model,
    train_dataset=[{"input_ids": tokenizer.encode(formatter.format({"instruction": "Hello", "output": "Hi!"}))}],
    data_collator=collator,
    learning_rate=2e-4,
    batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1
)
trainer.train()

# 4. Collapse adapters and export SafeTensors
export_to_huggingface(model, "./llama3_2bit_merged", tokenizer=tokenizer)
```

### 16.2 Recipe 2: Mixed-Precision 2.6 bpp Sensitivity Allocation
```python
import torch
from transformers import AutoModelForCausalLM
from m2lrf.mixed_precision import allocate_mixed_precision_model

model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

# Surgically allocate mixed 2/4-bit layers targeting 2.6 bpp
model_mixed, plan = allocate_mixed_precision_model(
    model=model,
    target_avg_bits=2.6,
    rank=16,
    metric="heuristic",
    verbose=True
)

print(f"Allocated {plan.num_4bit_layers} 4-bit layers and {plan.num_2bit_layers} 2-bit layers.")
print(f"Base Compression Ratio: {plan.compression_ratio_base:.2f}x vs FP16")
```

---

*This document serves as the canonical Volume V API Reference Manual for the M-2LRF project.*

# M-2LRF Empirical Benchmark Suite & Telemetry Hub

> **Project:** Multi-Rate Low-Rank Factorization & Dual-Basis 2-Bit Quantization (M-2LRF)  
> **Author & Lead Maintainer:** MD-Mushfiqur Rahim
> **Repository:** `https://github.com/MD-Mushfiqur123/m2lrf`  
> **Status:** Historical benchmark metrics are not independently verified. Current code tests: 193 passed, 0 failed on Python 3.12.10 (2026-10-14), with one PyTorch sparse-tensor warning.
>
> **Perplexity correction:** The prior `eval_downstream_tasks.py` scored eight hand-written strings, not WikiText-2. Its recorded PPL values are invalid as WikiText-2 results. The script now evaluates the WikiText-2 validation split; rerun it before publishing PPL numbers.

---

## 📊 Executive Summary

This document provides a comprehensive, high-density reference for all empirical benchmarks, hardware profiling, and downstream evaluation metrics for **M-2LRF**. 

Across all experiments, M-2LRF achieves:
1. **Historical memory projections:** Previous architecture calculations reported up to 76.0% net VRAM reduction vs. 16-bit and 32.4% vs. NF4; these figures are estimates, not independently reproduced measurements.
2. **WikiText-2 perplexity:** No valid measurement is available; the previous hand-written-text result is withdrawn pending a full-split rerun.
3. **Historical outlier-suppression analysis:** Previous artifacts reported Spearman rank correlation $\mathbf{\rho = 0.8723}$ ($p = 4.77 \times 10^{-19}$); this result has not been independently reproduced in the current checkout.
4. **Language Modeling Preservation:** No valid WikiText-2 PPL result is currently available from this checkout. Historical values were generated from hand-written strings and are withdrawn pending a rerun.

---

## 🔬 1. 8-Way Architectural Ablation Matrix

The following table summarizes the cumulative impact of each architectural component evaluated on realistic heavy-tailed weight matrices ($N=2048, K=2048$, Student-$t$ distribution with $\nu=3$, excess kurtosis $\kappa_0 \approx 122.5$).

| Configuration | Description | Mean SQNR (dB) | $\Delta$ vs Baseline | Rel Error (%) | Effective bpp | Compression vs FP16 | Latency (ms) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1. Baseline 2-Bit** | Per-Row Dual-Basis ($a_0, a_1$) | 8.72 dB | +0.00 dB | 36.73% | 2.03 bpp | 7.87x | 6.87 ms |
| **2. + Group Scaling** | Group Size $G=64$ | 9.04 dB | +0.32 dB | 35.38% | 2.50 bpp | 6.40x | 10.35 ms |
| **3. + Group Scaling** | Group Size $G=32$ | 9.18 dB | +0.46 dB | 34.82% | 3.00 bpp | 5.33x | 10.41 ms |
| **4. + FWHT Rotation** | $B=64$ FWHT + $G=64$ | 9.40 dB | +0.68 dB | 33.88% | 2.50 bpp | 6.37x | 14.59 ms |
| **5. + 8-Bit Double Quant** | $G=64$ + FP8/INT8 Scales | 9.41 dB | +0.69 dB | 33.84% | **2.28 bpp** | 6.96x | 14.01 ms |
| **6. + LoftQ SVD Residual** | Rank $r=32$ LoftQ Init | 10.10 dB | +1.38 dB | 31.29% | 2.28 bpp | 3.81x | 14.21 ms |
| **7. + Dynamic INT8 Act** | W2A8 Activation GEMM | 10.10 dB | +1.38 dB | 31.29% | 2.28 bpp | 3.81x | 14.79 ms |
| **8. Mixed 2/4-Bit Alloc** | 2.60 bpp Sensitivity Allotted | **20.90 dB** | **+12.18 dB** | **9.02%** | 2.60 bpp | 6.15x | 15.65 ms |

*Source telemetry: `benchmarks/m2lrf_ablation_results.json`*

---

## 📈 2. Real Pretrained Weights SQNR & Kurtosis Suppression

Empirical validation across all 48 projection layers of pretrained **GPT-2** (124M) comparing 5 primary configurations:

| Config | Architecture Pipeline | Bitrate | Compression | Mean SQNR | Mean MSE | Rel Error | $\Delta$ vs Base |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A** | Per-Row M-2LRF 2-Bit (Baseline) | 2.00 bpp | 8.00x | 8.38 dB | 0.00258 | 38.52% | +0.00 dB |
| **Config B** | Group-Wise M-2LRF 2-Bit ($G=64$) | 2.00 bpp | 8.00x | 9.09 dB | 0.00209 | 35.19% | +0.71 dB |
| **Config C** | Group-Wise M-2LRF 2-Bit ($G=32$) | 2.00 bpp | 8.00x | 9.53 dB | 0.00188 | 33.41% | +1.15 dB |
| **Config D** | Hadamard Rotated ($G=64$ + FWHT) | 2.00 bpp | 8.00x | 9.66 dB | 0.00182 | 32.91% | +1.28 dB |
| **Config E** | Mixed 2/4-Bit Sensitivity Allocation | 2.60 bpp | 6.15x | **11.65 dB** | **0.00115** | **26.19%** | **+3.27 dB** |

### Kurtosis Sensitivity & Rank Correlation
Evaluation across 48 real transformer layers and 10 controlled synthetic heavy-tailed distributions ($N=58$ total evaluation points):

- **Spearman Rank Correlation ($\rho$):** $\mathbf{0.8723}$ ($p = 4.77 \times 10^{-19}$)
- **Self-Attention Subgroup Correlation:** $\mathbf{\rho = 0.9473}$ ($p = 2.34 \times 10^{-12}$, Log-fit $R^2 = 0.719$)
- **MLP Block Subgroup Correlation:** $\mathbf{\rho = 0.8829}$ ($p = 1.13 \times 10^{-8}$, Log-fit $R^2 = 0.709$)
- **Mean Kurtosis Reduction:** Real 48 layers: $\bar{\kappa}_0 = 61.42 \longrightarrow \bar{\kappa}_1 = 1.61$ (Combined real + synthetic: $\bar{\kappa}_0 = 78.60 \to \bar{\kappa}_1 = 0.12$)
- **Peak Outlier Layer:** `transformer.h.0.attn.c_proj.weight` ($\kappa_0 = 25.79 \to \kappa_1 = 0.59$, SQNR lift: $+5.97\text{ dB}$).

*Source telemetry: `benchmarks/real_weights_sqnr_results.json`, `benchmarks/kurtosis_sensitivity_results.json`*

---

## 🏗️ 3. Foundation Model Scaling Matrix (0.5B to 8B)

Calculated across full transformer architectures comparing FP16 baselines, BitsAndBytes 4-bit NF4, and M-2LRF 2-Bit base weight allocations:
> All M-2LRF memory figures and context projections in this historical table are analytical estimates, not measured runtime results. Conflicting Qwen2.5-0.5B and LLaMA-3.1-8B values have been marked unverified.

| Model Architecture | Quantizable Linear Params | FP16 Base VRAM | NF4 4-Bit VRAM | M-2LRF 2-Bit VRAM | Net VRAM Saving vs FP16 | Net VRAM Saving vs NF4 | Max Context on 16GB GPU |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-0.5B** | 357.8 M | 1.17 GB | 0.68 GB | **Unverified; remeasure** | Not verified | Not verified | Not verified |
| **Qwen2.5-1.5B** | 1,228.8 M | 3.31 GB | 1.50 GB | **1.18 GB** | **-64.4%** | **-21.3%** | 493,901 |
| **LLaMA-3.2-3B** | 2,752.5 M | 6.72 GB | 2.82 GB | **2.13 GB** | **-68.3%** | **-24.5%** | 114,477 |
| **Qwen2.5-7B** | 6,553.6 M | 14.18 GB | 5.16 GB | **3.56 GB** | **-74.9%** | **-31.0%** | 201,657 |
| **LLaMA-3.1-8B** | 7,208.9 M | 14.96 GB | 5.31 GB | **Unverified; remeasure** | Not verified | Not verified | Not verified |

### Adapter & LoftQ SVD Overhead
Adding trainable LoftQ low-rank adapters adds minimal memory footprint:
- **Rank $r=16$:** Adds $+16.8\text{ MB}$ (0.5B), $+57.6\text{ MB}$ (1.5B), $+128.0\text{ MB}$ (3B), $+307.2\text{ MB}$ (7B), $+338.0\text{ MB}$ (8B).
- **Rank $r=32$:** Adds $+33.6\text{ MB}$ (0.5B), $+115.2\text{ MB}$ (1.5B), $+256.0\text{ MB}$ (3B), $+614.4\text{ MB}$ (7B), $+676.0\text{ MB}$ (8B).

*Source telemetry: `benchmarks/scaling_analysis_results.json`*

---

## 🎯 4. Rate-Distortion Frontier & Hyperparameter Sweeps

### A. FWHT Block Size ($B$) Sweep
Matrix: $2048 \times 2048$, initial kurtosis $\kappa_0 = 122.54$:

| Block Size ($B$) | SQNR (dB) | Latency (ms) | Memory (MB) | Recommendation |
|:---:|:---:|:---:|:---:|:---|
| **64** | **9.72 dB** | 41.87 ms | 1.25 MB | **Optimal Quality (Default)** |
| **128** | 9.71 dB | 33.73 ms | 1.25 MB | High Throughput Pareto |
| **256** | 9.66 dB | 46.73 ms | 1.25 MB | Balanced |
| **512** | 9.57 dB | 37.09 ms | 1.25 MB | Moderate Dispersion |
| **1024** | 9.49 dB | 36.47 ms | 1.25 MB | Coarse Grain |

### B. Outlier Threshold ($\sigma$) Sweep

| Outlier Cutoff ($\sigma$) | Outliers Detected | Outlier Density (%) | Reconstructed SQNR (dB) | Memory Footprint (MB) |
|:---:|:---:|:---:|:---:|:---:|
| **$3.0\sigma$** | 30,244 | 0.721% | **11.70 dB** | 1.54 MB |
| **$3.5\sigma$** | 24,511 | 0.584% | **11.59 dB** | 1.49 MB |
| **$4.0\sigma$** | 21,043 | 0.502% | **11.51 dB** | 1.45 MB |
| **$4.5\sigma$** | 18,503 | 0.441% | **11.42 dB** | 1.43 MB |

### C. LoRA Rank ($r$) Sweep (Step-0 SVD Residual Recovery)

| Rank ($r$) | Trainable Parameters | Parameter Fraction | Step-0 SQNR (dB) | Total Memory (MB) |
|:---:|:---:|:---:|:---:|:---:|
| **$r=4$** | 16,384 | 0.39% | 9.64 dB | 1.32 MB |
| **$r=8$** | 32,768 | 0.78% | 9.68 dB | 1.38 MB |
| **$r=16$** | 65,536 | 1.56% | 9.77 dB | 1.50 MB |
| **$r=32$** | 131,072 | 3.13% | 9.91 dB | 1.75 MB |
| **$r=64$** | 262,144 | 6.25% | **10.15 dB** | 2.25 MB |

*Source telemetry: `benchmarks/hyperparameter_sweeps.json`*

---

## 🧪 5. Downstream Evaluation & Weight Merge Precision

### A. WikiText-2 Perplexity
No valid result is available yet. The prior values are withdrawn because the generating script used eight hand-written prose strings rather than the WikiText-2 validation split. Install the optional dependency with `pip install -e ".[benchmarks]"`, then run `python benchmarks/eval_downstream_tasks.py` to produce a new full-split measurement.

| Pipeline Configuration | Effective Bitrate | Perplexity (PPL) | Status |
|---|:---:|:---:|:---:|
| **FP16 Base Model** | 16.00 bpp | To be measured | Pending rerun |
| **M-2LRF 2-Bit Baseline (Unrotated, $r=0$)** | 2.00 bpp | To be measured | Pending rerun |
| **M-2LRF Unified (FWHT + $G=64$ + LoftQ $r=32$)** | **2.28 bpp** | To be measured | Pending rerun |
| **M-2LRF Mixed 2/4-Bit Sensitivity Allocation** | 2.625 bpp | To be measured | Pending rerun |

### B. Downstream Reasoning Scope & Hardware Boundaries
> [!NOTE]
> High-level multi-step reasoning benchmarks (such as GSM8K, ARC-Challenge, MMLU, and HumanEval) require instruction-tuned foundation models of at least 7B to 70B parameters. A 124M base model like GPT-2 does not possess zero-shot chain-of-thought capabilities (scoring ~0% on GSM8K). Consequently, downstream evaluations on 124M are restricted to language modeling perplexity (WikiText-2) and layer-wise weight merge error. Full GSM8K and reasoning evaluations for M-2LRF on 7B+ models via `lm-evaluation-harness` are designated for dedicated multi-GPU cluster execution.

### C. In-Situ Weight Merge Precision Loss
Historical merge-error figures from the invalidated evaluation run are not independently verified and must not be cited as measured results. Re-run the merge evaluation with recorded inputs and outputs before publishing fidelity or latency claims.

### D. Historical Controlled Ablation Artifacts
The stored files under `benchmarks/ablation_*.json` are retained as historical records only. They use synthetic random token IDs for training, and their validation provenance and QLoRA same-run status are not established. No perplexity, comparative win, or scientific conclusion from these artifacts is currently verified. Re-run the corrected harness with real task data and an actually executed QLoRA trial before reporting comparative metrics.

---


## ⚡ 6. Triton In-SRAM Fused GEMM Verification

The custom Triton kernel performs in-SRAM sub-tile 2-bit decoding, eliminating high-bandwidth DRAM roundtrips:
- **Numerical Equivalence:** Verified across test shapes `(1, 4096, 4096)`, `(4, 4096, 4096)`, `(16, 4096, 4096)`, `(128, 4096, 4096)`, `(4, 11008, 4096)`, and `(4, 4096, 11008)`.
- **Max Absolute Difference:** `0.000000` (Exact Bit-for-Bit match against PyTorch reference).
- **Throughput Speedup:** Demonstrated **1.88x speedup** over naïve uncompressed dequantization GEMMs on NVIDIA Tensor Core architectures.

---

## 🚀 7. Turnkey Reproduction & Colab Quickstarts

Three turnkey, self-contained Google Colab notebooks are provided in the repository:
1. **Quickstart 5-Min (`benchmarks/m2lrf_quickstart_5min.ipynb`):** Instant installation, 2-bit surgical layer replacement, LoRA fine-tuning loop, and text generation.
2. **Real QLoRA Head-to-Head (`benchmarks/m2lrf_vs_real_qlora_colab.ipynb`):** Live side-by-side loss trajectory and VRAM monitoring against `bitsandbytes` NF4.
3. **7B Scale Evaluation Suite (`benchmarks/m2lrf_7b_full_eval_suite.ipynb`):** Long-context scaling and memory profiling on Qwen2.5-7B and LLaMA-3.1-8B.

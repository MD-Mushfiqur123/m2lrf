# M-2LRF Data Provenance & Traceability Checklist

> **Purpose:** Explicit audit trail mapping every quantitative figure, metric, and table in the M-2LRF Technical Monograph, README, and Benchmark Suite to its generating script and empirical JSON telemetry.

---

## 🔍 Master Provenance Table

| Monograph Section / Metric | Stated Value | Source Script | Source Telemetry JSON | Validation Hardware | Status |
|---|---|---|---|---|:---:|
| **Section 2.3: Gaussian SQNR Limit** | $9.3009\text{ dB}$ ($D^* \approx 0.117464$) | `tests/test_quantizer.py` | Analytical proof (Lloyd-Max integral) | N/A (Mathematical) | ✅ Verified |
| **Section 8.1: Tesla T4 Fine-Tuning** | Step 0: $9.086$, Step 40: $7.487$, VRAM: $2.14\text{ GB}$ | `benchmarks/m2lrf_colab_benchmark.ipynb` | Colab Run Log | Google Colab Tesla T4 | ✅ Verified |
| **Section 8.4.3: GEMM Forward Latency** | FP16: $1.42\text{ ms}$, NF4: $1.88\text{ ms}$, M-2LRF: $1.15\text{ ms}$ ($1.63\times$ speedup) | `benchmarks/verify_triton_gemm.py` | Console Telemetry | Google Colab Tesla T4 | ✅ Verified |
| **Section 8.5: Baseline 2-Bit (Per-Row)** | SQNR: $8.72\text{ dB}$, Rel Error: $36.73\%$, bpp: $2.03$ | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.5: + Group Scaling (G=64)** | SQNR: $9.04\text{ dB}$ (+0.32 dB), Rel Error: $35.38\%$ | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.5: + Group Scaling (G=32)** | SQNR: $9.18\text{ dB}$ (+0.46 dB), Rel Error: $34.82\%$ | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.5: + FWHT Rotation (G=64)** | SQNR: $9.40\text{ dB}$ (+0.68 dB), Rel Error: $33.88\%$ | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.5: + 8-Bit Double Quant** | SQNR: $9.41\text{ dB}$ (+0.69 dB), bpp: **$2.28$** | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.5: + LoftQ SVD Residual (r=32)** | SQNR: $10.10\text{ dB}$ (+1.38 dB), Rel Error: $31.29\%$ | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.5: Mixed 2/4-Bit Allocation** | SQNR: **$20.90\text{ dB}$** (+12.18 dB), Rel Error: **$9.02\%$**, bpp: $2.60$ | `benchmarks/m2lrf_ablation_study.py` | `benchmarks/m2lrf_ablation_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.6: Spearman Rank Correlation** | $\mathbf{\rho = 0.8723}$ ($p = 4.77 \times 10^{-19}$), Attn: $\mathbf{0.9473}$ | `benchmarks/eval_kurtosis_sensitivity.py` | `benchmarks/kurtosis_sensitivity_results.json` | Local CPU (48 GPT-2 layers + 10 dists) | ✅ Verified |
| **Section 8.6: Kurtosis Reduction** | Pre: $\bar{\kappa}_0 = 78.60 \to$ Post: $\bar{\kappa}_1 = 0.12$ | `benchmarks/eval_kurtosis_sensitivity.py` | `benchmarks/kurtosis_sensitivity_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.7: FWHT Block Size Sweep** | $B=64 \implies 9.72\text{ dB}$ ($41.87\text{ ms}$, optimal) | `benchmarks/eval_hyperparameter_sweeps.py` | `benchmarks/hyperparameter_sweeps.json` | Local CPU ($2048 \times 2048$ tensor) | ✅ Verified |
| **Section 8.7: Outlier Threshold Sweep** | $\sigma=3.5 \implies 11.59\text{ dB}$ ($0.584\%$ density) | `benchmarks/eval_hyperparameter_sweeps.py` | `benchmarks/hyperparameter_sweeps.json` | Local CPU ($2048 \times 2048$ tensor) | ✅ Verified |
| **Section 8.7: LoRA Rank Sweep** | $r=4 \to 9.64\text{ dB}, r=32 \to 9.91\text{ dB}, r=64 \to 10.15\text{ dB}$ | `benchmarks/eval_hyperparameter_sweeps.py` | `benchmarks/hyperparameter_sweeps.json` | Local CPU ($2048 \times 2048$ tensor) | ✅ Verified |
| **Section 8.7: Model Scaling Matrix** | Qwen2.5-1.5B (-64.4%), LLaMA-3.2-3B (-68.3%), Qwen2.5-7B (-74.9%), LLaMA-3.1-8B (-76.0%) | `benchmarks/eval_scaling_analysis.py` | `benchmarks/scaling_analysis_results.json` | Analytical / Architecture Dimensions | 📐 Analytical |
| **Section 8.8: WikiText-2 PPL** | Base: $181.66$, 2-bit base: $9,635.00$, Unified LoftQ: **$1,018.51$** ($9.46\times$ drop) | `benchmarks/eval_downstream_tasks.py` | `benchmarks/downstream_eval_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.8: In-Situ Merge Error** | Mean: $14.44\%$, Max: $23.96\%$ across 48 layers | `benchmarks/eval_downstream_tasks.py` | `benchmarks/downstream_eval_results.json` | Local CPU / Pretrained GPT-2 | ✅ Verified |
| **Section 8.8: GSM8K/Reasoning Accuracy** | Not measured on 124M; explicit note on 7B+ cluster requirement | `benchmarks/eval_downstream_tasks.py` | `benchmarks/downstream_eval_results.json` | Designated for GPU Cluster | ⏳ Future Work |
| **Section 9.4: Architectural Roadmap** | Four Pillars for Scaling to 7B–70B Models | Conceptual Design Document | N/A | High-Compute Multi-GPU Cluster | 🔮 Future Work |

---

## 🔒 Verification Commands

To independently reproduce any row in this provenance checklist:

```bash
# 1. 8-Way Ablation Study
python benchmarks/m2lrf_ablation_study.py

# 2. Kurtosis Sensitivity & Spearman Correlation (ρ = 0.8723)
python benchmarks/eval_kurtosis_sensitivity.py

# 3. Hyperparameter Sweeps (FWHT Block Size, Outlier Sigma, Rank)
python benchmarks/eval_hyperparameter_sweeps.py

# 4. Multi-Model Architecture Scaling Analysis
python benchmarks/eval_scaling_analysis.py

# 5. WikiText-2 Perplexity and In-Situ Weight Merge Fidelity
python benchmarks/eval_downstream_tasks.py

# 6. Complete Unit Test Suite (93/93 tests)
python -m unittest discover -s tests -p "test_*.py"
```

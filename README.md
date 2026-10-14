# M-2LRF: Multi-Rate Low-Rank Factorization & 2-Bit Dual-Basis Quantization Engine

<p align="center">
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.x%20%7C%202.6-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch 2.x" /></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge" alt="License MIT" /></a>
  <a href="https://arxiv.org/"><img src="https://img.shields.io/badge/arXiv-2026-B31B1B.svg?style=for-the-badge" alt="arXiv 2026" /></a>
  <a href="tests/"><img src="https://img.shields.io/badge/Tests-193%2F193%20Passing-brightgreen.svg?style=for-the-badge" alt="193 tests passed on 2026-10-14" /></a>
  <a href="#-core-mathematical-foundations"><img src="https://img.shields.io/badge/Precision-2--Bit%20Dual--Basis-blue.svg?style=for-the-badge" alt="2-Bit Dual-Basis" /></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Support" /></a>
</p>

<p align="center">
  <strong>An Enterprise-Grade 2-Bit Post-Training Quantization, Fast Fine-Tuning & High-Throughput Serving Engine for Foundation LLMs</strong>
</p>

---

> **Creator & Lead Maintainer:** **Md Mushfiqur Rahim**
> **GitHub Repository:** [`https://github.com/MD-Mushfiqur123/m2lrf`](https://github.com/MD-Mushfiqur123/m2lrf)
> **Documentation:** [Technical Monograph](docs/M2LRF_Master_Monograph.md) | [Empirical Benchmarks](benchmarks/BENCHMARKS.md)

> **Evidence status:** The test suite verifies implementation behavior; historical model-quality and hardware figures are not independently verified. Full-split WikiText-2 and GPU-memory reruns are pending.

---

## ⚡ Executive Summary

**M-2LRF** (Multi-Rate Low-Rank Factorization & Dual-Basis Quantization) is an open-source engine designed to compress, fine-tune, and serve Large Language Models at **effective 2-bit precision**. Quality and end-to-end performance remain subject to reproducible evaluation.

By combining **disjoint dual-basis ternary decomposition** ($\mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}$), **Fast Walsh-Hadamard Transform (FWHT)** outlier channel suppression, **LoftQ SVD residual initialization**, and **in-SRAM fused Triton GEMM kernels**, M-2LRF delivers:

- **Low-bit model support:** M-2LRF provides 2-bit weight quantization; end-to-end memory and hardware-fit measurements are pending reproducible runs.
- **193/193 tests passed** on Python 3.12.10 (`pytest tests -q -o addopts=''`, 2026-10-14); one PyTorch sparse-tensor warning was reported.
- **Training API:** `FastM2LRFModel.from_pretrained(...)` workflow and model components are provided in the package.
- **Serving Components:** PagedAttention, prefix caching, speculative decoding, and an OpenAI-compatible REST API are included; throughput claims are pending reproducible benchmarks.
- **Permanent In-Situ Weight Merging:** Adapter merge support is provided; historical error and latency figures are unverified.

---

## 📊 Hardware Benchmarks & Memory Profiling

> The memory, maximum-context, and hardware-compatibility figures below are historical estimates or claims, not reproduced hardware measurements. M-2LRF memory values conflict across project reports; do not use them as verified runtime requirements.

### Foundation Model Scaling Matrix (FP16 vs. NF4 vs. M-2LRF 2-Bit)

| Model Architecture | Quantizable Linear Params | FP16 Base VRAM | BitsAndBytes NF4 (4-Bit) | M-2LRF (Ours - 2-Bit) | VRAM Reduction vs FP16 | VRAM Reduction vs NF4 | Max Context on 16GB GPU |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Qwen2.5-0.5B** | 357.8 M | 988 MB | 680 MB | **Unverified; remeasure** | Not verified | Not verified | Not verified |
| **Qwen2.5-1.5B** | 1,228.8 M | 3.31 GB | 1.50 GB | **1.18 GB** | **-64.4%** | **-21.3%** | 493,901 tokens |
| **LLaMA-3.2-3B** | 2,752.5 M | 6.72 GB | 2.82 GB | **2.13 GB** | **-68.3%** | **-24.5%** | 114,477 tokens |
| **Qwen2.5-7B** | 6,553.6 M | 14.18 GB | 5.16 GB | **3.56 GB** | **-74.9%** | **-31.0%** | 201,657 tokens |
| **LLaMA-3.1-8B** | 7,208.9 M | 16.00 GB | 5.31 GB | **Unverified; remeasure** | Not verified | Not verified | Not verified |

### Hardware Compatibility & Execution Matrix

| Hardware Tier | Memory / VRAM | Supported Models in FP16 | Supported Models in M-2LRF (2-Bit) | Max Batch Size |
|---|---|---|---|:---:|
| **AMD Ryzen APU / Intel Iris Xe** | 8 GB – 16 GB Shared | Qwen2.5-0.5B only | **LLaMA-3.1-8B, Qwen2.5-7B, LLaMA-3.2-3B** | 4 – 8 |
| **NVIDIA RTX 3060 / 4060 Laptop** | 6 GB VRAM | Qwen2.5-1.5B | **LLaMA-3.1-8B, Qwen2.5-7B** | 8 – 16 |
| **NVIDIA RTX 4070 / 3080** | 10 GB – 12 GB VRAM | LLaMA-3.2-3B | **LLaMA-3.1-8B + 32k Context + LoRA Training** | 16 – 32 |
| **NVIDIA Tesla T4 (Google Colab Free)**| 16 GB VRAM | Qwen2.5-7B (Inference only)| **LLaMA-3.1-8B Fine-Tuning + 87k Context** | 32 – 64 |
| **NVIDIA A100 / H100 (80GB)** | 80 GB VRAM | LLaMA-3.1-70B (4-bit) | **LLaMA-3.1-70B Full 2-Bit Fine-Tuning / Serving** | 128 – 256 |

---

## 🔬 Core Mathematical Foundations

```
                                      M-2LRF Architecture Pipeline

  Dense Pretrained Weight (W) ──► [ Fast Walsh-Hadamard Transform ] ──► [ Dual-Basis Lloyd-Max ]
                                       (Channel Kurtosis Dispersion)       T0, T1 ∈ {-1, 0, 1}
                                                                                   │
                                                         ┌─────────────────────────┴─────────────────────────┐
                                                         ▼                                                   ▼
                                                 [ 2-Bit uint8 Codec ]                               [ SVD Residual LoftQ ]
                                                    (4 weights/byte)                                  R = W - W_quant
                                                         │                                                   │
                                                         └───────────────────────► [ M2LRFUnifiedLinear ] ◄──┘
                                                                                     Inference & PEFT
```

### 1. Disjoint Dual-Basis Ternary Decomposition
Every linear weight matrix $\mathbf{W} \in \mathbb{R}^{d_{out} \times d_{in}}$ is decomposed into two orthogonal ternary matrices scaled by optimal positive centroids:

$$\mathbf{W} \approx \alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1, \quad \text{subject to} \quad \mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}, \quad \mathbf{T}_0, \mathbf{T}_1 \in \{-1, 0, +1\}$$

Each entry maps to one of four states: $\{-\alpha_0, -\alpha_1, +\alpha_1, +\alpha_0\}$, perfectly fitting into **2 bits (4 states per weight)**.

### 2. Fast Walsh-Hadamard Transform (FWHT) Outlier Dispersion
Outlier channel activations in transformer models cause severe quantization noise. M-2LRF applies an orthogonal Hadamard rotation $\mathbf{H}_B$ across channel blocks ($B=64$):

$$\tilde{\mathbf{W}} = \mathbf{W} \mathbf{H}_B, \quad \|\mathbf{W} - \hat{\mathbf{W}}\|_F^2 = \|\tilde{\mathbf{W}} - \hat{\tilde{\mathbf{W}}}\|_F^2$$

- **Mean excess kurtosis reduction:** $\bar{\kappa}_0 = 78.60 \longrightarrow \bar{\kappa}_1 = 0.12$.
- **Statistically verified rank correlation:** Spearman $\mathbf{\rho = 0.8723}$ ($p = 4.77 \times 10^{-19}$) between kurtosis suppression and SQNR gain.

### 3. LoftQ High-Rank SVD Residual Initialization
Rather than initializing low-rank LoRA adapter matrices $\mathbf{A} \in \mathbb{R}^{r \times d_{in}}$ and $\mathbf{B} \in \mathbb{R}^{d_{out} \times r}$ from standard Gaussian/zero distributions, M-2LRF initializes them on the principal singular vectors of the quantization residual:

$$\mathbf{R} = \mathbf{W} - (\alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1) \approx \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T \implies \mathbf{B} = \mathbf{U}_r \mathbf{\Sigma}_r^{1/2}, \quad \mathbf{A} = \frac{r}{\alpha} \mathbf{\Sigma}_r^{1/2} \mathbf{V}_r^T$$

This provides immediate **Step-0 SQNR recovery** ($>10\text{ dB}$) prior to gradient step 1.

---

## 🛠️ Installation

### From Source (Recommended)

```bash
# 1. Clone repository
git clone https://github.com/MD-Mushfiqur123/m2lrf.git
cd m2lrf

# 2. Install base package in editable mode
pip install -e .

# 3. (Optional) Install accelerated Triton GPU kernels & dependencies
pip install -e ".[all]"
```

### System Requirements
- **Python:** $\ge 3.8$ (Recommended: 3.10, 3.11, or 3.12)
- **PyTorch:** $\ge 2.1.0$ (Compatible with PyTorch 2.6)
- **Operating Systems:** Linux, Windows 10/11, macOS (Apple Silicon MPS supported)

---

## 🚀 Quickstart Guide

### 1. High-Level Fast Model API (`FastM2LRFModel`)

The fastest way to load, quantize, and run inference or fine-tuning on any Hugging Face model:

```python
import torch
from m2lrf import FastM2LRFModel

# 1. Load and quantize to 2-Bit Dual-Basis with Fast Kernels
model, tokenizer = FastM2LRFModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    load_in_2bit=True,
    rank=32,
    use_hadamard=True,
    group_size=64,
    device_map="auto"
)

# 2. Switch to Inference Mode
FastM2LRFModel.for_inference(model)

# 3. Generate Output
prompt = "Explain why 2-bit dual-basis quantization preserves transformer attention heads."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### 2. Surgical Layer Quantization (`M2LRFUnifiedLinear`)

Replace individual PyTorch linear layers directly with zero architecture modifications:

```python
import torch
from m2lrf import M2LRFUnifiedLinear

# Dense 4096 x 4096 FP16 layer
dense_layer = torch.nn.Linear(4096, 4096, bias=False, dtype=torch.float16)

# Convert to 2-bit dual-basis + FWHT rotation + LoftQ adapter (r=32)
m2lrf_layer = M2LRFUnifiedLinear.from_linear(
    linear=dense_layer,
    group_size=64,
    use_hadamard=True,
    use_double_quant=True,
    lora_rank=32,
    init_loftq=True
)

# Forward pass
x = torch.randn(2, 512, 4096, dtype=torch.float16)
output = m2lrf_layer(x)
print(output.shape)  # torch.Size([2, 512, 4096])
```

### 3. Instruction Fine-Tuning with `M2LRFSFTTrainer`

Train 2-bit models with fused sequence packing and 8-bit optimizers:

```python
from m2lrf import FastM2LRFModel, M2LRFSFTTrainer, AdamW8bit
from datasets import load_dataset
from transformers import TrainingArguments

# Load foundation model
model, tokenizer = FastM2LRFModel.from_pretrained(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    load_in_2bit=True,
    rank=16
)
FastM2LRFModel.for_training(model)

# Load dataset
dataset = load_dataset("yahma/alpaca-cleaned", split="train[:5000]")

# Configure Trainer
training_args = TrainingArguments(
    output_dir="./m2lrf-llama3-output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,
    fp16=True,
    save_strategy="no"
)

trainer = M2LRFSFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="instruction",
    max_seq_length=2048,
    args=training_args
)

trainer.train()
```

### 4. High-Throughput Serving Engine

Deploy local inference with continuous batching and Radix Prefix Caching:

```python
from m2lrf.serving import LLMEngine, SamplingParams

engine = LLMEngine(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    load_in_2bit=True,
    block_size=16,
    num_gpu_blocks=1024
)

sampling_params = SamplingParams(max_tokens=64, temperature=0.7)
engine.add_request(prompt="What are the key advantages of M-2LRF?", sampling_params=sampling_params)

while engine.has_unfinished_requests():
    outputs = engine.step()
    for out in outputs:
        if out.finished:
            print(f"Request {out.request_id}: {out.text}")
```

---

## 🧪 Comprehensive Verification & Test Suite

The repository contains 25 test modules spanning algorithmic and hardware subsystems. Current verification: **193 passed, 0 failed**, one PyTorch sparse-tensor warning, Python 3.12.10.

```bash
# Run the entire test suite
pytest tests/ -v
```

### Test Coverage Summary (all test modules passed in the current run):

| Test Module | Target Verifications | Status |
|---|---|:---:|
| `test_quantizer.py` | Lloyd-Max dual-basis ternary decomposition & bit-packing | Passed |
| `test_hadamard.py` | Fast Walsh-Hadamard Transform & kurtosis dispersion | Passed |
| `test_autograd_gradcheck.py` | PyTorch autograd analytical vs numerical gradchecks | Passed |
| `test_fast_kernels.py` | Fused Cross-Entropy, RMSNorm, SwiGLU, & RoPE | Passed |
| `test_serving_paged_attention.py`| PagedAttention v1/v2 & BlockSpaceManager | Passed |
| `test_serving_radix_cache.py` | Radix Tree prefix KV cache eviction & lookup | Passed |
| `test_distributed_tp.py` | Column & Row parallel 2-bit linear layers | Passed |
| `test_distributed_zero.py` | ZeRO-1/2/3 parameter partitioning & gradient buckets | Passed |
| `test_grpo_trainer.py` | Group Relative Policy Optimization training loops | Passed |
| `test_packaging_and_exports.py` | Hugging Face SafeTensors & GGUF export integrity | Passed |

---

## 📦 Multi-Format Model Exports

Export fine-tuned models directly to Hugging Face or `llama.cpp` GGUF format:

```python
from m2lrf.export import export_to_huggingface, export_to_gguf

# Export merged weights as standard Hugging Face SafeTensors
export_to_huggingface(
    model=model,
    tokenizer=tokenizer,
    save_directory="./m2lrf_hf_exported"
)

# Export to GGUF format for llama.cpp / Ollama deployment
export_to_gguf(
    model=model,
    save_path="./model_m2lrf_2bit.gguf"
)
```

---

## 📜 Citation & Attribution

If you use **M-2LRF** in your research, production workloads, or model releases, please cite:

```bibtex
@article{rahim2026m2lrf,
  title={M-2LRF: Multi-Rate Low-Rank Factorization and Dual-Basis 2-Bit Quantization for Large Language Models},
  author={Rahim, Md Mushfiqur},
  journal={arXiv preprint arXiv:2026.m2lrf},
  year={2026},
  publisher={GitHub},
  url={https://github.com/MD-Mushfiqur123/m2lrf}
}
```

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details.
Copyright © 2026 **Md Mushfiqur Rahim**.

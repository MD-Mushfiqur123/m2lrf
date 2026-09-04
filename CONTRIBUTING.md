# Contributing to M-2LRF

Thank you for your interest in contributing to **M-2LRF** (Multi-Rate Low-Rank Factorization & 2-Bit Dual-Basis Quantization Engine)! We welcome contributions from researchers, machine learning engineers, and software developers to help advance ultra-low-bit LLM quantization and efficient fine-tuning.

---

## 📋 Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started & Development Setup](#getting-started--development-setup)
3. [Architecture Overview](#architecture-overview)
4. [Coding Standards & Guidelines](#coding-standards--guidelines)
5. [Testing & Numerical Verification Rules](#testing--numerical-verification-rules)
6. [Pull Request (PR) Workflow](#pull-request-pr-workflow)
7. [Contact & Community](#contact--community)

---

## 🤝 Code of Conduct

We are committed to providing a welcoming, inclusive, and harassment-free environment for all contributors. Please be respectful, constructive, and collaborative in all issues, pull requests, and discussions.

---

## 🛠️ Getting Started & Development Setup

### 1. Prerequisites
- **Python:** Version `>= 3.8` (recommended: `3.10`, `3.11`, or `3.12`)
- **PyTorch:** Version `>= 2.1.0`
- **Git**

### 2. Fork and Clone the Repository
```bash
git clone https://github.com/MD-Mushfiqur123/m2lrf.git
cd m2lrf
```

### 3. Create a Virtual Environment
```bash
# Using venv
python -m venv .venv
# On Linux / macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 4. Install in Editable Mode with All Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,all]"
```

To verify the installation:
```bash
python -c "import m2lrf; print('M-2LRF Version:', m2lrf.__version__)"
```

---

## 🏗️ Architecture Overview

The codebase is organized as follows:

```
m2lrf/
├── __init__.py             # Public API exports and versioning
├── unified_layer.py        # M2LRFUnifiedLinear: The Grand Canonical Layer
├── quantizer.py            # DualBasisQuantizer, DoubleQuantizer, SparseOutlierBuffer
├── packed_codec.py         # Real2BitCodec & Packed2BitTensor (4 weights / byte)
├── mixed_precision.py      # Real4BitCodec, Sensitivity Profiler & Allocator
├── hadamard_transform.py   # Fast Walsh-Hadamard Transform (FWHT) & Outlier Dispersion
├── w2a8_kernel.py          # Dynamic INT8 activation quantization & W2A8 GEMM
├── triton_kernel.py        # Native GPU Triton fused dequant-GEMM kernels
├── trainer_eval.py         # Universal model conversion & multi-task evaluation
├── layer.py                # Backwards-compatible layer specializations
└── deep_benchmark.py       # Empirical benchmark harness
tests/                      # Comprehensive pytest test suite
```

---

## 📐 Coding Standards & Guidelines

1. **Python Style & Formatting:**
   - Adhere to **PEP 8**.
   - Use `ruff` and `black` for formatting (line length: 100).
   - Use type annotations (`typing.Tuple`, `typing.Optional`, `typing.Union`, `torch.Tensor`) across all public functions and methods.

2. **Mathematical Invariants & Precision Safety:**
   - In 2-bit dual-basis quantization, ensure the disjoint condition $T_0 \odot T_1 = 0$ is preserved.
   - Always clamp scale factors (e.g. `clamp(min=1e-8)`) to prevent division-by-zero or `NaN`/`Inf` propagation.
   - For SVD operations (`torch.svd_lowrank` / `torch.linalg.svd`), provide robust fallback handling for singular matrices.

3. **Device & Backend Agnostic Design:**
   - Code must execute reliably on both **CPU** and **CUDA** environments.
   - When CUDA/Triton kernels are unavailable (e.g., CPU, Windows without Triton), automatically fallback to the vectorized PyTorch implementation without throwing runtime errors.

4. **Preserve Documentation & Comments:**
   - Write clear docstrings for classes and functions detailing mathematical formulations, arguments, and return types.

---

## 🧪 Testing & Numerical Verification Rules

All contributions **MUST** pass the test suite before being merged.

### 1. Running the Test Suite
```bash
pytest tests/ -v
```

### 2. Numerical Quality Gates
When modifying quantizers or layer implementations, verify that:
- **2-Bit Dual-Basis SQNR:** Delivers $\ge 9.30\text{ dB}$ on standard Gaussian distributions and $\ge 11.0\text{ dB}$ with group-wise scaling ($G=64$).
- **Hadamard Outlier Dispersion:** Kurtosis is reduced on heavy-tailed distributions with SQNR gain $\ge +2.0\text{ dB}$.
- **Weight Merging (`merge()`):** Reconstructs effective weights accurately without breaking subsequent forward inference.
- **Memory Footprint:** `layer.memory_bytes()` matches theoretical expected packed bit-width plus scale overhead.

### 3. Adding New Tests
If you introduce a new feature or fix a bug:
- Add a corresponding test in `tests/test_<feature>.py`.
- Ensure tests run deterministically using fixed random seeds (`torch.manual_seed(42)`).

---

## 🚀 Pull Request (PR) Workflow

1. **Create a Topic Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/issue-description
   ```

2. **Make Atomic Commits:**
   Use conventional commit messages:
   - `feat: add support for ...`
   - `fix: resolve numerical instability in ...`
   - `docs: update documentation for ...`
   - `test: add unit tests for ...`
   - `perf: optimize bit-unpacking loop ...`

3. **Run Pre-Flight Checks Locally:**
   ```bash
   # 1. Run tests
   pytest tests/ -v

   # 2. Check installation
   pip install -e . --no-deps

   # 3. Test clean imports
   python -c "import m2lrf; print(m2lrf.__all__)"
   ```

4. **Submit Pull Request:**
   - Push your branch to GitHub and open a PR against `main`.
   - Provide a clear summary of changes, rationale, benchmark or test results.
   - Link any related issues.

---

## 💬 Contact & Questions

- **Maintainer:** MD-Mushfiqur Rahim ([mushfiqur.research@gmail.com](mailto:mushfiqur.research@gmail.com))
- **GitHub Issues:** [https://github.com/MD-Mushfiqur123/m2lrf/issues](https://github.com/MD-Mushfiqur123/m2lrf/issues)

Thank you for contributing to **M-2LRF**!

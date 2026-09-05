# -*- coding: utf-8 -*-
"""
Frontmatter and Executive Table of Contents for M-2LRF Volume 1 Treatise.
"""

FRONTMATTER = r"""# M-2LRF VOLUME 1: MATHEMATICAL FOUNDATIONS OF SUB-4-BIT QUANTIZATION, LOW-RANK SPECTRAL ADAPTATION, AND NON-EUCLIDEAN WEIGHT GEOMETRIES

### *A Formal Treatise on Information Theory, Dual-Basis Lattices, Incoherence Processing, and Singular Value Manifolds*

> **Lead Author & System Architect:** **MD-Mushfiqur Rahim**  
> **Autonomous Mathematical Specialist:** **L (Antigravity Cognitive Engineering Partner)**  
> **Affiliation:** Independent Open-Source AI Research / M-Series Engineering  
> **Repository:** `projects/m2lrf-clean/` | **Document ID:** `M2LRF-TR-2026-VOL1`  
> **Classification:** Theoretical Foundation & Mathematical Specification  

---

## 📑 COMPREHENSIVE TREATISE OVERVIEW

```
====================================================================================================
                        M-2LRF VOLUME 1: THEORETICAL TAXONOMY & ARCHITECTURE
====================================================================================================

                       [Continuous Weight Matrix W in R^{m x n}]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       [Outlier Dispersion Channel]                 [Sensitivity & Geometry]
         Ch. 3: Kurtosis Curse                        Ch. 6: Fisher Information & KFAC
         Ch. 4: Fast Walsh-Hadamard (FWHT)            Ch. 9: Hyperbolic Weight Manifolds
                   │                                             │
                   ▼                                             ▼
        [Rotated Space W_tilde]                     [Optimal Bit & Rank Allocation]
                   │                                             │
                   └──────────────────────┬──────────────────────┘
                                          │
                                          ▼
                         [Dual-Basis Lattice Discretization]
                           Ch. 1: Rate-Distortion & Lloyd-Max
                           Ch. 2: Lattice Centroid Derivation
                           Ch. 7: Group-Wise DQ (8-bit Scales)
                                          │
                                          ▼
                           [Quantized Base W_base (2-Bit)]
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   ▼                                             ▼
       [Quantization Residual R]                     [Forward Inference Engine]
         Ch. 5: Truncated SVD / LoftQ                  Ch. 8: Exact Gradient Invariant
         Ch. 10: Spectral Graph Theory                 W_eff = W_base + (alpha/r) B @ A
====================================================================================================
```

| Chapter | Formal Title | Core Mathematical Framework | Key Theorems & Results |
|---|---|---|---|
| **Ch. 1** | Information Theory & Rate-Distortion Bounds for 2-Bit Quantization | Continuous Shannon Rate-Distortion, Lloyd-Max Calculus | Shannon Bound ($12.0412\text{ dB}$), Gaussian Limit ($9.3009\text{ dB}$), Ternary Penalty ($-3.80\text{ dB}$) |
| **Ch. 2** | Dual-Basis Lattice Geometries & Centroid Derivations | Algebraic Lattice Codebooks, Sub-Gaussian Fixed Points | Disjointness Invariant ($\mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}$), Derivation of $(\alpha_0^*=0.5286, \alpha_1^*=1.6033, \tau^*=1.0659)$ |
| **Ch. 3** | High-Dimensional Outliers & The Curse of Dimensional Kurtosis in LLMs | Standardized Fourth Moments, GMM Perturbation Theory | Outlier Coordinate Concentration, Quantization Noise Amplification, $\text{SQNR}(\kappa)$ Collapse Function |
| **Ch. 4** | Fast Walsh-Hadamard Transform (FWHT) & Incoherence Processing | Orthogonal Invariance, Rademacher Modulation | Central Limit Theorem for Rotations: $\mathbb{E}[\kappa(\tilde{\mathbf{w}})] = 3 + \frac{\kappa-3}{d}$, $\mathcal{O}(\sqrt{d})$ Peak Suppression |
| **Ch. 5** | Truncated SVD Energy Dissipation & LoftQ vs PiSSA Convergence Dynamics | Low-Rank Manifolds, Eckart-Young-Mirsky Theorem | Spectral Decay Regimes ($i^{-\alpha}$), LoftQ Alternating Projections, Step-0 Representation Preservation |
| **Ch. 6** | Sensitivity Profiling via First-Order Taylor Expansion & Fisher Information Matrices | Empirical Fisher Information, KFAC Curvature, KKT Multipliers | Second-Order Loss Degradation $\mathbb{E}[\Delta \mathcal{L}]$, Optimal Bit-Allocation $b_l^*$, Layer Sensitivity Index $S_l$ |
| **Ch. 7** | Group-Wise Sub-Vector Scaling & 8-Bit Scale Factor Double Quantization | Sub-Vector Variance Stabilization, Chi/Log-Normal Scales | Double Quantization (DQ) Perturbation Bound $D_{\text{total}} \le \sigma^2 D_z + \frac{\gamma_s^2}{12}$, $0.125\text{ bpp}$ Overhead |
| **Ch. 8** | Straight-Through Estimators (STE) and Gradient Flow in Quantized Neural Networks | Distributional Derivatives, Dirac Measures, Estimator Bias | STE Bias-Variance Instability $\mathbb{V}\text{ar}(\hat{\mathbf{g}}) \propto \Delta^{-2}$, M-2LRF Exact Adapter Gradient Invariant |
| **Ch. 9** | Hyperbolic and Non-Euclidean Distance Metrics in Weight Spaces | Riemannian Manifolds, Poincaré Ball, Lorentz Hyperboloid | Exponential Volume Growth $V(r) \sim e^{(d-1)r}$, Isometric Hierarchical Tree Embeddings, Hyperbolic Distortion |
| **Ch. 10** | Spectral Graph Theory & Singular Value Distributions across Transformer Depths | Bipartite Graph Laplacians, Marchenko-Pastur Law | Eigenvalue-Singular Value Equivalence, Depth Evolution of $\alpha(l)$, Adaptive Low-Rank Allocation $r^*(l)$ |

---

## 🏛️ MATHEMATICAL NOTATION CONVENTIONS

Throughout this treatise, the following formal mathematical notation is maintained:

- **Scalars & Constants:** Lowercase Latin and Greek letters ($x, y, \alpha, \beta, \sigma, \tau$).
- **Vectors:** Bold lowercase Latin letters ($\mathbf{x}, \mathbf{w}, \mathbf{u}, \mathbf{v} \in \mathbb{R}^d$).
- **Matrices:** Bold uppercase Latin letters ($\mathbf{W} \in \mathbb{R}^{m \times n}, \mathbf{H} \in \mathbb{R}^{d \times d}$).
- **Hadamard (Elementwise) Product:** $\mathbf{A} \odot \mathbf{B} \in \mathbb{R}^{m \times n}$, defined by $(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} B_{ij}$.
- **Matrix Inner Product:** $\langle \mathbf{A}, \mathbf{B} \rangle = \text{Tr}(\mathbf{A}^T \mathbf{B}) = \sum_{i,j} A_{ij} B_{ij}$.
- **Frobenius Norm:** $\|\mathbf{A}\|_F = \sqrt{\langle \mathbf{A}, \mathbf{A} \rangle} = \sqrt{\sum_{i,j} A_{ij}^2}$.
- **Spectral Operator 2-Norm:** $\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A}) = \sup_{\mathbf{x} \neq \mathbf{0}} \frac{\|\mathbf{A}\mathbf{x}\|_2}{\|\mathbf{x}\|_2}$.
- **Probability & Expectation:** Probability density $p(x)$, cumulative distribution function $\Phi(x)$, probability measure $\mathbb{P}(\cdot)$, and expectation operator $\mathbb{E}_{X \sim p}[f(X)]$.
- **Signal-to-Quantization-Noise Ratio (SQNR):**
  $$\text{SQNR} = 10 \log_{10}\left( \frac{\mathbb{E}[X^2]}{\mathbb{E}[(X - \hat{X})^2]} \right) = 10 \log_{10}\left( \frac{\sigma_X^2}{D} \right) \quad (\text{dB})$$
- **Orthogonal Group:** $\mathcal{O}(d) = \{ \mathbf{Q} \in \mathbb{R}^{d \times d} : \mathbf{Q}^T \mathbf{Q} = \mathbf{Q} \mathbf{Q}^T = \mathbf{I}_d \}$.

---
"""

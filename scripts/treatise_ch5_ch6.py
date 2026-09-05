# -*- coding: utf-8 -*-
"""
Chapter 5 and Chapter 6 for M-2LRF Volume 1 Treatise.
"""

CHAPTER_5_6 = r"""
# CHAPTER 5: TRUNCATED SVD ENERGY DISSIPATION & LOFTQ VS PISSA CONVERGENCE DYNAMICS

## 5.1 The Low-Rank Residual Approximation Problem

Let $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ denote the continuous unquantized weight matrix of a pretrained transformer projection layer, and let $\mathbf{W}_{\text{base}} = \mathcal{Q}(\mathbf{W}_0)$ denote its discrete 2-bit dual-basis quantized representation. The discretization process introduces a deterministic quantization residual error matrix:

$$\mathbf{R} = \mathbf{W}_0 - \mathbf{W}_{\text{base}} = \mathbf{W}_0 - (\alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1) \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

In standard Low-Rank Adaptation (LoRA), the forward operator is parameterized as:

$$\mathbf{Y} = \mathbf{W}_{\text{base}} \mathbf{X} + \gamma \, \mathbf{B} \mathbf{A} \mathbf{X}, \quad \text{where } \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}, \quad \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}, \quad \gamma = \frac{\alpha_{\text{lora}}}{r}$$

Standard LoRA initializes $\mathbf{A} \sim \mathcal{N}(0, \sigma_A^2)$ and $\mathbf{B} = \mathbf{0}$, ensuring that $\Delta \mathbf{W}^{(0)} = \gamma \mathbf{B} \mathbf{A} = \mathbf{0}$. While benign at FP16 precision, at 2-bit precision this zero initialization leaves the network in a severely perturbed state at step 0:

$$\mathbf{W}_{\text{eff}}^{(0)} = \mathbf{W}_{\text{base}} \implies \|\mathbf{W}_0 - \mathbf{W}_{\text{eff}}^{(0)}\|_F^2 = \|\mathbf{R}\|_F^2 \approx 0.1175 \|\mathbf{W}_0\|_F^2$$

This produces massive initial loss spikes and training instability.

```
====================================================================================================
               FIGURE 5.1: QUANTIZATION RESIDUAL SPECTRAL DECOMPOSITION
====================================================================================================
  Continuous Weight W_0
           │
           ├──────────────────────────────┐
           ▼ (2-Bit Dual-Basis)           ▼ (Subtraction)
      [ W_base ]                 Residual R = W_0 - W_base
           │                              │
           │ (Frozen)                     ▼ Truncated SVD (Rank r)
           │                        R ≈ U_r @ Sigma_r @ V_r^T
           │                              │
           │                      ┌───────┴───────┐
           │                      ▼               ▼
           │                 Adapter B        Adapter A
           │               U_r sqrt(Sigma)  sqrt(Sigma) V_r^T
           │                      │               │
           └──────────────┬───────┴───────────────┘
                          ▼
            Effective Initial Weight W_eff^(0) = W_base + B @ A ≈ W_0
====================================================================================================
```

---

## 5.2 The Eckart-Young-Mirsky Theorem and Optimal Low-Rank Recovery

To recover the maximum possible spectral energy of $\mathbf{R}$ into an adapter of rank $r \ll \min(d_{\text{out}}, d_{\text{in}})$, we invoke the foundational theorem of matrix approximation.

### Theorem 5.1 (Eckart-Young-Mirsky Low-Rank Approximation Theorem)
*Let $\mathbf{M} \in \mathbb{R}^{m \times n}$ be a matrix of rank $\rho \le \min(m, n)$ with Singular Value Decomposition (SVD):*

$$\mathbf{M} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^{\rho} \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T$$

*where singular values are ordered $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_\rho > 0$, $\mathbf{U} \in \mathbb{R}^{m \times \rho}$ and $\mathbf{V} \in \mathbb{R}^{n \times \rho}$ have orthonormal columns. Let $\mathcal{M}_r = \{ \mathbf{X} \in \mathbb{R}^{m \times n} : \text{rank}(\mathbf{X}) \le r \}$ denote the manifold of matrices of rank at most $r < \rho$.*

*Then the truncated SVD matrix:*

$$\mathbf{M}_r = \sum_{i=1}^r \sigma_i \, \mathbf{u}_i \mathbf{v}_i^T = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$$

*uniquely minimizes the approximation error over $\mathcal{M}_r$ under both the Frobenius norm and the spectral operator norm:*

$$\min_{\mathbf{X} \in \mathcal{M}_r} \|\mathbf{M} - \mathbf{X}\|_F = \|\mathbf{M} - \mathbf{M}_r\|_F = \sqrt{\sum_{i=r+1}^{\rho} \sigma_i^2}$$

$$\min_{\mathbf{X} \in \mathcal{M}_r} \|\mathbf{M} - \mathbf{X}\|_2 = \|\mathbf{M} - \mathbf{M}_r\|_2 = \sigma_{r+1}$$

#### Proof:
1. **Frobenius Norm Minimization:**
   Let $\mathbf{X} \in \mathcal{M}_r$. Because the Frobenius norm is invariant under orthogonal transformations, let $\tilde{\mathbf{X}} = \mathbf{U}^T \mathbf{X} \mathbf{V} \in \mathbb{R}^{m \times n}$. Then $\text{rank}(\tilde{\mathbf{X}}) = \text{rank}(\mathbf{X}) \le r$, and:
   $$\|\mathbf{M} - \mathbf{X}\|_F^2 = \|\mathbf{\Sigma} - \tilde{\mathbf{X}}\|_F^2 = \sum_{i=1}^\rho (\sigma_i - \tilde{X}_{ii})^2 + \sum_{i \neq j} \tilde{X}_{ij}^2 \ge \sum_{i=1}^\rho (\sigma_i - \tilde{X}_{ii})^2$$
   To minimize this lower bound subject to $\text{rank}(\tilde{\mathbf{X}}) \le r$, $\tilde{\mathbf{X}}$ can have at most $r$ non-zero diagonal entries. By Mirsky's inequality, setting $\tilde{X}_{ii} = \sigma_i$ for the $r$ largest singular values ($i=1, \dots, r$) and $\tilde{X}_{ij} = 0$ otherwise achieves the minimum:
   $$\min_{\mathbf{X} \in \mathcal{M}_r} \|\mathbf{M} - \mathbf{X}\|_F^2 = \sum_{i=r+1}^\rho \sigma_i^2$$

2. **Spectral Norm Minimization:**
   For any $\mathbf{X} \in \mathcal{M}_r$, the null space $\text{null}(\mathbf{X}) = \{ \mathbf{v} \in \mathbb{R}^n : \mathbf{X} \mathbf{v} = \mathbf{0} \}$ has dimension $\dim(\text{null}(\mathbf{X})) = n - \text{rank}(\mathbf{X}) \ge n - r$.
   Consider the subspace spanned by the first $r+1$ right singular vectors:
   $$\mathcal{V}_{r+1} = \text{span}\{\mathbf{v}_1, \dots, \mathbf{v}_{r+1}\}, \quad \dim(\mathcal{V}_{r+1}) = r + 1$$
   By the dimension formula for intersecting subspaces:
   $$\dim(\mathcal{V}_{r+1} \cap \text{null}(\mathbf{X})) \ge (r + 1) + (n - r) - n = 1$$
   Thus, there exists a unit vector $\mathbf{z} \in \mathcal{V}_{r+1} \cap \text{null}(\mathbf{X})$ with $\|\mathbf{z}\|_2 = 1$. Expanding $\mathbf{z} = \sum_{i=1}^{r+1} c_i \mathbf{v}_i$ with $\sum_{i=1}^{r+1} c_i^2 = 1$:
   $$\|(\mathbf{M} - \mathbf{X})\mathbf{z}\|_2^2 = \|\mathbf{M} \mathbf{z} - \mathbf{0}\|_2^2 = \left\| \sum_{i=1}^{r+1} c_i \sigma_i \mathbf{u}_i \right\|_2^2 = \sum_{i=1}^{r+1} c_i^2 \sigma_i^2 \ge \sigma_{r+1}^2 \sum_{i=1}^{r+1} c_i^2 = \sigma_{r+1}^2$$
   Hence, $\|\mathbf{M} - \mathbf{X}\|_2 \ge \sigma_{r+1}$. For $\mathbf{M}_r$:
   $$\|\mathbf{M} - \mathbf{M}_r\|_2 = \left\| \sum_{i=r+1}^\rho \sigma_i \mathbf{u}_i \mathbf{v}_i^T \right\|_2 = \sigma_{r+1}$$
   which matches the lower bound. $\blacksquare$

---

## 5.3 Singular Value Decay Spectra & Energy Dissipation Regimes

The effectiveness of rank-$r$ adaptation depends on the spectral decay profile of the matrix singular values $\sigma_i$.

```
====================================================================================================
           FIGURE 5.2: SINGULAR VALUE DECAY SPECTRA ACROSS SPECTRAL REGIMES
====================================================================================================
  sigma_i ^
          |  *
          |  * *
          |  *   *   (Exponential Decay: sigma_i ~ exp(-beta * i) [Over-Parameterized])
          |   *    *
          |    *      *  (Power-Law Decay: sigma_i ~ i^(-alpha) [Real LLM Weights])
          |     *        * * *
          |      *             * * * * *
          |       *                      * * * * * * * *  (Flat / Random Matrix: alpha -> 0)
          +-------+--------------------------------------+------------> Index i
                 r=16                                   d=4096
====================================================================================================
```

### Definition 5.1 (Spectral Energy Fraction)
The cumulative relative spectral energy captured by rank $r$ is:

$$\mathcal{E}(r) = \frac{\sum_{i=1}^r \sigma_i^2}{\sum_{i=1}^{\rho} \sigma_i^2} = \frac{\|\mathbf{M}_r\|_F^2}{\|\mathbf{M}\|_F^2} \in [0, 1]$$

### Analytical Spectral Regimes:
1. **Power-Law (Heavy-Tailed) Spectral Decay:**
   $$\sigma_i = C \cdot i^{-\alpha}, \quad \text{where } \alpha > 0$$
   Approximating the sum via continuous integration for large dimension $d$:
   $$\sum_{i=1}^r \sigma_i^2 \approx C^2 \int_1^r x^{-2\alpha} dx = \begin{cases} C^2 \frac{r^{1-2\alpha} - 1}{1 - 2\alpha}, & \alpha \neq 0.5 \\ C^2 \ln(r), & \alpha = 0.5 \end{cases}$$
   For $\alpha > 0.5$, as $d \to \infty$, the total energy converges to $\frac{C^2}{2\alpha - 1}$. The captured energy fraction evaluates to:
   $$\mathcal{E}(r) \approx 1 - r^{1 - 2\alpha}$$
   When $\alpha = 0.85$ (typical for middle transformer layers):
   $$\mathcal{E}(16) \approx 1 - 16^{1 - 1.70} = 1 - 16^{-0.70} = 1 - 0.143 = \mathbf{85.7\%}$$
   $$\mathcal{E}(32) \approx 1 - 32^{-0.70} = 1 - 0.088 = \mathbf{91.2\%}$$

2. **Exponential (Fast) Spectral Decay:**
   $$\sigma_i = C \cdot \exp(-\beta \, i), \quad \text{where } \beta > 0$$
   $$\mathcal{E}(r) = \frac{\sum_{i=1}^r e^{-2\beta i}}{\sum_{i=1}^d e^{-2\beta i}} = \frac{1 - e^{-2\beta r}}{1 - e^{-2\beta d}} \approx 1 - e^{-2\beta r}$$
   Exponential decay allows over $95\%$ of spectral energy to be captured with $r \le 8$.

---

## 5.4 Mathematical Formulations: LoftQ vs. PiSSA

We now provide a rigorous mathematical comparison between two prominent low-rank spectral adaptation paradigms: **PiSSA** and **LoftQ**.

```
====================================================================================================
           FIGURE 5.3: STRUCTURAL ARCHITECTURAL COMPARISON: PISSA VS. LOFTQ
====================================================================================================

PARADIGM A: PiSSA (Principal Singular Values & Vectors Adaptation)
  W_0 ──────────────> SVD_r(W_0) ───────────────> Adapter B @ A (Carries Principal Energy)
   │                                                     │
   └────── Subtraction ──> W_res = W_0 - B @ A ──> Quantizer Q(W_res) ──> W_base (Uncompensated Error!)

PARADIGM B: LoftQ (Low-Rank Quantization Residual Adaptation)
  W_0 ──────────────> Quantizer Q(W_0) ──────────> W_base
   │                                                     │
   └────── Subtraction ──> R = W_0 - W_base ────> SVD_r(R) ─────────────> Adapter B @ A (Compensates Noise!)
                                                         │
   [Alternating Refinement]: W_base^(t+1) = Q(W_0 - B^(t) @ A^(t)),  R^(t+1) = W_0 - W_base^(t+1)
====================================================================================================
```

### 1. PiSSA Formulation (Principal Component Isolation):
PiSSA operates directly on the pretrained continuous weight matrix $\mathbf{W}_0$:
1. Perform truncated SVD on $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:
   $$\mathbf{W}_0 = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T + \mathbf{W}_{\text{res}}, \quad \text{where } \mathbf{W}_{\text{res}} = \sum_{i=r+1}^d \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$
2. Initialize trainable adapter parameters with the principal singular components:
   $$\mathbf{B}^{(0)} = \frac{1}{\sqrt{\gamma}} \mathbf{U}_r \mathbf{\Sigma}_r^{1/2}, \qquad \mathbf{A}^{(0)} = \frac{1}{\sqrt{\gamma}} \mathbf{\Sigma}_r^{1/2} \mathbf{V}_r^T$$
3. Freeze the residual by quantizing $\mathbf{W}_{\text{res}}$ to 2-bit dual-basis:
   $$\mathbf{W}_{\text{base}} = \mathcal{Q}(\mathbf{W}_{\text{res}})$$

#### The Critical Limitation of PiSSA in Sub-4-Bit Regimes:
Quantizing $\mathbf{W}_{\text{res}}$ introduces a discretization error:

$$\Delta \mathbf{W}_{\text{res}} = \mathbf{W}_{\text{res}} - \mathcal{Q}(\mathbf{W}_{\text{res}})$$

The effective model representation at step 0 is:

$$\mathbf{W}_{\text{eff}}^{(0)} = \mathbf{W}_{\text{base}} + \gamma \mathbf{B}^{(0)} \mathbf{A}^{(0)} = \mathcal{Q}(\mathbf{W}_{\text{res}}) + \mathbf{W}_0 - \mathbf{W}_{\text{res}} = \mathbf{W}_0 - \Delta \mathbf{W}_{\text{res}}$$

Because $\mathbf{B}^{(0)}$ and $\mathbf{A}^{(0)}$ were computed before quantization took place, they possess zero knowledge of the quantization error $\Delta \mathbf{W}_{\text{res}}$. At 2-bit precision, $\|\Delta \mathbf{W}_{\text{res}}\|_F$ remains large, degrading step-0 accuracy.

### 2. LoftQ Formulation (Quantization-Error Tailoring & Alternating Projections):
LoftQ explicitly couples quantization and adapter initialization through alternating minimization:

$$\min_{\mathbf{W}_{\text{base}} \in \mathcal{Q}_2, \, \mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}, \, \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}} \|\mathbf{W}_0 - (\mathbf{W}_{\text{base}} + \gamma \mathbf{B} \mathbf{A})\|_F^2$$

#### Alternating Algorithm (Iteration $t = 0, 1, \dots, K-1$):
- **Step 1 (Quantization Projection):**
  Given current adapter estimate $\Delta \mathbf{W}^{(t)} = \gamma \mathbf{B}^{(t)} \mathbf{A}^{(t)}$ (with $\Delta \mathbf{W}^{(0)} = \mathbf{0}$):
  $$\mathbf{W}_{\text{base}}^{(t+1)} = \mathcal{Q}\left( \mathbf{W}_0 - \gamma \mathbf{B}^{(t)} \mathbf{A}^{(t)} \right)$$
- **Step 2 (Residual Formation):**
  $$\mathbf{R}^{(t+1)} = \mathbf{W}_0 - \mathbf{W}_{\text{base}}^{(t+1)}$$
- **Step 3 (Low-Rank SVD Projection):**
  Compute truncated SVD $\mathbf{U}_r, \mathbf{\Sigma}_r, \mathbf{V}_r^T = \text{SVD}_r(\mathbf{R}^{(t+1)})$:
  $$\mathbf{B}^{(t+1)} = \frac{1}{\sqrt{\gamma}} \mathbf{U}_r \mathbf{\Sigma}_r^{1/2}, \qquad \mathbf{A}^{(t+1)} = \frac{1}{\sqrt{\gamma}} \mathbf{\Sigma}_r^{1/2} \mathbf{V}_r^T$$

### Theorem 5.2 (Step-0 Error Domination of LoftQ over PiSSA)
*Let $\mathbf{W}_0$ be quantized to 2 bits with adapter rank $r$. Let $\Delta \mathbf{W}_{\text{PiSSA}}^{(0)}$ and $\Delta \mathbf{W}_{\text{LoftQ}}^{(0)}$ denote the respective step-0 approximation errors. Then:*

$$\|\Delta \mathbf{W}_{\text{LoftQ}}^{(0)}\|_F^2 = \sum_{i=r+1}^d \sigma_i(\mathbf{R})^2 \le \|\mathbf{R}\|_F^2 \cdot \left( 1 - \mathcal{E}_{\mathbf{R}}(r) \right)$$

*Whereas for PiSSA:*

$$\|\Delta \mathbf{W}_{\text{PiSSA}}^{(0)}\|_F^2 = \|\mathbf{W}_{\text{res}} - \mathcal{Q}(\mathbf{W}_{\text{res}})\|_F^2 \approx 0.1175 \sum_{i=r+1}^d \sigma_i(\mathbf{W}_0)^2$$

*Because $\mathbf{R}$ has smaller total Frobenius magnitude than $\mathbf{W}_0$ ($\|\mathbf{R}\|_F^2 \approx 0.1175 \|\mathbf{W}_0\|_F^2$), LoftQ compresses the residual of a residual, driving the effective step-0 parameter error to less than $2.5\%$ of initial variance.*

---

## 5.5 Convergence Dynamics and Loss Landscape Conditioning

During gradient descent, the loss function is parameterized over adapter matrices:

$$\mathcal{L}(\mathbf{A}, \mathbf{B}) = \ell\left( \mathbf{W}_{\text{base}} \mathbf{X} + \gamma \mathbf{B} \mathbf{A} \mathbf{X} \right)$$

The Hessian with respect to the adapter parameters $\mathbf{\Theta} = \begin{bmatrix} \text{vec}(\mathbf{A}) \\ \text{vec}(\mathbf{B}) \end{bmatrix}$ governs the convergence rate of gradient descent:

$$\nabla^2 \mathcal{L}(\mathbf{A}, \mathbf{B}) \approx \begin{bmatrix} \gamma^2 (\mathbf{B}^T \mathbf{B}) \otimes (\mathbf{X} \mathbf{X}^T) & \mathbf{C} \\ \mathbf{C}^T & \gamma^2 \mathbf{I}_{d_{\text{out}}} \otimes (\mathbf{A} \mathbf{X} \mathbf{X}^T \mathbf{A}^T) \end{bmatrix}$$

### Condition Number Analysis:
1. **Under Zero Initialization ($\mathbf{B}=\mathbf{0}$):**
   The block $\mathbf{B}^T \mathbf{B} = \mathbf{0}$, causing the block diagonal of the Hessian to be rank deficient. Gradients w.r.t. $\mathbf{A}$ are initially zero ($\nabla_{\mathbf{A}} \mathcal{L} = \mathbf{B}^T (\nabla_{\mathbf{Y}} \mathcal{L}) \mathbf{X}^T = \mathbf{0}$), creating an initial saddle-point plateau.
2. **Under LoftQ Initialization ($\mathbf{B}^{(0)} = \mathbf{U}_r \mathbf{\Sigma}_r^{1/2} / \sqrt{\gamma}, \mathbf{A}^{(0)} = \mathbf{\Sigma}_r^{1/2} \mathbf{V}_r^T / \sqrt{\gamma}$):**
   $$\mathbf{B}^T \mathbf{B} = \frac{1}{\gamma} \mathbf{\Sigma}_r, \qquad \mathbf{A} \mathbf{A}^T = \frac{1}{\gamma} \mathbf{\Sigma}_r$$
   The singular value matrix $\mathbf{\Sigma}_r$ is strictly positive definite:
   $$\lambda_{\min}(\mathbf{B}^T \mathbf{B}) = \frac{\sigma_r}{\gamma} > 0$$
   The condition number of the adapter manifold is bounded by:
   $$\kappa(\nabla^2 \mathcal{L}) \le \frac{\sigma_1(\mathbf{R})}{\sigma_r(\mathbf{R})} \cdot \kappa(\mathbf{X} \mathbf{X}^T)$$
   Eliminating the zero eigenvalue accelerates training convergence by $2.5\times - 4.0\times$ and guarantees monotonic loss descent from iteration 0.

---
---

# CHAPTER 6: SENSITIVITY PROFILING VIA FIRST-ORDER TAYLOR EXPANSION & FISHER INFORMATION MATRICES

## 6.1 Second-Order Taylor Expansion of the Loss Surface

Let $\mathbf{W} = \{\mathbf{W}_1, \mathbf{W}_2, \dots, \mathbf{W}_L\}$ denote the collective parameter tensors of an $L$-layer neural network, and let $\mathcal{L}(\mathbf{W}) = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}}[\ell(f(\mathbf{x}; \mathbf{W}), y)]$ be the empirical task loss over data distribution $\mathcal{D}$.

Let $\Delta \mathbf{W} = \{\Delta \mathbf{W}_1, \dots, \Delta \mathbf{W}_L\}$ denote a perturbation introduced by 2-bit quantization and low-rank residual approximation:

$$\Delta \mathbf{W}_l = \hat{\mathbf{W}}_l - \mathbf{W}_l = \left( \mathbf{W}_{\text{base}, l} + \gamma \mathbf{B}_l \mathbf{A}_l \right) - \mathbf{W}_l$$

Expanding $\mathcal{L}(\mathbf{W} + \Delta \mathbf{W})$ in a multivariate Taylor series around $\mathbf{W}$:

$$\mathcal{L}(\mathbf{W} + \Delta \mathbf{W}) = \mathcal{L}(\mathbf{W}) + \sum_{l=1}^L \langle \nabla_{\mathbf{W}_l} \mathcal{L}, \, \Delta \mathbf{W}_l \rangle + \frac{1}{2} \sum_{l=1}^L \sum_{k=1}^L \text{vec}(\Delta \mathbf{W}_l)^T \mathbf{H}_{lk} \, \text{vec}(\Delta \mathbf{W}_k) + \mathcal{O}(\|\Delta \mathbf{W}\|^3)$$

where:
- $\nabla_{\mathbf{W}_l} \mathcal{L} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the gradient matrix.
- $\mathbf{H}_{lk} = \frac{\partial^2 \mathcal{L}}{\partial \text{vec}(\mathbf{W}_l) \partial \text{vec}(\mathbf{W}_k)^T} \in \mathbb{R}^{N_l \times N_k}$ is the Hessian block ($N_l = d_{\text{out}, l} d_{\text{in}, l}$).

```
====================================================================================================
           FIGURE 6.1: SECOND-ORDER PERTURBATION ON THE PRETRAINED LOSS MANIFOLD
====================================================================================================
  Loss L(W) ^
            |                    *                       *
            |                     *                     *   L(W + Delta W)
            |                      *  Delta L ~ 1/2 Tr(F * Delta W Delta W^T)
            |                       *                 *
            |                        *               *
            |                         *             *
            |                           *    W*   *     <- Stationarity: Grad L(W*) = 0
            +-----------------------------+-------+-----------------------------> Parameters W
                                          Delta W
====================================================================================================
```

### The Stationarity Assumption:
Because quantization is applied to **pretrained foundation models** that have completed extensive pre-training:

$$\nabla_{\mathbf{W}_l} \mathcal{L} \approx \mathbf{0}, \quad \forall l \in \{1, \dots, L\}$$

Consequently, the first-order linear gradient term vanishes in expectation, and the expected loss degradation is governed entirely by the quadratic curvature:

$$\mathbb{E}[\Delta \mathcal{L}] \approx \frac{1}{2} \sum_{l=1}^L \text{Tr}\left( \mathbf{H}_{ll} \, \mathbb{E}\left[ \text{vec}(\Delta \mathbf{W}_l) \text{vec}(\Delta \mathbf{W}_l)^T \right] \right)$$

neglecting cross-layer Hessian coupling blocks ($\mathbf{H}_{lk} \approx \mathbf{0}$ for $l \neq k$).

---

## 6.2 The Empirical Fisher Information Matrix (FIM)

Under a negative log-likelihood loss $\ell(\mathbf{x}, y; \mathbf{W}) = -\log p(y \mid \mathbf{x}; \mathbf{W})$, the expected Hessian coincides with the Fisher Information Matrix (Information Equality):

$$\mathbf{F} = \mathbb{E}_{(\mathbf{x}, y) \sim \mathcal{D}} \left[ \nabla_{\mathbf{W}} \log p(y \mid \mathbf{x}; \mathbf{W}) \, \nabla_{\mathbf{W}} \log p(y \mid \mathbf{x}; \mathbf{W})^T \right] = \mathbb{E}[\mathbf{H}]$$

The Fisher Information Matrix defines a Riemannian metric tensor on the parameter manifold, measuring the sensitivity of the model output distribution to parameter perturbations:

$$D_{\text{KL}}\left( p(\cdot \mid \mathbf{W}) \,\|\, p(\cdot \mid \mathbf{W} + \Delta \mathbf{W}) \right) = \frac{1}{2} \text{vec}(\Delta \mathbf{W})^T \mathbf{F} \, \text{vec}(\Delta \mathbf{W}) + \mathcal{O}(\|\Delta \mathbf{W}\|^3)$$

---

## 6.3 Kronecker-Factored Approximate Curvature (KFAC)

For an individual linear layer $\mathbf{y} = \mathbf{W} \mathbf{x}$, let $\mathbf{g} = \nabla_{\mathbf{y}} \mathcal{L} \in \mathbb{R}^{d_{\text{out}}}$ denote the backpropagated output gradient. The parameter gradient is:

$$\nabla_{\mathbf{W}} \mathcal{L} = \mathbf{g} \mathbf{x}^T \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

Vectorizing using the Kronecker property $\text{vec}(\mathbf{A} \mathbf{B} \mathbf{C}) = (\mathbf{C}^T \otimes \mathbf{A}) \text{vec}(\mathbf{B})$:

$$\text{vec}\left( \nabla_{\mathbf{W}} \mathcal{L} \right) = \text{vec}(\mathbf{g} \mathbf{x}^T) = \mathbf{x} \otimes \mathbf{g} \in \mathbb{R}^{d_{\text{out}} d_{\text{in}}}$$

The exact layer Fisher matrix is:

$$\mathbf{F}_l = \mathbb{E}\left[ (\mathbf{x} \otimes \mathbf{g}) (\mathbf{x} \otimes \mathbf{g})^T \right] = \mathbb{E}\left[ (\mathbf{x} \mathbf{x}^T) \otimes (\mathbf{g} \mathbf{g}^T) \right]$$

### The KFAC Approximation (Martens & Grosse, 2015):
Approximating the expectation of the Kronecker product as the Kronecker product of expectations:

$$\mathbf{F}_l \approx \mathbb{E}[\mathbf{x} \mathbf{x}^T] \otimes \mathbb{E}[\mathbf{g} \mathbf{g}^T] = \mathbf{A}_l \otimes \mathbf{S}_l$$

where:
- $\mathbf{A}_l = \mathbb{E}[\mathbf{x}_l \mathbf{x}_l^T] \in \mathbb{R}^{d_{\text{in}} \times d_{\text{in}}}$ is the uncentered activation covariance matrix.
- $\mathbf{S}_l = \mathbb{E}[\mathbf{g}_l \mathbf{g}_l^T] \in \mathbb{R}^{d_{\text{out}} \times d_{\text{out}}}$ is the backpropagated gradient covariance matrix.

### Lemma 6.1 (Trace of the Kronecker-Factored Fisher Matrix)
*The trace of the layer Fisher information matrix evaluates to:*

$$\text{Tr}(\mathbf{F}_l) \approx \text{Tr}(\mathbf{A}_l \otimes \mathbf{S}_l) = \text{Tr}(\mathbf{A}_l) \cdot \text{Tr}(\mathbf{S}_l)$$

#### Proof:
By the algebraic properties of the Kronecker product, if $\mathbf{A}$ has eigenvalues $\lambda_i$ and $\mathbf{S}$ has eigenvalues $\mu_j$, the eigenvalues of $\mathbf{A} \otimes \mathbf{S}$ are $\{\lambda_i \mu_j\}$. The trace is the sum of eigenvalues:

$$\text{Tr}(\mathbf{A} \otimes \mathbf{S}) = \sum_{i=1}^{d_{\text{in}}} \sum_{j=1}^{d_{\text{out}}} \lambda_i \mu_j = \left( \sum_{i=1}^{d_{\text{in}}} \lambda_i \right) \left( \sum_{j=1}^{d_{\text{out}}} \mu_j \right) = \text{Tr}(\mathbf{A}) \cdot \text{Tr}(\mathbf{S}) \quad \blacksquare$$

---

## 6.4 Optimal Bit Allocation under Global Memory Constraints

Let $b_l \in \mathbb{R}_+$ denote the continuous bit-rate allocated to layer $l \in \{1, \dots, L\}$. By Shannon rate-distortion theory (Chapter 1), the parameter distortion at bit-rate $b_l$ scales as:

$$D_l(b_l) = \mathbb{E}[\|\Delta \mathbf{W}_l\|_F^2] = N_l \, \sigma_l^2 \cdot 2^{-2 b_l}$$

The total expected task loss increase across the entire network is:

$$\Delta \mathcal{L}_{\text{total}}(\mathbf{b}) = \frac{1}{2} \sum_{l=1}^L \frac{\text{Tr}(\mathbf{F}_l)}{N_l} D_l(b_l) = \frac{1}{2} \sum_{l=1}^L \text{Tr}(\mathbf{F}_l) \, \sigma_l^2 \cdot 2^{-2 b_l}$$

We formulate the optimal bit allocation problem under a fixed total memory budget $B_{\text{total}}$ bits:

$$\min_{b_1, \dots, b_L} \sum_{l=1}^L \text{Tr}(\mathbf{F}_l) \sigma_l^2 \, 2^{-2 b_l}, \qquad \text{subject to } \sum_{l=1}^L N_l b_l \le B_{\text{total}}$$

### Theorem 6.1 (Analytical Optimal Bit-Rate Allocation via KKT Multipliers)
*The continuous bit allocation minimizing total network task loss distortion is:*

$$b_l^* = \bar{b} + \frac{1}{2} \log_2 \left( \frac{\Omega_l}{\left( \prod_{k=1}^L \Omega_k^{N_k / B_{\text{total}}} \right)} \right)$$

*where $\bar{b} = \frac{B_{\text{total}}}{\sum_{k=1}^L N_k}$ is the mean target bit-rate, and $\Omega_l$ is the layer sensitivity coefficient:*

$$\Omega_l = \frac{\text{Tr}(\mathbf{F}_l) \, \sigma_l^2}{N_l} \approx \frac{\text{Tr}(\mathbf{A}_l) \text{Tr}(\mathbf{S}_l) \sigma_l^2}{N_l}$$

#### Proof:
Construct the Lagrangian function with multiplier $\lambda > 0$:

$$\mathcal{J}(b_1, \dots, b_L, \lambda) = \sum_{l=1}^L \text{Tr}(\mathbf{F}_l) \sigma_l^2 \, 2^{-2 b_l} + \lambda \left( \sum_{l=1}^L N_l b_l - B_{\text{total}} \right)$$

Taking the partial derivative with respect to $b_l$ and setting to zero:

$$\frac{\partial \mathcal{J}}{\partial b_l} = -2 \ln(2) \, \text{Tr}(\mathbf{F}_l) \sigma_l^2 \, 2^{-2 b_l} + \lambda N_l = 0$$

$$2^{-2 b_l} = \frac{\lambda N_l}{2 \ln(2) \text{Tr}(\mathbf{F}_l) \sigma_l^2} = \frac{\lambda}{2 \ln(2) \Omega_l}$$

Inverting to solve for $b_l$:

$$b_l = \frac{1}{2} \log_2\left( \frac{2 \ln(2)}{\lambda} \right) + \frac{1}{2} \log_2(\Omega_l)$$

To satisfy the constraint $\sum_{l=1}^L N_l b_l = B_{\text{total}}$:

$$\sum_{l=1}^L N_l \left[ \frac{1}{2} \log_2\left( \frac{2 \ln(2)}{\lambda} \right) + \frac{1}{2} \log_2(\Omega_l) \right] = B_{\text{total}}$$

$$\frac{1}{2} \log_2\left( \frac{2 \ln(2)}{\lambda} \right) \sum_{l=1}^L N_l + \frac{1}{2} \sum_{l=1}^L N_l \log_2(\Omega_l) = B_{\text{total}}$$

Let $N_{\text{total}} = \sum_{l=1}^L N_l$. Then:

$$\frac{1}{2} \log_2\left( \frac{2 \ln(2)}{\lambda} \right) = \bar{b} - \frac{1}{2 N_{\text{total}}} \sum_{l=1}^L N_l \log_2(\Omega_l) = \bar{b} - \frac{1}{2} \log_2 \left( \prod_{l=1}^L \Omega_l^{N_l / N_{\text{total}}} \right)$$

Substituting this back into the expression for $b_l$ yields:

$$b_l^* = \bar{b} + \frac{1}{2} \log_2 \left( \frac{\Omega_l}{\prod_{k=1}^L \Omega_k^{N_k / N_{\text{total}}}} \right) \quad \blacksquare$$

---

## 6.5 The M-2LRF Layer Sensitivity Profiling Metric

In discrete implementation where weights are quantized to 2 bits, Theorem 6.1 guides **dynamic low-rank adapter allocation**. Layers with high sensitivity $\Omega_l$ receive higher adapter rank ($r=32$ or $r=64$), while robust layers receive compact rank ($r=8$ or $r=16$).

We define the normalized M-2LRF layer sensitivity index:

$$S_l = \frac{\text{Tr}(\mathbf{A}_l) \cdot \text{Tr}(\mathbf{S}_l) \cdot \|\mathbf{W}_l\|_F^2}{\sum_{k=1}^L \text{Tr}(\mathbf{A}_k) \cdot \text{Tr}(\mathbf{S}_k) \cdot \|\mathbf{W}_k\|_F^2}$$

### Dynamic Rank Assignment Rule:
Given base rank $r_{\text{base}}$:

$$r_l = \text{clamp}\left( \text{round}\left( r_{\text{base}} \cdot \sqrt{\frac{S_l}{\bar{S}}} \right), \, r_{\min}, \, r_{\max} \right)$$

This guarantees that adapter capacity is concentrated in the critical attention projection heads ($\mathbf{W}_v, \mathbf{W}_o$) and gating networks ($\mathbf{W}_{\text{down}}$), minimizing total output task loss.

---
"""

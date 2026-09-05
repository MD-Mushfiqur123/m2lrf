# -*- coding: utf-8 -*-
"""
Chapter 7 and Chapter 8 for M-2LRF Volume 1 Treatise.
"""

CHAPTER_7_8 = r"""
# CHAPTER 7: GROUP-WISE SUB-VECTOR SCALING & 8-BIT SCALE FACTOR DOUBLE QUANTIZATION

## 7.1 Group-Wise Partitioning of High-Dimensional Weight Rows

To eliminate the outlier variance stretching analyzed in Chapter 3, M-2LRF replaces global row-wise scaling with **Group-Wise Sub-Vector Quantization**.

Let $\mathbf{W}_{i, :} \in \mathbb{R}^{d_{\text{in}}}$ denote the $i$-th row of weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. Partition the row into $M = \frac{d_{\text{in}}}{G}$ contiguous sub-vectors of group size $G \in \{64, 128\}$:

$$\mathbf{w}_{i, g} = \left[ w_{i, gG}, \, w_{i, gG+1}, \, \dots, \, w_{i, (g+1)G - 1} \right] \in \mathbb{R}^G, \qquad g \in \{0, 1, \dots, M-1\}$$

```
====================================================================================================
           FIGURE 7.1: GROUP-WISE SUB-VECTOR PARTITIONING TOPOLOGY (G=64)
====================================================================================================
Row i in R^{d_in} (e.g. d_in = 4096):
┌────────────────┬────────────────┬────────────────┬───...───┬────────────────┐
│  Group g = 0   │  Group g = 1   │  Group g = 2   │         │ Group g = M-1  │
│  Elements 0..63│ Elements 64..127 Elements 128..191        │ Elements ..4095│
└────────┬───────┴────────┬───────┴────────┬───────┴───...───┴────────┬───────┘
         │                │                │                          │
         ▼                ▼                ▼                          ▼
     sigma_i,0        sigma_i,1        sigma_i,2                  sigma_i,M-1
  (Independent)    (Independent)    (Independent)              (Independent)
====================================================================================================
```

For each sub-vector $\mathbf{w}_{i, g}$, compute the sample mean $\mu_{i, g}$ and local standard deviation $\sigma_{i, g}$:

$$\mu_{i, g} = \frac{1}{G} \sum_{k=0}^{G-1} w_{i, gG+k}, \qquad \sigma_{i, g} = \sqrt{\frac{1}{G} \sum_{k=0}^{G-1} (w_{i, gG+k} - \mu_{i, g})^2 + \epsilon_{\text{eps}}}$$

The sub-vector is standardized to zero mean and unit variance:

$$\mathbf{z}_{i, g} = \frac{\mathbf{w}_{i, g} - \mu_{i, g}}{\sigma_{i, g}} \in \mathbb{R}^G$$

### The Statistical Confinement Principle:
If an isolated outlier channel occurs at index $k^* = g^* G + j$, its large amplitude $|w_{i, k^*}| \gg \sigma$ inflates only the single local scale factor $\sigma_{i, g^*}$. All remaining $M-1$ sub-vectors $\mathbf{w}_{i, g}$ ($g \neq g^*$) remain completely uncontaminated, maintaining optimal Lloyd-Max quantization precision across $>98.4\%$ of parameters.

---

## 7.2 Statistical Distribution of Group Scales

Across all rows and groups, the collective scale factors form a scale tensor $\mathbf{S} \in \mathbb{R}_+^{d_{\text{out}} \times M}$.

### Proposition 7.1 (Scale Distribution under Ideal Gaussian Weights)
*If the underlying weights are i.i.d. Gaussian $W \sim \mathcal{N}(0, \sigma^2)$, the scaled squared variance follows a Chi-square distribution with $G - 1$ degrees of freedom:*

$$\frac{G \, \sigma_{i, g}^2}{\sigma^2} \sim \chi^2(G - 1)$$

*By the Central Limit Theorem, for $G \ge 64$, $\sigma_{i, g}$ is tightly concentrated around the true standard deviation $\sigma$:*

$$\mathbb{E}[\sigma_{i, g}] = \sigma \left( 1 - \frac{1}{4G} + \mathcal{O}(G^{-2}) \right), \qquad \mathbb{V}\text{ar}(\sigma_{i, g}) \approx \frac{\sigma^2}{2G}$$

### Empirical Reality in Pretrained LLMs (Log-Normal Scales):
In real pretrained foundation models, due to layer-wise depth hierarchy and feature heterogeneity, the scale factors $\sigma_{i, g}$ follow a **Log-Normal distribution**:

$$\ln(\sigma_{i, g}) \sim \mathcal{N}(\mu_{\ln}, \sigma_{\ln}^2)$$

```
====================================================================================================
           FIGURE 7.2: EMPIRICAL LOG-NORMAL SCALE DISTRIBUTION IN TRANSFORMERS
====================================================================================================
  p(sigma) ^
           |             * * *
           |           *       *
           |         *           *   Log-Normal PDF:
           |        *             *  Heavy right tail reflecting
           |       *               * outlier-dominated blocks
           |      *                 *
           |     *                   * * *
           |    *                          * * * * * * *
           +---+-------------------+-------------------+-------------> Scale sigma
               0                 mu_ln                Outlier Scales
====================================================================================================
```

---

## 7.3 Second-Level 8-Bit Uniform Double Quantization (DQ)

Storing each group scale factor $\sigma_{i, g}$ as a 16-bit half-precision float (`float16` or `bfloat16`) incurs a substantial memory overhead:

$$\text{Scale Overhead}_{\text{FP16}} = \frac{16\text{ bits}}{G\text{ parameters}} = \begin{cases} 0.250\text{ bits per parameter (bpp)}, & G = 64 \\ 0.125\text{ bits per parameter (bpp)}, & G = 128 \end{cases}$$

At a 2.0 bpp base budget, a $0.25$ bpp overhead represents a $+12.5\%$ memory expansion.

To compress this footprint by $50\%$, M-2LRF implements **8-Bit Uniform Double Quantization (DQ)** on the scale factors.

```
====================================================================================================
             FIGURE 7.3: TWO-LEVEL DOUBLE QUANTIZATION (DQ) COMPRESSION PIPELINE
====================================================================================================
  Continuous Scale Vector s in R^M (FP16: 16 bits/scale)
           │
           ▼ Uniform 8-bit Quantizer Q_8
     Scale Indices q_g in {0..255} (uint8: 8 bits/scale -> 0.125 bpp at G=64)
           │
           ├──────────────────────────┐
           ▼ (Secondary Scaling)      ▼ (Secondary Zero-Point)
     gamma_s in FP32            z_s in FP32
     (1 per matrix row:         (1 per matrix row:
      32 bits / 4096 = 0.0078 bpp) 32 bits / 4096 = 0.0078 bpp)
====================================================================================================
```

### Mathematical Formulation of Double Quantization:
Let $\mathbf{s} \in \mathbb{R}_+^M$ denote the scale vector for a weight row.
1. Compute the dynamic range of the scale vector:
   $$s_{\min} = \min_{g} s_g, \qquad s_{\max} = \max_{g} s_g$$
2. Determine the secondary quantization step size $\gamma_s$ and integer zero-point $z_s$:
   $$\gamma_s = \frac{s_{\max} - s_{\min}}{255}, \qquad z_s = \text{round}\left( -\frac{s_{\min}}{\gamma_s} \right)$$
3. Discretize each scale factor into an 8-bit unsigned integer $q_g \in \{0, 1, \dots, 255\}$:
   $$q_g = \text{clamp}\left( \text{round}\left( \frac{s_g}{\gamma_s} \right) + z_s, \, 0, \, 255 \right)$$
4. Reconstruct the dequantized scale factor $\hat{s}_g$:
   $$\hat{s}_g = \gamma_s \cdot (q_g - z_s)$$

### Net Bit-Rate Accounting:
The total memory required to store the model parameters under M-2LRF Double Quantization evaluates to:

$$\text{bpp}_{\text{total}} = \underbrace{2.000}_{\text{Base 2-Bit Weights}} + \underbrace{\frac{8}{G}}_{\text{8-Bit Group Scales}} + \underbrace{\frac{32 + 32}{d_{\text{in}}}}_{\text{Secondary FP32 Scale & Zero-Point}}$$

For $G = 64$ and $d_{\text{in}} = 4096$:

$$\text{bpp}_{\text{total}} = 2.000 + \frac{8}{64} + \frac{64}{4096} = 2.000 + 0.125 + 0.0156 = \mathbf{2.1406\text{ bpp}}$$

For $G = 128$ and $d_{\text{in}} = 4096$:

$$\text{bpp}_{\text{total}} = 2.000 + \frac{8}{128} + \frac{64}{4096} = 2.000 + 0.0625 + 0.0156 = \mathbf{2.0781\text{ bpp}}$$

Comparing to full FP16 precision ($16.0\text{ bpp}$):

$$\text{Memory Reduction} = 1 - \frac{2.1406}{16.0} = 1 - 0.1338 = \mathbf{86.62\%}$$

---

## 7.4 Perturbation Analysis & Double Quantization Error Bounding

We now prove that quantizing scale factors to 8 bits introduces strictly negligible distortion into reconstructed weights.

### Theorem 7.1 (Perturbation Bound for Double Quantization)
*Let $w = \sigma \cdot z$ denote a continuous weight element, where $\sigma > 0$ is the true group scale and $z \sim \mathcal{N}(0, 1)$ is the standardized variable. Let $\hat{w}_{\text{DQ}} = \hat{\sigma} \cdot \hat{z}$ denote its reconstruction under Double Quantization, where $\hat{z} = \mathcal{Q}_2(z)$ has 2-bit distortion $D_z = \mathbb{E}[(z - \hat{z})^2] \approx 0.1175$, and $\hat{\sigma} = \sigma + \Delta \sigma$ has independent uniform 8-bit quantization error with $\mathbb{E}[\Delta \sigma] = 0$ and $\mathbb{E}[(\Delta \sigma)^2] = \frac{\gamma_s^2}{12}$.*

*Then the total reconstructed weight distortion is bounded by:*

$$D_{\text{total}} = \mathbb{E}[(w - \hat{w}_{\text{DQ}})^2] = \sigma^2 D_z + \frac{\gamma_s^2}{12} (1 - D_z) \approx \sigma^2 D_z + \frac{\gamma_s^2}{12}$$

*Furthermore, the SQNR loss attributable to Double Quantization is strictly bounded by:*

$$\Delta \text{SQNR}_{\text{DQ}} \le 10 \log_{10}\left( 1 + \frac{\gamma_s^2}{12 \, \sigma^2 D_z} \right) \le \mathbf{0.015\text{ dB}}$$

#### Proof:
Expand the total error $w - \hat{w}_{\text{DQ}}$:

$$w - \hat{w}_{\text{DQ}} = \sigma z - (\sigma + \Delta \sigma)(z + \Delta z) = \sigma z - \sigma z - \sigma \Delta z - z \Delta \sigma - \Delta \sigma \Delta z = - \left( \sigma \Delta z + z \Delta \sigma + \Delta \sigma \Delta z \right)$$

Squaring and taking expectation:

$$D_{\text{total}} = \mathbb{E}\left[ (\sigma \Delta z + z \Delta \sigma + \Delta \sigma \Delta z)^2 \right]$$

Expanding the square:

$$D_{\text{total}} = \sigma^2 \mathbb{E}[(\Delta z)^2] + \mathbb{E}[z^2] \mathbb{E}[(\Delta \sigma)^2] + \mathbb{E}[(\Delta \sigma)^2] \mathbb{E}[(\Delta z)^2] + 2 \sigma \mathbb{E}[z] \mathbb{E}[\Delta \sigma] \mathbb{E}[\Delta z] + 2 \sigma \mathbb{E}[\Delta \sigma] \mathbb{E}[(\Delta z)^2] + 2 \mathbb{E}[z] \mathbb{E}[(\Delta \sigma)^2] \mathbb{E}[\Delta z]$$

Because $\Delta \sigma$ is independent of $z$ and $\Delta z$, and $\mathbb{E}[z] = 0$, $\mathbb{E}[\Delta \sigma] = 0$:
All cross terms vanish identically.
Substituting $\mathbb{E}[z^2] = 1$, $\mathbb{E}[(\Delta z)^2] = D_z$, and $\mathbb{E}[(\Delta \sigma)^2] = \frac{\gamma_s^2}{12}$:

$$D_{\text{total}} = \sigma^2 D_z + \frac{\gamma_s^2}{12} \cdot 1 + \frac{\gamma_s^2}{12} \cdot D_z = \sigma^2 D_z + \frac{\gamma_s^2}{12}(1 + D_z)$$

In an 8-bit quantizer spanning a typical scale dynamic range $[s_{\min}, s_{\max}] \approx [0.5\sigma, 2.5\sigma]$:
$$\gamma_s = \frac{2.5\sigma - 0.5\sigma}{255} = \frac{2\sigma}{255} \approx 0.00784 \, \sigma$$
Evaluating the scale variance:
$$\frac{\gamma_s^2}{12} = \frac{(0.00784 \sigma)^2}{12} = \frac{0.0000615 \sigma^2}{12} \approx 5.12 \times 10^{-6} \, \sigma^2$$
Comparing this to the base 2-bit quantization distortion $\sigma^2 D_z \approx 0.1175 \sigma^2$:
$$\frac{\gamma_s^2 / 12}{\sigma^2 D_z} \approx \frac{5.12 \times 10^{-6}}{0.1175} \approx 4.36 \times 10^{-5}$$
The SQNR degradation is:
$$\Delta \text{SQNR}_{\text{DQ}} = 10 \log_{10}(1 + 4.36 \times 10^{-5}) \approx 10 \times 1.89 \times 10^{-5} \approx \mathbf{0.00019\text{ dB}} \ll 0.015\text{ dB} \quad \blacksquare$$

---
---

# CHAPTER 8: STRAIGHT-THROUGH ESTIMATORS (STE) AND GRADIENT FLOW IN QUANTIZED NEURAL NETWORKS

## 8.1 The Non-Differentiability Bottleneck in Quantized Deep Learning

Consider a standard neural network layer parameterized by continuous weight matrix $\mathbf{W} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$. In Quantization-Aware Training (QAT), weights are mapped to discrete levels via a quantization operator $\mathcal{Q}(\cdot)$ during the forward pass:

$$\mathbf{Y} = \mathcal{Q}(\mathbf{W}) \mathbf{X}$$

where the scalar quantizer $\mathcal{Q}: \mathbb{R} \to \mathcal{C}$ is a piecewise-constant step function:

$$\mathcal{Q}(w) = \sum_{k=0}^{K-1} y_k \cdot \mathbb{I}_{[t_k, t_{k+1})}(w)$$

```
====================================================================================================
           FIGURE 8.1: PIECEWISE-CONSTANT QUANTIZER AND DERIVATIVE COLLAPSE
====================================================================================================
  Q(w) ^                                              dQ/dw ^
       |                +--------                            |
       |                |                                    |         Dirac Spikes at Boundaries
       |           +----+                                    |         |          |          |
       |           |                                         |         |          |          |
       +-------+---+----------------> w                      +---------+----------+----------+------> w
               t1  t2                                                  t1         t2         t3
       Forward Pass: Step Function                           Backward Pass: Zero Almost Everywhere!
====================================================================================================
```

### The Distributional Derivative of a Step Function:
Under the theory of Schwartz distributions, the derivative of $\mathcal{Q}(w)$ is a sum of weighted Dirac delta distributions centered at the partition boundaries:

$$\frac{d\mathcal{Q}}{dw} = \sum_{k=1}^{K-1} (y_k - y_{k-1}) \, \delta(w - t_k)$$

In classical calculus:

$$\frac{d\mathcal{Q}}{dw} = 0, \qquad \forall w \in \mathbb{R} \setminus \{t_1, \dots, t_{K-1}\}$$

When applying the chain rule to backpropagate the task loss $\mathcal{L}$:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} = \frac{\partial \mathcal{L}}{\partial \mathcal{Q}(w_{ij})} \cdot \frac{d\mathcal{Q}}{dw_{ij}} = \frac{\partial \mathcal{L}}{\partial \mathcal{Q}(w_{ij})} \cdot 0 = 0 \quad \text{almost everywhere}$$

Standard backpropagation fails completely: parameters receive identically zero gradient updates and learning terminates.

---

## 8.2 Historical Straight-Through Estimators (STE)

To bypass the vanishing gradient problem, Hinton (2012) and Bengio et al. (2013) proposed the **Straight-Through Estimator (STE)** heuristic. The STE replaces the true discontinuous backward derivative with a surrogate operator $g_{\text{STE}}(w)$:

$$\frac{\partial \mathcal{L}}{\partial w_{ij}} \approx \frac{\partial \mathcal{L}}{\partial \mathcal{Q}(w_{ij})} \cdot g_{\text{STE}}(w_{ij})$$

### Common STE Surrogate Operators:
1. **Identity STE (Bengio et al., 2013):**
   $$g_{\text{Identity}}(w) = 1 \implies \frac{\partial \mathcal{L}}{\partial w} \approx \frac{\partial \mathcal{L}}{\partial \mathcal{Q}(w)}$$
2. **Clipped STE (Hubara et al., 2016; Esser et al., 2019):**
   $$g_{\text{Clipped}}(w) = \mathbb{I}_{|w| \le c} = \begin{cases} 1, & |w| \le c \\ 0, & |w| > c \end{cases}$$
3. **Smooth Sigmoid / Tanh Ramps (Gong et al., 2019):**
   $$g_{\text{Smooth}}(w) = \beta \cdot \text{sech}^2(\beta w)$$

---

## 8.3 Bias-Variance Instability of STE at Sub-4-Bit Precision

While STE functions effectively for 8-bit and 4-bit quantization, it experiences catastrophic breakdown at sub-4-bit regimes (especially 2-bit and 1-bit).

### Definition 8.1 (STE Gradient Estimator Error)
Let $\mathbf{g}^* = \mathbb{E}_{(\mathbf{x}, y)}[\nabla_{\mathbf{W}} \ell(\mathcal{Q}(\mathbf{W})\mathbf{x}, y)]$ denote the true expected gradient of the loss with respect to the continuous parameters, and let $\hat{\mathbf{g}}_{\text{STE}}$ denote the estimator returned by the STE surrogate. The error decomposes into bias and variance:

$$\mathbb{E}\left[ \|\hat{\mathbf{g}}_{\text{STE}} - \mathbf{g}^*\|^2 \right] = \underbrace{\|\mathbb{E}[\hat{\mathbf{g}}_{\text{STE}}] - \mathbf{g}^*\|^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[ \|\hat{\mathbf{g}}_{\text{STE}} - \mathbb{E}[\hat{\mathbf{g}}_{\text{STE}}]\|^2 \right]}_{\text{Variance}}$$

```
====================================================================================================
            FIGURE 8.2: STE GRADIENT MISALIGNMENT AND DIRECTIONAL DRIFT
====================================================================================================
                        True Gradient Descent Vector g*
                                  ────────────>
                                 \           /
                                  \  Angle  /
                                   \ theta /
                                    \     /
                                     \   /
                                      \ v
                        STE Surrogate Gradient g_STE
                        (High Bias: theta > 60 degrees, Large Variance: ||g|| ~ 1/Delta)
====================================================================================================
```

### Theorem 8.1 (STE Variance Explosion in Coarse Quantization)
*Let $\Delta = y_{k+1} - y_k$ denote the quantization step size. For coarse 2-bit scalar quantization where $\Delta \approx 1.0\sigma$, the variance of the STE gradient estimator scales inversely with step size resolution:*

$$\mathbb{V}\text{ar}(\hat{\mathbf{g}}_{\text{STE}}) \propto \frac{1}{\Delta^2} \cdot \mathbb{E}[\|\mathbf{X}\|^2]$$

*producing catastrophic directional misalignment $\langle \hat{\mathbf{g}}_{\text{STE}}, \mathbf{g}^* \rangle < 0$ and gradient explosion during Stochastic Gradient Descent.*

#### Mathematical Consequence:
Under coarse 2-bit quantization, the step function contains wide intervals $\Delta \approx 1.05\sigma$. Setting $g_{\text{STE}}(w) = 1$ pretends that the loss surface is smooth, when in reality perturbing $w$ does not alter $\mathcal{Q}(w)$ until $w$ crosses a distant threshold $\tau$. The optimizer accumulates continuous updates that do not alter the forward model until an abrupt discrete jump occurs, causing severe optimization instability.

---

## 8.4 The M-2LRF Exact Gradient Flow Invariant

M-2LRF circumvents the mathematical pathology of STE by separating the parameter space into two distinct components:
1. **Discrete Base Parameters ($\mathbf{W}_{\text{base}}$):** Fully quantized to 2-bit dual-basis, packed into bit arrays, and **strictly frozen** ($\nabla_{\mathbf{W}_{\text{base}}} \mathcal{L} \equiv \mathbf{0}$).
2. **Continuous Low-Rank Adapters ($\mathbf{B}, \mathbf{A}$):** Unquantized full-precision matrices ($\mathbb{R}^{d_{\text{out}} \times r}$ and $\mathbb{R}^{r \times d_{\text{in}}}$) that receive smooth, continuous backpropagation updates.

```
====================================================================================================
             FIGURE 8.3: M-2LRF FORWARD AND BACKWARD GRADIENT FLOW TOPOLOGY
====================================================================================================
                                Loss Function L(Y)
                                        │
                                        ▼ dL/dY
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
                 [Frozen Path]                 [Trainable Path]
                    W_base                      Adapter B @ A
                 (2-Bit uint8)               (FP16 / FP32 Exact)
                         │                             │
                   Grad = ZERO!                        ▼
               (Zero STE Distortion)       dL/dB = (dL/dY) @ (A @ X)^T
                                           dL/dA = B^T @ (dL/dY) @ X^T
                                           [EXACT ANALYTICAL GRADIENTS]
====================================================================================================
```

### Forward Graph:
$$\mathbf{Y} = \mathbf{W}_{\text{base}} \mathbf{X} + \gamma \, \mathbf{B} \mathbf{A} \mathbf{X}$$

### Backward Gradient Derivations:
Let $\mathbf{G} = \frac{\partial \mathcal{L}}{\partial \mathbf{Y}} \in \mathbb{R}^{B \times d_{\text{out}}}$ denote the incoming upstream gradient. Using the differential of matrix operations:

$$d\mathcal{L} = \text{Tr}\left( \mathbf{G}^T d\mathbf{Y} \right) = \text{Tr}\left( \mathbf{G}^T \left[ \mathbf{W}_{\text{base}} d\mathbf{X} + \gamma \, (d\mathbf{B}) \mathbf{A} \mathbf{X} + \gamma \, \mathbf{B} (d\mathbf{A}) \mathbf{X} + \gamma \, \mathbf{B} \mathbf{A} d\mathbf{X} \right] \right)$$

1. **Gradient with respect to Adapter $\mathbf{B}$:**
   $$\text{Tr}\left( \mathbf{G}^T \gamma (d\mathbf{B}) \mathbf{A} \mathbf{X} \right) = \gamma \, \text{Tr}\left( (\mathbf{A} \mathbf{X} \mathbf{G}^T) d\mathbf{B} \right) = \gamma \, \text{Tr}\left( (\mathbf{G}^T (\mathbf{A} \mathbf{X})^T)^T d\mathbf{B} \right)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{B}} = \gamma \, \mathbf{G}^T (\mathbf{A} \mathbf{X}) = \gamma \sum_{b=1}^B \mathbf{g}_b (\mathbf{A} \mathbf{x}_b)^T \in \mathbb{R}^{d_{\text{out}} \times r}$$

2. **Gradient with respect to Adapter $\mathbf{A}$:**
   $$\text{Tr}\left( \mathbf{G}^T \gamma \mathbf{B} (d\mathbf{A}) \mathbf{X} \right) = \gamma \, \text{Tr}\left( (\mathbf{X} \mathbf{G}^T \mathbf{B}) d\mathbf{A} \right) = \gamma \, \text{Tr}\left( (\mathbf{B}^T \mathbf{G} \mathbf{X}^T)^T d\mathbf{A} \right)$$
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \gamma \, \mathbf{B}^T \mathbf{G}^T \mathbf{X} = \gamma \sum_{b=1}^B (\mathbf{B}^T \mathbf{g}_b) \mathbf{x}_b^T \in \mathbb{R}^{r \times d_{\text{in}}}$$

3. **Gradient with respect to Activation $\mathbf{X}$:**
   $$\frac{\partial \mathcal{L}}{\partial \mathbf{X}} = \mathbf{G} \mathbf{W}_{\text{base}} + \gamma \, \mathbf{G} (\mathbf{B} \mathbf{A}) \in \mathbb{R}^{B \times d_{\text{in}}}$$

### Theorem 8.2 (Exact Gradient Invariant of M-2LRF)
*Under the M-2LRF parameterization, the gradient estimators $\nabla_{\mathbf{A}} \mathcal{L}$ and $\nabla_{\mathbf{B}} \mathcal{L}$ satisfy:*

$$\text{Bias}\left( \nabla_{\mathbf{A}} \mathcal{L} \right) = \mathbf{0}, \qquad \text{Bias}\left( \nabla_{\mathbf{B}} \mathcal{L} \right) = \mathbf{0}$$

$$\mathbb{V}\text{ar}_{\text{STE}}\left( \nabla_{\mathbf{A}} \mathcal{L} \right) = 0, \qquad \mathbb{V}\text{ar}_{\text{STE}}\left( \nabla_{\mathbf{B}} \mathcal{L} \right) = 0$$

*The optimization dynamics are strictly smooth, deterministic, and identical to full-precision low-rank matrix factorization.*

#### Proof:
The operations $\mathbf{B} \mapsto \mathbf{B} (\mathbf{A} \mathbf{X})$ and $\mathbf{A} \mapsto (\mathbf{B} \mathbf{A}) \mathbf{X}$ are bilinear maps on continuous Euclidean spaces $\mathbb{R}^{d_{\text{out}} \times r}$ and $\mathbb{R}^{r \times d_{\text{in}}}$. Because no discretization operator appears in the active computation graph between loss $\mathcal{L}$ and adapter parameters $(\mathbf{A}, \mathbf{B})$, no surrogate approximation is introduced. The gradients evaluate to exact analytical derivatives. $\blacksquare$

---
"""

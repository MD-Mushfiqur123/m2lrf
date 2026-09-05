# -*- coding: utf-8 -*-
"""
Appendix, Index of Theorems, and Academic Bibliography for M-2LRF Volume 1 Treatise.
"""

APPENDIX = r"""
# APPENDIX A: INDEX OF FORMAL THEOREMS, LEMMAS & PROOFS

| Formal Statement | Chapter | Subject Matter | Key Mathematical Finding |
|---|---|---|---|
| **Theorem 1.1** | Ch. 1 | Differential Entropy Bound | Gaussian maximizes differential entropy $h(X) \le \frac{1}{2}\log_2(2\pi e \sigma^2)$ for fixed variance. |
| **Theorem 1.2** | Ch. 1 | Gaussian Rate-Distortion | Shannon rate-distortion function $R(D) = \frac{1}{2}\log_2(\sigma^2 / D)$, yielding $D(R) = \sigma^2 2^{-2R}$. |
| **Theorem 1.3** | Ch. 1 | Discrete Lloyd-Max Limit | Maximum 4-level scalar Gaussian SQNR without entropy coding is strictly bounded by $9.3009\text{ dB}$. |
| **Lemma 1.1** | Ch. 1 | First Gaussian Moments | Indefinite integral $\int x \phi(x) dx = -\phi(x) + C$, yielding closed-form centroid equations. |
| **Lemma 1.2** | Ch. 1 | Zero-th Gaussian Moments | Cumulative probability mass $\int_0^\tau \phi(x) dx = \Phi(\tau) - 0.5$. |
| **Theorem 2.1** | Ch. 2 | Disjointness Invariant | Construction $T_0(w) \cdot T_1(w) = 0$ holds identically $\forall w \in \mathbb{R}$, yielding a 4-state bijection. |
| **Theorem 2.2** | Ch. 2 | Sub-Gaussian Centroid Derivation | Sub-Gaussian lattice with $\beta=4.24, s=2.04$ yields unique centroids $a_0^*=0.5286, a_1^*=1.6033, \tau^*=1.0659$. |
| **Theorem 2.3** | Ch. 2 | Uniform Lattice Limit | Ratio of outer to inner centroids converges identically to $3.0000$ as $\beta \to \infty$. |
| **Theorem 3.1** | Ch. 3 | Kurtosis Inflation Law | Sparse GMM outlier mixture inflates kurtosis to $\kappa \approx \frac{3}{\epsilon} (\frac{\epsilon\gamma}{1+\epsilon\gamma})^2 \gg 3$. |
| **Theorem 3.2** | Ch. 3 | Inlier Noise Amplification | Global scale stretching inflates inlier MSE, collapsing empirical SQNR by $-10\log_{10}(1 + \frac{\kappa-3}{12})\text{ dB}$. |
| **Theorem 4.1** | Ch. 4 | Orthogonal Linear Invariance | Inner product identity: $\tilde{\mathbf{X}} \tilde{\mathbf{W}}^T = (\mathbf{X}\mathbf{Q})(\mathbf{W}\mathbf{Q})^T = \mathbf{X}\mathbf{W}^T = \mathbf{Y}$ holds algebraically. |
| **Theorem 4.2** | Ch. 4 | Residual Isometry | Unitary trace invariance preserves Frobenius quantization error: $\|\mathbf{W} - \hat{\mathbf{W}}\|_F^2 = \|\tilde{\mathbf{W}} - \hat{\tilde{\mathbf{W}}}\|_F^2$. |
| **Theorem 4.3** | Ch. 4 | Incoherence Kurtosis CLT | Expected sample kurtosis under randomized Hadamard rotation satisfies $\mathbb{E}[\kappa(\tilde{\mathbf{w}})] = 3 + \frac{\kappa(\mathbf{w})-3}{d}$. |
| **Theorem 4.4** | Ch. 4 | Peak Outlier Bounding | Peak coordinate amplitude is suppressed by $\mathcal{O}(\sqrt{d})$ factor into the interior Lloyd-Max envelope. |
| **Theorem 5.1** | Ch. 5 | Eckart-Young-Mirsky Theorem | Truncated SVD $\mathbf{M}_r = \mathbf{U}_r \mathbf{\Sigma}_r \mathbf{V}_r^T$ uniquely minimizes error in both Frobenius and spectral norms. |
| **Theorem 5.2** | Ch. 5 | Step-0 LoftQ Dominance | LoftQ fits adapters to the residual of a residual, reducing step-0 error below $2.5\%$ vs $11.7\%$ for PiSSA. |
| **Theorem 6.1** | Ch. 6 | Optimal Bit Allocation | KKT multipliers yield closed-form bit-rate allocation $b_l^* = \bar{b} + \frac{1}{2}\log_2(\Omega_l / \bar{\Omega})$. |
| **Lemma 6.1** | Ch. 6 | KFAC Trace Factorization | Trace of Kronecker-factored Fisher matrix factorizes as $\text{Tr}(\mathbf{F}) \approx \text{Tr}(\mathbf{A}) \cdot \text{Tr}(\mathbf{S})$. |
| **Theorem 7.1** | Ch. 7 | Double Quantization Perturbation | 8-bit scale factor quantization introduces $\le 0.00019\text{ dB}$ SQNR loss, bounded strictly by $0.015\text{ dB}$. |
| **Theorem 8.1** | Ch. 8 | STE Variance Explosion | At 2-bit precision, STE gradient variance explodes as $\mathcal{O}(\Delta^{-2})$, causing directional misalignment. |
| **Theorem 8.2** | Ch. 8 | M-2LRF Exact Gradient Invariant | Base weights are frozen; adapter gradients have zero STE bias ($\text{Bias}=\mathbf{0}$) and zero STE variance. |
| **Theorem 9.1** | Ch. 9 | Sarkar Tree Distortion | Embedding regular trees of depth $h$ into Euclidean space requires $d = \Omega(2^{h/2})$ to avoid exponential distortion. |
| **Theorem 9.2** | Ch. 9 | Boundary Hyperbolic Distortion | Hyperbolic distance expands logarithmically near the Poincaré boundary, necessitating high-precision embedding preservation. |
| **Theorem 10.1** | Ch. 10 | Graph-SVD Equivalence | The spectrum of bipartite weight graph $\mathbf{A}_G$ consists of symmetric pairs $\pm \sigma_i(\mathbf{W})$ and zero modes. |
| **Theorem 10.2** | Ch. 10 | Spectral Rank Allocation | Power-law spectral decay $\sigma_i \propto i^{-\alpha(l)}$ dictates optimal adapter rank scaling $r_l^* \propto \alpha(l)^{-\eta}$. |

---

# APPENDIX B: FORMAL MATHEMATICAL SYMBOL GLOSSARY

| Symbol | Mathematical Concept | Formal Domain / Definition |
|---|---|---|
| $X, W$ | Continuous random variables (weight source) | $\mathbb{R}$ with density $p(x)$ |
| $\sigma^2$ | Second-order parameter variance | $\sigma^2 = \mathbb{E}[(W - \mu)^2]$ |
| $\kappa$ | Fourth standardized moment (kurtosis) | $\kappa = \mathbb{E}[(W - \mu)^4] / \sigma^4$ |
| $\kappa_{\text{ex}}$ | Excess kurtosis | $\kappa_{\text{ex}} = \kappa - 3$ |
| $h(X)$ | Continuous differential entropy | $-\int p(x) \log_2 p(x) dx$ (bits) |
| $R(D)$ | Shannon rate-distortion function | $\inf I(X; \hat{X})$ subject to distortion $\le D$ |
| $D(R)$ | Distortion-rate function | $\sigma^2 2^{-2R}$ for Gaussian source |
| $\text{SQNR}$ | Signal-to-Quantization-Noise Ratio | $10 \log_{10}(\sigma^2 / D)$ (dB) |
| $\mathbf{T}_0, \mathbf{T}_1$ | Discrete ternary basis matrices | $\{-1, 0, +1\}^{d_{\text{out}} \times d_{\text{in}}}$ |
| $\alpha_0, \alpha_1$ | Dual-basis scaling centroids | Positive real scalars or group-wise vectors |
| $\tau$ | Nearest-neighbor decision threshold | $\tau = (\alpha_0 + \alpha_1) / 2$ |
| $\odot$ | Hadamard elementwise product | $(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} B_{ij}$ |
| $\otimes$ | Kronecker matrix product | Block matrix $(A_{ij} \mathbf{B})$ |
| $\mathbf{H}_d$ | Sylvester Walsh-Hadamard matrix | $\{-1, +1\}^{d \times d}, \mathbf{H}_d^2 = d \mathbf{I}_d$ |
| $\hat{\mathbf{H}}_d$ | Orthonormal Walsh-Hadamard matrix | $\frac{1}{\sqrt{d}} \mathbf{H}_d \in \mathcal{O}(d)$ |
| $\mathbf{D}$ | Diagonal Rademacher matrix | $\text{diag}(s_1, \dots, s_d), s_i \in \{-1, +1\}$ |
| $\mathbf{Q}$ | Randomized orthogonal rotation matrix | $\mathbf{Q} = \mathbf{D} \hat{\mathbf{H}}_d \in \mathcal{O}(d)$ |
| $\mathbf{R}$ | Quantization residual error matrix | $\mathbf{R} = \mathbf{W}_0 - \mathbf{W}_{\text{base}}$ |
| $\mathbf{U}_r, \mathbf{\Sigma}_r, \mathbf{V}_r^T$ | Rank-$r$ Truncated SVD components | $\mathbf{R}_r = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ |
| $\mathbf{B}, \mathbf{A}$ | Low-Rank Adaptation (LoRA) matrices | $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}, \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$ |
| $\gamma$ | LoRA scaling hyperparameter | $\gamma = \alpha_{\text{lora}} / r$ |
| $\mathbf{F}$ | Fisher Information Matrix (FIM) | $\mathbb{E}[\nabla \log p \nabla \log p^T]$ |
| $\mathbf{A}_l, \mathbf{S}_l$ | KFAC covariance factors | $\mathbf{A} = \mathbb{E}[\mathbf{x}\mathbf{x}^T], \mathbf{S} = \mathbb{E}[\mathbf{g}\mathbf{g}^T]$ |
| $G$ | Sub-vector group size | $G \in \{64, 128\}$ elements |
| $\gamma_s, z_s$ | Secondary scale factor and zero-point | FP32 parameters for 8-bit DQ scales |
| $g_{\text{STE}}$ | Straight-Through Estimator surrogate | Backward surrogate for $\frac{d\mathcal{Q}}{dw}$ |
| $\mathbb{B}^d$ | Poincaré ball model of hyperbolic space | $\{\mathbf{x} \in \mathbb{R}^d : \|\mathbf{x}\|_2 < 1\}$ |
| $\mathbb{H}^d$ | Lorentz hyperboloid model | $\{\mathbf{x} \in \mathbb{R}^{d+1} : \langle \mathbf{x}, \mathbf{x} \rangle_{\mathcal{L}} = -1, x_0 > 0\}$ |
| $d_{\mathbb{B}}(\mathbf{u}, \mathbf{v})$ | Hyperbolic geodesic distance | $\text{arcosh}(1 + 2 \frac{\|\mathbf{u}-\mathbf{v}\|^2}{(1-\|\mathbf{u}\|^2)(1-\|\mathbf{v}\|^2)})$ |
| $\mathbf{A}_G$ | Bipartite graph adjacency matrix | $\begin{bmatrix} \mathbf{0} & \mathbf{W} \\ \mathbf{W}^T & \mathbf{0} \end{bmatrix}$ |
| $p_{\text{MP}}(\lambda)$ | Marchenko-Pastur probability density | Random matrix spectral distribution |
| $\alpha(l)$ | Power-law spectral decay exponent | $\sigma_i \propto i^{-\alpha(l)}$ at layer depth $l$ |

---

# APPENDIX C: COMPREHENSIVE ACADEMIC BIBLIOGRAPHY

1. **Shannon, C. E.** (1948). "A Mathematical Theory of Communication." *Bell System Technical Journal*, 27(3), 379–423.
2. **Shannon, C. E.** (1959). "Coding Theorems for a Discrete Source with a Fidelity Criterion." *IRE National Convention Record*, 7(4), 142–163.
3. **Lloyd, S. P.** (1982). "Least Squares Quantization in PCM." *IEEE Transactions on Information Theory*, 28(2), 129–137. (Originally documented as internal Bell Labs technical memorandum in 1957).
4. **Max, J.** (1960). "Quantizing for Minimum Distortion." *IRE Transactions on Information Theory*, 6(1), 7–12.
5. **Eckart, C., & Young, G.** (1936). "The Approximation of One Matrix by Another of Lower Rank." *Psychometrika*, 1(3), 211–218.
6. **Mirsky, L.** (1960). "Symmetric Gauge Functions and Unitarily Invariant Norms." *The Quarterly Journal of Mathematics*, 11(1), 50–59.
7. **Hadamard, J.** (1893). "Résolution d'une question relative aux déterminants." *Bulletin des Sciences Mathématiques*, 17, 240–246.
8. **Sylvester, J. J.** (1867). "LX. Thoughts on inverse orthogonal matrices, reporting of equal sign, and other matters." *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of Science*, 34(232), 461–475.
9. **Walsh, J. L.** (1923). "A Closed Set of Normal Orthogonal Functions." *American Journal of Mathematics*, 45(1), 5–24.
10. **Bengio, Y., Léonard, N., & Courville, A.** (2013). "Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation." *arXiv preprint arXiv:1308.3432*.
11. **Hubara, I., Courbariaux, M., Soudry, D., El-Yaniv, R., & Bengio, Y.** (2016). "Binarized Neural Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 29, 4107–4115.
12. **Dettmers, T., Lewis, M., Belkada, Y., & Zettlemoyer, L.** (2022). "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale." *Advances in Neural Information Processing Systems (NeurIPS)*, 35, 30318–30332.
13. **Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L.** (2023). "QLoRA: Efficient Finetuning of Quantized LLMs." *Advances in Neural Information Processing Systems (NeurIPS)*, 36, 10088–10115.
14. **Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., & Han, S.** (2023). "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models." *International Conference on Machine Learning (ICML)*, PMLR 202, 38087–38104.
15. **Tseng, A., Chee, J., Sun, Q., Kuleshov, V., & De Sa, C.** (2023). "QuIP: 2-Bit Quantization of Large Language Models With Incoherence Processing." *Advances in Neural Information Processing Systems (NeurIPS)*, 36, 68123–68145.
16. **Chee, J., Cai, Y., Kuleshov, V., & De Sa, C.** (2024). "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Vector Quantization." *International Conference on Machine Learning (ICML)*, 2024.
17. **Li, Y., Yu, Y., Liang, C., He, P., Chen, W., & Zhao, T.** (2023). "LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models." *International Conference on Learning Representations (ICLR)*, 2024.
18. **Meng, F., Wang, Z., & Zhang, M.** (2024). "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models." *International Conference on Learning Representations (ICLR)*, 2024.
19. **Martens, J., & Grosse, R.** (2015). "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." *International Conference on Machine Learning (ICML)*, 2408–2417.
20. **Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ling, Z., Zhang, S., Chen, Y., & Wei, F.** (2024). "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits." *arXiv preprint arXiv:2402.17764*.
21. **Sarkar, R.** (2011). "Low Distortion Delaunay Embedding of Trees in Hyperbolic Plane." *International Symposium on Graph Drawing*, 355–366.
22. **Nickel, M., & Kiela, D.** (2017). "Poincaré Embeddings for Learning Hierarchical Representations." *Advances in Neural Information Processing Systems (NeurIPS)*, 30, 6338–6347.
23. **Marchenko, V. A., & Pastur, L. A.** (1967). "Distribution of eigenvalues for some sets of random matrices." *Matematicheskii Sbornik*, 114(4), 507–536.
24. **Martin, C. H., & Mahoney, M. W.** (2021). "Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning." *Journal of Machine Learning Research (JMLR)*, 22(165), 1–73.
25. **De Sa, C., Gu, A., Ré, C., & Sala, F.** (2018). "Representation Tradeoffs for Hyperbolic Embeddings." *Proceedings of the 35th International Conference on Machine Learning (ICML)*, PMLR 80, 4460–4469.

---
"""

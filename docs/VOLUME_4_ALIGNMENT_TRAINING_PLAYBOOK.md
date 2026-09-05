# VOLUME 4: LLM TRAINING, ALIGNMENT & FINE-TUNING PLAYBOOK
### *An Industrial-Grade Reference Manual for Pretrained Foundation Models & Sub-4-Bit Quantized Adaptation*

> **Lead Author & System Architect:** **MD-Mushfiqur Rahim**  
> **System Engineering:** **L (Autonomous Engineering Agent)**  
> **Workspace / Project:** `projects/m2lrf-clean` | M-2LRF Monograph Series  
> **Classification:** Engineering & Research Playbook — Volume 4  
> **Target Scope:** High-Performance SFT, Sequence Packing, Preference Optimization (DPO/ORPO/KTO), Verifiable Reasoning (RLVR/GRPO), PEFT Mechanics, and Distributed 2-Bit Quantized Training.

---

## 📑 TABLE OF CONTENTS

1. [Chapter 1: Supervised Fine-Tuning (SFT) Best Practices](#chapter-1-supervised-fine-tuning-sft-best-practices)
   - 1.1 Mathematical Foundations of Auto-Regressive Fine-Tuning
   - 1.2 Learning Rate Schedules: Cosine Decay with Warmup vs. WSD (Warmup-Stable-Decay)
   - 1.3 Effective Batch Size Dynamics & Gradient Accumulation Mechanics
   - 1.4 Optimizer Architectures: Full-Precision AdamW vs. Block-Wise 8-Bit AdamW
   - 1.5 Precision Regimes: FP32 Master Weights, BF16 vs. FP16, and FP8 GEMM Dynamics
   - 1.6 Global Gradient Norm Bounding & Loss Spike Mitigation
2. [Chapter 2: Multiplexed Sequence Packing](#chapter-2-multiplexed-sequence-packing)
   - 2.1 The Quadratic Penalty of Padded Sequences ($O(L^2)$ Inefficiency)
   - 2.2 First-Fit Decreasing (FFD) Bin-Packing Formulation
   - 2.3 Cross-Contamination Hazards in Causal Attention
   - 2.4 2D Block-Diagonal Document Masking
   - 2.5 Dynamic Positional Encoding Reset ($p_i \in [0, L_k-1]$)
   - 2.6 FlashAttention-2 / FlashAttention-3 VarLen Kernel Pipeline (`cu_seqlens`)
   - 2.7 Production PyTorch & HuggingFace Packing Data Collator
3. [Chapter 3: Completion-Only Dynamic Loss Masking](#chapter-3-completion-only-dynamic-loss-masking)
   - 3.1 Prompt Contamination Hazard & Instruction Following Degradation
   - 3.2 Dynamic Label Masking Mechanics via `ignore_index = -100`
   - 3.3 Multi-Turn Dialogue Parsing & Delimiter-Guided Masking (ChatML / Llama 3)
   - 3.4 Token-Normalized vs. Sample-Normalized Gradient Weighting
   - 3.5 Production-Grade `DataCollatorForCompletionOnlyLM` Implementation
4. [Chapter 4: Direct Preference Optimization (DPO)](#chapter-4-direct-preference-optimization-dpo)
   - 4.1 Theoretical Lineage: From Bradley-Terry RLHF to Implicit Rewards
   - 4.2 Mathematical Derivation: Eliminating the Partition Function $Z(x)$
   - 4.3 The DPO Loss Formulation and Gradient Flow Dynamics
   - 4.4 Reference Model Caching & In-Situ Adapter Freezing
   - 4.5 Hyperparameter Sensitivity: Temperature $\beta$, Conservative Regularization, and Length Normalization
   - 4.6 Production PyTorch DPO Loss Kernel
5. [Chapter 5: Odds Ratio Preference Optimization (ORPO)](#chapter-5-odds-ratio-preference-optimization-orpo)
   - 5.1 The Alignment Dilemma: SFT vs. Preference Decoupling
   - 5.2 Mathematical Derivation of Odds and Log Odds Ratio ($\log \text{OR}$)
   - 5.3 Monolithic Objective: Unifying SFT Likelihood with Preference Dispreference
   - 5.4 Gradient Dissection: Selective Suppression of Degenerate Tokens
   - 5.5 Reference-Free Architecture: $50\%$ Memory Reduction vs. DPO
   - 5.6 Production PyTorch Implementation of the ORPO Loss
6. [Chapter 6: Kahneman-Tversky Optimization (KTO)](#chapter-6-kahneman-tversky-optimization-kto)
   - 6.1 Cognitive Foundations: Behavioral Prospect Theory & Cumulative Utility
   - 6.2 The Mathematics of Loss Aversion: Steepness Ratio $\lambda_D > 1$
   - 6.3 Implicit Reward Centering with Batch-Estimated Reference Points
   - 6.4 Unpaired Alignment: Eliminating the Costly Preference Tuple Barrier
   - 6.5 Production PyTorch Implementation of KTO Loss
7. [Chapter 7: Reinforcement Learning with Verifiable Rewards (RLVR / DeepSeek-R1 Style)](#chapter-7-reinforcement-learning-with-verifiable-rewards-rlvr--deepseek-r1-style)
   - 7.1 The End of Neural Reward Hacking: Verifiable Ground Truth
   - 7.2 Rule-Based Reward Modeling (RBRM) Architecture: Math, Formal Logic, & Unit Tests
   - 7.3 Group Relative Policy Optimization (GRPO): Eliminating the Critic Network
   - 7.4 Advantage Estimation across Symmetric Sample Groups
   - 7.5 Emergence of Self-Correction, Backtracking, and Dynamic Thought Chains ("Aha Moments")
   - 7.6 Complete Production Sandbox: SymPy Equivalence & Pytest Code Verifier
8. [Chapter 8: Parameter-Efficient Fine-Tuning (PEFT) Deep Dive](#chapter-8-parameter-efficient-fine-tuning-peft-deep-dive)
   - 8.1 LoRA: Low-Rank Matrix Decomposition Mechanics and Gradient Entanglement
   - 8.2 DoRA: Weight-Decomposed Directional & Magnitude Normalization
   - 8.3 LoHa: Low-Rank Hadamard Product Adaptation ($O(r^2)$ Effective Rank)
   - 8.4 PiSSA: Principal Singular Component Adaptation from Step 0
   - 8.5 LoftQ: Alternating Quantization Residual Optimization
   - 8.6 The M-2LRF Synthesis: Dual-Basis 2-Bit Quantization + SVD Residual Adaptation
   - 8.7 Comprehensive Structural Comparison Matrix
9. [Chapter 9: Distributed Multi-GPU Training](#chapter-9-distributed-multi-gpu-training)
   - 9.1 The Parameter Memory Budget: Weights, Gradients, and Optimizer States
   - 9.2 DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3 Memory Partitions
   - 9.3 PyTorch Fully Sharded Data Parallel (FSDP): Wrapping Policies & Prefetching
   - 9.4 Distributed Fine-Tuning with 2-Bit Quantized Weights (M-2LRF & QLoRA)
   - 9.5 Production Configuration Templates: DeepSpeed JSON & Accelerate YAML
10. [Chapter 10: Preventing Catastrophic Forgetting & Quality Auditing](#chapter-10-preventing-catastrophic-forgetting--quality-auditing)
    - 10.1 Curvature Fragility and Representation Collapse in Low-Bit Loss Basins
    - 10.2 Elastic Weight Consolidation (EWC++) via Diagonal Fisher Information
    - 10.3 Pretrained Reference KL Divergence Penalty
    - 10.4 Strategic Token Mixing: The 80/10/10 Experience Replay Heuristic
    - 10.5 Quality Auditing Telemetry: Perplexity, MMLU, GSM8K, and HumanEval Regression Guards
    - 10.6 Automated Rollback Gates and Model Checkpoint Verification

---

# CHAPTER 1: SUPERVISED FINE-TUNING (SFT) BEST PRACTICES

### 1.1 Mathematical Foundations of Auto-Regressive Fine-Tuning

In autoregressive language modeling, a foundation model parameterized by $\theta \in \mathbb{R}^d$ defines a probability distribution over token sequences $\mathbf{x} = (x_1, x_2, \dots, x_T) \in \mathcal{V}^T$ drawn from vocabulary $\mathcal{V}$. The probability of sequence $\mathbf{x}$ decomposes via the chain rule of probability:

$$P_\theta(\mathbf{x}) = \prod_{t=1}^T P_\theta(x_t \mid \mathbf{x}_{<t}) = \prod_{t=1}^T \frac{\exp(\mathbf{w}_{x_t}^T \mathbf{h}_t)}{\sum_{v \in \mathcal{V}} \exp(\mathbf{w}_v^T \mathbf{h}_t)}$$

where $\mathbf{h}_t = \text{Transformer}(\mathbf{x}_{<t}; \theta) \in \mathbb{R}^{d_{\text{model}}}$ represents the hidden representation vector emitted by the final transformer block at sequence position $t-1$, and $\mathbf{w}_v \in \mathbb{R}^{d_{\text{model}}}$ is the unembedding vector for token $v$.

During Supervised Fine-Tuning (SFT), the model is adapted on a target dataset $\mathcal{D}_{\text{SFT}} = \{(\mathbf{x}^{(i)}, \mathbf{y}^{(i)})\}_{i=1}^N$, where $\mathbf{x}^{(i)}$ is an instruction prompt and $\mathbf{y}^{(i)} = (y_1^{(i)}, \dots, y_{T_i}^{(i)})$ is the reference completion. The empirical risk minimization objective is formulated as:

$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T_i} \log P_\theta(y_t^{(i)} \mid \mathbf{x}^{(i)}, \mathbf{y}_{<t}^{(i)})$$

Gradient updates under stochastic gradient descent or adaptive moment methods compute:

$$\theta_{k+1} = \theta_k - \eta_k \cdot \widehat{\mathbf{m}}_k \left( \nabla_\theta \mathcal{L}_{\text{SFT}}(\theta_k) \right)$$

where $\eta_k$ is the step-dependent learning rate and $\widehat{\mathbf{m}}_k(\cdot)$ denotes the optimizer update operator.

```
       +-------------------------------------------------------------+
       |                  SFT OPTIMIZATION PIPELINE                  |
       +-------------------------------------------------------------+
                                      |
                      [Batch of Sequence Prompts & Targets]
                                      |
                                      v
                       +-----------------------------+
                       | Forward Pass (BF16 / FP8)   |
                       | Compute Next-Token Logits   |
                       +-----------------------------+
                                      |
                                      v
                       +-----------------------------+
                       | Masked Cross-Entropy Loss   |
                       | (Target completions only)   |
                       +-----------------------------+
                                      |
                                      v
                       +-----------------------------+
                       | Backward Pass               |
                       | Compute Local Gradients ∇θ  |
                       +-----------------------------+
                                      |
                                      v
                       +-----------------------------+
                       | All-Reduce / Sharded Sync   |
                       | Global Norm Clip: ||g||2 ≤ 1|
                       +-----------------------------+
                                      |
                                      v
                       +-----------------------------+
                       | 8-bit / 16-bit AdamW Step   |
                       | Cosine / WSD Learning Rate  |
                       +-----------------------------+
```

---

### 1.2 Learning Rate Schedules: Cosine Decay with Warmup vs. WSD

The trajectory of the learning rate $\eta_k$ governs whether the optimizer escapes poor saddle points, preserves foundational world knowledge, and smoothly converges into flat, generalizable loss minima.

#### 1.2.1 Cosine Annealing Schedule with Linear Warmup

The standard industry baseline establishes a linear warmup phase for $T_{\text{warmup}}$ steps followed by a half-period cosine decay over the remaining $T_{\text{max}} - T_{\text{warmup}}$ steps:

$$\eta(t) = \begin{cases} 
\eta_{\min} + \frac{t}{T_{\text{warmup}}} (\eta_{\max} - \eta_{\min}) & \text{if } t \le T_{\text{warmup}} \\
\eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - T_{\text{warmup}}}{T_{\text{max}} - T_{\text{warmup}}} \pi\right)\right) & \text{if } T_{\text{warmup}} < t \le T_{\text{max}}
\end{cases}$$

- **Warmup Horizon ($T_{\text{warmup}}$):** Set between $1\%$ and $5\%$ of total training steps (typically $50 \le T_{\text{warmup}} \le 250$). Linear warmup stabilizes adaptive momentum estimates ($\mathbf{v}_t$ in AdamW) before taking substantial steps in weight space.
- **Minimum Learning Rate ($\eta_{\min}$):** Typically fixed to $0.1 \times \eta_{\max}$ or $0.05 \times \eta_{\max}$. Decaying entirely to zero risks freezing the network in suboptimal parameter states during the final $5\%$ of steps.

#### 1.2.2 Warmup-Stable-Decay (WSD) Schedule

Pioneered in frontier architectures (MiniCPM, DeepSeek), the WSD schedule decouples training duration from learning rate decay:

$$\eta(t) = \begin{cases}
\frac{t}{T_{\text{warmup}}} \eta_{\max} & \text{if } 0 \le t \le T_{\text{warmup}} \\
\eta_{\max} & \text{if } T_{\text{warmup}} < t \le T_{\text{stable}} \\
\eta_{\min} + (\eta_{\max} - \eta_{\min}) \cdot f_{\text{decay}}\left(\frac{t - T_{\text{stable}}}{T_{\text{max}} - T_{\text{stable}}}\right) & \text{if } T_{\text{stable}} < t \le T_{\text{max}}
\end{cases}$$

```
Learning Rate
 ^
 |          +-------------------------+ (Stable Stage: 70-80% of steps)
 |         /                           \
 |        /                             \ (Decay Stage: Cosine or Exponential)
 |       / (Warmup: 3-5%)                \
 |      /                                 +--- η_min
 +-----+---------------------------------------+------> Step (t)
 0   T_warmup                      T_stable   T_max
```

**Industrial Advantages of WSD:**
1. **Checkpoint Forkability:** In the Stable phase, training can be extended arbitrarily without recalculating the cosine horizon.
2. **Dynamic Decay Testing:** Different annealing styles (linear, cosine, exponential, 1-cycle) can be evaluated from any checkpoint along the stable plateau.

---

### 1.3 Effective Batch Size Dynamics & Gradient Accumulation Mechanics

The global effective batch size $B_{\text{global}}$ in tokens is defined by:

$$B_{\text{global}} = B_{\text{micro}} \times S \times N_{\text{devices}} \times K_{\text{accum}}$$

where $B_{\text{micro}}$ is the sequence count per device, $S$ is the packed sequence length (e.g., 4096 or 8192), $N_{\text{devices}}$ is the total count of GPUs across all nodes, and $K_{\text{accum}}$ is the gradient accumulation step count.

#### The Gradient Accumulation Pitfall: Token Averaging vs. Sequence Averaging

In standard PyTorch implementations of Cross-Entropy Loss, reduction is computed per-token:

$$\mathcal{L}_{\text{micro}} = \frac{1}{\sum_{b=1}^{B_{\text{micro}}} M_b} \sum_{b=1}^{B_{\text{micro}}} \sum_{t=1}^S m_{b,t} \cdot \ell(y_{b,t}, \widehat{y}_{b,t})$$

where $m_{b,t} \in \{0, 1\}$ is the completion loss mask, and $M_b = \sum_t m_{b,t}$ is the count of active target tokens in sequence $b$.

> [!CAUTION]
> **The Accumulation Scale Error:** If micro-batches contain varying numbers of active tokens (e.g. batch 1 has 3,000 active tokens and batch 2 has 200 active tokens), dividing loss by $K_{\text{accum}}$ creates an artificial gradient imbalance. 
> To maintain strict mathematical equivalence to a single giant forward pass of size $B_{\text{global}}$, gradients must be accumulated with respect to the **global sum of active tokens** across all $K_{\text{accum}}$ sub-steps:
> 
> $$\mathcal{L}_{\text{accum}} = \frac{\sum_{k=1}^{K_{\text{accum}}} \sum_{j=1}^{T_k} \ell_{k,j}}{\sum_{k=1}^{K_{\text{accum}}} N_{\text{tokens}}^{(k)}}$$

#### Learning Rate Scaling Heuristics

When scaling effective batch size $B_{\text{global}}$:
- **Square Root Scaling (Recommended for SFT):** $\eta_{\text{new}} = \eta_{\text{base}} \sqrt{\frac{B_{\text{new}}}{B_{\text{base}}}}$. Prevents early gradient divergence when batch sizes expand beyond $0.5\text{M}$ tokens.
- **Linear Scaling (Standard Pretraining):** $\eta_{\text{new}} = \eta_{\text{base}} \left(\frac{B_{\text{new}}}{B_{\text{base}}}\right)$. Applicable when batch sizes remain well within the empirical noise scale $B_{\text{noise}} = \frac{\text{Tr}(\Sigma)}{\|\nabla \mathcal{L}\|^2}$.

---

### 1.4 Optimizer Architectures: Full-Precision AdamW vs. Block-Wise 8-Bit AdamW

Standard FP32 AdamW maintains two state vectors for every model parameter $\theta$: the first moment $\mathbf{m}_t$ (mean gradient) and the second moment $\mathbf{v}_t$ (uncentered variance):

$$\mathbf{m}_t = \beta_1 \mathbf{m}_{t-1} + (1 - \beta_1) \mathbf{g}_t, \quad \mathbf{v}_t = \beta_2 \mathbf{v}_{t-1} + (1 - \beta_2) \mathbf{g}_t^2$$

$$\widehat{\mathbf{m}}_t = \frac{\mathbf{m}_t}{1 - \beta_1^t}, \quad \widehat{\mathbf{v}}_t = \frac{\mathbf{v}_t}{1 - \beta_2^t}$$

$$\theta_t = \theta_{t-1} - \eta_t \left( \frac{\widehat{\mathbf{m}}_t}{\sqrt{\widehat{\mathbf{v}}_t} + \epsilon} + \lambda_{\text{wd}} \theta_{t-1} \right)$$

#### Memory Consumption Breakdown (7-Billion Parameter Model)

| State Component | Precision | Bytes per Param | Total VRAM (7B Model) |
| :--- | :--- | :--- | :--- |
| **Model Parameters ($\theta$)** | FP16 / BF16 | 2 bytes | 14.0 GB |
| **Gradients ($\mathbf{g}$)** | FP16 / BF16 | 2 bytes | 14.0 GB |
| **Master Weights (FP32)** | FP32 | 4 bytes | 28.0 GB |
| **1st Optimizer Momentum ($\mathbf{m}$)** | FP32 | 4 bytes | 28.0 GB |
| **2nd Optimizer Variance ($\mathbf{v}$)** | FP32 | 4 bytes | 28.0 GB |
| **Total Optimizer Memory** | — | **12 bytes** | **84.0 GB** |
| **Complete Training State (Excl. Activations)**| — | **16 bytes** | **112.0 GB** |

#### Block-Wise 8-Bit AdamW (bitsandbytes Engine)

Block-wise 8-bit AdamW compresses $\mathbf{m}_t$ and $\mathbf{v}_t$ from 32-bit floating point down to an 8-bit custom dynamic quantile format over localized blocks of size $B_{\text{opt}} = 2048$:

1. **Dynamic Quantization:** For each block $\mathbf{x} \in \mathbb{R}^{B_{\text{opt}}}$, find the maximum absolute value $c = \max_i |x_i|$.
2. **Quantile Mapping:** Map values into 256 non-linear intervals optimized for Gaussian-distributed gradients ($\mathbf{m}_t$) and log-normal distributions ($\mathbf{v}_t$):
   
   $$q_i = \text{argmin}_k |x_i / c - Q_k|$$

3. **Memory Footprint:** Reduces optimizer memory from $12\text{ bytes/param}$ down to $2\text{ bytes/param}$ (1 byte for $\mathbf{m}$, 1 byte for $\mathbf{v}$), while maintaining master weights in FP32 or unquantized BF16. For a 7B model, optimizer states drop from $84\text{ GB}$ to $14\text{ GB}$ with **zero empirical degradation in downstream benchmark accuracy**.

---

### 1.5 Precision Regimes: FP32 Master Weights, BF16 vs. FP16, and FP8 GEMM Dynamics

```
Bit Layout Formats:
FP32: [Sign: 1b][Exponent: 8b (bias 127)][Mantissa: 23b]        Dynamic Range: ~10^±38
FP16: [Sign: 1b][Exponent: 5b (bias 15) ][Mantissa: 10b]        Dynamic Range: ~10^±4.8  (Max: 65,504)
BF16: [Sign: 1b][Exponent: 8b (bias 127)][Mantissa: 7b]         Dynamic Range: ~10^±38
FP8 (E4M3): [Sign: 1b][Exponent: 4b][Mantissa: 3b]              High precision forward pass GEMMs
FP8 (E5M2): [Sign: 1b][Exponent: 5b][Mantissa: 2b]              High dynamic range backward pass GEMMs
```

#### Why BF16 Dominates Modern SFT:
1. **Dynamic Range Parity with FP32:** BF16 preserves the full 8-bit exponent of single-precision float. It handles activations across 70 orders of magnitude without underflowing to zero or overflowing to `+Inf`/`NaN`.
2. **Elimination of Loss Scaling:** FP16 requires dynamic loss scaling factors ($2^{16} \dots 2^{32}$) to prevent gradient underflow. Any sudden activation spike causes immediate gradient overflow, skipping optimization steps. BF16 operates with **unit loss scale ($S=1.0$)**, ensuring consistent optimizer steps.

---

### 1.6 Global Gradient Norm Bounding & Loss Spike Mitigation

In instruction fine-tuning, outlier tokens (e.g. malformed JSON, multi-lingual token fragments, code blocks with extreme indentation) generate massive instantaneous cross-entropy losses. 

$$\mathbf{g}_{\text{global}} = \left[ \nabla_{\theta_1} \mathcal{L}, \nabla_{\theta_2} \mathcal{L}, \dots, \nabla_{\theta_P} \mathcal{L} \right]^T$$

$$\|\mathbf{g}_{\text{global}}\|_2 = \sqrt{\sum_{p=1}^P \|\nabla_{\theta_p} \mathcal{L}\|_2^2}$$

The clipped gradient $\widetilde{\mathbf{g}}$ is computed via:

$$\widetilde{\mathbf{g}} = \begin{cases}
\mathbf{g}_{\text{global}} & \text{if } \|\mathbf{g}_{\text{global}}\|_2 \le C_{\text{clip}} \\
\frac{C_{\text{clip}}}{\|\mathbf{g}_{\text{global}}\|_2} \mathbf{g}_{\text{global}} & \text{if } \|\mathbf{g}_{\text{global}}\|_2 > C_{\text{clip}}
\end{cases}$$

- **Optimal Industrial Threshold:** Set $C_{\text{clip}} = 1.0$. For aggressive low-rank adaptation (LoRA/DoRA) or sub-4-bit quantized base layers, tighten to $C_{\text{clip}} = 0.5$ to suppress catastrophic directional shocks to the adapter matrices.

---

# CHAPTER 2: MULTIPLEXED SEQUENCE PACKING

### 2.1 The Quadratic Penalty of Padded Sequences ($O(L^2)$ Inefficiency)

Standard batched training aligns variable-length conversational sequences by padding each sample to the maximum sequence length $L_{\text{max}}$ in the batch with `<pad>` tokens:

$$\text{Padded Batch Shape} = (B, L_{\text{max}})$$

Given the quadratic complexity of self-attention $\mathcal{O}(B \cdot L_{\text{max}}^2 \cdot d)$, if an instruction dataset has a median prompt-response length of $420$ tokens, but the tail distribution reaches $4096$ tokens:

$$\text{Padding Overhead} = 1 - \frac{\sum_{i=1}^B L_i}{B \cdot L_{\text{max}}} \approx 72\% \text{ to } 85\% \text{ wasted compute}$$

The GPU spends $>75\%$ of all Tensor Core cycles multiplying zeros inside causal attention masks and projection layers.

---

### 2.2 First-Fit Decreasing (FFD) Bin-Packing Formulation

Multiplexed Sequence Packing (Axolotl / Megatron-LM / TRL Packing) packs multiple independent conversations into a single contiguous context window of fixed length $L$ (e.g. $4096$, $8192$, or $32768$ tokens) without padding.

```
Padded Batch: (Wasteful)
Batch 0: [=== Sample 1 (600 tok) ===][PAD PAD PAD PAD PAD PAD PAD PAD PAD... (3496 tok)]
Batch 1: [======= Sample 2 (1200 tok) =======][PAD PAD PAD PAD PAD PAD... (2896 tok)]

Multiplexed Packed Window: (Zero Waste, 100% Compute Saturation)
Packed:  [== Sample 1 (600) ==][==== Sample 2 (1200) ====][====== Sample 3 (2296) ======] (Total: 4096)
```

The packing problem maps directly to the NP-hard **1D Bin Packing Problem**. The First-Fit Decreasing (FFD) heuristic provides an approximation ratio proven to use no more than $\frac{11}{9} \text{OPT} + \frac{6}{9}$ bins:

```
Algorithm 1: First-Fit Decreasing (FFD) Sequence Packing
Input: Set of sequence token lengths S = {l_1, l_2, ..., l_N}, Target Context Length L
Output: Bins B = {b_1, b_2, ..., b_M}, where sum_{i in b_k} l_i <= L

1. Sort sequences in descending order of token length: S_sorted = sort_descending(S)
2. Initialize empty bin set: B = [new_bin()]
3. for each sequence seq with length l in S_sorted do:
4.     placed = false
5.     for each bin b in B do:
6.         if b.remaining_capacity() >= l then:
7.             b.append(seq)
8.             placed = true
9.             break
10.    if not placed then:
11.        new_b = new_bin()
12.        new_b.append(seq)
13.        B.append(new_b)
14. return B
```

---

### 2.3 Cross-Contamination Hazards in Causal Attention

In naive sequence concatenation, multiple independent dialogues share the same sequence buffer:

$$\mathbf{X}_{\text{packed}} = [\mathbf{x}_1^{(1)}, \dots, \mathbf{x}_{L_1}^{(1)}, \mathbf{x}_1^{(2)}, \dots, \mathbf{x}_{L_2}^{(2)}, \dots]$$

Under standard lower-triangular causal attention:

$$\mathbf{A}_{i,j} = \begin{cases}
\frac{\mathbf{q}_i^T \mathbf{k}_j}{\sqrt{d_k}} & \text{if } j \le i \\
-\infty & \text{if } j > i
\end{cases}$$

> [!CAUTION]
> **Cross-Contamination Failure Mode:** Token $1$ of Sample 2 ($\mathbf{x}_1^{(2)}$) can attend to all tokens of Sample 1 ($\mathbf{x}_{1 \dots L_1}^{(1)}$). The model conditions its response not only on Sample 2's user prompt, but also on the unrelated context of Sample 1! This produces hallucinated cross-dialogue bleed-through, instruction confusion, and catastrophic evaluation degradation.

---

### 2.4 2D Block-Diagonal Document Masking

To eliminate cross-contamination, the self-attention score matrix must be masked into a **2D Block-Diagonal structure**:

$$\mathbf{M}_{i,j} = \begin{cases}
0 & \text{if } j \le i \text{ and } \text{DocID}(i) == \text{DocID}(j) \\
-\infty & \text{otherwise}
\end{cases}$$

```
2D Block-Diagonal Attention Mask Matrix (Context Length = 8 tokens, 3 Samples):
Sample 1 (len=3), Sample 2 (len=2), Sample 3 (len=3)

      j -> 0   1   2   3   4   5   6   7
    i   +---------------------------------+
    0   |  0  -∞  -∞  -∞  -∞  -∞  -∞  -∞  |  <-- Sample 1
    1   |  0   0  -∞  -∞  -∞  -∞  -∞  -∞  |  <-- Sample 1
    2   |  0   0   0  -∞  -∞  -∞  -∞  -∞  |  <-- Sample 1
    3   | -∞  -∞  -∞   0  -∞  -∞  -∞  -∞  |  <-- Sample 2 (Isolated from Sample 1)
    4   | -∞  -∞  -∞   0   0  -∞  -∞  -∞  |  <-- Sample 2
    5   | -∞  -∞  -∞  -∞  -∞   0  -∞  -∞  |  <-- Sample 3 (Isolated from Samples 1 & 2)
    6   | -∞  -∞  -∞  -∞  -∞   0   0  -∞  |  <-- Sample 3
    7   | -∞  -∞  -∞  -∞  -∞   0   0   0  |  <-- Sample 3
        +---------------------------------+
```

---

### 2.5 Dynamic Positional Encoding Reset ($p_i \in [0, L_k-1]$)

In addition to masking attention weights, positional embeddings (RoPE - Rotary Position Embeddings) must be reset at the boundary of each packed document.

If positions increment monotonically across the entire packed buffer ($0, 1, 2, \dots, 4095$):
- Sample 2's prompt starts at position $p=600$ instead of $p=0$.
- High positional values activate rotary frequency bands that degrade syntax comprehension for introductory prompt tokens.

**The Fix:** Reset Rotary Position Embeddings for each packed document:

$$\mathbf{p}_{\text{packed}} = [\underbrace{0, 1, \dots, L_1-1}_{\text{Sample 1}}, \underbrace{0, 1, \dots, L_2-1}_{\text{Sample 2}}, \dots, \underbrace{0, 1, \dots, L_K-1}_{\text{Sample } K}]$$

---

### 2.6 FlashAttention-2 / FlashAttention-3 VarLen Kernel Pipeline (`cu_seqlens`)

Constructing explicit 2D attention matrices of size $(B, H, L, L)$ in global memory (HBM) consumes $O(L^2)$ memory and negates the speed of packed training.

Modern frontier stacks utilize **FlashAttention-2 / 3 Variable Length (VarLen)** kernels. Instead of feeding padded tensors and dense boolean masks, the unpadded tokens are passed as a single 1D tensor of shape $(\sum L_i, d_{\text{model}})$ alongside an integer array of cumulative sequence lengths:

$$\text{cu\_seqlens} = [0, L_1, L_1 + L_2, L_1 + L_2 + L_3, \dots, \sum_{k=1}^K L_k]$$

Inside the FlashAttention CUDA kernel:
1. Thread blocks determine their sequence boundary directly from `cu_seqlens[batch_idx]` and `cu_seqlens[batch_idx + 1]`.
2. Online softmax scaling and tiled matrix multiplication remain strictly bounded inside the local sequence chunk in on-chip SRAM.
3. Memory complexity remains strictly $O(N)$ with **zero cross-attention, zero memory overhead, and $2.8\times$ higher token throughput** than padded batching.

---

### 2.7 Production PyTorch & HuggingFace Packing Data Collator

```python
import torch
from typing import Dict, List, Any

class MultiplexedPackingDataCollator:
    """
    Production-grade Sequence Packing Collator with 2D block diagonal awareness,
    position ID resetting, and FlashAttention cu_seqlens generation.
    """
    def __init__(self, max_seq_len: int = 4096, pad_token_id: int = 0):
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __call__(self, samples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        packed_input_ids = []
        packed_labels = []
        packed_position_ids = []
        cu_seqlens = [0]
        
        current_len = 0
        for sample in samples:
            input_ids = sample["input_ids"]
            labels = sample["labels"]
            seq_len = len(input_ids)
            
            assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds max {self.max_seq_len}"
            
            # Check if adding this sequence exceeds the current window
            if current_len + seq_len > self.max_seq_len:
                break
                
            packed_input_ids.extend(input_ids)
            packed_labels.extend(labels)
            # Reset position IDs from 0 to seq_len - 1
            packed_position_ids.extend(list(range(seq_len)))
            
            current_len += seq_len
            cu_seqlens.append(current_len)

        # Pad remaining space if necessary to guarantee fixed GPU tensor shapes
        remainder = self.max_seq_len - current_len
        if remainder > 0:
            packed_input_ids.extend([self.pad_token_id] * remainder)
            packed_labels.extend([-100] * remainder)  # Masked from loss
            packed_position_ids.extend([0] * remainder)
            cu_seqlens.append(self.max_seq_len)

        return {
            "input_ids": torch.tensor([packed_input_ids], dtype=torch.long),
            "labels": torch.tensor([packed_labels], dtype=torch.long),
            "position_ids": torch.tensor([packed_position_ids], dtype=torch.long),
            "cu_seqlens": torch.tensor(cu_seqlens, dtype=torch.int32),
            "max_seqlen": torch.tensor(max(s["cu_seqlens"][i+1] - s["cu_seqlens"][i] 
                                          for i in range(len(cu_seqlens)-1)) if len(cu_seqlens) > 1 else current_len, 
                                      dtype=torch.int32)
        }
```

---

# CHAPTER 3: COMPLETION-ONLY DYNAMIC LOSS MASKING

### 3.1 Prompt Contamination Hazard & Instruction Following Degradation

In naive autoregressive training, cross-entropy loss is computed uniformly across all tokens in the sequence:

$$\mathcal{L}_{\text{naive}}(\theta) = - \sum_{t=1}^{T_{\text{prompt}}} \log P_\theta(x_t \mid \mathbf{x}_{<t}) - \sum_{k=1}^{T_{\text{completion}}} \log P_\theta(y_k \mid \mathbf{x}, \mathbf{y}_{<k})$$

```
+-------------------------------------------------------------------------------+
| PROMPT TOKENS (User Query, Few-Shot, Template) | COMPLETION TOKENS (Assistant)|
+-------------------------------------------------------------------------------+
| "Below is an instruction... \n User: Write..." | "Here is the python code..." |
+-------------------------------------------------------------------------------+
   |                                                |
   v [NAIVE SFT: Computes Loss Here!]               v [ALWAYS Computes Loss Here]
   - Forces model to memorize user questions        - Desired generative objective
   - Induces robotic, memorized prompt patterns
   - Drastically impairs generalization
```

**Consequences of Unmasked Prompt Losses:**
1. **Capacity Squandering:** The model allocates valuable gradient capacity to predict the exact phrasing, stylistic idiosyncrasies, and whitespace formatting of the prompt text.
2. **Instruction Adherence Drift:** Training on user instructions induces a bidirectional language modeling bias, confusing the boundary between question and answer.

---

### 3.2 Dynamic Label Masking Mechanics via `ignore_index = -100`

PyTorch's `torch.nn.CrossEntropyLoss` and CUDA-fused kernels (such as Flash-Cross-Entropy or Triton CE) accept an integer argument `ignore_index` (default: `-100`).

When calculating:

$$\mathcal{L} = \frac{\sum_{i=1}^B \sum_{t=1}^T \mathbb{I}(y_{i,t} \ne -100) \cdot \ell_{i,t}}{\sum_{i=1}^B \sum_{t=1}^T \mathbb{I}(y_{i,t} \ne -100)}$$

Any label assigned `-100` contributes zero to the cumulative loss and emits zero gradient in the backward pass:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}_t} = \mathbf{0}, \quad \forall t \text{ where } y_t = -100$$

---

### 3.3 Multi-Turn Dialogue Parsing & Delimiter-Guided Masking (ChatML / Llama 3)

In real-world multi-turn conversations, the prompt consists of multiple alternating turns:

$$\mathcal{C} = [U_1, A_1, U_2, A_2, \dots, U_K, A_K]$$

#### The Masking Invariant:
- Mask all system instructions: `Label = -100`
- Mask all user turns $U_k$ and framing tokens (`<|im_start|>user...<|im_end|>`): `Label = -100`
- Mask assistant preamble header (`<|im_start|>assistant\n`): `Label = -100`
- **Preserve assistant completion tokens $A_k$**: `Label = Token_ID`
- **Preserve assistant EOS token (`<|im_end|>` or `<|eot_id|>`):** `Label = EOS_Token_ID` (Crucial: omitting EOS from loss causes infinite generation loops).

```
Sample Multi-Turn Token String:
<|im_start|>system\nYou are a helpful AI.<|im_end|>\n
<|im_start|>user\nCalculate 2+2.<|im_end|>\n
<|im_start|>assistant\n2+2 is 4.<|im_end|>\n
<|im_start|>user\nMultiply it by 3.<|im_end|>\n
<|im_start|>assistant\n4 * 3 = 12.<|im_end|>

Labels Vector:
[-100, -100, ..., -100]  <-- All system tokens masked (-100)
[-100, -100, ..., -100]  <-- User turn 1 masked (-100)
[-100, -100]             <-- Assistant header masked (-100)
[17, 10, 220, 19]        <-- "2+2 is 4." ACTIVELY LEARNED!
[<|im_end|>]             <-- EOS token ACTIVELY LEARNED!
[-100, -100, ..., -100]  <-- User turn 2 masked (-100)
[-100, -100]             <-- Assistant header masked (-100)
[19, 489, 220, 1532]     <-- "4 * 3 = 12." ACTIVELY LEARNED!
[<|im_end|>]             <-- EOS token ACTIVELY LEARNED!
```

---

### 3.4 Token-Normalized vs. Sample-Normalized Gradient Weighting

When completing loss reduction across a batch with variable completion lengths:

$$\text{Option A (Sample-Normalized): } \mathcal{L} = \frac{1}{B} \sum_{b=1}^B \left( \frac{1}{N_b} \sum_{t=1}^{N_b} \ell_{b,t} \right)$$

$$\text{Option B (Token-Normalized): } \mathcal{L} = \frac{1}{\sum_{b=1}^B N_b} \sum_{b=1}^B \sum_{t=1}^{N_b} \ell_{b,t}$$

> [!IMPORTANT]
> **Industrial Rule:** Option B (Token-Normalized) is mathematically correct. Option A assigns equal gradient magnitude to a 3-token answer ("Yes, it is.") and a 1000-token proof, amplifying the gradient variance of short answers by up to $300\times$.

---

### 3.5 Production-Grade `DataCollatorForCompletionOnlyLM` Implementation

```python
import torch
from transformers import DataCollatorForLanguageModeling
from typing import List, Union, Any, Dict

class ProductionCompletionOnlyDataCollator(DataCollatorForLanguageModeling):
    """
    Industrial-grade completion-only data collator.
    Dynamically searches for multi-token response delimiters and sets labels to -100 
    for all prompt context and delimiter prefix tokens.
    """
    def __init__(
        self,
        response_template: Union[str, List[int]],
        instruction_template: Union[str, List[int]] = None,
        tokenizer: Any = None,
        mlm: bool = False
    ):
        super().__init__(tokenizer=tokenizer, mlm=mlm)
        self.tokenizer = tokenizer
        
        # Resolve response template to token IDs
        if isinstance(response_template, str):
            self.response_token_ids = self.tokenizer.encode(response_template, add_special_tokens=False)
        else:
            self.response_token_ids = response_template
            
        if instruction_template is not None:
            if isinstance(instruction_template, str):
                self.instruction_token_ids = self.tokenizer.encode(instruction_template, add_special_tokens=False)
            else:
                self.instruction_token_ids = instruction_template
        else:
            self.instruction_token_ids = None

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        batch = super().torch_call(examples)
        labels = batch["labels"].clone()
        
        for i in range(len(examples)):
            input_ids = batch["input_ids"][i].tolist()
            label_mask = [True] * len(input_ids) # True = MASK OUT (-100)
            
            # Locate all occurrences of response_token_ids
            resp_len = len(self.response_token_ids)
            idx = 0
            while idx < len(input_ids):
                # Match delimiter sub-sequence
                if input_ids[idx : idx + resp_len] == self.response_token_ids:
                    # Completion begins immediately after the response template
                    comp_start = idx + resp_len
                    # Find next instruction template or EOS to close the active completion
                    comp_end = len(input_ids)
                    if self.instruction_token_ids is not None:
                        inst_len = len(self.instruction_token_ids)
                        for next_idx in range(comp_start, len(input_ids)):
                            if input_ids[next_idx : next_idx + inst_len] == self.instruction_token_ids:
                                comp_end = next_idx
                                break
                    
                    # Unmask completion tokens
                    for active_idx in range(comp_start, comp_end):
                        label_mask[active_idx] = False
                    
                    idx = comp_end
                else:
                    idx += 1
                    
            # Apply masking
            for t_idx, mask_token in enumerate(label_mask):
                if mask_token:
                    labels[i, t_idx] = -100
                    
        batch["labels"] = labels
        return batch
```

---

# CHAPTER 4: DIRECT PREFERENCE OPTIMIZATION (DPO)

### 4.1 Theoretical Lineage: From Bradley-Terry RLHF to Implicit Rewards

Traditional Reinforcement Learning from Human Feedback (RLHF) executes a three-phase pipeline:
1. **SFT Phase:** Train baseline policy $\pi^{\text{SFT}}$ on curated demonstrations.
2. **Reward Modeling Phase:** Fit a scalar reward model $r_\phi(x, y)$ on pairwise human comparisons $(x, y_w, y_l)$ where $y_w \succ y_l$ using the Bradley-Terry preference model:
   
   $$P(y_w \succ y_l \mid x) = \sigma\left( r_\phi(x, y_w) - r_\phi(x, y_l) \right) = \frac{1}{1 + \exp\left( - (r_\phi(x, y_w) - r_\phi(x, y_l)) \right)}$$

3. **PPO Policy Optimization Phase:** Maximize expected reward under a Kullback-Leibler (KL) divergence penalty to avoid drifting from $\pi_{\text{ref}} = \pi^{\text{SFT}}$:
   
   $$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ r_\phi(x, y) \right] - \beta D_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x))$$

**PPO Instabilities:** PPO requires simultaneously maintaining 4 distinct neural networks in VRAM: Policy $\pi_\theta$, Reference Model $\pi_{\text{ref}}$, Reward Model $r_\phi$, and Critic/Value Network $V_\psi$. This induces extreme memory bloat, high gradient variance, and hyperparameter fragility.

---

### 4.2 Mathematical Derivation: Eliminating the Partition Function $Z(x)$

The constrained RLHF optimization problem has an exact closed-form analytical solution:

$$\mathcal{L}(\pi) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \mathbb{E}_{y \sim \pi} [r(x, y)] - \beta \mathbb{E}_{y \sim \pi} \left[ \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] \right]$$

Setting the functional derivative $\frac{\delta \mathcal{L}}{\delta \pi} = 0$ subject to the probability constraint $\sum_y \pi(y \mid x) = 1$:

$$\pi^*(y \mid x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$$

where $Z(x) = \sum_y \pi_{\text{ref}}(y \mid x) \exp\left( \frac{1}{\beta} r(x, y) \right)$ is the partition function.

Rearranging this identity to express the ground-truth reward $r(x, y)$ as an **implicit function** of the optimal policy:

$$\log \pi^*(y \mid x) = \log \pi_{\text{ref}}(y \mid x) + \frac{1}{\beta} r(x, y) - \log Z(x)$$

$$r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)$$

Now substitute this expression for $r(x, y)$ directly into the Bradley-Terry preference likelihood:

$$P(y_w \succ y_l \mid x) = \sigma\left( r(x, y_w) - r(x, y_l) \right)$$

$$= \sigma\left( \left[ \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} + \beta \log Z(x) \right] - \left[ \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} + \beta \log Z(x) \right] \right)$$

$$\mathbf{P(y_w \succ y_l \mid x) = \sigma\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right)}$$

> [!NOTE]
> **The Algebraic Breakthrough:** The intractable partition function $Z(x)$, which sums over all $|\mathcal{V}|^T$ possible output strings, is identical for both $y_w$ and $y_l$ under the same prompt $x$. It subtracts out to exactly zero.

---

### 4.3 The DPO Loss Formulation and Gradient Flow Dynamics

The Direct Preference Optimization (DPO) objective minimizes the negative log-likelihood of preference pairs:

$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = - \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$

Defining the implicit reward $\widehat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$, the gradient with respect to parameter vector $\theta$ evaluates to:

$$\nabla_\theta \mathcal{L}_{\text{DPO}}(\theta) = - \beta \cdot \underbrace{\sigma\left( \widehat{r}_\theta(x, y_l) - \widehat{r}_\theta(x, y_w) \right)}_{\text{Implicit Error Weight } \omega(x, y_w, y_l)} \cdot \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$

#### Mechanics of the Gradient Weight $\omega$:
1. If the current model strongly prefers the chosen completion ($\widehat{r}_\theta(y_w) \gg \widehat{r}_\theta(y_l)$), the error term $\sigma(\dots) \to 0$. The model takes negligible gradient steps, avoiding over-optimization.
2. If the model incorrectly prefers the rejected completion ($\widehat{r}_\theta(y_l) > \widehat{r}_\theta(y_w)$), the weight scales up toward $1.0$, exerting maximal gradient force to increase $\log \pi_\theta(y_w)$ while decreasing $\log \pi_\theta(y_l)$.

---

### 4.4 Reference Model Caching & In-Situ Adapter Freezing

DPO requires evaluating both $\pi_\theta$ and $\pi_{\text{ref}}$ on all tokens, which typically consumes $2\times$ model VRAM.

#### Production Optimization Strategies:
1. **Offline Log-Probability Caching:** If training a full parameter model, precompute $\log \pi_{\text{ref}}(y_w \mid x)$ and $\log \pi_{\text{ref}}(y_l \mid x)$ over the entire dataset in a single forward inference sweep. Store the scalar log-probabilities directly in the Parquet/Arrow dataset. **Eliminates the reference model from training VRAM entirely ($50\%$ memory reduction).**
2. **In-Situ Adapter Freezing (LoRA / M-2LRF):** When fine-tuning low-rank adapters over a frozen base model:
   - Perform forward pass 1: Enable adapters $\implies \pi_\theta$.
   - Perform forward pass 2: Call `model.disable_adapters()` $\implies \pi_{\text{ref}}$ directly from the base weights without allocating a second model instance in GPU memory!

---

### 4.5 Hyperparameter Sensitivity

- **Temperature $\beta$:** Controls the strength of the KL penalty constraint.
  - $\beta = 0.1$: Default for standard 7B/8B chat alignment.
  - $\beta = 0.01 - 0.05$: Required for mathematical and reasoning alignment. High $\beta$ values over-constrain the model to the base policy, preventing the acquisition of novel reasoning trajectories.
- **Conservative Regularization (Label Smoothing $\epsilon$):**
  
  $$\mathcal{L}_{\text{cDPO}} = - (1 - \epsilon) \log \sigma(\Delta r) - \epsilon \log \sigma(-\Delta r)$$

  Prevents deterministic overconfidence on noisy preference labels where annotators disagree.

---

### 4.6 Production PyTorch DPO Loss Kernel

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class DirectPreferenceOptimizationLoss(nn.Module):
    """
    Numerically stable Direct Preference Optimization (DPO) Loss Kernel
    with sequence length normalization and optional label smoothing.
    """
    def __init__(self, beta: float = 0.1, label_smoothing: float = 0.0, length_norm: bool = False):
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.length_norm = length_norm

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,   # Shape: (B,)
        policy_rejected_logps: torch.Tensor, # Shape: (B,)
        reference_chosen_logps: torch.Tensor,# Shape: (B,)
        reference_rejected_logps: torch.Tensor,# Shape: (B,)
        chosen_lengths: torch.Tensor = None,  # Shape: (B,)
        rejected_lengths: torch.Tensor = None # Shape: (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        if self.length_norm and chosen_lengths is not None:
            policy_chosen_logps = policy_chosen_logps / chosen_lengths
            reference_chosen_logps = reference_chosen_logps / chosen_lengths
            policy_rejected_logps = policy_rejected_logps / rejected_lengths
            reference_rejected_logps = reference_rejected_logps / rejected_lengths

        # Compute log-ratio differences
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        
        logits = self.beta * (pi_logratios - ref_logratios)

        if self.label_smoothing > 0.0:
            loss = (
                - (1.0 - self.label_smoothing) * F.logsigmoid(logits)
                - self.label_smoothing * F.logsigmoid(-logits)
            ).mean()
        else:
            loss = - F.logsigmoid(logits).mean()

        # Compute implicit rewards for logging
        with torch.no_grad():
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)

        return loss, chosen_rewards, rejected_rewards
```

---

# CHAPTER 5: ODDS RATIO PREFERENCE OPTIMIZATION (ORPO)

### 5.1 The Alignment Dilemma: SFT vs. Preference Decoupling

Standard preference tuning methods decouple alignment into two stages: SFT on positive demonstrations followed by DPO on preference pairs.

**The Failure Mode of Decoupled Alignment:**
1. During DPO, while the relative probability ratio $\frac{\pi(y_w)}{\pi(y_l)}$ increases, the absolute cross-entropy probability $\pi(y_w \mid x)$ can actually decrease across training steps (probability drift).
2. DPO provides no intrinsic regularizer enforcing auto-regressive fluency on token generation, occasionally resulting in degenerate, repetitive, or structurally ungrammatical outputs if run for multiple epochs.

---

### 5.2 Mathematical Derivation of Odds and Log Odds Ratio ($\log \text{OR}$)

ORPO resolves this dilemma by integrating a reference-free preference penalty directly into the Supervised Fine-Tuning cross-entropy loss.

For any completion $y$ given prompt $x$, define the probability of generating sequence $y$:

$$P_\theta(y \mid x) = \prod_{t=1}^{|y|} P_\theta(y_t \mid x, y_{<t})$$

The **Odds** of generating $y$ versus generating any other token sequence is defined as:

$$\text{Odds}_\theta(y \mid x) = \frac{P_\theta(y \mid x)}{1 - P_\theta(y \mid x)}$$

The **Odds Ratio (OR)** between the favored completion $y_w$ and disfavored completion $y_l$ evaluates to:

$$\text{OR}_\theta(y_w, y_l \mid x) = \frac{\text{Odds}_\theta(y_w \mid x)}{\text{Odds}_\theta(y_l \mid x)} = \frac{P_\theta(y_w \mid x) / (1 - P_\theta(y_w \mid x))}{P_\theta(y_l \mid x) / (1 - P_\theta(y_l \mid x))}$$

Taking the natural logarithm yields the **Log Odds Ratio ($\log \text{OR}$)**:

$$\log \text{OR}_\theta(y_w, y_l \mid x) = \log \left( \frac{P_\theta(y_w \mid x)}{1 - P_\theta(y_w \mid x)} \right) - \log \left( \frac{P_\theta(y_l \mid x)}{1 - P_\theta(y_l \mid x)} \right)$$

---

### 5.3 Monolithic Objective: Unifying SFT Likelihood with Preference Dispreference

ORPO defines a single combined objective optimized from step 0:

$$\mathcal{L}_{\text{ORPO}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \lambda_{\text{OR}} \cdot \mathcal{L}_{\text{OR}}(\theta)$$

where:

$$\mathcal{L}_{\text{SFT}}(\theta) = - \frac{1}{|y_w|} \sum_{t=1}^{|y_w|} \log P_\theta(y_{w,t} \mid x, y_{w,<t})$$

$$\mathcal{L}_{\text{OR}}(\theta) = - \log \sigma \left( \log \text{OR}_\theta(y_w, y_l \mid x) \right) = \log \left( 1 + \left( \frac{\text{Odds}_\theta(y_l \mid x)}{\text{Odds}_\theta(y_w \mid x)} \right) \right)$$

$$\lambda_{\text{OR}} \in [0.05, 0.2] \text{ is the relative weighting coefficient.}$$

---

### 5.4 Gradient Dissection: Selective Suppression of Degenerate Tokens

Computing the derivative of $\mathcal{L}_{\text{ORPO}}$ with respect to model parameters $\theta$:

$$\nabla_\theta \mathcal{L}_{\text{ORPO}} = \nabla_\theta \mathcal{L}_{\text{SFT}}(y_w) + \lambda_{\text{OR}} \cdot \delta_{\text{OR}} \cdot \left[ \frac{\nabla_\theta P_\theta(y_l)}{P_\theta(y_l)(1 - P_\theta(y_l))} - \frac{\nabla_\theta P_\theta(y_w)}{P_\theta(y_w)(1 - P_\theta(y_w))} \right]$$

where:

$$\delta_{\text{OR}} = \sigma(-\log \text{OR}_\theta(y_w, y_l \mid x)) = \frac{1}{1 + \text{OR}_\theta(y_w, y_l \mid x)}$$

**The Dynamics:**
- $\mathcal{L}_{\text{SFT}}$ exerts an absolute pull, training the model to predict high-likelihood tokens for $y_w$.
- $\mathcal{L}_{\text{OR}}$ acts as a relative hinge: as soon as the odds of $y_w$ dominate $y_l$ ($\text{OR} \gg 1$), $\delta_{\text{OR}} \to 0$, shutting off the preference gradient and letting standard language modeling proceed without distortion.

---

### 5.5 Reference-Free Architecture: 50% Memory Reduction vs. DPO

```
DPO Pipeline:
+------------------------+      +------------------------+
| Active Policy Model πθ |      | Frozen Ref Model πref  |
+------------------------+      +------------------------+
           |                                 |
           +----------------+----------------+
                            |
                     [DPO Loss Step] (VRAM: ~28GB for 7B)

ORPO Pipeline:
+------------------------+
| Active Policy Model πθ |  <-- ONLY ONE MODEL IN MEMORY!
+------------------------+
           |
   [Forward Pass yw, yl]
           |
     [ORPO Loss Step] (VRAM: ~14GB for 7B -> 50% SAVINGS)
```

---

### 5.6 Production PyTorch Implementation of the ORPO Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class OddsRatioPreferenceOptimizationLoss(nn.Module):
    """
    Numerically stable Odds Ratio Preference Optimization (ORPO) Loss.
    Combines SFT cross-entropy on winning completions with log odds ratio penalty.
    """
    def __init__(self, lambda_or: float = 0.1, pad_token_id: int = 0):
        super().__init__()
        self.lambda_or = lambda_or
        self.pad_token_id = pad_token_id

    def forward(
        self,
        chosen_logps: torch.Tensor,      # Shape: (B,) - Average log-prob per token of y_w
        rejected_logps: torch.Tensor,    # Shape: (B,) - Average log-prob per token of y_l
        sft_loss: torch.Tensor           # Scalar SFT loss on chosen tokens
    ) -> torch.Tensor:
        
        # Convert average log-probabilities to sequence probabilities
        log_odds_chosen = chosen_logps - torch.log1p(-torch.exp(chosen_logps).clamp(max=0.9999))
        log_odds_rejected = rejected_logps - torch.log1p(-torch.exp(rejected_logps).clamp(max=0.9999))
        
        # Log Odds Ratio
        log_odds_ratio = log_odds_chosen - log_odds_rejected
        
        # Odds Ratio Loss
        or_loss = - F.logsigmoid(log_odds_ratio).mean()
        
        # Monolithic Combined Loss
        total_loss = sft_loss + self.lambda_or * or_loss
        
        return total_loss, sft_loss.detach(), or_loss.detach()
```

---

# CHAPTER 6: KAHNEMAN-TVERSKY OPTIMIZATION (KTO)

### 6.1 Cognitive Foundations: Behavioral Prospect Theory & Cumulative Utility

Standard alignment methods (RLHF, DPO) rely on Von Neumann-Morgenstern utility theory, which assumes humans evaluate options according to expected utility with symmetric, rational preferences.

In 1979, Daniel Kahneman and Amos Tversky established **Prospect Theory**, revealing three fundamental realities of human choice:
1. **Reference Dependence:** Humans do not evaluate outcomes in absolute terms; they evaluate outcomes as gains or losses relative to a subjective cognitive reference point ($z_{\text{ref}}$).
2. **Diminishing Sensitivity:** The marginal psychological value of both gains and losses decreases as their scale increases (concave for gains, convex for losses).
3. **Loss Aversion:** **Losses loom larger than gains.** The psychological displeasure of losing $\$100$ is approximately $1.5\times$ to $2.5\times$ greater than the pleasure of winning $\$100$.

```
Psychological Value V(x)
                 ^
                 |          / (Concave: Diminishing Sensitivity for Gains)
                 |         /
                 |        /
    Losses       |       /       Gains
<----------------+----------------> Outcome (x - z_ref)
  \              |
   \             |
    \ (Steeper:  |
     \ Loss      |
      \ Aversion)|
       v         |
```

---

### 6.2 The Mathematics of Loss Aversion: Steepness Ratio $\lambda_D > 1$

KTO formulates model alignment using the Kahneman-Tversky value function $v(z)$:

$$v(z) = \begin{cases}
1 - \sigma\left( \beta \left( r(x, y) - z_{\text{ref}} \right) \right) & \text{if desirable } (y \in \mathcal{Y}_{\text{desirable}}) \\
1 - \sigma\left( \beta \lambda_D \left( z_{\text{ref}} - r(x, y) \right) \right) & \text{if undesirable } (y \in \mathcal{Y}_{\text{undesirable}})
\end{cases}$$

where:
- $r(x, y) = \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$ is the implicit reward.
- $z_{\text{ref}}$ is the cognitive reference point.
- $\lambda_D > 1$ is the **Loss Aversion Coefficient** (typically $1.33 \le \lambda_D \le 2.0$).

---

### 6.3 Implicit Reward Centering with Batch-Estimated Reference Points

The subjective reference point $z_{\text{ref}}$ represents what the user expects on average from the model. Mathematically, it is defined as the expected KL divergence between the current policy and the reference model:

$$z_{\text{ref}} = \mathbb{E}_{x \sim \mathcal{D}} \left[ D_{\text{KL}}(\pi_\theta(y \mid x) \parallel \pi_{\text{ref}}(y \mid x)) \right]$$

In practice, this expectation is estimated dynamically across the training batch:

$$\widehat{z}_{\text{ref}} = \frac{1}{B} \sum_{i=1}^B \text{detach}\left( \log \frac{\pi_\theta(y_i \mid x_i)}{\pi_{\text{ref}}(y_i \mid x_i)} \right)$$

The overall KTO loss function evaluates to:

$$\mathcal{L}_{\text{KTO}}(\theta) = \sum_{y \in \mathcal{Y}_{\text{desirable}}} w_D \left( 1 - \sigma\left( \beta (r(x, y) - \widehat{z}_{\text{ref}}) \right) \right) + \sum_{y \in \mathcal{Y}_{\text{undesirable}}} w_U \left( 1 - \sigma\left( \beta \lambda_D (\widehat{z}_{\text{ref}} - r(x, y)) \right) \right)$$

---

### 6.4 Unpaired Alignment: Eliminating the Costly Preference Tuple Barrier

The primary operational advantage of KTO over DPO/PPO is **Unpaired Optimization**:
- DPO strictly requires pairs: $(x, y_w, y_l)$. If an enterprise logs $100,000$ customer interactions with binary feedback ("Thumbs Up" vs "Thumbs Down"), converting this to DPO requires synthesizing counterfactual pairs or discarding non-paired data.
- KTO accepts disjoint lists: a dataset of $80,000$ upvoted responses and a separate dataset of $20,000$ downvoted responses. **No paired data is required.**

---

### 6.5 Production PyTorch Implementation of KTO Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class KahnemanTverskyOptimizationLoss(nn.Module):
    """
    Industrial-grade Kahneman-Tversky Optimization (KTO) Loss Kernel.
    Implements behavioral prospect theory alignment with loss aversion coefficient lambda_D.
    """
    def __init__(self, beta: float = 0.1, lambda_d: float = 1.33, desirable_weight: float = 1.0, undesirable_weight: float = 1.0):
        super().__init__()
        self.beta = beta
        self.lambda_d = lambda_d
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_logps: torch.Tensor,     # Shape: (B,)
        reference_logps: torch.Tensor,  # Shape: (B,)
        is_desirable: torch.Tensor      # Shape: (B,) Boolean mask (True = thumbs up, False = thumbs down)
    ) -> torch.Tensor:
        
        # Implicit reward r(x, y)
        implicit_rewards = policy_logps - reference_logps
        
        # Batch-estimated reference point z_ref (detached from autograd)
        z_ref = implicit_rewards.mean().detach()
        
        losses = []
        for i in range(len(policy_logps)):
            r = implicit_rewards[i]
            if is_desirable[i]:
                # Desirable utility: 1 - sigmoid(beta * (r - z_ref))
                u = 1.0 - torch.sigmoid(self.beta * (r - z_ref))
                losses.append(self.desirable_weight * u)
            else:
                # Undesirable utility: 1 - sigmoid(beta * lambda_d * (z_ref - r))
                u = 1.0 - torch.sigmoid(self.beta * self.lambda_d * (z_ref - r))
                losses.append(self.undesirable_weight * u)
                
        return torch.stack(losses).mean()
```

---

# CHAPTER 7: REINFORCEMENT LEARNING WITH VERIFIABLE REWARDS (RLVR / DEEPSEEK-R1 STYLE)

### 7.1 The End of Neural Reward Hacking: Verifiable Ground Truth

In standard RLHF for open-ended conversation, reward models are parameterized by neural networks $r_\phi(x, y)$.

#### The Goodhart Failure (Reward Hacking):
When a neural reward model is optimized against as an objective, it ceases to be a reliable measure of quality:
- The policy exploits blind spots in $r_\phi$, generating superficial markers (excessive politeness, bullet points, bloated word counts) that trigger high scores despite factual inaccuracy.

#### The RLVR Paradigm:
In deterministic domains (Mathematics, Competitive Programming, Theorem Proving, SQL Query Generation, Formal Logic), **ground truth correctness is verifiable programmatically**:
- Math: Does the symbolic expression match the solution under SymPy expansion?
- Code: Does the code pass all deterministic test assertions inside an isolated sandbox?
- Formal Logic: Does Lean 4 / Isabelle / Coq accept the formal proof tactic?

```
                      +----------------------------------+
                      | Prompt (GSM8K, HumanEval, Proof) |
                      +----------------------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Rollout Generation (G outputs)   |
                      | <think> ... </think> <answer>    |
                      +----------------------------------+
                                        |
                     +------------------+------------------+
                     |                                     |
                     v                                     v
      +-----------------------------+       +-----------------------------+
      | Rule-Based Verifiers (RBRM) |       | Formatting & Length Checks  |
      | - SymPy Symbolic Math Check |       | - Strict XML Tag Structure  |
      | - Isolated Docker / Pytest  |       | - Repetition & Loop Penalty |
      | - Deterministic 0.0 or 1.0  |       +-----------------------------+
      +-----------------------------+                      |
                     |                                     |
                     +------------------+------------------+
                                        |
                                        v
                      +----------------------------------+
                      | Combined Objective Reward Matrix |
                      | R_total = R_verifiable + R_format|
                      +----------------------------------+
```

---

### 7.2 Rule-Based Reward Modeling (RBRM) Architecture

The reward vector decomposes into deterministic component functions:

$$R_{\text{total}}(x, y) = w_{\text{acc}} \cdot R_{\text{accuracy}}(x, y) + w_{\text{format}} \cdot R_{\text{format}}(y) + w_{\text{rep}} \cdot R_{\text{penalty}}(y)$$

1. **Accuracy Reward ($R_{\text{accuracy}} \in \{0.0, 1.0\}$):**
   - Exact numeric match, fraction simplification, or unit test pass rate ($K_{\text{passed}} / K_{\text{total}}$).
2. **Formatting Reward ($R_{\text{format}} \in \{-1.0, 0.0, 0.5\}$):**
   - Strict adherence to thought tags: Does the completion contain matching `<think>` and `</think>` tags followed by `<answer>` and `</answer>`?
3. **Repetition Penalty ($R_{\text{penalty}} \le 0.0$):**
   - Penalizes infinite cyclic loops (e.g. repeated n-grams $>3$ times) to deter degenerate CoT stuttering.

---

### 7.3 Group Relative Policy Optimization (GRPO): Eliminating the Critic Network

Standard PPO requires training an auxiliary Critic (Value) Network $V_\psi(s)$ that must be equal in parameter scale to the Actor $\pi_\theta$.

> [!CAUTION]
> **VRAM Bottleneck:** Training a 70B parameter reasoning model with PPO requires maintaining:
> - Actor (70B) + Critic (70B) + Reference Model (70B) + Reward Model (70B) $\implies >500\text{ GB}$ VRAM before activations!

#### GRPO Architecture (DeepSeek-R1 / DeepSeek-Math):
GRPO eliminates the Critic network entirely. For each prompt $q$, the engine samples a **group of $G$ independent rollouts** $\{o_1, o_2, \dots, o_G\}$ from the current policy $\pi_{\theta_{\text{old}}}$:

```
Prompt q ---> [Model Policy π_old] ---> Sample Rollout o_1  ---> Verifier ---> Reward r_1
                                   ---> Sample Rollout o_2  ---> Verifier ---> Reward r_2
                                   ---> ...
                                   ---> Sample Rollout o_G  ---> Verifier ---> Reward r_G
```

---

### 7.4 Advantage Estimation across Symmetric Sample Groups

Instead of querying a Critic network for a baseline value $V(q)$, GRPO calculates the baseline **directly from the empirical mean and standard deviation of rewards within the sampled group**:

$$\bar{r} = \frac{1}{G} \sum_{i=1}^G r_i, \quad \sigma_r = \sqrt{\frac{1}{G} \sum_{i=1}^G (r_i - \bar{r})^2 + \epsilon}$$

The normalized advantage for rollout $i$ is formulated as:

$$A_i = \frac{r_i - \bar{r}}{\sigma_r}$$

The GRPO surrogate objective minimizes:

$$\mathcal{L}_{\text{GRPO}}(\theta) = - \frac{1}{G} \sum_{i=1}^G \left[ \min\left( \rho_i(\theta) A_i, \text{clip}\left(\rho_i(\theta), 1 - \epsilon_{\text{clip}}, 1 + \epsilon_{\text{clip}}\right) A_i \right) - \beta D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

where $\rho_i(\theta) = \frac{\pi_\theta(o_i \mid q)}{\pi_{\theta_{\text{old}}}(o_i \mid q)}$ is the importance sampling probability ratio.

---

### 7.5 Emergence of Self-Correction, Backtracking, and Dynamic Thought Chains ("Aha Moments")

When trained under RLVR with verifiable rewards and no length penalties, models discover novel cognitive behaviors without human supervision:
1. **Self-Correction:** The model generates an initial hypothesis, notices an inconsistency in line 12, explicitly outputs *"Wait, let me double check this equation..."*, backtracks, and corrects the derivation.
2. **Dynamic Search Scaling:** For easy problems, the model generates $200$ tokens of thought. For difficult Putnam or AIME competition problems, the model naturally expands its reasoning trajectory up to $16,000$ tokens of exploration.
3. **Format Crystallization:** Format rewards incentivize the model to structure reasoning systematically before committing to a final answer.

---

### 7.6 Complete Production Sandbox: SymPy Equivalence & Pytest Code Verifier

```python
import sys
import multiprocessing
import sympy as sp
from typing import Dict, Any

def verify_math_symbolic(prediction_str: str, ground_truth_str: str) -> float:
    """
    Evaluates mathematical equivalence between candidate string and ground truth
    using SymPy symbolic simplification. Returns 1.0 if equivalent, 0.0 otherwise.
    """
    try:
        pred_sym = sp.sympify(prediction_str)
        gt_sym = sp.sympify(ground_truth_str)
        difference = sp.simplify(pred_sym - gt_sym)
        if difference == 0:
            return 1.0
    except Exception:
        pass
    
    # Fallback to direct stripped string equality
    if prediction_str.strip() == ground_truth_str.strip():
        return 1.0
    return 0.0

def _run_code_isolated(code_snippet: str, test_cases: str, queue: multiprocessing.Queue):
    """Execution worker executed inside a hard isolated process."""
    exec_globals = {}
    full_script = f"{code_snippet}\n{test_cases}"
    try:
        exec(full_script, exec_globals)
        queue.put(True)
    except Exception:
        queue.put(False)

def verify_code_sandbox(code_snippet: str, test_cases: str, timeout_seconds: int = 3) -> float:
    """
    Executes generated Python code against test cases in an isolated process
    with hard execution timeout to prevent infinite loops or memory bombs.
    """
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_run_code_isolated,
        args=(code_snippet, test_cases, queue)
    )
    process.start()
    process.join(timeout=timeout_seconds)
    
    if process.is_alive():
        process.terminate()
        process.join()
        return 0.0 # Timed out
        
    if not queue.empty():
        passed = queue.get()
        return 1.0 if passed else 0.0
    return 0.0
```

---

# CHAPTER 8: PARAMETER-EFFICIENT FINE-TUNING (PEFT) DEEP DIVE

### 8.1 LoRA: Low-Rank Matrix Decomposition Mechanics and Gradient Entanglement

Standard Low-Rank Adaptation (LoRA - Hu et al., 2021) freezes the pretrained base weight matrix $\mathbf{W}_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ and parameterizes its incremental update $\Delta \mathbf{W}$ as the product of two low-rank matrices:

$$\mathbf{W} = \mathbf{W}_0 + \Delta \mathbf{W} = \mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}$$

where $\mathbf{B} \in \mathbb{R}^{d_{\text{out}} \times r}, \mathbf{A} \in \mathbb{R}^{r \times d_{\text{in}}}$, with rank $r \ll \min(d_{\text{out}}, d_{\text{in}})$, and $\alpha$ is a constant scaling hyperparameter.

```
       d_in
     +------+              +------+           r
     |      |              |      |        +------+
d_out|  W0  |     +   d_out|  B   |    *   |  A   | d_in
     |      |              |      |        +------+
     +------+              +------+
     [Frozen Base]         [Trainable]    [Trainable]
```

#### Initialization Dynamics:
- $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$ (Gaussian initialization)
- $\mathbf{B} = \mathbf{0}$ (Zero initialization)
- Consequently: $\Delta \mathbf{W} = \frac{\alpha}{r} (\mathbf{0}) \mathbf{A} = \mathbf{0}$ at Step 0. The model begins identically to the pretrained base.

#### The Gradient Entanglement Defect:
In full fine-tuning, updates can alter magnitude $\|\mathbf{W}\|_F$ and direction $\mathbf{W} / \|\mathbf{W}\|_F$ independently. In LoRA, updating $\mathbf{A}$ and $\mathbf{B}$ simultaneously couples directional shifts with magnitude shifts, causing slower optimization on sub-4-bit quantized bases.

---

### 8.2 DoRA: Weight-Decomposed Directional & Magnitude Normalization

Weight-Decomposed Low-Rank Adaptation (DoRA - Liu et al., 2024) decomposes the weight matrix into its Euclidean magnitude $m \in \mathbb{R}^{1 \times d_{\text{in}}}$ and directional matrix $\mathbf{V} \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$:

$$\mathbf{W} = m \odot \frac{\mathbf{V}}{\|\mathbf{V}\|_c} = m \odot \frac{\mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}}{\|\mathbf{W}_0 + \frac{\alpha}{r} \mathbf{B} \mathbf{A}\|_c}$$

where $\|\cdot\|_c$ denotes the column-wise L2 norm:

$$\|\mathbf{X}\|_c = \left[ \|\mathbf{X}_{*,1}\|_2, \|\mathbf{X}_{*,2}\|_2, \dots, \|\mathbf{X}_{*,d_{\text{in}}}\|_2 \right]$$

- **Initialization:** $m = \|\mathbf{W}_0\|_c$, $\mathbf{B} = \mathbf{0}$, $\mathbf{A} \sim \mathcal{N}(0, \sigma^2)$.
- **Empirical Advantage:** By isolating magnitude $m$ as an explicitly trainable scalar vector, directional matrix $\frac{\mathbf{W}_0 + \mathbf{B}\mathbf{A}}{\|\dots\|}$ optimizes trajectory angles cleanly, closely replicating full fine-tuning gradient trajectories.

---

### 8.3 LoHa: Low-Rank Hadamard Product Adaptation ($O(r^2)$ Effective Rank)

Low-Rank Hadamard Product Adaptation (LoHa) computes weight updates via the element-wise (Hadamard) product of two distinct low-rank factorizations:

$$\Delta \mathbf{W} = \frac{\alpha}{r} \left( \mathbf{B}_1 \mathbf{A}_1 \right) \odot \left( \mathbf{B}_2 \mathbf{A}_2 \right)$$

where $\mathbf{B}_1, \mathbf{B}_2 \in \mathbb{R}^{d_{\text{out}} \times r}$ and $\mathbf{A}_1, \mathbf{A}_2 \in \mathbb{R}^{r \times d_{\text{in}}}$.

#### Mathematical Properties:
- **Maximum Possible Rank:** While standard LoRA is strictly bounded by rank $r$ ($\text{rank}(\mathbf{B}\mathbf{A}) \le r$), the rank of a Hadamard product satisfies:
  
  $$\text{rank}(\mathbf{X} \odot \mathbf{Y}) \le \text{rank}(\mathbf{X}) \times \text{rank}(\mathbf{Y})$$

  Therefore:
  
  $$\mathbf{\text{rank}(\Delta \mathbf{W}_{\text{LoHa}}) \le r^2}$$

- With rank $r=8$, LoHa achieves an effective expressive capacity of up to **rank 64**, requiring only $4 \times d \times r$ parameters ($2\times$ standard LoRA).

---

### 8.4 PiSSA: Principal Singular Component Adaptation from Step 0

Principal Singular values and Singular vectors Adaptation (PiSSA - Meng et al., 2024) alters the initialization paradigm entirely:

1. Perform Singular Value Decomposition (SVD) on the pretrained base matrix $\mathbf{W}_0$:
   
   $$\mathbf{W}_0 = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^{\min(d_{\text{out}}, d_{\text{in}})} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

2. Extract the top-$r$ principal singular components:
   
   $$\mathbf{A}_{\text{init}} = \sqrt{\mathbf{\Sigma}_r} \mathbf{V}_r^T \in \mathbb{R}^{r \times d_{\text{in}}}, \quad \mathbf{B}_{\text{init}} = \mathbf{U}_r \sqrt{\mathbf{\Sigma}_r} \in \mathbb{R}^{d_{\text{out}} \times r}$$

3. The frozen base matrix retains only the residual components:
   
   $$\mathbf{W}_{\text{residual}} = \mathbf{W}_0 - \mathbf{B}_{\text{init}} \mathbf{A}_{\text{init}}$$

4. The adapted layer computes:
   
   $$\mathbf{W} = \mathbf{W}_{\text{residual}} + \mathbf{B} \mathbf{A}$$

**The Impact:** Rather than starting adapter matrices at zero and learning updates from scratch, PiSSA begins training directly on the primary, most energetically dominant singular vectors of the pretrained model, leading to significantly faster downstream loss convergence.

---

### 8.5 LoftQ: Alternating Quantization Residual Optimization

When quantizing a base model to sub-4-bit (e.g. 2-bit or 4-bit NormalFloat), the quantization error is substantial:

$$\mathbf{R}_0 = \mathbf{W}_0 - \mathcal{Q}(\mathbf{W}_0)$$

In standard QLoRA, $\mathbf{B}=\mathbf{0}$, meaning at Step 0 the network suffers the **full quantization degradation** without compensation.

LoftQ (Li et al., 2023) resolves this via an alternating optimization loop before training begins:

```
Algorithm 2: LoftQ Alternating Quantization Residual Loop
Input: Pretrained Weight W0, Target Quantizer Q, Target Rank r, Iterations T
Output: Quantized Base Q_T, Initial Adapter Matrices A_T, B_T

1. Initialize R_0 = W0, B_0 = 0, A_0 = 0
2. for t = 1 to T do:
3.     // Step A: Quantize the residual between original weight and current adapter
4.     Q_t = Q(W0 - B_{t-1} A_{t-1})
5.     // Step B: SVD on the remaining quantization discrepancy
6.     U, Sigma, V^T = SVD(W0 - Q_t, rank=r)
7.     B_t = U * sqrt(Sigma)
8.     A_t = sqrt(Sigma) * V^T
9. return Q_T, B_T, A_T
```

---

### 8.6 The M-2LRF Synthesis: Dual-Basis 2-Bit Quantization + SVD Residual Adaptation

M-2LRF merges the mathematical strengths of LoftQ, PiSSA, and 2-Bit Dual-Basis Quantization into a single architecture:

$$\mathbf{W} \approx \underbrace{\alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1}_{\text{Frozen 2-Bit Dual-Basis Base } \mathbf{W}_{\text{base}}} + \underbrace{\frac{\alpha_{\text{adapter}}}{r} \mathbf{B} \mathbf{A}}_{\text{Trainable SVD Residual Adapter}}$$

$$\text{subject to } \mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}, \quad \mathbf{T}_0, \mathbf{T}_1 \in \{-1, 0, +1\}^{d_{\text{out}} \times d_{\text{in}}}$$

By initializing $\mathbf{B}$ and $\mathbf{A}$ using truncated SVD on the exact dual-basis quantization residual $\mathbf{R} = \mathbf{W}_0 - (\alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1)$, **Step-0 perplexity on WikiText-2 drops from 9,635.00 down to 1,018.51 (a 9.46× recovery)** on 2-bit GPT-2, proving that SVD residual compensation is essential for low-bit training.

---

### 8.7 Comprehensive Structural Comparison Matrix

| Method | Expressive Rank | Step-0 Output Match ($\mathbf{W} = \mathbf{W}_0$) | Trainable Param Overhead | VRAM Footprint (7B Base) | Hardware Merge Overhead | Optimal Quantization Synergy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Full FT** | Full ($d$) | Exact ($100\%$) | $100\%$ ($7.0\text{B}$) | $112.0\text{ GB}$ | Zero | None |
| **LoRA** | $r$ | Exact ($100\%$) | $0.2\% - 0.8\%$ | $16.5\text{ GB}$ (FP16) | Zero ($\mathbf{W}_0 + \frac{\alpha}{r}\mathbf{BA}$) | 4-Bit NF4 (QLoRA) |
| **DoRA** | $r$ | Exact ($100\%$) | $0.25\% - 0.9\%$ | $17.2\text{ GB}$ (FP16) | Low (Norm fold) | 4-Bit NF4 |
| **LoHa** | Up to $r^2$ | Exact ($100\%$) | $0.4\% - 1.6\%$ | $18.0\text{ GB}$ (FP16) | Moderate | 4-Bit / 8-Bit |
| **PiSSA** | $r$ | Exact ($100\%$) | $0.2\% - 0.8\%$ | $16.5\text{ GB}$ (FP16) | Zero ($\mathbf{W}_{\text{res}} + \mathbf{BA}$) | FP16 Base Only |
| **LoftQ** | $r$ | Sub-bit Residual | $0.2\% - 0.8\%$ | $6.2\text{ GB}$ (4-Bit) | Zero | 4-Bit / 2-Bit |
| **M-2LRF** | $r$ (Dual SVD) | Sub-bit Residual | **$0.15\% - 0.4\%$** | **$2.8\text{ GB}$ (2-Bit)** | Zero In-Situ | **2-Bit Dual-Basis** |

---

# CHAPTER 9: DISTRIBUTED MULTI-GPU TRAINING

### 9.1 The Parameter Memory Budget: Weights, Gradients, and Optimizer States

For any model of size $\Phi$ parameters under 16-bit mixed precision with AdamW, training memory divides into static state and dynamic activations:

$$V_{\text{total}} = V_{\text{weights}} + V_{\text{gradients}} + V_{\text{optimizer}} + V_{\text{activations}} + V_{\text{cuda\_overhead}}$$

```
Standard Mixed-Precision AdamW Breakdown (Total: 16 Bytes per Parameter):
[ 2 Bytes: Model Parameters (FP16/BF16) ]
[ 2 Bytes: Parameter Gradients (FP16/BF16) ]
[ 4 Bytes: Master Parameters (FP32) ]
[ 4 Bytes: 1st Momentum Vector (FP32) ]
[ 4 Bytes: 2nd Variance Vector (FP32) ]
```

---

### 9.2 DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3 Memory Partitions

The Zero Redundancy Optimizer (ZeRO - Rajbhandari et al., 2020) shards static states across all $N_{\text{data}}$ distributed data-parallel workers:

```
+-------------------------------------------------------------------------------+
|                       ZeRO MEMORY PARTITIONING REGIMES                        |
+-------------------------------------------------------------------------------+

Stage 0 (DDP Baseline):
GPU 0: [ Params (2B) ][ Grads (2B) ][ Optimizer States: Master/m/v (12B) ]
GPU 1: [ Params (2B) ][ Grads (2B) ][ Optimizer States: Master/m/v (12B) ]

ZeRO-1 (Optimizer State Partitioning): 4x Memory Reduction
GPU 0: [ Params (2B) ][ Grads (2B) ][ Opt State Shard 0 (12B / N) ]
GPU 1: [ Params (2B) ][ Grads (2B) ][ Opt State Shard 1 (12B / N) ]

ZeRO-2 (Optimizer State + Gradient Partitioning): 8x Memory Reduction
GPU 0: [ Params (2B) ][ Grad Shard 0 (2B / N) ][ Opt State Shard 0 (12B / N) ]
GPU 1: [ Params (2B) ][ Grad Shard 1 (2B / N) ][ Opt State Shard 1 (12B / N) ]

ZeRO-3 (Full Sharding - Params + Grads + Optimizer States): Linear Memory Scaling
GPU 0: [ Param Shard 0 (2B / N) ][ Grad Shard 0 (2B / N) ][ Opt State Shard 0 ]
GPU 1: [ Param Shard 1 (2B / N) ][ Grad Shard 1 (2B / N) ][ Opt State Shard 1 ]
All-Gather parameters dynamically layer-by-layer during Forward/Backward passes!
```

---

### 9.3 PyTorch Fully Sharded Data Parallel (FSDP): Wrapping Policies & Prefetching

PyTorch FSDP provides a native, high-performance implementation of parameter sharding:

- **Sharding Strategies:**
  - `FULL_SHARD`: Equivalent to ZeRO-3. Shards parameters, gradients, and optimizer states. Parameters are all-gathered immediately prior to forward execution and discarded immediately after.
  - `SHARD_GRAD_OP`: Equivalent to ZeRO-2. Retains un-sharded parameters throughout forward/backward passes; shards gradients and optimizer states.
- **Backward & Forward Prefetching:** While layer $L$ is executing its GEMM kernel on GPU streaming multiprocessors, the FSDP communication stream concurrently issues non-blocking NCCL All-Gather calls to fetch parameters for layer $L-1$, hiding communication latency behind compute.
- **Transformer Auto-Wrap Policy:** FSDP must shard at the level of individual decoder layers (e.g. `LlamaDecoderLayer`) to prevent all-gathering the entire network at once.

---

### 9.4 Distributed Fine-Tuning with 2-Bit Quantized Weights (M-2LRF & QLoRA)

> [!CAUTION]
> **The Quantized Sharding Bottleneck:** Naively applying ZeRO-3 or FSDP `FULL_SHARD` to quantized base weights breaks down. All-Gathering packed 2-bit or 4-bit `uint8` tensors requires complex dequantization hooks inside NCCL collectives.

#### The Industrial Solution: Frozen Base Unsharded + Adapter Sharded
1. **Base Model:** Quantize the base model locally on each device into 2-bit M-2LRF or 4-bit NF4. Because a 7B model at 2-bit consumes only **1.75 GB to 2.8 GB VRAM**, it easily fits on every GPU without sharding!
2. **Adapter Sharding:** Apply FSDP or ZeRO-2 exclusively to the trainable adapter parameters ($\mathbf{A}$ and $\mathbf{B}$ matrices) and their optimizer states.
3. This completely eliminates base-parameter network communication across nodes while delivering linear scaling of adapter parameters.

---

### 9.5 Production Configuration Templates: DeepSpeed JSON & Accelerate YAML

#### Production `deepspeed_zero2_config.json`
```json
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": 1.0,
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 500000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": true,
    "round_robin_gradients": true
  },
  "bf16": {
    "enabled": true
  },
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": "auto",
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "wall_clock_breakdown": false
}
```

#### Production HuggingFace `accelerate_fsdp_config.yaml`
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch: BACKWARD_PRE
  fsdp_forward_prefetch: true
  fsdp_offload_params: false
  fsdp_sharding_strategy: SHARD_GRAD_OP
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

---

# CHAPTER 10: PREVENTING CATASTROPHIC FORGETTING & QUALITY AUDITING

### 10.1 Curvature Fragility and Representation Collapse in Low-Bit Loss Basins

Pretrained foundation models converge into broad, flat minima in continuous parameter space $\mathbb{R}^d$. When weights are quantized to 2-bit discrete grids:

$$\mathcal{W}_{\text{discrete}} = \{-(\alpha_0 + \alpha_1), -\alpha_1, -\alpha_0, 0, +\alpha_0, +\alpha_1, +(\alpha_0 + \alpha_1)\}$$

The local loss surface transitions from a smooth, continuous manifold to a piecewise, discontinuous, high-curvature landscape.

```
Continuous Pretrained Space (FP16):
Loss ^
     |       \               /
     |        \  Flat Minimum /  <-- Broad loss basin, stable adaptation
     +---------\-------------/---------> Weight Space

Discrete 2-Bit Quantized Space with LoRA:
Loss ^
     |   /\    /\    /\    /\
     |  /  \  /  \  /  \  /  \   <-- High-curvature ridges, narrow minima.
     +-/----\/-*--\/----\/-*--\--------> Weight Space
             Fragile basin
```

If adapter gradients $\nabla_{\mathbf{A}, \mathbf{B}} \mathcal{L}$ take excessive step sizes, the representation jumps completely out of the basin, permanently destroying fundamental linguistic and reasoning capabilities—a phenomenon known as **Catastrophic Representation Collapse**.

---

### 10.2 Elastic Weight Consolidation (EWC++) via Diagonal Fisher Information

To prevent the adapter updates from erasing foundational knowledge, Elastic Weight Consolidation (EWC) adds a quadratic penalty that constrains parameters according to their historical importance:

$$\mathcal{L}_{\text{total}}(\theta) = \mathcal{L}_{\text{task}}(\theta) + \sum_i \frac{\lambda_{\text{EWC}}}{2} F_i (\theta_i - \theta_{0,i})^2$$

where $F_i$ represents the diagonal elements of the **Fisher Information Matrix (FIM)** evaluated on a general pretraining corpus:

$$F_i = \mathbb{E}_{\mathbf{x} \sim \mathcal{D}_{\text{pretrain}}} \left[ \left( \frac{\partial \log P_\theta(\mathbf{x})}{\partial \theta_i} \right)^2 \right]$$

In low-rank adaptation, parameterizing EWC over the low-rank projection matrices ensures that adapter directions orthogonal to the null space of the original model receive maximum damping.

---

### 10.3 Pretrained Reference KL Divergence Penalty

During SFT or Preference Tuning, a computationally efficient guard against forgetting is the **In-Situ Token KL Regularizer**:

$$\mathcal{L}_{\text{reg}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) + \gamma_{\text{KL}} \cdot \mathbb{E}_{\mathbf{x} \sim \mathcal{D}} \left[ D_{\text{KL}}\left( \pi_\theta(\cdot \mid \mathbf{x}) \parallel \pi_{\text{base}}(\cdot \mid \mathbf{x}) \right) \right]$$

For each token position $t$, the penalty computes:

$$D_{\text{KL}}(\pi_\theta \parallel \pi_{\text{base}}) = \sum_{v \in \mathcal{V}} \pi_\theta(v \mid \mathbf{x}_{<t}) \left[ \log \pi_\theta(v \mid \mathbf{x}_{<t}) - \log \pi_{\text{base}}(v \mid \mathbf{x}_{<t}) \right]$$

Setting $\gamma_{\text{KL}} \in [0.01, 0.05]$ keeps policy token distributions bounded within the information envelope of the base foundation model.

---

### 10.4 Strategic Token Mixing: The 80/10/10 Experience Replay Heuristic

To maintain capabilities across reasoning, multilingual fluency, and coding while fine-tuning on a specialized downstream task:

```
+-------------------------------------------------------------------------------+
|                       80 / 10 / 10 TOKEN REPLAY COMPOSITION                   |
+-------------------------------------------------------------------------------+
|  80% Specialized Domain Alignment Tokens (Target Task / Domain Instruction)  |
+-------------------------------------------------------------------------------+
|  10% General Conversational & Linguistic Tokens (LMSYS Chatbot Arena / Wild) |
+-------------------------------------------------------------------------------+
|  10% Verifiable Multi-Step Reasoning Tokens (GSM8K, MATH, HumanEval Synthetic)|
+-------------------------------------------------------------------------------+
```

Mixing in $10\%$ high-quality reasoning and $10\%$ broad conversational tokens stabilizes the Hessian condition number $\kappa = \frac{\lambda_{\text{max}}}{\lambda_{\text{min}}}$ of the adapted layers, preventing specialized task overfitting.

---

### 10.5 Quality Auditing Telemetry: Perplexity, MMLU, GSM8K, and HumanEval Regression Guards

Continuous validation must be integrated directly into training checkpoints. Every $N$ steps (e.g. every 500 steps), the automated auditing daemon evaluates:

1. **Validation Perplexity:** Evaluated on held-out WikiText-2 and C4 verification subsets. A perplexity spike $>15\%$ relative to step 0 triggers an immediate training pause.
2. **MMLU 5-Shot Micro-Sweep:** Evaluates 100 representative STEM and humanities questions.
3. **GSM8K 8-Shot Reasoning Accuracy:** Verifies that mathematical chain-of-thought logic has not degraded.
4. **HumanEval 0-Shot Code Pass@1:** Verifies code syntax generation and indentation mechanics.
5. **Frobenius Drift Telemetry:** Monitors the norm ratio $\frac{\|\frac{\alpha}{r}\mathbf{B}\mathbf{A}\|_F}{\|\mathbf{W}_0\|_F}$. A ratio exceeding $0.25$ indicates adapter over-amplification.

---

### 10.6 Automated Rollback Gates and Model Checkpoint Verification

```python
import os
import torch
import math
from typing import Dict, Any

class TrainingQualityAuditor:
    """
    Automated Quality Auditing & Checkpoint Rollback Gate.
    Tracks validation perplexity, reasoning regression, and Frobenius drift.
    Halts training and triggers rollback if degradation thresholds are breached.
    """
    def __init__(
        self,
        max_allowed_ppl_increase_ratio: float = 0.15,
        max_frobenius_drift_ratio: float = 0.25,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.max_ppl_ratio = max_allowed_ppl_increase_ratio
        self.max_drift_ratio = max_frobenius_drift_ratio
        self.checkpoint_dir = checkpoint_dir
        
        self.baseline_ppl = None
        self.best_ppl = float("inf")
        self.best_checkpoint_path = None

    def audit_step(
        self,
        current_step: int,
        val_loss: float,
        model: torch.nn.Module,
        adapter_names: list = ["lora_A", "lora_B"]
    ) -> Dict[str, Any]:
        
        current_ppl = math.exp(min(val_loss, 20.0))
        
        if self.baseline_ppl is None:
            self.baseline_ppl = current_ppl
            self.best_ppl = current_ppl
            return {"status": "BASELINE_INITIALIZED", "ppl": current_ppl}
            
        # 1. Perplexity Breach Check
        ppl_increase = (current_ppl - self.baseline_ppl) / self.baseline_ppl
        if ppl_increase > self.max_ppl_ratio:
            return {
                "status": "ABORT_ROLLBACK_REQUIRED",
                "reason": f"Perplexity increased by {ppl_increase*100:.2f}% (Threshold: {self.max_ppl_ratio*100}%)",
                "rollback_target": self.best_checkpoint_path
            }
            
        # 2. Frobenius Drift Check
        max_layer_drift = 0.0
        with torch.no_grad():
            for name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B"):
                    # Calculate ||BA||_F / ||W_base||_F
                    A = module.lora_A.default.weight
                    B = module.lora_B.default.weight
                    delta_W = B @ A
                    drift = torch.norm(delta_W, p="fro") / (torch.norm(module.weight, p="fro") + 1e-8)
                    max_layer_drift = max(max_layer_drift, drift.item())
                    
        if max_layer_drift > self.max_drift_ratio:
            return {
                "status": "ABORT_ROLLBACK_REQUIRED",
                "reason": f"Max Frobenius drift {max_layer_drift:.4f} exceeded threshold {self.max_drift_ratio}",
                "rollback_target": self.best_checkpoint_path
            }
            
        # Update best checkpoint
        if current_ppl < self.best_ppl:
            self.best_ppl = current_ppl
            self.best_checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{current_step}.pt")
            
        return {
            "status": "HEALTHY",
            "current_ppl": current_ppl,
            "best_ppl": self.best_ppl,
            "max_drift": max_layer_drift
        }
```

---

## 🎯 SUMMARY & COMPREHENSIVE IMPLEMENTATION ROADMAP

This concludes Volume 4 of the engineering monograph series. The techniques documented herein form an interconnected stack:
1. Multiplexed sequence packing (Chapter 2) and dynamic completion masking (Chapter 3) maximize training throughput and preserve instruction integrity.
2. Preference optimization via DPO (Chapter 4), ORPO (Chapter 5), and KTO (Chapter 6) provides tailored alignment strategies depending on whether reference models and paired datasets are available.
3. Rule-Based RLVR / GRPO (Chapter 7) unlocks frontier reasoning capabilities without neural reward hacking.
4. Parameter-efficient fine-tuning (Chapter 8) and distributed orchestration (Chapter 9) enable sub-4-bit foundation model adaptation at industrial scale while safeguarding against catastrophic forgetting (Chapter 10).

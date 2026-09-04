# M-2LRF: Multi-Rate Low-Rank Factorization & 2-Bit Dual-Basis Engine

> **Lead Architect & Creator:** MD-Mushfiqur Rahim  
> **Repository:** `m2lrf` | **Version:** `1.0.0-PROD`  

---

## ⚡ Key Features

- **True 2-Bit Bit-Packed Storage:** Weights are packed densely at **4 weights per uint8 byte** on GPU/CPU buffers, delivering an **87.5% reduction** in static weight memory over FP16.
- **Dual-Basis Decomposition:** Decomposes weight tensors into two disjoint ternary bases:
  $$\mathbf{W} \approx \alpha_0 \mathbf{T}_0 + \alpha_1 \mathbf{T}_1, \quad \mathbf{T}_0 \odot \mathbf{T}_1 = \mathbf{0}, \quad \mathbf{T}_0, \mathbf{T}_1 \in \{-1, 0, +1\}$$
- **Closed-Form Lloyd-Max Scaling:** Exact Gaussian scaling centroids ($\alpha_0^* \approx 0.4528\sigma, \alpha_1^* \approx 1.5104\sigma, \tau^* \approx 0.9816\sigma$) achieving optimal $\text{SQNR} \approx 9.30\text{ dB}$.
- **LoftQ SVD Residual Initialization:** Initializes Low-Rank Adapters (LoRA) directly on the principal singular vectors of the quantization residual $\mathbf{R} = \mathbf{W} - \mathbf{W}_{\text{base}}$ to recover Step-0 representation fidelity.
- **In-Situ Permanent Merger:** Fuses trained LoRA adapters into packed base weights after training with zero runtime latency overhead.

---

## 🏗️ Canonical Architecture Specification

To ensure clear usage, M-2LRF defines the following canonical classes:

1. **`M2LRF2BitLinear` (`m2lrf.layer` or `m2lrf.m2lrf_core_v1`):**
   The primary production linear layer. Stores frozen `uint8` packed weights (`packed_weights`), per-row FP16 scale vectors (`a0`, `a1`), and trainable FP32 LoRA adapters (`lora_A`, `lora_B`).
2. **`DualBasisQuantizer` (`m2lrf.quantizer`):**
   The mathematical quantizer providing `quantize_2_00b(w)` and guaranteed disjointness ($T_0 \odot T_1 = 0$).
3. **`Real2BitCodec` (`m2lrf.packed_codec`):**
   Low-level LSB-first bit-packing encoder/decoder (`pack` and `unpack_and_dequantize`).
4. **`prepare_m2lrf_model` (`m2lrf.trainer_eval`):**
   Automated model converter that replaces `nn.Linear` layers in any HuggingFace model (Qwen, Llama, Mistral, GPT-2).

---

## 🚀 Quickstart Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from m2lrf import prepare_m2lrf_model

# 1. Load Foundation Model
model_id = "Qwen/Qwen2.5-7B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Convert to M-2LRF 2-Bit with Rank-16 LoRA
model = prepare_m2lrf_model(
    model,
    rank=16,
    alpha=16.0,
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
)

# 3. Fine-Tune Trainable Parameters Only
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params, lr=2e-4)
```

---

## 🧪 Running Unit Tests

```bash
python -m unittest tests/test_quantizer.py
```

---

## 📜 Documentation

- Complete Technical Monograph: `docs/M2LRF_Master_Monograph.md`
- Printable Publication Vector PDF: `docs/M2LRF_Master_Technical_Monograph.pdf`

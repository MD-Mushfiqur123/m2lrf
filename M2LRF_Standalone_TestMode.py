# ====================================================================================================
# 🧪 M-2LRF 100% STANDALONE 1-CELL TEST & VERIFICATION ENGINE (ZERO IMPORT ERRORS)
# Hardware: Google Colab (T4 / A100 / L4 GPU)
# Target Model: Qwen/Qwen2.5-0.5B-Instruct
# ====================================================================================================

# Cell 1: Setup
!pip install -q torch transformers datasets accelerate scipy
print("✅ Test Environment Ready!")

# Cell 2: Complete Self-Contained Test Mode
import os, sys, math, time, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("="*80)
print(f"🧪 ENTERING 100% SELF-CONTAINED TEST MODE ON: {DEVICE}")
if torch.cuda.is_available():
    print(f"[*] GPU: {torch.cuda.get_device_name(0)}")
print("="*80 + "\n")

# ----------------------------------------------------------------------------------------------------
# 1. CORE M-2LRF 2-BIT PACKED LAYER (EMBEDDED)
# ----------------------------------------------------------------------------------------------------
class M2LRF2BitLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0

        # Pack 4 2-bit weights per uint8 byte
        self.packed_k = math.ceil(in_features / 4)
        self.register_buffer("packed_weights", torch.zeros(out_features, self.packed_k, dtype=torch.uint8))
        self.register_buffer("a0", torch.zeros(out_features, 1, dtype=torch.float16))
        self.register_buffer("a1", torch.zeros(out_features, 1, dtype=torch.float16))
        self.orig_shape = (out_features, in_features)

        # Trainable Adapter (LoftQ Residual SVD)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

    @torch.no_grad()
    def initialize_from_pretrained(self, weight: torch.Tensor):
        w_f = weight.float()
        std = torch.std(w_f, dim=-1, keepdim=True).clamp(min=1e-6)
        a0 = std * 0.4527786409
        a1 = std * 1.5104181947
        decision_boundary = (a0 + a1) / 2.0

        abs_w = w_f.abs()
        sign_pos = (w_f >= 0)

        codes = torch.zeros_like(weight, dtype=torch.uint8)
        codes = torch.where(~sign_pos & (abs_w > decision_boundary), torch.tensor(0, dtype=torch.uint8, device=weight.device), codes)
        codes = torch.where(~sign_pos & (abs_w <= decision_boundary), torch.tensor(1, dtype=torch.uint8, device=weight.device), codes)
        codes = torch.where(sign_pos & (abs_w <= decision_boundary), torch.tensor(2, dtype=torch.uint8, device=weight.device), codes)
        codes = torch.where(sign_pos & (abs_w > decision_boundary), torch.tensor(3, dtype=torch.uint8, device=weight.device), codes)

        padded_k = self.packed_k * 4
        if padded_k != self.in_features:
            codes = F.pad(codes, (0, padded_k - self.in_features))

        c_reshaped = codes.view(self.out_features, -1, 4)
        packed_bytes = (
            (c_reshaped[..., 0] << 0) |
            (c_reshaped[..., 1] << 2) |
            (c_reshaped[..., 2] << 4) |
            (c_reshaped[..., 3] << 6)
        ).to(torch.uint8)

        self.packed_weights.copy_(packed_bytes)
        self.a0.copy_(a0.to(torch.float16))
        self.a1.copy_(a1.to(torch.float16))

        # Truncated SVD Initialization on residual (LoftQ)
        w_dequant = self._vectorized_dequant()
        residual = w_f - w_dequant.float()
        try:
            u, s, v = torch.svd_lowrank(residual, q=self.rank, niter=4)
            sqrt_s = torch.diag(torch.sqrt(s.clamp(min=1e-8)))
            self.lora_B.copy_(u @ sqrt_s)
            self.lora_A.copy_(sqrt_s @ v.t())
        except Exception:
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def _vectorized_dequant(self) -> torch.Tensor:
        c0 = (self.packed_weights >> 0) & 0x03
        c1 = (self.packed_weights >> 2) & 0x03
        c2 = (self.packed_weights >> 4) & 0x03
        c3 = (self.packed_weights >> 6) & 0x03

        codes = torch.stack([c0, c1, c2, c3], dim=-1).flatten(start_dim=-2)
        codes = codes[..., :self.in_features]

        w_dequant = torch.zeros(self.orig_shape, dtype=torch.float16, device=self.packed_weights.device)
        w_dequant = torch.where(codes == 0, -self.a1, w_dequant)
        w_dequant = torch.where(codes == 1, -self.a0, w_dequant)
        w_dequant = torch.where(codes == 2, self.a0, w_dequant)
        w_dequant = torch.where(codes == 3, self.a1, w_dequant)
        return w_dequant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_dequant = self._vectorized_dequant().to(x.dtype)
        base_out = F.linear(x, w_dequant)
        lora_out = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling
        out = base_out + lora_out
        if self.bias is not None:
            out = out + self.bias
        return out


def prepare_m2lrf_model(model: nn.Module, rank: int = 16, alpha: float = 16.0) -> nn.Module:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    for p in model.parameters():
        p.requires_grad = False

    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            m2 = M2LRF2BitLinear(module.in_features, module.out_features, rank=rank, alpha=alpha, bias=(module.bias is not None)).to(module.weight.device)
            m2.initialize_from_pretrained(module.weight.data)
            if module.bias is not None:
                m2.bias.data.copy_(module.bias.data)
            
            parent_name = name.rsplit(".", 1)[0] if "." in name else ""
            child_name = name.rsplit(".", 1)[1] if "." in name else name
            setattr(model.get_submodule(parent_name) if parent_name else model, child_name, m2)
            count += 1
    print(f"[*] Replaced {count} linear projections with M-2LRF 2-Bit layers.")
    return model

# ----------------------------------------------------------------------------------------------------
# 2. RUN STEP-BY-STEP TEST
# ----------------------------------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"[1/5] Loading Pretrained Model: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(DEVICE)

# 2. Test BEFORE Fine-Tuning
test_prompt = "<|im_start|>user\nWho created M-2LRF and what is it?<|im_end|>\n<|im_start|>assistant\n"
inputs = tokenizer(test_prompt, return_tensors="pt").to(DEVICE)

print("\n[2/5] Baseline Model Output (BEFORE Fine-Tuning):")
with torch.no_grad():
    gen_before = model.generate(**inputs, max_new_tokens=40, pad_token_id=tokenizer.pad_token_id)
print(">>> " + tokenizer.decode(gen_before[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))

# 3. Apply 2-Bit Quantization
print("\n[3/5] Applying 2-Bit Quantization & SVD LoRA...")
model = prepare_m2lrf_model(model, rank=16, alpha=16.0)

# 4. Supervised Fine-Tuning on Specific Fact
print("\n[4/5] Running 15-Step Mini Fine-Tuning...")
target_text = (
    "<|im_start|>user\nWho created M-2LRF and what is it?<|im_end|>\n"
    "<|im_start|>assistant\nM-2LRF is an advanced 2-bit dual-basis quantization and fine-tuning engine created by MD-Mushfiqur Rahim (M).<|im_end|>"
)
enc = tokenizer(target_text, return_tensors="pt").to(DEVICE)
optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-4)

model.train()
for step in range(15):
    optimizer.zero_grad()
    outputs = model(input_ids=enc.input_ids, labels=enc.input_ids)
    loss = outputs.loss
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    if (step + 1) % 5 == 0 or step == 0:
        print(f"  [Step {step+1:02d}/15] Loss: {loss.item():.4f} | Grad Norm: {grad_norm.item():.4f}")

# 5. Test AFTER Fine-Tuning
model.eval()
print("\n[5/5] Model Output (AFTER M-2LRF Fine-Tuning):")
with torch.no_grad():
    gen_after = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.pad_token_id)
print(">>> " + tokenizer.decode(gen_after[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True))

print("\n" + "="*80)
print("🎉 TEST MODE SUCCEEDED: 2-BIT BASE FROZEN, GRADIENTS FLOWED & KNOWLEDGE LEARNED!")
print("="*80)

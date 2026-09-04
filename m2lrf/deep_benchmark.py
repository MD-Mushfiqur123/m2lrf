"""
M-2LRF Deep Empirical Benchmark Suite
======================================
Automated controlled comparison between:
  1. Full FP16 Baseline
  2. 4-Bit Uniform (QLoRA Equivalent)
  3. M-2LRF 2-Bit Packed + LoftQ SVD Residual Initialization
"""

import os
import sys
import math
import time
import gc
import json
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.layer import M2LRF2BitLinear


class Uniform4BitLinearLoRA(nn.Module):
    """Reference 4-bit uniform quantization baseline for comparative benchmarking."""
    def __init__(self, in_features: int, out_features: int, rank: int = 16, alpha: float = 16.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = alpha / rank if rank > 0 else 1.0

        self.register_buffer("packed_weight", torch.zeros(out_features, math.ceil(in_features / 2), dtype=torch.uint8))
        self.register_buffer("scale", torch.zeros(out_features, 1, dtype=torch.float16))
        self.orig_shape = (out_features, in_features)

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, dtype=torch.float32))

    @torch.no_grad()
    def initialize_from_fp16(self, fp_weight: torch.Tensor):
        w_f = fp_weight.float()
        max_val = torch.max(w_f.abs(), dim=-1, keepdim=True).values.clamp(min=1e-6)
        scale = max_val / 7.0

        q = torch.clamp(torch.round(w_f / scale), -7, 7).to(torch.int8) + 7
        q_even = q[..., 0::2]
        q_odd = q[..., 1::2]
        packed = (q_even & 0x0F) | ((q_odd & 0x0F) << 4)

        self.packed_weight.copy_(packed.to(torch.uint8))
        self.scale.copy_(scale.to(torch.float16))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_even = (self.packed_weight & 0x0F).to(torch.float16) - 7.0
        q_odd = ((self.packed_weight >> 4) & 0x0F).to(torch.float16) - 7.0

        w_dequant = torch.empty(self.orig_shape, dtype=torch.float16, device=self.packed_weight.device)
        w_dequant[..., 0::2] = q_even * self.scale
        w_dequant[..., 1::2] = q_odd * self.scale

        base_out = F.linear(x, w_dequant.to(x.dtype))
        lora_out = F.linear(F.linear(x.float(), self.lora_A), self.lora_B).to(x.dtype) * self.scaling
        return base_out + lora_out


def run_benchmark_comparison(
    in_features: int = 1024,
    out_features: int = 4096,
    batch_size: int = 4,
    seq_len: int = 128,
    steps: int = 40,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Runs a controlled micro-benchmark comparing FP16, 4-bit, and M-2LRF 2-bit layers."""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dtype = torch.float16 if (device.type == "cuda" and torch.cuda.is_available()) else torch.float32

    w_orig = torch.randn(out_features, in_features, dtype=dtype, device=device)
    x = torch.randn(batch_size, seq_len, in_features, dtype=dtype, device=device)
    target = torch.randn(batch_size, seq_len, out_features, dtype=dtype, device=device)

    # 1. FP16 Baseline (Matched dtype and device)
    linear_fp16 = nn.Linear(in_features, out_features, bias=False).to(device=device, dtype=dtype)
    linear_fp16.weight.data.copy_(w_orig)
    opt_fp16 = torch.optim.AdamW(linear_fp16.parameters(), lr=2e-4)

    t0 = time.time()
    for _ in range(steps):
        opt_fp16.zero_grad()
        out = linear_fp16(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt_fp16.step()
    time_fp16 = time.time() - t0

    # 2. 4-Bit Baseline
    linear_4b = Uniform4BitLinearLoRA(in_features, out_features, rank=16).to(device)
    linear_4b.initialize_from_fp16(w_orig)
    opt_4b = torch.optim.AdamW([linear_4b.lora_A, linear_4b.lora_B], lr=2e-4)

    t0 = time.time()
    for _ in range(steps):
        opt_4b.zero_grad()
        out = linear_4b(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt_4b.step()
    time_4b = time.time() - t0

    # 3. M-2LRF 2-Bit
    linear_2b = M2LRF2BitLinear(in_features, out_features, rank=16).to(device)
    linear_2b.initialize_from_pretrained(w_orig)
    opt_2b = torch.optim.AdamW([linear_2b.lora_A, linear_2b.lora_B], lr=2e-4)

    t0 = time.time()
    for _ in range(steps):
        opt_2b.zero_grad()
        out = linear_2b(x)
        loss = F.mse_loss(out, target)
        loss.backward()
        opt_2b.step()
    time_2b = time.time() - t0

    return {
        "device": str(device),
        "dtype": str(dtype),
        "steps": steps,
        "time_fp16_s": round(time_fp16, 3),
        "time_4bit_s": round(time_4b, 3),
        "time_m2lrf_2bit_s": round(time_2b, 3),
        "speedup_over_fp16": round(time_fp16 / time_2b, 2) if time_2b > 0 else 0.0
    }


if __name__ == "__main__":
    results = run_benchmark_comparison()
    print(json.dumps(results, indent=2))

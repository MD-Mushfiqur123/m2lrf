"""
M-2LRF 8-Bit Block-Wise Quantized AdamW Optimizer (BitsAndBytes-Inspired)
==========================================================================
Reduces optimizer state memory by 75% (from 8 bytes/param to 2 bytes/param)
using block-wise dynamic 8-bit quantization for first (m) and second (v) moments.
"""

from typing import Optional, Tuple, List, Dict, Any, Iterable
import math
import torch
from torch.optim.optimizer import Optimizer


class AdamW8bit(Optimizer):
    """
    Block-wise quantized 8-bit AdamW optimizer.
    Maintains first moment in INT8 and second moment in UINT8 with per-block scaling (block size = 256).
    """
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
        block_size: int = 256
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]
            block_size = group["block_size"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW8bit does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    n_elements = p.numel()
                    # Calculate padding to multiple of block_size
                    pad_len = (block_size - (n_elements % block_size)) % block_size
                    state["pad_len"] = pad_len
                    total_padded = n_elements + pad_len
                    n_blocks = total_padded // block_size

                    # INT8 first moment (m)
                    state["q_m"] = torch.zeros(total_padded, dtype=torch.int8, device=p.device)
                    state["scale_m"] = torch.zeros(n_blocks, dtype=torch.float32, device=p.device)

                    # UINT8 second moment (v)
                    state["q_v"] = torch.zeros(total_padded, dtype=torch.uint8, device=p.device)
                    state["scale_v"] = torch.zeros(n_blocks, dtype=torch.float32, device=p.device)

                state["step"] += 1
                step = state["step"]
                pad_len = state["pad_len"]
                block_size = group["block_size"]

                q_m = state["q_m"]
                scale_m = state["scale_m"]
                q_v = state["q_v"]
                scale_v = state["scale_v"]

                # 1. Flatten and pad gradient
                flat_grad = grad.flatten()
                if pad_len > 0:
                    flat_grad = torch.cat([flat_grad, torch.zeros(pad_len, dtype=flat_grad.dtype, device=flat_grad.device)])

                n_blocks = q_m.shape[0] // block_size

                # Reshape into [n_blocks, block_size]
                grad_blocks = flat_grad.view(n_blocks, block_size).float()
                qm_blocks = q_m.view(n_blocks, block_size)
                qv_blocks = q_v.view(n_blocks, block_size)

                # 2. Dequantize moments
                m = qm_blocks.float() * scale_m.unsqueeze(1)
                v = qv_blocks.float() * scale_v.unsqueeze(1)

                # 3. Update moments
                m = beta1 * m + (1.0 - beta1) * grad_blocks
                v = beta2 * v + (1.0 - beta2) * (grad_blocks * grad_blocks)

                # 4. Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1

                denom = (v.sqrt() / math.sqrt(bias_correction2)) + eps
                update = (m / denom)

                # Unpad and apply weight decay + step
                flat_update = update.flatten()
                if pad_len > 0:
                    flat_update = flat_update[:-pad_len]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                p.data.add_(flat_update.view_as(p.data), alpha=-step_size)

                # 5. Re-quantize moments to 8-bit
                # First moment m -> INT8 [-127, 127]
                max_abs_m = m.abs().max(dim=1).values.clamp(min=1e-8)
                new_scale_m = max_abs_m / 127.0
                scale_m.copy_(new_scale_m)
                qm_blocks.copy_((m / new_scale_m.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8))

                # Second moment v -> UINT8 [0, 255]
                max_v = v.max(dim=1).values.clamp(min=1e-8)
                new_scale_v = max_v / 255.0
                scale_v.copy_(new_scale_v)
                qv_blocks.copy_((v / new_scale_v.unsqueeze(1)).round().clamp(0, 255).to(torch.uint8))

        return loss

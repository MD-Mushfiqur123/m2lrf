"""
M-2LRF LLaMA Architecture Patcher (Unsloth-Inspired)
====================================================
In-place surgical optimization and 2-bit quantization for LLaMA-2, LLaMA-3, 3.1, and 3.2 models.
"""

from typing import Optional
import torch
import torch.nn as nn

from m2lrf.models.base_patcher import BaseArchitecturePatcher
from m2lrf.trainer_eval import prepare_m2lrf_model


class LlamaPatcher(BaseArchitecturePatcher):
    target_architectures = ["llama", "llamaformodellm", "llamaconfig"]

    @classmethod
    def patch_model(
        cls,
        model: nn.Module,
        load_in_2bit: bool = True,
        rank: int = 16,
        alpha: Optional[float] = None,
        use_hadamard: bool = False,
        group_size: Optional[int] = 128,
        loftq_iters: int = 1,
        target_avg_bits: Optional[float] = None,
        patch_kernels: bool = True,
        verbose: bool = True
    ) -> nn.Module:
        if verbose:
            print("[*] [M-2LRF Patcher] Initializing LLaMA optimization engine...")

        # 1. Patch RMSNorm modules with FastRMSNorm
        if patch_kernels:
            cls.patch_norm_modules(model, verbose=verbose)

        # 2. Surgically quantize linear projections into M-2LRF 2-Bit + LoftQ
        if load_in_2bit:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
            model = prepare_m2lrf_model(
                model=model,
                rank=rank,
                alpha=alpha,
                target_modules=target_modules,
                loftq_iters=loftq_iters,
                group_size=group_size,
                use_hadamard=use_hadamard,
                target_avg_bits=target_avg_bits,
                verbose=verbose
            )

        return model

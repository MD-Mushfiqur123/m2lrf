"""
M-2LRF Fast Model Loader (Unsloth-Inspired)
============================================
Unified user entry point: FastM2LRFModel.from_pretrained(...)
"""

from typing import Optional, Tuple, Union, Any, Dict
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from m2lrf.models.patch_llama import LlamaPatcher
from m2lrf.models.patch_qwen import QwenPatcher
from m2lrf.models.patch_mistral import MistralPatcher
from m2lrf.models.base_patcher import BaseArchitecturePatcher
from m2lrf.trainer_eval import prepare_m2lrf_model


PATCHER_REGISTRY = [
    LlamaPatcher,
    QwenPatcher,
    MistralPatcher
]


class FastM2LRFModel:
    """
    Main model factory for M-2LRF accelerated models (inspired by Unsloth FastLanguageModel).
    """
    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        max_seq_length: int = 4096,
        dtype: Optional[torch.dtype] = None,
        load_in_2bit: bool = True,
        rank: int = 16,
        alpha: Optional[float] = None,
        use_hadamard: bool = False,
        group_size: Optional[int] = 128,
        loftq_iters: int = 1,
        target_avg_bits: Optional[float] = None,
        device_map: Optional[str] = "auto",
        trust_remote_code: bool = True,
        patch_kernels: bool = True,
        verbose: bool = True,
        **kwargs
    ) -> Tuple[nn.Module, Any]:
        """
        Loads, patches, and quantizes a foundation LLM into M-2LRF 2-Bit with fast kernels.
        """
        if verbose:
            print("=" * 80)
            print(f"🚀 [M-2LRF FastModel] Loading foundation model: {model_name}")
            print(f"[*] Target Bitrate  : {target_avg_bits or 2.0} bpp")
            print(f"[*] FWHT Dispersion : {use_hadamard}")
            print(f"[*] LoRA Rank       : {rank}")
            print("=" * 80)

        # 1. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            padding_side="right"
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 2. Determine Compute Dtype
        if dtype is None:
            dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

        # 3. Load Model
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        
        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        actual_device_map = device_map if torch.cuda.is_available() else None

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=dtype,
            device_map=actual_device_map,
            trust_remote_code=trust_remote_code,
            **kwargs
        )
        if actual_device_map is None:
            model = model.to(device)

        # 4. Find Matching Architecture Patcher
        matched_patcher = None
        for patcher in PATCHER_REGISTRY:
            if patcher.supports(model):
                matched_patcher = patcher
                break

        if matched_patcher is not None:
            model = matched_patcher.patch_model(
                model=model,
                load_in_2bit=load_in_2bit,
                rank=rank,
                alpha=alpha,
                use_hadamard=use_hadamard,
                group_size=group_size,
                loftq_iters=loftq_iters,
                target_avg_bits=target_avg_bits,
                patch_kernels=patch_kernels,
                verbose=verbose
            )
        else:
            if verbose:
                print("[!] Architecture not matched with custom patcher; using universal M-2LRF patcher.")
            if patch_kernels:
                BaseArchitecturePatcher.patch_norm_modules(model, verbose=verbose)
            if load_in_2bit:
                model = prepare_m2lrf_model(
                    model=model,
                    rank=rank,
                    alpha=alpha,
                    loftq_iters=loftq_iters,
                    group_size=group_size,
                    use_hadamard=use_hadamard,
                    target_avg_bits=target_avg_bits,
                    verbose=verbose
                )

        return model, tokenizer

    @classmethod
    def for_training(cls, model: nn.Module):
        """Prepares model for gradient training."""
        model.train()
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.contiguous()
        return model

    @classmethod
    def for_inference(cls, model: nn.Module):
        """Prepares model for evaluation / generation."""
        model.eval()
        return model

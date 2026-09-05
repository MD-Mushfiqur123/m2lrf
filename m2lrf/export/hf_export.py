"""
M-2LRF HuggingFace Production Exporter (Unsloth + Axolotl-Inspired)
===================================================================
Merges LoRA adapters in-situ and exports production SafeTensors checkpoints
compatible with standard HuggingFace pipelines.
"""

import os
from typing import Optional, Any
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer

from m2lrf.unified_layer import M2LRFUnifiedLinear, M2LRF2BitLinear


def export_to_huggingface(
    model_or_dir: Any,
    output_dir: str,
    tokenizer: Optional[Any] = None,
    save_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True
) -> str:
    """
    Collapses LoRA adapters permanently into base weights and writes standard
    HuggingFace SafeTensors / PyTorch bin format.
    """
    if verbose:
        print("=" * 80)
        print(f"📦 [M-2LRF Export] Exporting model to HuggingFace format: {output_dir}")
        print("=" * 80)

    os.makedirs(output_dir, exist_ok=True)

    if isinstance(model_or_dir, nn.Module):
        model = model_or_dir
    else:
        raise ValueError("model_or_dir must be an initialized nn.Module instance.")

    # 1. In-situ merge all M-2LRF and LoRA layers
    merged_count = 0
    for name, module in model.named_modules():
        if hasattr(module, "merge") and callable(module.merge):
            module.merge()
            merged_count += 1

    if verbose:
        print(f"[*] In-situ collapsed {merged_count} adapter layers permanently.")

    # 2. Convert to target compute dtype
    model.to(save_dtype)

    # 3. Save model using standard save_pretrained
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(output_dir, safe_serialization=True)
    else:
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

    # 4. Save tokenizer if provided
    if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(output_dir)

    if verbose:
        print(f"[✓] Successfully exported HuggingFace checkpoint to: {output_dir}")

    return output_dir

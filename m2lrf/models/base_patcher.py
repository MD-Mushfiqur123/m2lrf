"""
M-2LRF Base Architecture Patcher (Unsloth-Inspired)
===================================================
Base class providing in-place surgical model inspection, module replacement,
and kernel optimization dispatch.
"""

from typing import Dict, List, Optional, Tuple, Any
import torch
import torch.nn as nn

from m2lrf.kernels.fast_rms_norm import FastRMSNorm
from m2lrf.kernels.fast_swiglu import fast_swiglu
from m2lrf.kernels.fast_cross_entropy import FastCrossEntropyLoss
from m2lrf.trainer_eval import prepare_m2lrf_model


class BaseArchitecturePatcher:
    """
    Abstract base patcher for in-place transformer layer replacement.
    """
    target_architectures: List[str] = []

    @classmethod
    def supports(cls, model: nn.Module) -> bool:
        model_type = getattr(getattr(model, "config", None), "model_type", "").lower()
        class_name = model.__class__.__name__.lower()
        for target in cls.target_architectures:
            if target.lower() in model_type or target.lower() in class_name:
                return True
        return False

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
        raise NotImplementedError("Subclasses must implement patch_model")

    @staticmethod
    def patch_norm_modules(model: nn.Module, verbose: bool = False) -> int:
        """
        Replaces standard RMSNorm modules with FastRMSNorm.
        """
        count = 0
        for name, module in list(model.named_modules()):
            cls_name = module.__class__.__name__
            if "RMSNorm" in cls_name and not isinstance(module, FastRMSNorm):
                hidden_size = module.weight.shape[0]
                eps = getattr(module, "variance_epsilon", getattr(module, "eps", 1e-6))
                fast_norm = FastRMSNorm(hidden_size, eps=eps).to(module.weight.device)
                fast_norm.weight.data.copy_(module.weight.data)
                
                # In-place replace in parent
                parent, child_name = BaseArchitecturePatcher._get_parent_and_child(model, name)
                if parent is not None:
                    setattr(parent, child_name, fast_norm)
                    count += 1
        if verbose and count > 0:
            print(f"[*] Patched {count} RMSNorm layers with FastRMSNorm.")
        return count

    @staticmethod
    def _get_parent_and_child(root: nn.Module, full_name: str) -> Tuple[Optional[nn.Module], str]:
        parts = full_name.split(".")
        if len(parts) == 1:
            return root, parts[0]
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None, parts[-1]
        return parent, parts[-1]

"""
M-2LRF In-Place Architecture Patchers (Unsloth-Inspired)
========================================================
"""

from m2lrf.models.base_patcher import BaseArchitecturePatcher
from m2lrf.models.patch_llama import LlamaPatcher
from m2lrf.models.patch_qwen import QwenPatcher
from m2lrf.models.patch_mistral import MistralPatcher
from m2lrf.models.loader import FastM2LRFModel
from m2lrf.models import zoo

__all__ = [
    "BaseArchitecturePatcher",
    "LlamaPatcher",
    "QwenPatcher",
    "MistralPatcher",
    "FastM2LRFModel",
    "zoo",
]

"""
M-2LRF Training Paradigms (Axolotl-Inspired)
==============================================
"""

from m2lrf.trainers.sft_trainer import M2LRFSFTTrainer
from m2lrf.trainers.dpo_trainer import M2LRFDPOTrainer
from m2lrf.trainers.orpo_trainer import M2LRFORPOTrainer

__all__ = [
    "M2LRFSFTTrainer",
    "M2LRFDPOTrainer",
    "M2LRFORPOTrainer"
]

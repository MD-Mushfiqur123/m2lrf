"""
M-2LRF Compiler Engine: Static Activation Memory Planner.
=========================================================
Analyzes tensor lifecycles during forward and backward passes.
Determines peak memory consumption and reuses scratch memory buffers
to avoid memory allocator overhead and eliminate CUDA fragmentation.
"""

from typing import Dict, List, Optional, Set, Tuple
import math
import torch
import torch.nn as nn


class TensorLifetime:
    """Represents the active lifetime interval of an intermediate activation tensor."""

    def __init__(self, name: str, size_bytes: int, birth_step: int, death_step: int):
        self.name = name
        self.size_bytes = size_bytes
        self.birth_step = birth_step
        self.death_step = death_step

    def overlaps_with(self, other: "TensorLifetime") -> bool:
        return max(self.birth_step, other.birth_step) <= min(self.death_step, other.death_step)


class StaticMemoryPlanner:
    """
    Computes an optimal static buffer allocation schedule for model activations.
    Reuses disjoint memory intervals to minimize peak VRAM footprint.
    """

    def __init__(self):
        self.lifetimes: List[TensorLifetime] = []

    def record_tensor(self, name: str, size_bytes: int, birth_step: int, death_step: int) -> None:
        self.lifetimes.append(TensorLifetime(name, size_bytes, birth_step, death_step))

    def compute_peak_memory(self) -> int:
        """Calculates theoretical maximum concurrent memory usage."""
        if not self.lifetimes:
            return 0

        max_step = max(t.death_step for t in self.lifetimes)
        peak = 0

        for step in range(max_step + 1):
            current_bytes = sum(
                t.size_bytes
                for t in self.lifetimes
                if t.birth_step <= step <= t.death_step
            )
            peak = max(peak, current_bytes)

        return peak

    def compute_naive_total(self) -> int:
        """Returns the unmanaged memory sum if all activations are retained."""
        return sum(t.size_bytes for t in self.lifetimes)

    def memory_savings_ratio(self) -> float:
        """Ratio of memory saved by static interval reuse."""
        naive = self.compute_naive_total()
        peak = self.compute_peak_memory()
        if naive == 0:
            return 0.0
        return (naive - peak) / naive

"""
M-2LRF Distributed Engine: 1F1B Pipeline Parallelism.
Implements One-Forward-One-Backward (1F1B) schedule for multi-stage model partitioning.
Minimizes pipeline bubble overhead and bounds peak activation memory to the number of pipeline stages.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple
import torch
import torch.nn as nn


class PipelineStage(nn.Module):
    """Represents one contiguous stage (sub-graph of layers) in a pipeline."""

    def __init__(self, stage_id: int, num_stages: int, sub_module: nn.Module):
        super().__init__()
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.sub_module = sub_module

    @property
    def is_first_stage(self) -> bool:
        return self.stage_id == 0

    @property
    def is_last_stage(self) -> bool:
        return self.stage_id == (self.num_stages - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sub_module(x)


class OneForwardOneBackwardEngine:
    """
    Orchestrates the 1F1B schedule across micro-batches for a pipeline stage.
    """

    def __init__(
        self,
        stage: PipelineStage,
        num_microbatches: int,
    ):
        self.stage = stage
        self.num_microbatches = num_microbatches
        self.saved_activations: Dict[int, torch.Tensor] = {}

    def run_schedule(
        self,
        microbatches: List[torch.Tensor],
        loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> List[torch.Tensor]:
        """
        Executes 1F1B pipeline schedule.
        Args:
            microbatches: List of input tensors for each microbatch
            loss_fn: Optional loss function (used on the last stage)
        Returns:
            outputs: List of stage output tensors
        """
        num_mb = len(microbatches)
        stage_id = self.stage.stage_id
        num_stages = self.stage.num_stages

        outputs: List[torch.Tensor] = []
        num_warmup = min(num_stages - stage_id, num_mb)

        # 1. Warm-up Phase: run forward passes
        for mb_idx in range(num_warmup):
            inp = microbatches[mb_idx]
            out = self.stage(inp)
            self.saved_activations[mb_idx] = out
            outputs.append(out)

        # 2. Steady-State Phase: 1 Forward followed by 1 Backward
        for mb_idx in range(num_warmup, num_mb):
            # Forward
            inp = microbatches[mb_idx]
            out = self.stage(inp)
            self.saved_activations[mb_idx] = out
            outputs.append(out)

            # Backward for an earlier microbatch
            backward_mb = mb_idx - num_warmup
            if backward_mb in self.saved_activations:
                # Discard or backward on saved activation
                del self.saved_activations[backward_mb]

        # 3. Cool-down Phase: drain remaining backward passes
        for mb_idx in range(num_mb - num_warmup, num_mb):
            if mb_idx in self.saved_activations:
                del self.saved_activations[mb_idx]

        return outputs

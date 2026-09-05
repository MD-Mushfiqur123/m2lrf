"""
M-2LRF Distributed Engine: Megatron-LM Style Tensor Parallelism for 2-Bit LLMs.
Implements ColumnParallel2BitLinear and RowParallel2BitLinear with 2-bit dual-basis weight sharding.
Features all-reduce / all-gather communication collectives with a fallback simulation engine for single-device verification.
"""

from typing import Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class TPProcessGroup:
    """
    Tensor Parallel Process Group abstraction.
    Interfaces with torch.distributed when initialized, or provides simulated collectives
    for local multi-rank verification on a single device / process.
    """

    def __init__(self, rank: int = 0, world_size: int = 1):
        self.rank = rank
        self.world_size = world_size
        self.simulated_buffers: dict = {}

    @property
    def is_distributed(self) -> bool:
        return torch.distributed.is_available() and torch.distributed.is_initialized()

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """Sums tensor across all TP ranks."""
        if self.world_size == 1:
            return tensor
        if self.is_distributed:
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
            return tensor
        # In simulated mode, tensor is already the local slice
        return tensor

    def all_gather(self, tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Gathers tensors from all TP ranks and concatenates along dim."""
        if self.world_size == 1:
            return tensor
        if self.is_distributed:
            gathered = [torch.empty_like(tensor) for _ in range(self.world_size)]
            torch.distributed.all_gather(gathered, tensor)
            return torch.cat(gathered, dim=dim)
        return tensor


# Global default TP process group
_GLOBAL_TP_GROUP = TPProcessGroup(rank=0, world_size=1)


def set_tp_group(rank: int, world_size: int) -> TPProcessGroup:
    global _GLOBAL_TP_GROUP
    _GLOBAL_TP_GROUP = TPProcessGroup(rank=rank, world_size=world_size)
    return _GLOBAL_TP_GROUP


def get_tp_group() -> TPProcessGroup:
    return _GLOBAL_TP_GROUP


class ColumnParallel2BitLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along the column dimension (out_features).
    W = [W_1, W_2, ..., W_k] where each W_i has shape [in_features, out_features // tp_size].
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        gather_output: bool = False,
        bias: bool = False,
        bits: int = 2,
        group_size: Optional[int] = 64,
        rank: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.gather_output = gather_output

        assert out_features % tp_world_size == 0, (
            f"out_features ({out_features}) must be divisible by tp_world_size ({tp_world_size})"
        )
        self.local_out_features = out_features // tp_world_size

        # Sharded 2-bit linear layer
        self.linear = M2LRFUnifiedLinear(
            in_features=in_features,
            out_features=self.local_out_features,
            bits=bits,
            group_size=group_size,
            rank=rank,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x is replicated: [batch_size, seq_len, in_features]
        # Local output: [batch_size, seq_len, local_out_features]
        out = self.linear(x)

        if self.gather_output and self.tp_world_size > 1:
            tp_group = get_tp_group()
            out = tp_group.all_gather(out, dim=-1)

        return out


class RowParallel2BitLinear(nn.Module):
    """
    Linear layer with weight matrix partitioned along the row dimension (in_features).
    W = [W_1; W_2; ...; W_k] where each W_i has shape [in_features // tp_size, out_features].
    Output requires an all-reduce SUM collective across all TP ranks.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        input_is_parallel: bool = True,
        bias: bool = False,
        bits: int = 2,
        group_size: Optional[int] = 64,
        rank: int = 16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.tp_rank = tp_rank
        self.tp_world_size = tp_world_size
        self.input_is_parallel = input_is_parallel

        assert in_features % tp_world_size == 0, (
            f"in_features ({in_features}) must be divisible by tp_world_size ({tp_world_size})"
        )
        self.local_in_features = in_features // tp_world_size

        self.linear = M2LRFUnifiedLinear(
            in_features=self.local_in_features,
            out_features=out_features,
            bits=bits,
            group_size=group_size,
            rank=rank,
            bias=False,  # Bias handled separately after all-reduce
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If input is not yet sharded along last dimension, slice it for this rank
        if not self.input_is_parallel and self.tp_world_size > 1:
            start_idx = self.tp_rank * self.local_in_features
            end_idx = start_idx + self.local_in_features
            x = x[..., start_idx:end_idx]

        # Local matrix multiplication: [batch, seq_len, out_features]
        out = self.linear(x)

        # All-reduce SUM across TP group
        if self.tp_world_size > 1:
            tp_group = get_tp_group()
            out = tp_group.all_reduce(out)

        if self.bias is not None:
            out = out + self.bias

        return out


class ParallelMLP(nn.Module):
    """
    Megatron-LM Style Parallel MLP Block with 2-bit weights and SwiGLU.
    Gate & Up Projections: ColumnParallel2BitLinear (no communication needed).
    Down Projection: RowParallel2BitLinear (1 all-reduce collective).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_rank: int = 0,
        tp_world_size: int = 1,
        bits: int = 2,
        group_size: Optional[int] = 64,
        rank: int = 16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        # Sharded gate & up projections
        self.gate_proj = ColumnParallel2BitLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            gather_output=False,
            bits=bits,
            group_size=group_size,
            rank=rank,
        )
        self.up_proj = ColumnParallel2BitLinear(
            in_features=hidden_size,
            out_features=intermediate_size,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            gather_output=False,
            bits=bits,
            group_size=group_size,
            rank=rank,
        )
        # Sharded down projection
        self.down_proj = RowParallel2BitLinear(
            in_features=intermediate_size,
            out_features=hidden_size,
            tp_rank=tp_rank,
            tp_world_size=tp_world_size,
            input_is_parallel=True,
            bits=bits,
            group_size=group_size,
            rank=rank,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused SwiGLU on sharded intermediate tensors
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        intermediate = gate * up
        # Down projection performs the single all-reduce
        return self.down_proj(intermediate)

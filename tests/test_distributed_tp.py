"""
Unit tests for M-2LRF Distributed Tensor Parallelism, Ring Attention, and Pipeline Parallelism.
"""

import unittest
import torch
import torch.nn as nn

from m2lrf.distributed.tensor_parallel import (
    ColumnParallel2BitLinear,
    RowParallel2BitLinear,
    ParallelMLP,
    set_tp_group,
)
from m2lrf.distributed.sequence_parallel import RingAttention
from m2lrf.distributed.pipeline_parallel import (
    PipelineStage,
    OneForwardOneBackwardEngine,
)


class TestDistributedTP(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_column_parallel_linear(self):
        in_dim = 64
        out_dim = 128
        tp_world_size = 2

        col_layer = ColumnParallel2BitLinear(
            in_features=in_dim,
            out_features=out_dim,
            tp_rank=0,
            tp_world_size=tp_world_size,
            gather_output=False,
            bits=2,
            rank=4,
        )

        x = torch.randn(2, 8, in_dim)
        out = col_layer(x)
        # Partitioned output: 128 // 2 = 64
        self.assertEqual(out.shape, (2, 8, 64))
        self.assertFalse(torch.isnan(out).any())

    def test_row_parallel_linear(self):
        in_dim = 64
        out_dim = 32
        tp_world_size = 2

        row_layer = RowParallel2BitLinear(
            in_features=in_dim,
            out_features=out_dim,
            tp_rank=0,
            tp_world_size=tp_world_size,
            input_is_parallel=False,
            bias=True,
            bits=2,
            rank=4,
        )

        x = torch.randn(2, 8, in_dim)
        out = row_layer(x)
        self.assertEqual(out.shape, (2, 8, out_dim))
        self.assertFalse(torch.isnan(out).any())

    def test_parallel_mlp(self):
        hidden_size = 64
        intermediate_size = 128
        mlp = ParallelMLP(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            tp_rank=0,
            tp_world_size=1,
            bits=2,
            rank=4,
        )

        x = torch.randn(2, 4, hidden_size)
        out = mlp(x)
        self.assertEqual(out.shape, (2, 4, hidden_size))
        self.assertFalse(torch.isnan(out).any())

    def test_ring_attention(self):
        ring_attn = RingAttention(sp_rank=0, sp_world_size=1, is_causal=True)
        q = torch.randn(2, 4, 8, 16)
        k = torch.randn(2, 4, 8, 16)
        v = torch.randn(2, 4, 8, 16)

        out = ring_attn.forward(q, k, v)
        self.assertEqual(out.shape, (2, 4, 8, 16))
        self.assertFalse(torch.isnan(out).any())

        # Test multi-rank simulation
        ring_attn_sp2 = RingAttention(sp_rank=0, sp_world_size=2, is_causal=True)
        out_sp2 = ring_attn_sp2.forward(q, k, v)
        self.assertEqual(out_sp2.shape, (2, 4, 8, 16))
        self.assertFalse(torch.isnan(out_sp2).any())

    def test_pipeline_parallel_1f1b(self):
        sub_module = nn.Linear(32, 32)
        stage = PipelineStage(stage_id=0, num_stages=2, sub_module=sub_module)
        engine = OneForwardOneBackwardEngine(stage=stage, num_microbatches=4)

        microbatches = [torch.randn(2, 32) for _ in range(4)]
        outputs = engine.run_schedule(microbatches)
        self.assertEqual(len(outputs), 4)
        for out in outputs:
            self.assertEqual(out.shape, (2, 32))


if __name__ == "__main__":
    unittest.main()

"""
Unit tests for M-2LRF DeepSpeed ZeRO-1, ZeRO-2, and ZeRO-3 Memory Partitioners.
"""

import unittest
import torch
import torch.nn as nn

from m2lrf.distributed.zero import (
    ZeROStage1Optimizer,
    ZeROStage2Optimizer,
    ZeROStage3Partitioner,
)


class TestDistributedZeRO(unittest.TestCase):

    def test_zero_stage_1_optimizer(self):
        torch.manual_seed(42)
        layer1 = nn.Linear(10, 10)
        layer2 = nn.Linear(10, 10)
        params = list(layer1.parameters()) + list(layer2.parameters())

        # Rank 0 in a world size of 2
        opt_r0 = ZeROStage1Optimizer(params=params, lr=0.01, dp_rank=0, dp_world_size=2)
        opt_r1 = ZeROStage1Optimizer(params=params, lr=0.01, dp_rank=1, dp_world_size=2)

        self.assertEqual(len(opt_r0.assigned_params) + len(opt_r1.assigned_params), len(params))

        # Perform mock gradient step
        for p in params:
            p.grad = torch.ones_like(p.data) * 0.1

        init_param_data = [p.clone() for p in params]
        opt_r0.step()
        opt_r1.step()

        # Check that parameters updated
        for idx, p in enumerate(params):
            self.assertFalse(torch.equal(p, init_param_data[idx]))

    def test_zero_stage_2_gradient_partitioning(self):
        layer = nn.Linear(8, 8)
        params = list(layer.parameters())

        opt_r0 = ZeROStage2Optimizer(params=params, lr=0.01, dp_rank=0, dp_world_size=2)
        for p in params:
            p.grad = torch.ones_like(p.data)

        opt_r0.reduce_gradients()
        # Rank 0 only retains grad for assigned param 0 (idx % 2 == 0)
        for idx, p in enumerate(params):
            if idx % 2 == 0:
                self.assertIsNotNone(p.grad)
            else:
                self.assertIsNone(p.grad)

    def test_zero_stage_3_parameter_partitioning(self):
        model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 4))
        partitioner = ZeROStage3Partitioner(module=model, dp_rank=0, dp_world_size=2)
        partitioner.partition_parameters()

        # Ensure parameters were partitioned into shards
        self.assertGreater(len(partitioner.sharded_params), 0)
        for name, shard in partitioner.sharded_params.items():
            self.assertGreater(shard.numel(), 0)


if __name__ == "__main__":
    unittest.main()

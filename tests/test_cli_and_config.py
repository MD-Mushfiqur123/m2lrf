"""
Unit Tests for M-2LRF Declarative Config, Patchers & CLI (Axolotl + Unsloth)
=============================================================================
"""

import unittest
import os
import torch
import torch.nn as nn

from m2lrf.config.schema import M2LRFConfig, QuantConfig
from m2lrf.models.base_patcher import BaseArchitecturePatcher
from m2lrf.models.patch_llama import LlamaPatcher
from m2lrf.models.patch_qwen import QwenPatcher
from m2lrf.models.patch_mistral import MistralPatcher


class DummyLlamaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama"})()
        self.norm = nn.LayerNorm(32)
        self.q_proj = nn.Linear(32, 32)


class TestConfigAndCLI(unittest.TestCase):
    def test_config_from_dict(self):
        raw_dict = {
            "base_model": "meta-llama/Meta-Llama-3-8B",
            "quantization": {
                "method": "m2lrf_2bit",
                "rank": 64,
                "use_hadamard": True,
                "target_avg_bits": 2.0
            },
            "training": {
                "batch_size": 4,
                "learning_rate": 1e-4
            }
        }
        cfg = M2LRFConfig.from_dict(raw_dict)
        self.assertEqual(cfg.base_model, "meta-llama/Meta-Llama-3-8B")
        self.assertEqual(cfg.quantization.rank, 64)
        self.assertTrue(cfg.quantization.use_hadamard)
        self.assertEqual(cfg.quantization.target_avg_bits, 2.0)
        self.assertEqual(cfg.training.batch_size, 4)
        self.assertEqual(cfg.training.learning_rate, 1e-4)

    def test_patcher_supports_detection(self):
        model = DummyLlamaModel()
        self.assertTrue(LlamaPatcher.supports(model))
        self.assertFalse(QwenPatcher.supports(model))
        self.assertFalse(MistralPatcher.supports(model))


if __name__ == "__main__":
    unittest.main()

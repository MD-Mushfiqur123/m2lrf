"""
Unit tests for M-2LRF Native Foundation Model Zoo.
Verifies forward pass and loss computation across native 2-bit dual-basis architectures.
"""

import unittest
import torch

from m2lrf.models.zoo.llama import LLaMAConfig, LLaMAForCausalLM
from m2lrf.models.zoo.qwen2 import Qwen2Config, Qwen2ForCausalLM
from m2lrf.models.zoo.deepseek_v2 import DeepSeekV2Config, DeepSeekV2ForCausalLM
from m2lrf.models.zoo.mistral import MistralConfig, MistralForCausalLM
from m2lrf.models.zoo.gemma2 import Gemma2Config, Gemma2ForCausalLM


class TestModelZoo(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_llama_forward_pass(self):
        config = LLaMAConfig(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bits=2,
            rank=4,
        )
        model = LLaMAForCausalLM(config)
        input_ids = torch.tensor([[1, 5, 12, 25]], dtype=torch.long)
        labels = torch.tensor([[5, 12, 25, 0]], dtype=torch.long)

        loss, logits = model(input_ids, labels=labels)
        self.assertEqual(logits.shape, (1, 4, 50))
        self.assertFalse(torch.isnan(loss).item())
        self.assertGreater(loss.item(), 0.0)

    def test_qwen2_forward_pass(self):
        config = Qwen2Config(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bits=2,
            rank=4,
        )
        model = Qwen2ForCausalLM(config)
        input_ids = torch.tensor([[2, 8, 19]], dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(logits.shape, (1, 3, 50))
        self.assertFalse(torch.isnan(logits).any())

    def test_deepseek_v2_forward_pass(self):
        config = DeepSeekV2Config(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bits=2,
            rank=4,
        )
        model = DeepSeekV2ForCausalLM(config)
        input_ids = torch.tensor([[3, 7, 15, 20]], dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(logits.shape, (1, 4, 50))

    def test_mistral_forward_pass(self):
        config = MistralConfig(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bits=2,
            rank=4,
        )
        model = MistralForCausalLM(config)
        input_ids = torch.tensor([[4, 9, 22]], dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(logits.shape, (1, 3, 50))

    def test_gemma2_forward_pass(self):
        config = Gemma2Config(
            vocab_size=50,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            bits=2,
            rank=4,
        )
        model = Gemma2ForCausalLM(config)
        input_ids = torch.tensor([[5, 11, 33]], dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(logits.shape, (1, 3, 50))


if __name__ == "__main__":
    unittest.main()

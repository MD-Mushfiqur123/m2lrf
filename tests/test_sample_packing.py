"""
Unit Tests for M-2LRF Multiplexed Sequence Packing & Data Collators (Axolotl-Inspired)
========================================================================================
Validates constant-length sequence packing and 2D block-diagonal attention mask isolation.
"""

import unittest
import torch

from m2lrf.data.sample_packing import SequencePacker
from m2lrf.data.collators import CompletionOnlyDataCollator
from m2lrf.data.prompt_formatters import (
    AlpacaFormatter,
    ChatMLFormatter,
    Llama3Formatter,
    DPOFormatter
)


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        # Deterministic dummy tokenizer mapping words to ints
        return [hash(w) % 1000 + 2 for w in text.split()]


class TestSamplePackingAndData(unittest.TestCase):
    def test_sequence_packer_packing_and_mask_isolation(self):
        packer = SequencePacker(max_seq_length=16, pad_token_id=0, ignore_index=-100)
        
        # Sample 1: length 6
        sample1 = {"input_ids": [10, 11, 12, 13, 14, 15], "labels": [10, 11, 12, 13, 14, 15]}
        # Sample 2: length 8
        sample2 = {"input_ids": [20, 21, 22, 23, 24, 25, 26, 27], "labels": [20, 21, 22, 23, 24, 25, 26, 27]}
        # Sample 3: length 5 (should go to batch 2 because 6 + 8 + 5 = 19 > 16)
        sample3 = {"input_ids": [30, 31, 32, 33, 34], "labels": [30, 31, 32, 33, 34]}

        batches = packer.pack([sample1, sample2, sample3])
        self.assertEqual(len(batches), 2)

        # Batch 1 checks
        b1 = batches[0]
        self.assertEqual(b1["input_ids"].shape[0], 16)
        self.assertEqual(b1["attention_mask"].shape, (16, 16))

        # Position IDs should reset: 0..5 for sample 1, 0..7 for sample 2, then padded 0
        expected_pos = [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 7, 0, 0]
        self.assertEqual(b1["position_ids"].tolist(), expected_pos)

        # Check block diagonal attention mask:
        # Sample 2 (idx 6..13) should NEVER attend to Sample 1 (idx 0..5)
        mask = b1["attention_mask"]
        self.assertFalse(mask[6, 5].item())  # token 6 cannot attend to token 5!
        self.assertTrue(mask[5, 4].item())   # within sample 1: token 5 attends to token 4
        self.assertTrue(mask[10, 8].item())  # within sample 2: token 10 attends to token 8
        self.assertFalse(mask[10, 3].item()) # across samples: token 10 CANNOT attend to token 3

    def test_prompt_formatters(self):
        alpaca = AlpacaFormatter()
        formatted = alpaca.format({"instruction": "Say hello", "output": "Hello world"})
        self.assertIn("### Instruction:\nSay hello", formatted)
        self.assertIn("### Response:\nHello world", formatted)

        chatml = ChatMLFormatter()
        c_formatted = chatml.format({"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]})
        self.assertIn("<|im_start|>user\nHi<|im_end|>", c_formatted)
        self.assertIn("<|im_start|>assistant\nHello<|im_end|>", c_formatted)

        llama3 = Llama3Formatter()
        l_formatted = llama3.format({"messages": [{"role": "user", "content": "Hi"}]})
        self.assertIn("<|start_header_id|>user<|end_header_id|>", l_formatted)

        dpo = DPOFormatter()
        p, c, r = dpo.format({"prompt": "P", "chosen": "C", "rejected": "R"})
        self.assertEqual((p, c, r), ("P", "C", "R"))


if __name__ == "__main__":
    unittest.main()

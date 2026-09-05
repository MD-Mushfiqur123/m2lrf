"""
Unit tests for M-2LRF RadixPrefixCache and SpeculativeEngine.
"""

import unittest
import torch

from m2lrf.serving.radix_cache import RadixPrefixCache, RadixTreeNode
from m2lrf.serving.speculative import SpeculativeEngine


class TestServingRadixCache(unittest.TestCase):

    def test_radix_cache_insert_and_match(self):
        cache = RadixPrefixCache(block_size=4)
        self.assertEqual(cache.total_queries, 0)

        # Empty tree match
        node, matched_len, blocks = cache.match_prefix([1, 2, 3, 4])
        self.assertIsNone(node)
        self.assertEqual(matched_len, 0)
        self.assertEqual(blocks, [])

        # Insert first sequence: tokens [1, 2, 3, 4, 5, 6, 7, 8], blocks [10, 11]
        cache.insert([1, 2, 3, 4, 5, 6, 7, 8], [10, 11])

        # Exact match
        node, matched_len, blocks = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(matched_len, 8)
        self.assertEqual(blocks, [10, 11])
        self.assertIsNotNone(node)

        # Prefix match: query [1, 2, 3, 4, 5, 6, 7, 8, 99, 100]
        node, matched_len, blocks = cache.match_prefix([1, 2, 3, 4, 5, 6, 7, 8, 99, 100])
        self.assertEqual(matched_len, 8)
        self.assertEqual(blocks, [10, 11])

    def test_radix_cache_branching_and_splitting(self):
        cache = RadixPrefixCache(block_size=4)

        # Insert path A: [1, 2, 3, 4, 5] -> blocks [10, 11]
        cache.insert([1, 2, 3, 4, 5], [10, 11])

        # Insert path B: [1, 2, 3, 4, 6] -> blocks [10, 12]
        # Should split at common prefix [1, 2, 3, 4]
        cache.insert([1, 2, 3, 4, 6], [10, 12])

        # Match path A
        node_a, len_a, blocks_a = cache.match_prefix([1, 2, 3, 4, 5])
        self.assertEqual(len_a, 5)

        # Match path B
        node_b, len_b, blocks_b = cache.match_prefix([1, 2, 3, 4, 6])
        self.assertEqual(len_b, 5)

    def test_radix_cache_lru_eviction(self):
        cache = RadixPrefixCache(block_size=4)

        # Insert two disjoint branches
        cache.insert([10, 11], [101])
        cache.insert([20, 21], [201])

        # Evict 1 block: should evict the oldest unreferenced leaf
        freed = cache.evict_lru(num_blocks_to_free=1)
        self.assertEqual(len(freed), 1)

    def test_speculative_engine(self):
        vocab_size = 100

        # Deterministic draft: always predicts [5, 6]
        def mock_draft(input_ids, k):
            return torch.tensor([[5, 6]], dtype=torch.long)

        # Target model: assigns high logit to 5 for first pos, 6 for second pos, and 7 for third pos
        def mock_target(input_ids):
            seq_len = input_ids.size(1)
            logits = torch.zeros(1, seq_len, vocab_size)
            # Make sure greedy picks [5, 6, 7]
            for i in range(seq_len):
                logits[0, i, 5] = 10.0
                logits[0, i, 6] = 10.0
                logits[0, i, 7] = 10.0
            return logits

        engine = SpeculativeEngine(
            draft_generate_fn=mock_draft,
            target_eval_fn=mock_target,
            k_speculative_tokens=2,
            temperature=0.0,  # greedy
        )

        prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
        new_prefix, num_accepted = engine.step(prefix)

        # Both draft tokens accepted + 1 bonus token
        self.assertGreaterEqual(num_accepted, 2)
        self.assertEqual(new_prefix.shape[1], 3 + num_accepted)


if __name__ == "__main__":
    unittest.main()

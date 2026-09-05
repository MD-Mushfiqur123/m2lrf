"""
Unit tests for M-2LRF Serving Engine: BlockSpaceManager and PagedAttention v1/v2.
"""

import math
import unittest
import torch
import torch.nn.functional as F

from m2lrf.serving.block_manager import BlockAllocator, BlockSpaceManager, Sequence
from m2lrf.serving.paged_attention import PagedKVCache, paged_attention_v1, paged_attention_v2
from m2lrf.serving.engine import LLMEngine, SamplingParams


class TestServingPagedAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)

    def test_block_allocator_and_free(self):
        allocator = BlockAllocator(num_blocks=4, block_size=16, device="cpu")
        self.assertEqual(allocator.num_free_blocks, 4)

        b0 = allocator.allocate()
        self.assertEqual(b0.block_id, 0)
        self.assertEqual(b0.ref_count, 1)
        self.assertEqual(allocator.num_free_blocks, 3)

        allocator.free(b0.block_id)
        self.assertEqual(allocator.num_free_blocks, 4)
        self.assertEqual(b0.ref_count, 0)

    def test_block_space_manager_allocation_and_cow(self):
        manager = BlockSpaceManager(num_gpu_blocks=16, block_size=4, device="cpu")
        # Sequence of length 7 requires ceil(7/4) = 2 blocks
        seq1 = Sequence(seq_id=1, prompt_token_ids=[10, 20, 30, 40, 50, 60, 70], block_size=4)
        self.assertTrue(manager.can_allocate(seq1))
        manager.allocate(seq1)

        table1 = manager.get_block_table(seq1)
        self.assertEqual(len(table1), 2)
        self.assertEqual(manager.get_num_free_blocks(), 14)

        # Fork seq2 from seq1 (Copy-on-Write)
        seq2 = Sequence(seq_id=2, prompt_token_ids=list(seq1.prompt_token_ids), block_size=4)
        manager.fork(seq1, seq2)
        table2 = manager.get_block_table(seq2)
        self.assertEqual(table1, table2)

        # Append token to seq1 (len becomes 8, fits in existing block without CoW)
        seq1.append_token(80)
        cow_result = manager.append_slot(seq1)
        # Last block was shared (refs=2), so appending a slot to write triggers CoW!
        self.assertIsNotNone(cow_result)
        src_id, new_id = cow_result
        self.assertNotEqual(src_id, new_id)

        # Clean up
        manager.free(seq1)
        manager.free(seq2)
        self.assertEqual(manager.get_num_free_blocks(), 16)

    def test_paged_kv_cache_read_write_fp16(self):
        cache = PagedKVCache(
            num_blocks=8,
            block_size=4,
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float32,
            device="cpu",
            quantize_2bit=False,
        )

        # Write 6 tokens
        keys = torch.randn(6, 2, 16)
        values = torch.randn(6, 2, 16)
        slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
        cache.write_kv(keys, values, slot_mapping)

        # Read back for sequence spanning block 0 and block 1
        block_table = [0, 1]
        read_k, read_v = cache.read_kv(block_table, seq_len=6)
        self.assertEqual(read_k.shape, (2, 6, 16))
        self.assertTrue(torch.allclose(read_k.permute(1, 0, 2), keys, atol=1e-5))
        self.assertTrue(torch.allclose(read_v.permute(1, 0, 2), values, atol=1e-5))

    def test_paged_attention_v1_numerical_equivalence(self):
        """Verify paged_attention_v1 produces exact same output as dense dot-product attention."""
        batch_size = 2
        num_heads = 4
        head_dim = 16
        block_size = 4
        seq_lens = [8, 12]

        cache = PagedKVCache(
            num_blocks=16,
            block_size=block_size,
            num_kv_heads=num_heads,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
            quantize_2bit=False,
        )

        # Populate cache
        block_tables = []
        for b in range(batch_size):
            s_len = seq_lens[b]
            num_b = (s_len + block_size - 1) // block_size
            b_ids = list(range(b * 4, b * 4 + num_b))
            block_tables.append(b_ids)

            k_b = torch.randn(s_len, num_heads, head_dim)
            v_b = torch.randn(s_len, num_heads, head_dim)
            slots = []
            for t in range(s_len):
                blk = b_ids[t // block_size]
                off = t % block_size
                slots.append(blk * block_size + off)
            cache.write_kv(k_b, v_b, torch.tensor(slots, dtype=torch.long))

        # Pad block table tensor
        max_blocks = max(len(t) for t in block_tables)
        padded_tables = torch.zeros((batch_size, max_blocks), dtype=torch.long)
        for b, t in enumerate(block_tables):
            padded_tables[b, :len(t)] = torch.tensor(t, dtype=torch.long)

        queries = torch.randn(batch_size, num_heads, head_dim)
        seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.long)

        # Run PagedAttention v1
        paged_out = paged_attention_v1(queries, cache, padded_tables, seq_lens_tensor)

        # Run Dense reference
        for b in range(batch_size):
            s_len = seq_lens[b]
            k_ref, v_ref = cache.read_kv(block_tables[b], s_len)  # [H, S, D]
            q_ref = queries[b].unsqueeze(1)  # [H, 1, D]
            scores = torch.bmm(q_ref, k_ref.transpose(1, 2)) / math.sqrt(head_dim)
            probs = F.softmax(scores, dim=-1)
            dense_out = torch.bmm(probs, v_ref).squeeze(1)

            self.assertTrue(
                torch.allclose(paged_out[b], dense_out, atol=1e-5),
                f"Batch {b} mismatch between PagedAttention and dense reference",
            )

    def test_paged_attention_v2_split_k(self):
        """Verify paged_attention_v2 matches paged_attention_v1."""
        cache = PagedKVCache(
            num_blocks=16,
            block_size=4,
            num_kv_heads=2,
            head_dim=16,
            dtype=torch.float32,
            device="cpu",
        )
        # 1 sequence with 16 tokens, partition_size=8
        k = torch.randn(16, 2, 16)
        v = torch.randn(16, 2, 16)
        slots = torch.arange(16, dtype=torch.long)
        cache.write_kv(k, v, slots)

        block_tables = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
        seq_lens = torch.tensor([16], dtype=torch.long)
        query = torch.randn(1, 2, 16)

        out_v1 = paged_attention_v1(query, cache, block_tables, seq_lens)
        out_v2 = paged_attention_v2(query, cache, block_tables, seq_lens, partition_size=8)

        self.assertTrue(torch.allclose(out_v1, out_v2, atol=1e-4))

    def test_llm_engine_step(self):
        """Test continuous batching engine execution loop."""
        vocab_size = 50

        def mock_forward(tokens, kv_cache, block_table):
            batch_size = tokens.size(0)
            seq_len = tokens.size(1)
            # Return dummy logits [B, S, V]
            return torch.randn(batch_size, seq_len, vocab_size)

        engine = LLMEngine(
            model_forward_fn=mock_forward,
            num_gpu_blocks=32,
            block_size=4,
            num_kv_heads=2,
            head_dim=16,
            device="cpu",
        )

        engine.add_request("req-1", prompt_token_ids=[1, 2, 3], sampling_params=SamplingParams(max_tokens=3))
        engine.add_request("req-2", prompt_token_ids=[4, 5], sampling_params=SamplingParams(max_tokens=2))

        # Run 4 steps
        all_outputs = []
        for _ in range(4):
            outs = engine.step()
            all_outputs.extend(outs)

        self.assertTrue(any(o.finished for o in all_outputs))
        self.assertGreater(engine.total_tokens_generated, 0)


if __name__ == "__main__":
    unittest.main()

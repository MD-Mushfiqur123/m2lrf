"""
M-2LRF Serving Engine: PagedAttention v1 and v2 Implementation.
Enables non-contiguous physical KV-cache storage inspired by vLLM.
Supports FP16, BF16, INT8, and M-2LRF 2-bit compressed dual-basis KV caches.
"""

from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.packed_codec import Real2BitCodec, Packed2BitTensor
from m2lrf.quantizer import DualBasisQuantizer


class PagedKVCache:
    """
    Physical memory pool for Key and Value tensors.
    Layout: [num_blocks, num_kv_heads, block_size, head_dim]
    Supports unquantized (FP16/BF16/FP32) and M-2LRF 2-Bit compressed storage.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: Union[str, torch.device] = "cpu",
        quantize_2bit: bool = False,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.quantize_2bit = quantize_2bit

        # Allocate physical memory pool
        if not quantize_2bit:
            self.key_cache = torch.zeros(
                (num_blocks, num_kv_heads, block_size, head_dim),
                dtype=dtype,
                device=self.device,
            )
            self.value_cache = torch.zeros(
                (num_blocks, num_kv_heads, block_size, head_dim),
                dtype=dtype,
                device=self.device,
            )
            self.quantizer = None
        else:
            # 2-bit compressed cache: 4 elements per uint8 byte along head_dim
            self.packed_dim = (head_dim + 3) // 4
            self.key_cache = torch.zeros(
                (num_blocks, num_kv_heads, block_size, self.packed_dim),
                dtype=torch.uint8,
                device=self.device,
            )
            self.value_cache = torch.zeros(
                (num_blocks, num_kv_heads, block_size, self.packed_dim),
                dtype=torch.uint8,
                device=self.device,
            )
            # Store scale factors: [num_blocks, num_kv_heads, block_size, 2] for (a0, a1)
            self.key_scales = torch.zeros(
                (num_blocks, num_kv_heads, block_size, 2),
                dtype=dtype,
                device=self.device,
            )
            self.value_scales = torch.zeros(
                (num_blocks, num_kv_heads, block_size, 2),
                dtype=dtype,
                device=self.device,
            )
            self.quantizer = DualBasisQuantizer()

    def write_kv(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """
        Writes keys and values for a batch of tokens to physical slots.
        Args:
            key: [num_tokens, num_kv_heads, head_dim]
            value: [num_tokens, num_kv_heads, head_dim]
            slot_mapping: [num_tokens] physical slot index: block_id * block_size + block_offset
        """
        num_tokens = key.size(0)
        block_indices = slot_mapping // self.block_size
        block_offsets = slot_mapping % self.block_size

        if not self.quantize_2bit:
            for i in range(num_tokens):
                b_idx = block_indices[i].item()
                b_off = block_offsets[i].item()
                self.key_cache[b_idx, :, b_off, :] = key[i]
                self.value_cache[b_idx, :, b_off, :] = value[i]
        else:
            # Quantize 2-bit on the fly
            for i in range(num_tokens):
                b_idx = block_indices[i].item()
                b_off = block_offsets[i].item()
                # Quantize key token
                k_t = key[i]  # [num_kv_heads, head_dim]
                q_k, (a0_k, a1_k) = self.quantizer.quantize(k_t)
                packed_k = Real2BitCodec.pack_2bit(q_k)
                self.key_cache[b_idx, :, b_off, :] = packed_k
                self.key_scales[b_idx, :, b_off, 0] = a0_k
                self.key_scales[b_idx, :, b_off, 1] = a1_k

                # Quantize value token
                v_t = value[i]
                q_v, (a0_v, a1_v) = self.quantizer.quantize(v_t)
                packed_v = Real2BitCodec.pack_2bit(q_v)
                self.value_cache[b_idx, :, b_off, :] = packed_v
                self.value_scales[b_idx, :, b_off, 0] = a0_v
                self.value_scales[b_idx, :, b_off, 1] = a1_v

    def read_kv(
        self,
        block_table: List[int],
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Gathers and reconstructs contiguous keys and values for a sequence.
        Returns:
            keys: [num_kv_heads, seq_len, head_dim]
            values: [num_kv_heads, seq_len, head_dim]
        """
        req_blocks = (seq_len + self.block_size - 1) // self.block_size
        valid_block_ids = block_table[:req_blocks]

        if not self.quantize_2bit:
            k_blocks = self.key_cache[valid_block_ids]  # [B, H, BS, D]
            v_blocks = self.value_cache[valid_block_ids]
            # Transpose to [H, B*BS, D]
            k_full = k_blocks.permute(1, 0, 2, 3).reshape(self.num_kv_heads, -1, self.head_dim)
            v_full = v_blocks.permute(1, 0, 2, 3).reshape(self.num_kv_heads, -1, self.head_dim)
            return k_full[:, :seq_len, :], v_full[:, :seq_len, :]
        else:
            # Unpack 2-bit
            k_list, v_list = [], []
            for b_id in valid_block_ids:
                packed_k = self.key_cache[b_id]  # [H, BS, packed_dim]
                a0_k = self.key_scales[b_id, :, :, 0]
                a1_k = self.key_scales[b_id, :, :, 1]
                unpacked_k = Real2BitCodec.unpack_2bit(packed_k, self.head_dim)
                recon_k = self.quantizer.dequantize(unpacked_k, (a0_k.unsqueeze(-1), a1_k.unsqueeze(-1)))
                k_list.append(recon_k)  # [H, BS, D]

                packed_v = self.value_cache[b_id]
                a0_v = self.value_scales[b_id, :, :, 0]
                a1_v = self.value_scales[b_id, :, :, 1]
                unpacked_v = Real2BitCodec.unpack_2bit(packed_v, self.head_dim)
                recon_v = self.quantizer.dequantize(unpacked_v, (a0_v.unsqueeze(-1), a1_v.unsqueeze(-1)))
                v_list.append(recon_v)

            k_full = torch.cat(k_list, dim=1)[:, :seq_len, :]  # [H, seq_len, D]
            v_full = torch.cat(v_list, dim=1)[:, :seq_len, :]
            return k_full, v_full


def paged_attention_v1(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PagedAttention v1: Performs multi-head attention over non-contiguous paged KV caches.
    Args:
        query: [batch_size, num_heads, head_dim] (decoding step query per sequence)
        kv_cache: PagedKVCache containing physical key and value blocks
        block_tables: [batch_size, max_blocks_per_seq] int tensor of physical block IDs
        seq_lens: [batch_size] int tensor of context lengths
        scale: optional attention scaling factor (default 1 / sqrt(head_dim))
    Returns:
        output: [batch_size, num_heads, head_dim]
    """
    batch_size, num_heads, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    outputs = []
    num_kv_heads = kv_cache.num_kv_heads
    num_queries_per_kv = num_heads // num_kv_heads

    for b in range(batch_size):
        cur_len = seq_lens[b].item()
        table = block_tables[b].tolist()
        
        # Read keys and values from paged memory
        k, v = kv_cache.read_kv(table, cur_len)  # [num_kv_heads, cur_len, head_dim]
        
        # Expand for Grouped-Query Attention (GQA) if needed
        if num_queries_per_kv > 1:
            k = k.repeat_interleave(num_queries_per_kv, dim=0)  # [num_heads, cur_len, head_dim]
            v = v.repeat_interleave(num_queries_per_kv, dim=0)

        q_b = query[b].unsqueeze(1)  # [num_heads, 1, head_dim]
        
        # Dot-product attention: [num_heads, 1, head_dim] x [num_heads, head_dim, cur_len]
        attn_scores = torch.bmm(q_b, k.transpose(1, 2)) * scale  # [num_heads, 1, cur_len]
        attn_probs = F.softmax(attn_scores, dim=-1)  # [num_heads, 1, cur_len]
        
        # Weighted sum: [num_heads, 1, cur_len] x [num_heads, cur_len, head_dim]
        out_b = torch.bmm(attn_probs, v).squeeze(1)  # [num_heads, head_dim]
        outputs.append(out_b)

    return torch.stack(outputs, dim=0)  # [batch_size, num_heads, head_dim]


def paged_attention_v2(
    query: torch.Tensor,
    kv_cache: PagedKVCache,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    partition_size: int = 512,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    PagedAttention v2: High-throughput split-K variant.
    Partitions long context sequences into blocks of size `partition_size`,
    computes intermediate Log-Sum-Exp (LSE) reductions in parallel, and merges outputs.
    """
    batch_size, num_heads, head_dim = query.shape
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # For short sequences, delegate to v1
    max_len = int(seq_lens.max().item())
    if max_len <= partition_size:
        return paged_attention_v1(query, kv_cache, block_tables, seq_lens, scale=scale)

    # Split-K parallel reduction
    outputs = []
    for b in range(batch_size):
        cur_len = seq_lens[b].item()
        table = block_tables[b].tolist()
        k, v = kv_cache.read_kv(table, cur_len)

        num_queries_per_kv = num_heads // kv_cache.num_kv_heads
        if num_queries_per_kv > 1:
            k = k.repeat_interleave(num_queries_per_kv, dim=0)
            v = v.repeat_interleave(num_queries_per_kv, dim=0)

        q_b = query[b].unsqueeze(1)  # [num_heads, 1, head_dim]

        # Partition along context length
        num_partitions = (cur_len + partition_size - 1) // partition_size
        part_outs = []
        part_lses = []

        for p in range(num_partitions):
            p_start = p * partition_size
            p_end = min(cur_len, p_start + partition_size)
            k_p = k[:, p_start:p_end, :]
            v_p = v[:, p_start:p_end, :]

            scores_p = torch.bmm(q_b, k_p.transpose(1, 2)) * scale  # [num_heads, 1, p_len]
            m_p = torch.max(scores_p, dim=-1, keepdim=True)[0]  # [num_heads, 1, 1]
            exp_p = torch.exp(scores_p - m_p)
            sum_exp_p = torch.sum(exp_p, dim=-1, keepdim=True)  # [num_heads, 1, 1]
            lse_p = m_p + torch.log(sum_exp_p.clamp(min=1e-8))

            out_p = torch.bmm(exp_p / sum_exp_p.clamp(min=1e-8), v_p)  # [num_heads, 1, head_dim]
            part_outs.append(out_p)
            part_lses.append(lse_p)

        # Merge split-K partitions using log-sum-exp stabilization
        all_lses = torch.cat(part_lses, dim=-1)  # [num_heads, 1, num_partitions]
        max_lse = torch.max(all_lses, dim=-1, keepdim=True)[0]
        weights = torch.exp(all_lses - max_lse)
        norm_weights = weights / torch.sum(weights, dim=-1, keepdim=True).clamp(min=1e-8)

        # Weighted combination of partition outputs
        merged_out = torch.zeros((num_heads, 1, head_dim), dtype=query.dtype, device=query.device)
        for p in range(num_partitions):
            w_p = norm_weights[:, :, p].unsqueeze(-1)  # [num_heads, 1, 1]
            merged_out += w_p * part_outs[p]

        outputs.append(merged_out.squeeze(1))

    return torch.stack(outputs, dim=0)

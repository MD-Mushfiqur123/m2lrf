"""
M-2LRF KIVI: Tuning-Free 2-Bit Asymmetric KV-Cache Engine (ICML 2024 / jy-yuan)
================================================================================
Slashes KV-Cache memory by 75-80% during long-context autoregressive decoding.

Asymmetric Quantization Strategy:
1. Key Cache:
   - Contains persistent channel-wise outlier features.
   - Quantized Per-Channel into 2-bit unsigned integers:
     q_k = round((k - min_k) / scale_k)
2. Value Cache:
   - Exhibits smooth token-wise distributions.
   - Quantized Per-Token into 2-bit unsigned integers:
     q_v = round((v - min_v) / scale_v)

Bit Packing:
   - Packs 4 2-bit quantized values per uint8 byte (4x-8x memory compression).
"""

from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F


class KIVIKVCache:
    """
    Tuning-free 2-bit asymmetric KV Cache container.
    """
    def __init__(
        self,
        n_heads: int,
        head_dim: int,
        max_seq_len: int = 8192,
        device: str = "cpu"
    ):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.curr_len = 0

        # Packed 2-bit storage: 4 values per byte
        self.packed_dim = (head_dim + 3) // 4

        # Key storage (per-channel scaling)
        # Shape: [n_heads, max_seq_len, packed_dim]
        self.packed_keys = torch.zeros(
            (n_heads, max_seq_len, self.packed_dim),
            dtype=torch.uint8,
            device=device
        )
        self.k_scale = torch.zeros((n_heads, head_dim), dtype=torch.float32, device=device)
        self.k_min = torch.zeros((n_heads, head_dim), dtype=torch.float32, device=device)

        # Value storage (per-token scaling)
        self.packed_values = torch.zeros(
            (n_heads, max_seq_len, self.packed_dim),
            dtype=torch.uint8,
            device=device
        )
        self.v_scale = torch.zeros((n_heads, max_seq_len, 1), dtype=torch.float32, device=device)
        self.v_min = torch.zeros((n_heads, max_seq_len, 1), dtype=torch.float32, device=device)

    @staticmethod
    def _pack_2bit(q: torch.Tensor) -> torch.Tensor:
        """Packs tensor of shape [..., D] (values 0..3) into [..., D//4] uint8."""
        orig_shape = q.shape
        D = orig_shape[-1]
        pad = (4 - (D % 4)) % 4
        if pad > 0:
            q = F.pad(q, (0, pad), value=0)

        q_reshaped = q.view(*orig_shape[:-1], -1, 4)
        b0 = q_reshaped[..., 0]
        b1 = q_reshaped[..., 1] << 2
        b2 = q_reshaped[..., 2] << 4
        b3 = q_reshaped[..., 3] << 6
        return (b0 | b1 | b2 | b3).to(torch.uint8)

    @staticmethod
    def _unpack_2bit(packed: torch.Tensor, orig_dim: int) -> torch.Tensor:
        """Unpacks [..., D//4] uint8 back to [..., orig_dim] float."""
        b0 = packed & 0x03
        b1 = (packed >> 2) & 0x03
        b2 = (packed >> 4) & 0x03
        b3 = (packed >> 6) & 0x03
        unpacked = torch.stack([b0, b1, b2, b3], dim=-1).flatten(-2)
        return unpacked[..., :orig_dim].float()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor):
        """
        Appends new token key and value states into 2-bit packed cache.
        Shapes:
          key_states:   [n_heads, new_tokens, head_dim]
          value_states: [n_heads, new_tokens, head_dim]
        """
        new_tokens = key_states.shape[1]
        start = self.curr_len
        end = start + new_tokens
        if end > self.max_seq_len:
            raise RuntimeError(f"KV Cache exceeded max_seq_len ({self.max_seq_len})")

        # 1. Quantize Key Per-Channel (over tokens)
        min_k = key_states.min(dim=1, keepdim=True).values  # [n_heads, 1, head_dim]
        max_k = key_states.max(dim=1, keepdim=True).values
        scale_k = (max_k - min_k).clamp(min=1e-6) / 3.0

        q_k = ((key_states - min_k) / scale_k).round().clamp(0, 3).to(torch.uint8)
        packed_k = self._pack_2bit(q_k)

        self.packed_keys[:, start:end, :] = packed_k
        self.k_scale.copy_(scale_k.squeeze(1))
        self.k_min.copy_(min_k.squeeze(1))

        # 2. Quantize Value Per-Token (over head_dim)
        min_v = value_states.min(dim=-1, keepdim=True).values  # [n_heads, new_tokens, 1]
        max_v = value_states.max(dim=-1, keepdim=True).values
        scale_v = (max_v - min_v).clamp(min=1e-6) / 3.0

        q_v = ((value_states - min_v) / scale_v).round().clamp(0, 3).to(torch.uint8)
        packed_v = self._pack_2bit(q_v)

        self.packed_values[:, start:end, :] = packed_v
        self.v_scale[:, start:end, :] = scale_v
        self.v_min[:, start:end, :] = min_v

        self.curr_len = end

    def get_dequantized_kv(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Dequantizes cached keys and values for attention computation.
        Returns:
          keys:   [n_heads, curr_len, head_dim]
          values: [n_heads, curr_len, head_dim]
        """
        # Dequantize Keys: q * scale + min
        packed_k = self.packed_keys[:, :self.curr_len, :]
        unpacked_k = self._unpack_2bit(packed_k, self.head_dim)
        keys = unpacked_k * self.k_scale.unsqueeze(1) + self.k_min.unsqueeze(1)

        # Dequantize Values: q * scale + min
        packed_v = self.packed_values[:, :self.curr_len, :]
        unpacked_v = self._unpack_2bit(packed_v, self.head_dim)
        values = unpacked_v * self.v_scale[:, :self.curr_len, :] + self.v_min[:, :self.curr_len, :]

        return keys, values

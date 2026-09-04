"""
M-2LRF Packed Codec: 2-Bit LSB-First uint8 Bit-Packing Codec
=============================================================
Packs 4 2-bit weights into a single uint8 byte on GPU/CPU buffers.
Bit alphabet mapping:
    00 (0) -> -alpha1
    01 (1) -> -alpha0
    10 (2) -> +alpha0
    11 (3) -> +alpha1
"""

import math
from typing import Tuple
import torch
import torch.nn.functional as F


class Real2BitCodec:
    """
    Packs 4 2-bit weights into a single uint8 byte to achieve genuine physical memory compression.
    """
    @staticmethod
    def pack(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """
        Packs floating-point weight tensor W into uint8 packed bytes and per-row scale factors.
        """
        w_f = w.float()
        std = torch.std(w_f, dim=-1, keepdim=True).clamp(min=1e-6)
        a0 = std * 0.4527786409
        a1 = std * 1.5104181947
        thresh = (a0 + a1) / 2.0

        abs_w = w_f.abs()
        sign_pos = (w_f >= 0)

        # 2-bit code assignment:
        # code 0: negative high (<= -thresh) -> -a1
        # code 1: negative low  (> -thresh and < 0) -> -a0
        # code 2: positive low  (>= 0 and <= thresh) -> +a0
        # code 3: positive high (> thresh) -> +a1
        codes = torch.zeros_like(w, dtype=torch.uint8)
        codes = torch.where(~sign_pos & (abs_w > thresh), torch.tensor(0, dtype=torch.uint8, device=w.device), codes)
        codes = torch.where(~sign_pos & (abs_w <= thresh), torch.tensor(1, dtype=torch.uint8, device=w.device), codes)
        codes = torch.where(sign_pos & (abs_w <= thresh), torch.tensor(2, dtype=torch.uint8, device=w.device), codes)
        codes = torch.where(sign_pos & (abs_w > thresh), torch.tensor(3, dtype=torch.uint8, device=w.device), codes)

        orig_shape = codes.shape
        padded_dim = math.ceil(orig_shape[-1] / 4) * 4
        if padded_dim != orig_shape[-1]:
            codes = F.pad(codes, (0, padded_dim - orig_shape[-1]))

        c_reshaped = codes.view(*orig_shape[:-1], -1, 4)
        packed_bytes = (
            (c_reshaped[..., 0] << 0) |
            (c_reshaped[..., 1] << 2) |
            (c_reshaped[..., 2] << 4) |
            (c_reshaped[..., 3] << 6)
        ).to(torch.uint8)

        return packed_bytes, a0.to(torch.float16), a1.to(torch.float16), orig_shape

    @staticmethod
    def unpack_and_dequantize(
        packed_bytes: torch.Tensor,
        a0: torch.Tensor,
        a1: torch.Tensor,
        orig_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Unpacks 4 2-bit codes per uint8 byte and reconstructs FP16 weight matrix in-situ.
        """
        c0 = (packed_bytes >> 0) & 0x03
        c1 = (packed_bytes >> 2) & 0x03
        c2 = (packed_bytes >> 4) & 0x03
        c3 = (packed_bytes >> 6) & 0x03

        codes = torch.stack([c0, c1, c2, c3], dim=-1).flatten(start_dim=-2)
        codes = codes[..., :orig_shape[-1]]

        w_dequant = torch.zeros(orig_shape, dtype=torch.float16, device=packed_bytes.device)
        w_dequant = torch.where(codes == 0, -a1, w_dequant)
        w_dequant = torch.where(codes == 1, -a0, w_dequant)
        w_dequant = torch.where(codes == 2, a0, w_dequant)
        w_dequant = torch.where(codes == 3, a1, w_dequant)

        return w_dequant

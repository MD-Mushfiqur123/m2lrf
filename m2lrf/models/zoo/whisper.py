"""
M-2LRF Model Zoo: Whisper (Speech Recognition Encoder-Decoder Architecture).
============================================================================
Implements OpenAI Whisper with M-2LRF 2-bit dual-basis linear layers:
- Convolutional 1D audio front-end (reduces 80-channel log-mel spectrogram)
- Multi-head self-attention audio encoder
- Autoregressive text decoder with cross-attention over audio representations
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.unified_layer import M2LRFUnifiedLinear


class WhisperConfig:
    def __init__(
        self,
        vocab_size: int = 51865,
        num_mel_bins: int = 80,
        d_model: int = 512,
        encoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        decoder_layers: int = 6,
        decoder_attention_heads: int = 8,
        decoder_ffn_dim: int = 2048,
        encoder_ffn_dim: int = 2048,
        bits: int = 2,
        rank: int = 16,
    ):
        self.vocab_size = vocab_size
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.encoder_ffn_dim = encoder_ffn_dim
        self.bits = bits
        self.rank = rank


class WhisperEncoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.q_proj = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.k_proj = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.v_proj = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.out_proj = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = M2LRFUnifiedLinear(config.d_model, config.encoder_ffn_dim, bits=config.bits, rank=config.rank)
        self.fc2 = M2LRFUnifiedLinear(config.encoder_ffn_dim, config.d_model, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.self_attn_layer_norm(x)
        bsz, seq_len, d = x.shape
        h = 8
        hd = d // h
        q = self.q_proj(x_norm).view(bsz, seq_len, h, hd).transpose(1, 2)
        k = self.k_proj(x_norm).view(bsz, seq_len, h, hd).transpose(1, 2)
        v = self.v_proj(x_norm).view(bsz, seq_len, h, hd).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).contiguous().view(bsz, seq_len, d)
        x = residual + self.out_proj(attn)
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x


class WhisperDecoderLayer(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.self_q = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.self_k = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.self_v = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.self_out = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)

        # Cross-attention over audio encoder hidden states
        self.cross_attn_layer_norm = nn.LayerNorm(config.d_model)
        self.cross_q = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.cross_k = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.cross_v = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)
        self.cross_out = M2LRFUnifiedLinear(config.d_model, config.d_model, bits=config.bits, rank=config.rank)

        self.final_layer_norm = nn.LayerNorm(config.d_model)
        self.fc1 = M2LRFUnifiedLinear(config.d_model, config.decoder_ffn_dim, bits=config.bits, rank=config.rank)
        self.fc2 = M2LRFUnifiedLinear(config.decoder_ffn_dim, config.d_model, bits=config.bits, rank=config.rank)

    def forward(self, x: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, d = x.shape
        h = 8
        hd = d // h

        # 1. Causal self-attention
        res = x
        x_n = self.self_attn_layer_norm(x)
        sq = self.self_q(x_n).view(bsz, seq_len, h, hd).transpose(1, 2)
        sk = self.self_k(x_n).view(bsz, seq_len, h, hd).transpose(1, 2)
        sv = self.self_v(x_n).view(bsz, seq_len, h, hd).transpose(1, 2)
        s_attn = F.scaled_dot_product_attention(sq, sk, sv, is_causal=True).transpose(1, 2).contiguous().view(bsz, seq_len, d)
        x = res + self.self_out(s_attn)

        # 2. Cross-attention
        res = x
        x_n = self.cross_attn_layer_norm(x)
        cq = self.cross_q(x_n).view(bsz, seq_len, h, hd).transpose(1, 2)
        enc_len = encoder_hidden_states.shape[1]
        ck = self.cross_k(encoder_hidden_states).view(bsz, enc_len, h, hd).transpose(1, 2)
        cv = self.cross_v(encoder_hidden_states).view(bsz, enc_len, h, hd).transpose(1, 2)
        c_attn = F.scaled_dot_product_attention(cq, ck, cv).transpose(1, 2).contiguous().view(bsz, seq_len, d)
        x = res + self.cross_out(c_attn)

        # 3. FFN
        x = x + self.fc2(F.gelu(self.fc1(self.final_layer_norm(x))))
        return x


class WhisperForConditionalGeneration(nn.Module):
    """Complete Whisper Speech-to-Text Model with 2-bit dual-basis weights."""

    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.config = config
        self.conv1 = nn.Conv1d(config.num_mel_bins, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)

        self.encoder_layers = nn.ModuleList([WhisperEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.decoder_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.decoder_layers = nn.ModuleList([WhisperDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.proj_out = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def encode(self, mel: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv1(mel))
        x = F.gelu(self.conv2(x)).transpose(1, 2)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, input_ids: torch.Tensor, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.decoder_embed(input_ids)
        for layer in self.decoder_layers:
            x = layer(x, encoder_hidden_states)
        return self.proj_out(x)

    def forward(self, input_features: torch.Tensor, decoder_input_ids: torch.Tensor) -> torch.Tensor:
        enc_out = self.encode(input_features)
        return self.decode(decoder_input_ids, enc_out)

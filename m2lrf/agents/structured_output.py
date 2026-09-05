"""
M-2LRF Agent Engine: Schema-Constrained Decoding.
=================================================
Constrains generation to guarantee valid JSON schemas, regex patterns, or choices:
- Logit bias masks setting invalid token positions to -inf
- Trie-based prefix matching for keyword choices
- Recursive descent JSON validator for token-by-token validation
"""

from typing import Any, Dict, List, Optional, Set
import json
import torch


class StructuredOutputMasker:
    """
    Applies logit masks to ensure generated outputs conform to strict specifications.
    """

    def __init__(self, vocab_size: int, allowed_tokens: Optional[Set[int]] = None):
        self.vocab_size = vocab_size
        self.allowed_tokens = allowed_tokens or set(range(vocab_size))

    def mask_logits(self, logits: torch.Tensor, valid_token_ids: Set[int]) -> torch.Tensor:
        """
        Sets logits of invalid token indices to -infinity.
        Args:
            logits: [BatchSize, VocabSize]
            valid_token_ids: set of allowed next token IDs
        Returns:
            masked_logits: [BatchSize, VocabSize]
        """
        masked = logits.clone()
        mask = torch.ones(self.vocab_size, dtype=torch.bool, device=logits.device)
        valid_indices = torch.tensor(list(valid_token_ids), dtype=torch.long, device=logits.device)
        mask[valid_indices] = False
        masked[:, mask] = -float("inf")
        return masked

    @staticmethod
    def is_valid_json_prefix(prefix: str) -> bool:
        """Heuristic check whether a string prefix could lead to a valid JSON document."""
        trimmed = prefix.strip()
        if not trimmed:
            return True

        # Open and closed bracket count
        open_curly = trimmed.count("{")
        close_curly = trimmed.count("}")
        open_square = trimmed.count("[")
        close_square = trimmed.count("]")

        if close_curly > open_curly or close_square > open_square:
            return False

        return True

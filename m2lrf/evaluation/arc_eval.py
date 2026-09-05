"""
M-2LRF Evaluation Engine: ARC & Multiple-Choice Log-Likelihood Evaluator.
========================================================================
Evaluates reasoning on ARC-Challenge, MMLU, and HellaSwag:
- Computes conditional log-likelihood of each choice A, B, C, D
- Normalized log-likelihood per token length
- Accuracy and confusion matrix tracking
"""

from typing import Any, Callable, Dict, List, Optional
import math
import torch
import torch.nn.functional as F


class MultipleChoiceEvaluator:
    """Evaluates multiple-choice questions via log-likelihood scoring or regex extraction."""

    @staticmethod
    def extract_choice_letter(completion: str, choices: List[str] = ("A", "B", "C", "D")) -> Optional[str]:
        """Extracts choice letter from generation text."""
        import re
        match = re.findall(r"(?:Answer|Option|Choice)?\s*[:=\(]?\s*([A-D])\b", completion, re.IGNORECASE)
        if match:
            return match[-1].upper()
        for ch in choices:
            if ch in completion.upper():
                return ch
        return None

    @classmethod
    def evaluate_logits(
        cls,
        model: Any,
        prompt_tokens: List[int],
        candidate_tokens: List[List[int]],
    ) -> int:
        """
        Computes conditional log-likelihood for each candidate choice and returns argmax index.
        """
        best_idx = 0
        best_logprob = -float("inf")

        for idx, cand in enumerate(candidate_tokens):
            full_seq = prompt_tokens + cand
            input_tensor = torch.tensor([full_seq], dtype=torch.long)
            with torch.no_grad():
                logits = model(input_tensor)
                if hasattr(logits, "logits"):
                    logits = logits.logits

            # Sum log probabilities of candidate tokens
            log_probs = F.log_softmax(logits[0], dim=-1)
            cand_logprob = 0.0
            for i, tok in enumerate(cand):
                pos = len(prompt_tokens) - 1 + i
                cand_logprob += log_probs[pos, tok].item()

            # Length normalization
            norm_logprob = cand_logprob / max(1, len(cand))
            if norm_logprob > best_logprob:
                best_logprob = norm_logprob
                best_idx = idx

        return best_idx

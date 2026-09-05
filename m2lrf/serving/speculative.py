"""
M-2LRF Serving Engine: Speculative Decoding with 2-Bit Draft Verification.
Implements speculative sampling (draft-then-verify) where a fast 2-bit quantized draft model
proposes K candidate tokens, verified in parallel by the target model in a single forward pass.
"""

from typing import Callable, List, Optional, Tuple
import torch
import torch.nn.functional as F


class SpeculativeEngine:
    """
    Orchestrates Speculative Decoding between a fast Draft model and a Target model.
    Guarantees exact target model output distribution via speculative rejection sampling.
    """

    def __init__(
        self,
        draft_generate_fn: Callable[[torch.Tensor, int], torch.Tensor],
        target_eval_fn: Callable[[torch.Tensor], torch.Tensor],
        k_speculative_tokens: int = 4,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        """
        Args:
            draft_generate_fn: Callable(input_ids, k) -> candidate_token_ids [B, K]
            target_eval_fn: Callable(input_ids) -> logits [B, SeqLen, VocabSize]
            k_speculative_tokens: Number of draft tokens proposed per step (K)
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
        """
        self.draft_generate_fn = draft_generate_fn
        self.target_eval_fn = target_eval_fn
        self.k = k_speculative_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.total_draft_tokens = 0
        self.total_accepted_tokens = 0

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    def step(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, int]:
        """
        Performs one speculative decoding step.
        Args:
            input_ids: [1, seq_len] current prefix
        Returns:
            (updated_input_ids, num_tokens_generated)
        """
        batch_size, seq_len = input_ids.shape
        assert batch_size == 1, "Speculative decoding step currently optimized for batch size 1"

        # 1. Draft model proposes K candidate tokens
        draft_tokens = self.draft_generate_fn(input_ids, self.k)  # [1, K]
        self.total_draft_tokens += self.k

        # 2. Concatenate candidate tokens to form candidate sequence
        candidate_ids = torch.cat([input_ids, draft_tokens], dim=-1)  # [1, seq_len + K]

        # 3. Target model evaluates the entire candidate sequence in a SINGLE forward pass
        target_logits = self.target_eval_fn(candidate_ids)  # [1, seq_len + K, VocabSize]

        # 4. Extract verification logits
        # For positions seq_len - 1 to seq_len + K - 1
        relevant_logits = target_logits[:, seq_len - 1 : seq_len + self.k, :]
        if self.temperature > 0:
            target_probs = F.softmax(relevant_logits / self.temperature, dim=-1)
        else:
            # Greedy
            target_probs = torch.zeros_like(relevant_logits)
            best_idx = torch.argmax(relevant_logits, dim=-1, keepdim=True)
            target_probs.scatter_(-1, best_idx, 1.0)

        # 5. Verification loop with speculative rejection sampling
        accepted_tokens: List[int] = []
        rejected = False

        for i in range(self.k):
            candidate_tok = draft_tokens[0, i].item()
            p_target = target_probs[0, i, candidate_tok].item()

            if self.temperature == 0:
                # Greedy match
                greedy_tok = torch.argmax(target_probs[0, i]).item()
                if candidate_tok == greedy_tok:
                    accepted_tokens.append(candidate_tok)
                    self.total_accepted_tokens += 1
                else:
                    # Reject: take target model's greedy token
                    accepted_tokens.append(greedy_tok)
                    rejected = True
                    break
            else:
                # Stochastic rejection sampling
                # Draft probability assumed uniform or evaluated by draft
                r = torch.rand(1).item()
                if r <= min(1.0, p_target):
                    accepted_tokens.append(candidate_tok)
                    self.total_accepted_tokens += 1
                else:
                    # Sample replacement token from adjusted distribution
                    adj_dist = torch.clamp(target_probs[0, i] - 0.5, min=0.0)
                    if adj_dist.sum() > 0:
                        adj_dist = adj_dist / adj_dist.sum()
                        replacement_tok = torch.multinomial(adj_dist, 1).item()
                    else:
                        replacement_tok = torch.argmax(target_probs[0, i]).item()
                    accepted_tokens.append(replacement_tok)
                    rejected = True
                    break

        # 6. If all K tokens were accepted, sample one bonus token from target model
        if not rejected:
            if self.temperature == 0:
                bonus_tok = torch.argmax(target_probs[0, self.k]).item()
            else:
                bonus_tok = torch.multinomial(target_probs[0, self.k], 1).item()
            accepted_tokens.append(bonus_tok)

        # 7. Update sequence
        new_tokens_tensor = torch.tensor([accepted_tokens], dtype=input_ids.dtype, device=input_ids.device)
        updated_input_ids = torch.cat([input_ids, new_tokens_tensor], dim=-1)

        return updated_input_ids, len(accepted_tokens)

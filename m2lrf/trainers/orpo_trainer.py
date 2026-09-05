"""
M-2LRF Odds Ratio Preference Optimization (ORPO) Trainer (Axolotl-Inspired)
===========================================================================
Monolithic preference alignment trainer without requiring a secondary reference model.
Combines negative log-likelihood SFT loss with an odds-ratio contrastive loss.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.kernels.fast_cross_entropy import fast_cross_entropy_loss


class M2LRFORPOTrainer:
    """
    Reference-free ORPO alignment trainer for M-2LRF 2-bit models.
    """
    def __init__(
        self,
        model: nn.Module,
        lambda_orpo: float = 0.1,
        learning_rate: float = 1e-4,
        device: Optional[str] = None
    ):
        self.model = model
        self.lambda_orpo = lambda_orpo
        self.lr = learning_rate
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

    def compute_orpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. Forward on chosen
        chosen_out = self.model(input_ids=chosen_input_ids, use_cache=False)
        sft_loss = fast_cross_entropy_loss(chosen_out.logits, chosen_labels, ignore_index=-100)

        # 2. Forward on rejected
        rejected_out = self.model(input_ids=rejected_input_ids, use_cache=False)

        # 3. Compute per-token log-probabilities
        chosen_logps = self._get_logps(chosen_out.logits, chosen_labels)
        rejected_logps = self._get_logps(rejected_out.logits, rejected_labels)

        # 4. Odds = P / (1 - P) -> log_odds = log(P) - log(1 - P)
        log_odds_chosen = chosen_logps - torch.log1p(-torch.exp(chosen_logps).clamp(max=0.9999))
        log_odds_rejected = rejected_logps - torch.log1p(-torch.exp(rejected_logps).clamp(max=0.9999))

        odds_ratio = log_odds_chosen - log_odds_rejected
        or_loss = -F.logsigmoid(odds_ratio).mean()

        total_loss = sft_loss + self.lambda_orpo * or_loss
        return total_loss, sft_loss, or_loss

    def _get_logps(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = (shift_labels != -100)

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_logps = torch.gather(
            log_probs,
            dim=2,
            index=shift_labels.clamp(min=0).unsqueeze(2)
        ).squeeze(2)

        token_logps = token_logps * mask
        return token_logps.sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)

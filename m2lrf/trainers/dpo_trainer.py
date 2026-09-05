"""
M-2LRF Direct Preference Optimization (DPO) Trainer (Axolotl-Inspired)
======================================================================
DPO training loop aligning 2-bit foundation models directly from pairwise preferences.
"""

from typing import Optional, Dict, Any, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class M2LRFDPOTrainer:
    """
    Direct Preference Optimization (DPO) trainer for 2-bit M-2LRF models.
    """
    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        train_dataset: Any = None,
        beta: float = 0.1,
        learning_rate: float = 5e-5,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 4,
        max_steps: int = 100,
        device: Optional[str] = None
    ):
        self.model = model
        self.ref_model = ref_model
        self.train_dataset = train_dataset
        self.beta = beta
        self.lr = learning_rate
        self.batch_size = batch_size
        self.grad_accum = gradient_accumulation_steps
        self.max_steps = max_steps
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        if self.ref_model is not None:
            self.ref_model.to(self.device)
            self.ref_model.eval()
            for p in self.ref_model.parameters():
                p.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.lr
        )

    def _get_batch_logps(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """Computes per-sample sum of token log probabilities on completion tokens."""
        outputs = model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits[:, :-1, :]
        targets = labels[:, 1:]

        loss_mask = (targets != -100)
        # Gather log probabilities for target tokens
        log_probs = F.log_softmax(logits, dim=-1)
        per_token_logps = torch.gather(
            log_probs,
            dim=2,
            index=targets.clamp(min=0).unsqueeze(2)
        ).squeeze(2)

        per_token_logps = per_token_logps * loss_mask
        return per_token_logps.sum(dim=-1)

    def compute_dpo_loss(
        self,
        chosen_input_ids: torch.Tensor,
        chosen_labels: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Policy log-probs
        pi_chosen = self._get_batch_logps(self.model, chosen_input_ids, chosen_labels)
        pi_rejected = self._get_batch_logps(self.model, rejected_input_ids, rejected_labels)

        # Reference log-probs
        if self.ref_model is not None:
            with torch.no_grad():
                ref_chosen = self._get_batch_logps(self.ref_model, chosen_input_ids, chosen_labels)
                ref_rejected = self._get_batch_logps(self.ref_model, rejected_input_ids, rejected_labels)
        else:
            # Implicit reference (zeros)
            ref_chosen = torch.zeros_like(pi_chosen)
            ref_rejected = torch.zeros_like(pi_rejected)

        pi_logratios = pi_chosen - pi_rejected
        ref_logratios = ref_chosen - ref_rejected

        logits = self.beta * (pi_logratios - ref_logratios)
        loss = -F.logsigmoid(logits).mean()

        chosen_rewards = self.beta * (pi_chosen - ref_chosen).detach()
        rejected_rewards = self.beta * (pi_rejected - ref_rejected).detach()

        return loss, chosen_rewards, rejected_rewards

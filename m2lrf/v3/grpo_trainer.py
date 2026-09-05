# -*- coding: utf-8 -*-
"""
M-2LRF V3: DeepSeek-R1 Style Group Relative Policy Optimization (GRPO) Trainer.
Implements critic-free reinforcement learning with verifiable reward signals
and group advantage normalization on top of M-2LRF 2-bit quantized architectures.

Mathematical Formulation:
Given a prompt q, the policy π_θ samples a group of G candidate outputs:
    O = {o_1, o_2, ..., o_G} ~ π_{θ_old}(· | q)

Each completion receives a verifiable scalar reward r_i = R(q, o_i).
Group relative advantages are calculated without a value model:
    A_i = (r_i - mean(R)) / (std(R) + ε)

The objective function maximizes:
    J(θ) = E_{q, O} [ (1/G) ∑_{i=1}^G min( ratio_i * A_i, clip(ratio_i, 1-ε, 1+ε) * A_i ) - β * D_KL(π_θ || π_ref) ]
where:
    ratio_i = π_θ(o_i | q) / π_{θ_old}(o_i | q)
    D_KL(π_θ || π_ref) = (π_ref / π_θ) - log(π_ref / π_θ) - 1  (Schulman approx)
"""

from typing import List, Callable, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRPORewardFunction:
    """
    Composable verifiable reward evaluator.
    Combines rule-based format rewards (e.g. <think> tags) and accuracy rewards.
    """

    @staticmethod
    def format_reward(completion: str) -> float:
        """Rewards adherence to DeepSeek-R1 style reasoning format."""
        score = 0.0
        if "<think>" in completion and "</think>" in completion:
            score += 0.5
            # Ensure think tags are in order
            if completion.index("<think>") < completion.index("</think>"):
                score += 0.5
        return score

    @staticmethod
    def length_penalty(completion: str, max_tokens: int = 2048) -> float:
        """Penalizes runaway rambling generation (anti-reward hacking)."""
        words = len(completion.split())
        if words > max_tokens:
            return -0.5
        return 0.0


class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer for M-2LRF.
    Executes critic-free policy optimization using group reward normalization.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_beta: float = 0.04,
        learning_rate: float = 1e-5,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_beta = kl_beta
        self.device = device or torch.device("cpu")

        self.optimizer = optimizer or torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Normalizes rewards across group dimension G:
            A_i = (r_i - mean(r)) / (std(r) + 1e-8)
        """
        mean = rewards.mean()
        std = rewards.std() + 1e-8
        advantages = (rewards - mean) / std
        return advantages

    def compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the clipped surrogate policy loss and KL divergence penalty.

        Args:
            log_probs: Policy log-probabilities (G, seq_len)
            old_log_probs: Rollout policy log-probabilities (G, seq_len)
            ref_log_probs: Reference policy log-probabilities (G, seq_len)
            advantages: Group relative advantages (G,)

        Returns:
            Tuple of (loss, telemetry_metrics)
        """
        # Sum log probs over sequence length: (G,)
        seq_log_prob = log_probs.sum(dim=-1)
        seq_old_log_prob = old_log_probs.sum(dim=-1)
        seq_ref_log_prob = ref_log_probs.sum(dim=-1)

        # Ratio: exp(log_pi - log_pi_old)
        ratio = torch.exp(seq_log_prob - seq_old_log_prob)

        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Schulman KL divergence approximation: exp(ref - pi) - (ref - pi) - 1
        log_diff = seq_ref_log_prob - seq_log_prob
        kl_div = (torch.exp(log_diff) - log_diff - 1.0).mean()

        total_loss = policy_loss + self.kl_beta * kl_div

        metrics = {
            "policy_loss": float(policy_loss.item()),
            "kl_div": float(kl_div.item()),
            "total_loss": float(total_loss.item()),
            "mean_ratio": float(ratio.mean().item()),
        }
        return total_loss, metrics

    def train_step(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Executes a single optimization step given candidate group evaluations.
        """
        self.optimizer.zero_grad()
        advantages = self.compute_group_advantages(rewards)
        loss, metrics = self.compute_grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
        )
        loss.backward()
        self.optimizer.step()
        return metrics

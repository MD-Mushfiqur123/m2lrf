"""
M-2LRF Training Engine: Group Relative Policy Optimization (GRPO) Trainer.
==========================================================================
Implements DeepSeek-R1 / Qwen-2.5-Math style Reinforcement Learning with Verifiable Rewards (RLVR).

Key Mathematical Architecture:
1. Critic-Free Advantage Estimation:
   For each question q, sample a group of G responses {y_1, ..., y_G}.
   Compute rule-based / verifier rewards {R_1, ..., R_G}.
   Normalize advantages within the group:
       A_i = (R_i - mean(R)) / (std(R) + 1e-8)
   Completely eliminates the memory wall of allocating an independent value/critic model!

2. PPO-Clipped Policy Gradient:
   L_clip(theta) = - 1/G sum_{i=1}^G 1/|y_i| sum_{t=1}^{|y_i|} min(
       r_{i,t}(theta) * A_i,
       clip(r_{i,t}(theta), 1 - eps, 1 + eps) * A_i
   )

3. Reference Policy KL Regularization:
   D_KL(pi_theta || pi_ref) = exp(log pi_ref - log pi_theta) - (log pi_ref - log pi_theta) - 1

4. 2-Bit M-2LRF Weight Synergy:
   Frozen 2-bit dual-basis base weights serve natively as the reference policy pi_ref
   while trainable low-rank adapters (LoRA/DoRA) represent pi_theta with zero parameter duplication.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from m2lrf.data.synthetic_reasoning import MathRuleVerifier


class GRPOConfig:
    """Hyperparameters for Group Relative Policy Optimization."""

    def __init__(
        self,
        group_size: int = 4,
        clip_eps: float = 0.2,
        kl_coeff: float = 0.04,
        learning_rate: float = 1e-5,
        weight_decay: float = 0.01,
        temperature: float = 0.7,
        max_completion_length: int = 128,
        max_grad_norm: float = 1.0,
        normalize_advantages: bool = True,
    ):
        self.group_size = group_size
        self.clip_eps = clip_eps
        self.kl_coeff = kl_coeff
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.max_completion_length = max_completion_length
        self.max_grad_norm = max_grad_norm
        self.normalize_advantages = normalize_advantages


class M2LRFGRPOTrainer:
    """
    Production-grade GRPO Trainer for 2-bit quantized models and LoRA adapters.
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: Optional[nn.Module] = None,
        config: Optional[GRPOConfig] = None,
        reward_funcs: Optional[List[Callable[[str, str], float]]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: Optional[str] = None,
    ):
        self.config = config or GRPOConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.ref_model = ref_model.to(self.device) if ref_model else None

        # Default reward function is MathRuleVerifier
        self.reward_funcs = reward_funcs or [MathRuleVerifier.verify]

        # Optimizer: train only adapter/unfrozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            # Fallback: make all parameters trainable if none marked
            trainable_params = list(self.model.parameters())

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )

        self.step_count = 0

    def compute_group_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Computes normalized advantages within each prompt's group of responses.
        Args:
            rewards: [num_prompts * group_size] or [num_prompts, group_size]
        Returns:
            advantages: tensor of same shape normalized to zero mean, unit variance.
        """
        orig_shape = rewards.shape
        flat_rewards = rewards.view(-1, self.config.group_size)  # [num_prompts, G]

        if not self.config.normalize_advantages or self.config.group_size <= 1:
            return flat_rewards.view(orig_shape)

        mean = flat_rewards.mean(dim=-1, keepdim=True)
        std = flat_rewards.std(dim=-1, keepdim=True)
        # Avoid division by zero when all completions receive identical reward
        normalized = (flat_rewards - mean) / (std + 1e-8)
        return normalized.view(orig_shape)

    def evaluate_rewards(
        self,
        prompts: List[str],
        completions: List[str],
        ground_truths: List[str],
    ) -> torch.Tensor:
        """
        Scores completions across all registered reward verifier functions.
        """
        total_rewards = []
        for prompt, completion, gt in zip(prompts, completions, ground_truths):
            score = 0.0
            for r_fn in self.reward_funcs:
                score += r_fn(completion, gt)
            total_rewards.append(score)
        return torch.tensor(total_rewards, dtype=torch.float32, device=self.device)

    def compute_per_token_log_probs(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes log probabilities for tokens where label != -100.
        Args:
            model: transformer model
            input_ids: [B, SeqLen]
            labels: [B, SeqLen] with -100 for prompt tokens
        Returns:
            per_token_log_probs: [B, SeqLen] (masked positions set to 0.0)
        """
        logits = model(input_ids)
        if hasattr(logits, "logits"):
            logits = logits.logits

        # Shift logits and labels for autoregressive next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        # Gather log prob of target tokens
        clamped_labels = shift_labels.clone()
        mask = (clamped_labels != -100)
        clamped_labels[~mask] = 0

        token_log_probs = torch.gather(log_probs, dim=-1, index=clamped_labels.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * mask.float()
        return token_log_probs

    def compute_kl_divergence(
        self,
        log_probs: torch.Tensor,
        ref_log_probs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unbiased Schulman KL divergence approximation:
        KL = exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        """
        log_ratio = ref_log_probs - log_probs
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        return (kl * mask.float()).sum(dim=-1) / (mask.float().sum(dim=-1) + 1e-8)

    def grpo_loss(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor],
        advantages: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Computes the complete GRPO clipped surrogate objective + KL regularization.
        Args:
            log_probs: [B, SeqLen] current policy log probabilities
            old_log_probs: [B, SeqLen] rollout policy log probabilities
            ref_log_probs: [B, SeqLen] reference policy log probabilities (optional)
            advantages: [B] normalized group advantages
            mask: [B, SeqLen] boolean mask for completion tokens
        """
        # Per-token probability ratio: r_t(theta) = exp(log pi_theta - log pi_old)
        ratio = torch.exp(log_probs - old_log_probs)

        # Broadcast advantages [B, 1] across token positions
        adv = advantages.unsqueeze(-1)

        # Clipped surrogate objective
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - self.config.clip_eps, 1.0 + self.config.clip_eps) * adv
        policy_loss_per_token = -torch.min(surr1, surr2)

        # Average over valid completion tokens per sequence, then average over batch
        seq_lengths = mask.float().sum(dim=-1) + 1e-8
        policy_loss = ((policy_loss_per_token * mask.float()).sum(dim=-1) / seq_lengths).mean()

        # KL Divergence against reference policy
        if ref_log_probs is not None:
            kl_div = self.compute_kl_divergence(log_probs, ref_log_probs, mask).mean()
        else:
            kl_div = torch.tensor(0.0, device=self.device)

        total_loss = policy_loss + self.config.kl_coeff * kl_div

        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_divergence": kl_div.item(),
            "total_loss": total_loss.item(),
            "mean_ratio": ratio[mask].mean().item() if mask.any() else 1.0,
        }
        return total_loss, metrics

    def train_step_batch(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Executes a single optimization step given a batch of rolled-out sequences and rewards.
        Args:
            input_ids: [B, SeqLen] full prompt + completion token ids
            labels: [B, SeqLen] with prompt tokens masked to -100
            rewards: [B] scalar reward per sequence
        """
        self.model.train()
        mask = (labels[:, 1:] != -100)

        # 1. Compute group advantages
        advantages = self.compute_group_advantages(rewards)

        # 2. Get old log-probabilities without grad
        with torch.no_grad():
            old_log_probs = self.compute_per_token_log_probs(self.model, input_ids, labels)
            ref_log_probs = None
            if self.ref_model is not None:
                self.ref_model.eval()
                ref_log_probs = self.compute_per_token_log_probs(self.ref_model, input_ids, labels)

        # 3. Compute current policy log-probabilities with grad
        log_probs = self.compute_per_token_log_probs(self.model, input_ids, labels)

        # 4. Compute GRPO loss
        loss, metrics = self.grpo_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=advantages,
            mask=mask,
        )

        # 5. Backpropagate & step optimizer
        self.optimizer.zero_grad()
        loss.backward()
        if self.config.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
        self.optimizer.step()

        self.step_count += 1
        metrics["step"] = self.step_count
        metrics["mean_reward"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item() if rewards.numel() > 1 else 0.0

        return metrics

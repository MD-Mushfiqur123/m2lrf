"""
M-2LRF Fused KL Divergence Kernel (Liger-Kernel-Inspired)
==========================================================
Computes Kullback-Leibler divergence between policy and reference model log-probabilities
in-place for DPO, PPO, and RLHF alignment.
"""

import torch
import torch.nn.functional as F


def fast_kl_divergence(
    log_probs_p: torch.Tensor,
    log_probs_q: torch.Tensor,
    reduction: str = "batchmean"
) -> torch.Tensor:
    """
    Computes D_KL(P || Q) = sum(P * (log P - log Q)) directly from log-probabilities.
    """
    p = torch.exp(log_probs_p)
    kl = p * (log_probs_p - log_probs_q)
    if reduction == "batchmean":
        return kl.sum() / log_probs_p.shape[0]
    elif reduction == "sum":
        return kl.sum()
    elif reduction == "mean":
        return kl.mean()
    return kl

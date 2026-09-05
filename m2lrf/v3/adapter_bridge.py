# -*- coding: utf-8 -*-
"""
M-2LRF V3: Unified Architecture Bridge & ADR-001 Deduplication Layer.
Unifies SONA, AgentDB Vector Memory, GRPO, and Lifecycle Hooks into a single
high-level coordination interface.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn

from m2lrf.v3.sona import SONAEngine
from m2lrf.v3.agentdb_sync import AgentDBCoordinator
from m2lrf.v3.grpo_trainer import GRPOTrainer, GRPORewardFunction
from m2lrf.v3.hooks import HookManager, HookType


class V3DeepIntegrationBridge:
    """
    Unified V3 Enterprise Bridge consolidating:
      1. SONA 5-mode self-optimizing engine
      2. AgentDB HNSW vector memory coordinator (1536 dims)
      3. DeepSeek-R1 style GRPO trainer
      4. 19 standard lifecycle hooks
      5. M-2LRF 2-bit dual-basis base layer support
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        default_sona_mode: str = "balanced",
        agentdb_dims: int = 1536,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.device = device or torch.device("cpu")

        # 1. SONA Engine
        self.sona = SONAEngine(default_mode=default_sona_mode)

        # 2. AgentDB Vector Coordinator
        self.agentdb = AgentDBCoordinator(dimensions=agentdb_dims, device=self.device)

        # 3. Hook Manager (19 Hook Types)
        self.hooks = HookManager()

        # 4. GRPO Trainer (Initialized on demand or with model)
        self.grpo: Optional[GRPOTrainer] = None
        if model is not None:
            self.grpo = GRPOTrainer(model=model, device=self.device)

        # Attach default safety hooks
        self.hooks.register(HookType.NAN_INF_GUARD, HookManager.create_nan_guard())
        self.hooks.register(HookType.GRADIENT_NORM_GUARD, HookManager.create_gradient_norm_guard(max_norm=1.0))

    def set_learning_mode(self, mode: str) -> float:
        """Switches SONA operational mode and notifies hooks."""
        latency_ms = self.sona.set_mode(mode)
        self.hooks.trigger(HookType.PRE_TASK, step=0, mode=mode, latency_ms=latency_ms)
        return latency_ms

    def record_reasoning_pattern(self, pattern_id: str, embedding: torch.Tensor, metadata: Dict[str, Any]):
        """Caches successful reasoning trajectories into AgentDB."""
        self.agentdb.insert(entry_id=pattern_id, embedding=embedding, metadata=metadata)
        self.hooks.trigger(HookType.POST_STEP, step=self.agentdb.size(), pattern_id=pattern_id)

    def search_similar_patterns(self, query_embedding: torch.Tensor, top_k: int = 5):
        """Searches past successful reasoning patterns from AgentDB."""
        return self.agentdb.search(query_embedding=query_embedding, top_k=top_k)

    def get_status_report(self) -> Dict[str, Any]:
        """Returns comprehensive V3 system health and integration telemetry."""
        return {
            "sona_mode": self.sona.current_mode_name,
            "sona_config": {
                "lr": self.sona.current_config.learning_rate,
                "rank": self.sona.current_config.target_rank,
                "hadamard": self.sona.current_config.use_hadamard,
                "lora_pro": self.sona.current_config.use_lora_pro,
            },
            "agentdb_entries": self.agentdb.size(),
            "agentdb_dims": self.agentdb.dimensions,
            "total_registered_hooks": self.hooks.total_registered_hooks,
            "hook_executions_count": len(self.hooks.execution_log),
            "grpo_active": self.grpo is not None,
        }

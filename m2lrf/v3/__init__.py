# -*- coding: utf-8 -*-
"""
M-2LRF V3: Deep Agentic Integration Engine (ADR-001).
Consolidates SONA learning modes, AgentDB vector memory, GRPO reinforcement learning,
and 19 lifecycle hooks into a unified framework.
"""

from m2lrf.v3.sona import SONAEngine, SONAModeConfig
from m2lrf.v3.agentdb_sync import AgentDBCoordinator, AgentDBVectorEntry
from m2lrf.v3.grpo_trainer import GRPOTrainer, GRPORewardFunction
from m2lrf.v3.hooks import HookManager, HookType, HookContext
from m2lrf.v3.adapter_bridge import V3DeepIntegrationBridge

__all__ = [
    "SONAEngine",
    "SONAModeConfig",
    "AgentDBCoordinator",
    "AgentDBVectorEntry",
    "GRPOTrainer",
    "GRPORewardFunction",
    "HookManager",
    "HookType",
    "HookContext",
    "V3DeepIntegrationBridge",
]

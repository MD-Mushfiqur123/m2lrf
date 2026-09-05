# -*- coding: utf-8 -*-
"""
M-2LRF V3: 19 Lifecycle Hooks Automation System.
Implements the 19 standard hook types from agentic-flow@alpha / ADR-001
for safety checking, telemetry recording, memory guards, and training interception.
"""

from typing import Callable, Dict, List, Any, Optional
from enum import Enum
import time
import torch


class HookType(str, Enum):
    # Task lifecycle (4)
    PRE_TASK = "pre-task"
    POST_TASK = "post-task"
    PRE_STEP = "pre-step"
    POST_STEP = "post-step"

    # Neural forward/backward pass (4)
    PRE_FORWARD = "pre-forward"
    POST_FORWARD = "post-forward"
    PRE_BACKWARD = "pre-backward"
    POST_BACKWARD = "post-backward"

    # Quantization & Model Surgery (4)
    PRE_QUANTIZE = "pre-quantize"
    POST_QUANTIZE = "post-quantize"
    PRE_MERGE = "pre-merge"
    POST_MERGE = "post-merge"

    # Persistence & Checkpoints (4)
    PRE_SAVE = "pre-save"
    POST_SAVE = "post-save"
    PRE_LOAD = "pre-load"
    POST_LOAD = "post-load"

    # Safety, Stability & Memory Guards (3)
    MEMORY_SPIKE_GUARD = "memory-spike-guard"
    GRADIENT_NORM_GUARD = "gradient-norm-guard"
    NAN_INF_GUARD = "nan-inf-guard"


class HookContext:
    """Context object passed to hooks during invocation."""

    def __init__(
        self,
        hook_type: HookType,
        step: int = 0,
        data: Optional[Dict[str, Any]] = None,
    ):
        self.hook_type = hook_type
        self.step = step
        self.data = data or {}
        self.timestamp = time.time()


class HookManager:
    """
    Registry and execution manager for the 19 V3 lifecycle hook types.
    """

    def __init__(self):
        self._hooks: Dict[HookType, List[Callable[[HookContext], None]]] = {
            ht: [] for ht in HookType
        }
        self.execution_log = []

    def register(self, hook_type: HookType, callback: Callable[[HookContext], None]):
        """Registers a callback for a specific hook type."""
        self._hooks[hook_type].append(callback)

    def trigger(self, hook_type: HookType, step: int = 0, **kwargs) -> HookContext:
        """
        Triggers all registered callbacks for hook_type.
        """
        ctx = HookContext(hook_type=hook_type, step=step, data=kwargs)
        callbacks = self._hooks.get(hook_type, [])
        for cb in callbacks:
            cb(ctx)

        self.execution_log.append((hook_type.value, step, time.time()))
        return ctx

    def clear(self):
        for ht in HookType:
            self._hooks[ht].clear()
        self.execution_log.clear()

    @property
    def total_registered_hooks(self) -> int:
        return sum(len(cbs) for cbs in self._hooks.values())

    @staticmethod
    def create_nan_guard() -> Callable[[HookContext], None]:
        """Built-in hook callback that checks for NaN / Inf in tensors."""
        def _guard(ctx: HookContext):
            tensor = ctx.data.get("tensor")
            if tensor is not None and isinstance(tensor, torch.Tensor):
                if torch.isnan(tensor).any():
                    raise ValueError(f"[NaN Guard Triggered] at step {ctx.step}: Tensor contains NaNs!")
                if torch.isinf(tensor).any():
                    raise ValueError(f"[Inf Guard Triggered] at step {ctx.step}: Tensor contains Infs!")
        return _guard

    @staticmethod
    def create_gradient_norm_guard(max_norm: float = 1.0) -> Callable[[HookContext], None]:
        """Built-in hook that clips gradient norms during post-backward."""
        def _guard(ctx: HookContext):
            model = ctx.data.get("model")
            if model is not None and isinstance(model, torch.nn.Module):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        return _guard

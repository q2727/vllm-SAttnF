"""
Runtime helper that wires configuration to registered SAttnF components.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from vllm.config.sattnf import SparseFrameworkConfig

from .executor import BlockExecutor
from .hooks import SchedulerHook
from .kv import KVManager
from .mask import MaskStrategy
from .registry import executor_registry, hook_registry, kv_registry, mask_registry


@dataclass
class SparseComponents:
    mask: MaskStrategy | None
    kv_manager: KVManager | None
    executor: BlockExecutor | None
    hooks: list[SchedulerHook]


class SparseFrameworkManager:
    """Creates and owns the configured components."""

    def __init__(self, config: SparseFrameworkConfig | None):
        self.config = config
        self.components: SparseComponents | None = None

    def is_enabled(self) -> bool:
        return bool(self.config and self.config.enabled)

    def initialize(self) -> SparseComponents:
        if not self.is_enabled():
            self.components = SparseComponents(None, None, None, [])
            return self.components

        cfg = self.config or SparseFrameworkConfig()
        mask = (
            mask_registry.create(cfg.mask_strategy, **cfg.mask_strategy_config)
            if cfg.mask_strategy
            else None
        )
        kv_manager = (
            kv_registry.create(cfg.kv_manager, **cfg.kv_manager_config)
            if cfg.kv_manager
            else None
        )
        executor = (
            executor_registry.create(cfg.block_executor, **cfg.block_executor_config)
            if cfg.block_executor
            else None
        )
        hooks: list[SchedulerHook] = []
        for hook_name in cfg.scheduler_hooks:
            hook_cfg = cfg.scheduler_hook_configs.get(hook_name, {})
            hooks.append(hook_registry.create(hook_name, **hook_cfg))

        self.components = SparseComponents(mask, kv_manager, executor, hooks)
        return self.components

    def get_components(self) -> SparseComponents:
        if self.components is None:
            return self.initialize()
        return self.components

    def shutdown(self) -> None:
        if not self.components:
            return
        mask, kv_manager, executor, hooks = (
            self.components.mask,
            self.components.kv_manager,
            self.components.executor,
            self.components.hooks,
        )
        if mask and hasattr(mask, "teardown"):
            mask.teardown()
        if kv_manager and hasattr(kv_manager, "teardown"):
            kv_manager.teardown()  # type: ignore[attr-defined]
        if executor and hasattr(executor, "teardown"):
            executor.teardown()
        for hook in hooks:
            if hasattr(hook, "teardown"):
                hook.teardown()


_ACTIVE_MANAGER: SparseFrameworkManager | None = None
_ACTIVE_CONFIG_ID: int | None = None


def get_manager_for_config(
    config: SparseFrameworkConfig | None,
) -> SparseFrameworkManager | None:
    """
    Return a cached SparseFrameworkManager for the provided config.
    """
    global _ACTIVE_MANAGER, _ACTIVE_CONFIG_ID
    if not config or not config.enabled:
        return None
    cfg_id = id(config)
    if _ACTIVE_MANAGER is None or _ACTIVE_CONFIG_ID != cfg_id:
        _ACTIVE_MANAGER = SparseFrameworkManager(config)
        _ACTIVE_MANAGER.initialize()
        _ACTIVE_CONFIG_ID = cfg_id
    return _ACTIVE_MANAGER

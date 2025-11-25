# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sparse Attention Framework configuration."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic import Field
from pydantic.dataclasses import dataclass

from vllm.logger import init_logger

from .utils import config

logger = init_logger(__name__)


@config
@dataclass
class SparseFrameworkConfig:
    """
    Controls SAttnF integration.  It is disabled by default so existing
    workloads continue to use vanilla dense attention without extra overhead.
    """

    enabled: bool = False
    """Whether the sparse framework should be materialised."""

    mask_strategy: str | None = None
    """Registry name of the MaskStrategy to use."""

    mask_strategy_config: dict[str, Any] = field(default_factory=dict)
    """JSON-serialisable kwargs forwarded to the selected MaskStrategy."""

    kv_manager: str | None = None
    kv_manager_config: dict[str, Any] = field(default_factory=dict)

    block_executor: str | None = None
    block_executor_config: dict[str, Any] = field(default_factory=dict)

    scheduler_hooks: list[str] = Field(default_factory=list)
    scheduler_hook_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    def compute_hash(self) -> str:
        """Participates in VllmConfig hashing so compilation caches remain valid."""
        from vllm.config.utils import hash_factors

        if not self.enabled:
            return hash_factors(["sattnf-disabled"])

        hooks = tuple(sorted(self.scheduler_hooks))
        hook_cfg = {name: self.scheduler_hook_configs.get(name, {}) for name in hooks}
        factors = [
            "sattnf",
            self.mask_strategy,
            self.mask_strategy_config,
            self.kv_manager,
            self.kv_manager_config,
            self.block_executor,
            self.block_executor_config,
            hooks,
            hook_cfg,
        ]
        return hash_factors(factors)

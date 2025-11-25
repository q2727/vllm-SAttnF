"""
Public exports for the Sparse Attention Framework utilities.
"""

from .contexts import (
    BatchContext,
    BlockTableView,
    DecodeMaskRequest,
    KVFetchRequest,
    PrefillMaskRequest,
    RuntimeState,
)
from .executor import BlockExecutionOutput, BlockExecutionRequest, BlockExecutor
from .hooks import SchedulerEvent, SchedulerHook
from .kv import AllocationRecord, KVManager
from .mask import BlockMask, MaskStrategy
from .registry import (
    ComponentNotFoundError,
    executor_registry,
    hook_registry,
    kv_registry,
    mask_registry,
    register_block_executor,
    register_hook,
    register_kv_manager,
    register_mask,
)
from .runtime import SparseComponents, SparseFrameworkManager

# Pre-register built-in integrations.
__all__ = [
    "BatchContext",
    "BlockTableView",
    "DecodeMaskRequest",
    "KVFetchRequest",
    "PrefillMaskRequest",
    "RuntimeState",
    "BlockExecutionOutput",
    "BlockExecutionRequest",
    "BlockExecutor",
    "SchedulerEvent",
    "SchedulerHook",
    "AllocationRecord",
    "KVManager",
    "BlockMask",
    "MaskStrategy",
    "ComponentNotFoundError",
    "executor_registry",
    "hook_registry",
    "kv_registry",
    "mask_registry",
    "register_block_executor",
    "register_hook",
    "register_kv_manager",
    "register_mask",
    "SparseComponents",
    "SparseFrameworkManager",
]

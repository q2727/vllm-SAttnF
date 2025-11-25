"""
Block level executor abstractions for SAttnF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .mask import BlockMask


@dataclass(slots=True)
class BlockExecutionRequest:
    """Context passed to a BlockExecutor implementation."""

    mask: BlockMask
    query: Any
    key: Any
    value: Any
    additional_tensors: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockExecutionOutput:
    """Return value for BlockExecutor.run."""

    output: Any
    aux: Mapping[str, Any] = field(default_factory=dict)


class BlockExecutor:
    """
    Plug-in friendly API around attention kernels.  Implementations can define
    custom Triton/CUDA kernels, CPU fallbacks or hybrid execution paths.
    """

    def configure(self, **kwargs: Any) -> None:
        """Optional hook executed once after instantiation."""

    def run(self, request: BlockExecutionRequest) -> BlockExecutionOutput:
        raise NotImplementedError

    def maybe_execute_attention(
        self,
        *,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
    ):
        """
        Optional shortcut to override the full attention forward path.
        Returns the computed tensor if handled, otherwise ``None`` so the
        caller can fall back to the default backend.
        """
        return None

    def teardown(self) -> None:
        """Optional cleanup hook."""

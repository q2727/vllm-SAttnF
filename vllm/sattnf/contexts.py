"""
Runtime context objects shared by the Sparse Attention Framework (SAttnF).

The real attention stack already maintains rich metadata about requests,
KV-cache blocks and runtime state.  The classes defined here are lightweight
containers designed to standardise the subset of information that a custom
MaskStrategy/KVManager/BlockExecutor is expected to handle.  They avoid hard
dependencies on the current scheduler implementation so that the framework
can evolve without constantly touching algorithm plugins.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Sequence


@dataclass(slots=True)
class HeadMetadata:
    """Describes logical attention head settings used by the current batch."""

    num_q_heads: int
    num_kv_heads: int
    head_size: int
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BlockTableView:
    """Minimal view over the block-table maintained by vLLM's KV cache."""

    block_size: int
    max_context_blocks: int
    kv_group_count: int
    # The concrete block table is backend specific, so we keep it generic.
    table: Any
    # Derived information (e.g. prefix cache hit lengths) may be injected here.
    annotations: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BatchContext:
    """High-level information about the sequences involved in the current step."""

    request_ids: Sequence[str]
    head_meta: HeadMetadata
    block_table: BlockTableView
    runtime_tags: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RuntimeState:
    """
    Lightweight snapshot of executor level state.

    Attributes:
        step: Monotonic counter maintained by the driver (prefill/ decode step).
        stage: Human readable label ("prefill", "decode", "prefill_finalize", ...)
        device: Optional device identifier for the compute stream.
        stream: Optional backend specific stream / queue handle.
        extras: Free-form metadata (profiling handles, allocator stats, etc.).
    """

    step: int
    stage: str
    device: Any | None = None
    stream: Any | None = None
    extras: MutableMapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PrefillMaskRequest:
    """All information a mask strategy receives during prefill."""

    batch_ctx: BatchContext
    runtime_state: RuntimeState
    # Additional knobs such as sink token ranges or chunked prefill hints.
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DecodeMaskRequest:
    """Information passed to the mask strategy for decode/streaming steps."""

    batch_ctx: BatchContext
    runtime_state: RuntimeState
    # Request level offsets or LOOKAHEAD metadata for speculative decoding.
    cursor_positions: Mapping[str, int]
    attributes: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class KVFetchRequest:
    """Describes the KV blocks required by a mask strategy/executor."""

    block_ids: Sequence[int]
    kv_group: int = 0
    hints: Mapping[str, Any] = field(default_factory=dict)

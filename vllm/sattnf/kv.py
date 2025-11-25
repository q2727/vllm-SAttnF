"""
Interfaces for KV cache allocation / lifetime management used by SAttnF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .contexts import KVFetchRequest


@dataclass(slots=True)
class AllocationRecord:
    """Tracks a set of KV blocks associated with a logical request."""

    request_id: str
    block_ids: list[int] = field(default_factory=list)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class KVManager:
    """
    Abstract interface that mediates between SAttnF strategies and the
    underlying KV cache implementations (paged attention, prefix cache, offload).
    """

    def configure(self, **kwargs: Any) -> None:
        """Optional lifecycle hook with config specific keyword arguments."""

    # Allocation / eviction -------------------------------------------------
    def allocate(self, request_id: str, num_blocks: int, **kwargs: Any) -> AllocationRecord:
        raise NotImplementedError

    def evict(self, request_id: str, block_ids: Sequence[int], reason: str | None = None) -> None:
        raise NotImplementedError

    def release(self, request_id: str) -> None:
        """Called when a request finishes."""
        # Most managers just free bookkeeping; override when needed.
        return None

    # Fetch / staging -------------------------------------------------------
    def fetch(self, fetch_request: KVFetchRequest) -> Any:
        """
        Returns a backend specific view (tensor pointer, CPU staging buffer,
        etc.) for the requested blocks.  Implementations decide whether the data
        lives on GPU, CPU or needs on-demand migration.
        """
        raise NotImplementedError

    # Instrumentation -------------------------------------------------------
    def snapshot_metrics(self) -> Mapping[str, Any]:
        """Return diagnostic counters for observability/reporting."""
        return {}

    def teardown(self) -> None:
        """Optional cleanup hook."""
        return None

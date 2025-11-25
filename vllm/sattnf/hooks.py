"""
Scheduler level hooks that allow sparse strategies to observe/control runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .contexts import BatchContext, RuntimeState


@dataclass(slots=True)
class SchedulerEvent:
    """Canonical representation of scheduler callbacks."""

    name: str
    batch_ctx: BatchContext
    runtime_state: RuntimeState
    payload: Mapping[str, Any] | None = None


class SchedulerHook:
    """
    Hooks run before/after scheduler critical sections so strategies can
    prefetch blocks, alter priorities or gather statistics.
    """

    def configure(self, **kwargs: Any) -> None:
        """Optional hook executed after creation."""

    def on_prefill_start(self, event: SchedulerEvent) -> None:
        """Called when the scheduler enters the prefill phase for a batch."""

    def before_decode_step(self, event: SchedulerEvent) -> None:
        """Called prior to launching decode kernels for a batch."""

    def after_decode_step(self, event: SchedulerEvent) -> None:
        """Called when decode kernels finish for the batch."""

    def on_requests_finished(self, request_ids: Sequence[str]) -> None:
        """Called when requests leave the system (cleanup opportunity)."""

    def teardown(self) -> None:
        """Cleanup hook invoked when shutting down the runtime."""

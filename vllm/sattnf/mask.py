"""
Mask selection primitives for the Sparse Attention Framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Mapping, Sequence

from .contexts import DecodeMaskRequest, PrefillMaskRequest


@dataclass(slots=True)
class BlockMask:
    """
    Result returned by a MaskStrategy.

    block_ids captures coarse grained block/page level requests.  Token level
    selections can optionally be captured in sparse_token_indices, allowing the
    runtime to fallback to gather/scatter paths before block aggregation.
    """

    block_ids: Sequence[int]
    priorities: Sequence[float] | None = None
    sparse_token_indices: Sequence[int] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def is_empty(self) -> bool:
        return len(self.block_ids) == 0 and not self.sparse_token_indices


class MaskStrategy:
    """
    Strategy interface for determining which KV blocks should participate in
    attention.

    Implementations may keep internal state (e.g. cache heatmaps) and can
    override configure/teardown hooks for lifecycle management.
    """

    NAME: ClassVar[str] = "base"

    def configure(self, **kwargs: Any) -> None:
        """Optional hook executed after instantiation.  kwargs mirror config."""

    def generate_prefill_mask(self, request: PrefillMaskRequest) -> BlockMask:
        raise NotImplementedError

    def generate_decode_mask(self, request: DecodeMaskRequest) -> BlockMask:
        raise NotImplementedError

    def teardown(self) -> None:
        """Optional cleanup when the runtime shuts down."""

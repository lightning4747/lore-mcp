"""Chronology port interface for constructing temporal context from evidence."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from lore_mcp.domain.chronology import Evidence, TemporalContext


@runtime_checkable
class ChronologyBuilder(Protocol):
    """Port for organizing retrieved evidence into chronological context."""

    def build_context(
        self,
        evidence_list: Sequence[Evidence],
    ) -> TemporalContext:
        """Construct structured TemporalContext from retrieved evidence."""
        ...

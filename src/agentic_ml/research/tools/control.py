"""The ``finish_research`` control tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strands import tool

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


def make_finish_research(ctx: ResearchContext):
    """Build a ``finish_research`` tool bound to ``ctx``."""

    @tool
    def finish_research(reason: str = "") -> str:
        """Signal that the research loop is complete.

        Call this when the iteration budget is spent or no further improvement seems likely.
        """
        return f"Research finished. {reason}".strip()

    return finish_research

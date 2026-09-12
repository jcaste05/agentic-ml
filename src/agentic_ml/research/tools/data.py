"""The ``profile_dataset`` tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strands import tool

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


def make_profile_dataset(ctx: ResearchContext):
    """Build a ``profile_dataset`` tool bound to ``ctx``."""

    @tool
    def profile_dataset() -> str:
        """Return an aggregate description of the dataset (schema, dtypes, missing values).

        Only summary statistics are exposed; the raw rows are never revealed.
        """
        return ctx.profile_text

    return profile_dataset

"""The ``read_leaderboard`` tool."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strands import tool

from agentic_ml.core.engine import read_leaderboard_action

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


def make_read_leaderboard(ctx: ResearchContext):
    """Build a ``read_leaderboard`` tool bound to ``ctx``."""

    @tool
    def read_leaderboard() -> str:
        """Return the current leaderboard as CSV text.

        Columns: ID, Status, one column per tracked metric, runtime and description.
        Use it to decide what to try next and to avoid repeating failed ideas. If you
        only want to inspect a specific trial, use `read_leaderboard_row(trial_id)`.
        """
        return read_leaderboard_action(ctx)

    return read_leaderboard


def make_read_leaderboard_row(ctx: ResearchContext):
    """Build a ``read_leaderboard_row`` tool bound to ``ctx``."""

    @tool
    def read_leaderboard_row(trial_id: str) -> str:
        """Return the leaderboard row for a given trial ID as CSV text.

        Columns: ID, Status, one column per tracked metric, runtime and description.
        Use it to inspect the results of a specific trial.
        """
        return read_leaderboard_action(ctx, trial_id=trial_id)

    return read_leaderboard_row

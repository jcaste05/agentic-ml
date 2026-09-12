"""Strands tools exposed to the research agent.

Each tool is a thin wrapper over an LLM-agnostic action in
:mod:`agentic_ml.core.engine`. Tools are built by factory functions that close over the
:class:`~agentic_ml.core.engine.ResearchContext`, plus any task-specific tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_ml.research.tools.control import make_finish_research
from agentic_ml.research.tools.data import make_profile_dataset
from agentic_ml.research.tools.experiment import make_create_trial, make_read_trial
from agentic_ml.research.tools.leaderboard import make_read_leaderboard, make_read_leaderboard_row

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


def build_tools(ctx: ResearchContext) -> list:
    """Compose the base tools with the task's extra tools."""
    tools = [
        make_profile_dataset(ctx),
        make_create_trial(ctx),
        make_read_trial(ctx),
        make_read_leaderboard(ctx),
        make_read_leaderboard_row(ctx),
        make_finish_research(ctx),
    ]
    tools.extend(ctx.task.extra_tools(ctx))
    return tools


__all__ = [
    "build_tools",
    "make_create_trial",
    "make_finish_research",
    "make_profile_dataset",
    "make_read_leaderboard",
    "make_read_leaderboard_row",
    "make_read_trial",
]

"""Build a Strands agent for a research run (requires the ``research`` extra)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_ml.research.prompts import build_system_prompt
from agentic_ml.research.tools import build_tools

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


def build_research_agent(ctx: ResearchContext, model: object):
    """Create a Strands ``Agent`` wired with the research tools and the composed prompt."""
    from strands import Agent

    system_prompt = build_system_prompt(ctx.task, ctx.profile_text, ctx.config)
    return Agent(model=model, system_prompt=system_prompt, tools=build_tools(ctx))

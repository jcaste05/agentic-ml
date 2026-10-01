"""Build a Strands agent for a research run (requires the ``research`` extra)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from agentic_ml.research.prompts import build_system_prompt
from agentic_ml.research.tools import build_tools

if TYPE_CHECKING:
    from strands.hooks import AfterModelCallEvent
    from strands.session import SnapshotSessionManager

    from agentic_ml.core.engine import ResearchContext

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Strands-specific knobs for the research agent's conversation manager and context offloader.

    Kept separate from :class:`~agentic_ml.core.engine.ResearchConfig`, which must stay usable
    without Strands installed.
    """

    pin_first: int = 1
    """Number of messages at the very start of the conversation permanently protected from
    summarization. Message 0 is always the kickoff prompt; message 1 (only if ``pin_first >= 2``)
    is the agent's first response. The system prompt is sent separately on every model call and
    is never part of this managed history, so it needs no protection."""

    preserve_recent_messages: int = 4
    """Minimum number of most-recent messages the conversation manager always keeps uncompressed."""

    summary_ratio: float = 0.3
    """Fraction of the older, non-pinned messages summarized whenever compression triggers."""

    compression_threshold: float = 0.85
    """Fraction of the model's context window that triggers proactive summarization."""

    offload_max_result_tokens: int = 1_500
    """Tool results larger than this (estimated tokens) are truncated with a retrievable preview."""

    offload_preview_tokens: int = 750
    """Size of the truncated preview kept in place of an offloaded tool result."""


def build_session_manager(ctx: ResearchContext) -> SnapshotSessionManager:
    """Create the session manager that persists conversation state under the run's workspace."""
    from strands.session import SnapshotSessionManager
    from strands.storage import LocalFileStorage

    storage = LocalFileStorage(str(ctx.workspace.session_dir))
    # A single fixed id: one research_dir holds exactly one research session.
    return SnapshotSessionManager(session_id="research", storage=storage)


def _log_token_usage(event: AfterModelCallEvent) -> None:
    """Log real input/output/total tokens spent on this single model call (not cumulative)."""
    if event.stop_response is None:
        return  # the call raised; nothing to report
    usage = event.stop_response.message["metadata"]["usage"]
    logger.info(
        "LLM call tokens: input=%d output=%d total=%d",
        usage.get("inputTokens", 0),
        usage.get("outputTokens", 0),
        usage.get("totalTokens", 0),
    )
    logger.debug(
        f"agent.messages after call ({len(event.agent.messages)} total): {event.agent.messages}"
    )


def build_research_agent(ctx: ResearchContext, model: object, session_manager: object):
    """Create a Strands ``Agent`` wired with the research tools and the composed prompt.

    This reproduces what ``context_manager="auto"`` builds (a ``SummarizingConversationManager``
    with proactive compression plus a ``ContextOffloader`` truncating large tool results with a
    retrievable preview), but tuned via ``ctx.agent_config`` (see :class:`AgentConfig`) — the
    "auto" defaults (``preserve_recent_messages=10``) never let summarization act on the short
    conversations a research run typically has (kickoff + a handful of tool calls; no system
    message is ever part of this managed history), so it always failed with "insufficient
    messages for summarization". ``session_manager`` persists/restores the conversation across
    process restarts; see :func:`build_session_manager`.
    """
    from strands import Agent
    from strands.agent.conversation_manager import SummarizingConversationManager
    from strands.hooks import AfterModelCallEvent
    from strands.vended_plugins.context_offloader import ContextOffloader

    agent_config = ctx.agent_config
    system_prompt = build_system_prompt(ctx.task, ctx.profile_text, ctx.config)
    conversation_manager = SummarizingConversationManager(
        summary_ratio=agent_config.summary_ratio,
        proactive_compression={"compression_threshold": agent_config.compression_threshold},
        preserve_recent_messages=agent_config.preserve_recent_messages,
        pin_first=agent_config.pin_first,
    )
    agent = Agent(
        model=model,
        system_prompt=system_prompt,
        tools=build_tools(ctx),
        conversation_manager=conversation_manager,
        plugins=[
            ContextOffloader(
                max_result_tokens=agent_config.offload_max_result_tokens,
                preview_tokens=agent_config.offload_preview_tokens,
            )
        ],
        session_manager=session_manager,
    )
    agent.hooks.add_callback(AfterModelCallEvent, _log_token_usage)
    return agent

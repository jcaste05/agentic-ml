"""Tests for the research agent wiring."""

from __future__ import annotations

from agentic_ml.core.engine import ResearchConfig, setup_research
from agentic_ml.estimators.regression.tabular import TabularRegressionTask


def _context(tmp_path, regression_data, regression_schema):
    task = TabularRegressionTask()
    config = ResearchConfig(
        metrics=task.default_metrics(),
        primary_metric=task.primary_metric(),
        maximize=task.maximize(),
    )
    return setup_research(task, regression_data, regression_schema, config, str(tmp_path / "run"))


def test_build_tools_returns_expected_tools(tmp_path, regression_data, regression_schema):
    from agentic_ml.research.tools import build_tools

    ctx = _context(tmp_path, regression_data, regression_schema)
    tools = build_tools(ctx)
    assert len(tools) == 6


def test_system_prompt_contains_task_sections(tmp_path, regression_data, regression_schema):
    from agentic_ml.research.prompts import build_system_prompt

    ctx = _context(tmp_path, regression_data, regression_schema)
    prompt = build_system_prompt(ctx.task, ctx.profile_text, ctx.config)
    assert "REGRESSION" in prompt
    assert "model.py" in prompt
    assert "rmse" in prompt


def test_build_research_agent_tunes_conversation_manager(tmp_path, regression_data, regression_schema):
    from strands.agent.conversation_manager import SummarizingConversationManager
    from strands.vended_plugins.context_offloader import ContextOffloader

    from agentic_ml.research.agent import AgentConfig, build_research_agent

    task = TabularRegressionTask()
    config = ResearchConfig(
        metrics=task.default_metrics(),
        primary_metric=task.primary_metric(),
        maximize=task.maximize(),
    )
    agent_config = AgentConfig(preserve_recent_messages=4, pin_first=2)
    ctx = setup_research(
        task, regression_data, regression_schema, config, str(tmp_path / "run"), agent_config
    )

    agent = build_research_agent(ctx, model=None, session_manager=None)

    manager = agent.conversation_manager
    assert isinstance(manager, SummarizingConversationManager)
    assert manager.preserve_recent_messages == 4
    assert manager.pin_first == 2
    assert isinstance(agent._plugin_registry._plugins.get(ContextOffloader.name), ContextOffloader)


def test_agent_config_defaults():
    from agentic_ml.research.agent import AgentConfig

    agent_config = AgentConfig()
    assert agent_config.pin_first == 1
    assert agent_config.preserve_recent_messages == 4
    assert agent_config.summary_ratio == 0.3
    assert agent_config.compression_threshold == 0.85
    assert agent_config.offload_max_result_tokens == 1_500
    assert agent_config.offload_preview_tokens == 750


def test_build_research_agent_uses_default_agent_config(tmp_path, regression_data, regression_schema):
    from strands.vended_plugins.context_offloader import ContextOffloader

    from agentic_ml.research.agent import build_research_agent

    ctx = _context(tmp_path, regression_data, regression_schema)

    agent = build_research_agent(ctx, model=None, session_manager=None)

    manager = agent.conversation_manager
    assert manager.preserve_recent_messages == 4
    assert manager.pin_first == 1
    plugin = agent._plugin_registry._plugins.get(ContextOffloader.name)
    assert plugin._max_result_tokens == 1_500
    assert plugin._preview_tokens == 750

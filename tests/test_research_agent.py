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

"""Tests for the research Workspace, including the relative-path regression."""

from __future__ import annotations

from agentic_ml.core.engine import ResearchConfig, create_trial_action, setup_research
from agentic_ml.core.workspace import Workspace
from agentic_ml.estimators.regression.tabular import TabularRegressionTask
from tests.conftest import VALID_MODEL_PY


def test_root_is_resolved_to_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    workspace = Workspace("relative_run_dir")
    assert workspace.root.is_absolute()
    assert workspace.dataset_path.is_absolute()
    assert workspace.trial_dir("trial_1").is_absolute()


def test_relative_research_dir_survives_subprocess_cwd_change(
    tmp_path, monkeypatch, regression_data, regression_schema
):
    """Regression test: the sandbox subprocess runs in its own temp cwd, so a workspace built
    from a relative path must still resolve dataset/trial paths correctly."""
    monkeypatch.chdir(tmp_path)
    task = TabularRegressionTask()
    config = ResearchConfig(
        metrics=task.default_metrics(),
        primary_metric=task.primary_metric(),
        maximize=task.maximize(),
        n_splits=3,
        timeout=120,
    )
    ctx = setup_research(task, regression_data, regression_schema, config, "relative_run_dir")

    result = create_trial_action(ctx, "linear baseline", VALID_MODEL_PY)

    assert result["status"] == "ok"
    assert result["error"] is None


def test_write_error_persists_failure_reason(tmp_path):
    workspace = Workspace(tmp_path / "run")
    workspace.create()
    workspace.write_trial("trial_1", VALID_MODEL_PY)
    workspace.write_error("trial_1", "boom")
    assert (workspace.trial_dir("trial_1") / "error.txt").read_text() == "boom"

"""End-to-end tests of the research engine actions without any LLM."""

from __future__ import annotations

import numpy as np

from agentic_ml.core.engine import (
    ResearchConfig,
    create_trial_action,
    read_leaderboard_action,
    select_best,
    setup_research,
)
from agentic_ml.core.model import AgenticModel
from agentic_ml.estimators.regression.tabular import TabularRegressionTask
from tests.conftest import RAISING_MODEL_PY, VALID_MODEL_PY


def _config(iterations: int = 10) -> ResearchConfig:
    task = TabularRegressionTask()
    return ResearchConfig(
        metrics=task.default_metrics(),
        primary_metric=task.primary_metric(),
        maximize=task.maximize(),
        n_splits=3,
        timeout=120,
        iterations=iterations,
    )


def test_create_trial_records_and_export_best(tmp_path, regression_data, regression_schema):
    task = TabularRegressionTask()
    ctx = setup_research(task, regression_data, regression_schema, _config(), str(tmp_path / "run"))

    ok = create_trial_action(ctx, "linear baseline", VALID_MODEL_PY)
    failed = create_trial_action(ctx, "broken model", RAISING_MODEL_PY)

    assert ok["status"] == "ok"
    assert ok["trial_id"] == "trial_1"
    assert failed["status"] == "failed"
    assert failed["trial_id"] == "trial_2"

    csv_text = read_leaderboard_action(ctx)
    assert "trial_1" in csv_text
    assert "trial_2" in csv_text

    best = select_best(ctx)
    assert best == "trial_1"

    model = AgenticModel.from_trial(ctx.workspace.trial_dir(best))
    x = regression_data.drop(columns=["price"])
    y = regression_data["price"]
    model.fit(x, y)
    assert np.asarray(model.predict(x)).shape[0] == len(x)


def test_failed_trial_persists_error_file(tmp_path, regression_data, regression_schema):
    task = TabularRegressionTask()
    ctx = setup_research(task, regression_data, regression_schema, _config(), str(tmp_path / "run"))

    failed = create_trial_action(ctx, "broken model", RAISING_MODEL_PY)

    error_file = ctx.workspace.trial_dir(failed["trial_id"]) / "error.txt"
    assert error_file.exists()
    assert "intentional failure" in error_file.read_text()


def test_leaderboard_file_has_expected_columns(tmp_path, regression_data, regression_schema):
    task = TabularRegressionTask()
    ctx = setup_research(task, regression_data, regression_schema, _config(), str(tmp_path / "run"))
    create_trial_action(ctx, "linear baseline", VALID_MODEL_PY)

    frame = ctx.leaderboard.read()
    assert list(frame.columns) == ["ID", "Status", "rmse", "mae", "r2", "runtime", "description"]


def test_create_trial_refuses_once_iteration_budget_is_spent(
    tmp_path, regression_data, regression_schema
):
    task = TabularRegressionTask()
    config = _config(iterations=2)
    ctx = setup_research(task, regression_data, regression_schema, config, str(tmp_path / "run"))

    trial_1 = create_trial_action(ctx, "trial 1", VALID_MODEL_PY)
    last = create_trial_action(ctx, "trial 2", VALID_MODEL_PY)
    rejected = create_trial_action(ctx, "trial 3", VALID_MODEL_PY)

    assert trial_1["status"] == "ok"
    assert not trial_1["error"]
    assert trial_1["iterations_remaining"] == 1
    assert "budget_note" not in trial_1

    # The trial that reaches the budget still runs normally; the reminder goes in a dedicated
    # field, not "error", so the agent doesn't waste a call proposing another one.
    assert last["status"] == "ok"
    assert last["trial_id"] == "trial_2"
    assert not last["error"]
    assert last["iterations_remaining"] == 0
    assert "finish_research" in last["budget_note"]

    assert rejected["status"] == "budget_exhausted"
    assert rejected["trial_id"] is None
    assert rejected["iterations_remaining"] == 0
    assert "finish_research" in rejected["budget_note"]
    assert len(ctx.history) == 2
    assert not ctx.workspace.trial_dir("trial_3").exists()
    assert "trial_3" not in ctx.leaderboard.read()["ID"].tolist()

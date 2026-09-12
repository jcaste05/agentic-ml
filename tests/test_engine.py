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


def _config() -> ResearchConfig:
    task = TabularRegressionTask()
    return ResearchConfig(
        metrics=task.default_metrics(),
        primary_metric=task.primary_metric(),
        maximize=task.maximize(),
        n_splits=3,
        timeout=120,
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

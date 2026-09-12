"""Tests for the Task abstraction and the tabular regression task."""

from __future__ import annotations

from agentic_ml.estimators.regression.tabular import (
    TabularRegressionResearcher,
    TabularRegressionTask,
)


def test_task_metadata():
    task = TabularRegressionTask()
    assert task.name == "tabular_regression"
    assert task.primary_metric() == "rmse"
    assert task.maximize() is False
    assert "rmse" in task.default_metrics()


def test_build_partitions_default_kfold(regression_data, regression_schema):
    task = TabularRegressionTask()
    x = regression_data[regression_schema.feature_names(regression_data)]
    y = regression_data[regression_schema.target]
    partitions = task.build_partitions(x, y, None, n_splits=4, seed=0)
    assert len(partitions) == 4
    all_val = sorted(idx for _, val in partitions for idx in val)
    assert all_val == list(range(len(x)))


def test_build_partitions_uses_user_partitions(regression_data, regression_schema):
    task = TabularRegressionTask()
    x = regression_data[regression_schema.feature_names(regression_data)]
    y = regression_data[regression_schema.target]
    user = [([0, 1, 2], [3, 4])]
    partitions = task.build_partitions(x, y, user, n_splits=5, seed=0)
    assert partitions == [([0, 1, 2], [3, 4])]


def test_researcher_requires_model(regression_data, regression_schema):
    researcher = TabularRegressionResearcher()
    try:
        researcher.research(regression_data, regression_schema)
    except ValueError as exc:
        assert "Strands model" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError when no model is provided")

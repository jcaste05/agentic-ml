"""Tests for the sandboxed evaluation runner (spawns a subprocess)."""

from __future__ import annotations

from sklearn.model_selection import KFold

from agentic_ml.core.evaluation import evaluate_trial
from agentic_ml.core.workspace import Workspace
from tests.conftest import RAISING_MODEL_PY, ROUNDTRIP_FAIL_MODEL_PY, VALID_MODEL_PY


def _prepare(tmp_path, model_py, regression_data, regression_schema):
    workspace = Workspace(tmp_path / "run")
    workspace.create()
    x = regression_data[regression_schema.feature_names(regression_data)]
    y = regression_data[regression_schema.target]
    workspace.save_dataset(x, y)
    workspace.write_trial("trial_1", model_py)
    splitter = KFold(n_splits=3, shuffle=True, random_state=0)
    partitions = [(tr.tolist(), va.tolist()) for tr, va in splitter.split(x)]
    return workspace, partitions


def test_valid_trial_scores_ok(tmp_path, regression_data, regression_schema):
    workspace, partitions = _prepare(tmp_path, VALID_MODEL_PY, regression_data, regression_schema)
    result = evaluate_trial(
        workspace.trial_dir("trial_1"),
        workspace.dataset_path,
        metrics=["rmse", "r2"],
        partitions=partitions,
        tolerance=1e-6,
        timeout=120,
    )
    assert result.status == "ok"
    assert result.roundtrip_ok is True
    assert result.runtime is not None
    assert result.metrics["rmse"] > 0
    assert result.metrics["r2"] <= 1.0
    assert len(result.per_fold) == 3


def test_roundtrip_failure_is_failed(tmp_path, regression_data, regression_schema):
    workspace, partitions = _prepare(
        tmp_path, ROUNDTRIP_FAIL_MODEL_PY, regression_data, regression_schema
    )
    result = evaluate_trial(
        workspace.trial_dir("trial_1"),
        workspace.dataset_path,
        metrics=["rmse"],
        partitions=partitions,
        tolerance=1e-6,
        timeout=120,
    )
    assert result.status == "failed"
    assert result.roundtrip_ok is False
    assert result.metrics["rmse"] is None
    assert result.runtime is None


def test_raising_trial_is_failed_with_error(tmp_path, regression_data, regression_schema):
    workspace, partitions = _prepare(tmp_path, RAISING_MODEL_PY, regression_data, regression_schema)
    result = evaluate_trial(
        workspace.trial_dir("trial_1"),
        workspace.dataset_path,
        metrics=["rmse"],
        partitions=partitions,
        tolerance=1e-6,
        timeout=120,
    )
    assert result.status == "failed"
    assert result.error
    assert "intentional failure" in result.error

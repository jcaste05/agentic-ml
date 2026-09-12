"""Tests for the light AgenticModel production wrapper."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_ml.core.model import AgenticModel, NotFittedError
from tests.conftest import VALID_MODEL_PY


def _write_trial(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "model.py").write_text(VALID_MODEL_PY)
    return directory


def test_fit_predict_save_load_roundtrip(tmp_path, regression_data, regression_schema):
    trial_dir = _write_trial(tmp_path / "trial_1")

    x = regression_data.drop(columns=["price"])
    y = regression_data["price"]

    model = AgenticModel.from_trial(trial_dir)
    assert not model.is_fitted
    model.fit(x, y)
    assert model.is_fitted
    predictions = np.asarray(model.predict(x))

    save_dir = tmp_path / "artifact"
    model.save(save_dir)

    reloaded = AgenticModel.load(save_dir)
    reloaded_predictions = np.asarray(reloaded.predict(x))
    assert np.allclose(predictions, reloaded_predictions)


def test_predict_before_fit_raises(tmp_path):
    trial_dir = tmp_path / "trial_1"
    trial_dir.mkdir()
    (trial_dir / "model.py").write_text(VALID_MODEL_PY)
    model = AgenticModel.from_trial(trial_dir)
    with pytest.raises(NotFittedError):
        model.predict([[1, 2]])


def test_save_before_fit_raises(tmp_path):
    trial_dir = tmp_path / "trial_1"
    trial_dir.mkdir()
    (trial_dir / "model.py").write_text(VALID_MODEL_PY)
    model = AgenticModel.from_trial(trial_dir)
    with pytest.raises(NotFittedError):
        model.save(tmp_path / "artifact")

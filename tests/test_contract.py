"""Tests for the trial contract loader."""

from __future__ import annotations

import pytest

from agentic_ml.core.contract import ContractError, load_model_class, validate_model_class
from tests.conftest import VALID_MODEL_PY


def test_load_valid_model_class(tmp_path):
    (tmp_path / "model.py").write_text(VALID_MODEL_PY)
    cls = load_model_class(tmp_path)
    assert cls.__name__ == "Model"
    for method in ("fit", "predict", "save", "load"):
        assert callable(getattr(cls, method))


def test_missing_model_py_raises(tmp_path):
    with pytest.raises(ContractError):
        load_model_class(tmp_path)


def test_model_without_required_method_raises(tmp_path):
    (tmp_path / "model.py").write_text(
        "class Model:\n    def fit(self, X, y):\n        return self\n"
    )
    with pytest.raises(ContractError):
        load_model_class(tmp_path)


def test_model_can_import_helpers(tmp_path):
    (tmp_path / "helpers.py").write_text("BASELINE = 3.0\n")
    (tmp_path / "model.py").write_text(
        "import helpers\n"
        "class Model:\n"
        "    def fit(self, X, y):\n"
        "        return self\n"
        "    def predict(self, X):\n"
        "        return [helpers.BASELINE] * len(X)\n"
        "    def save(self, path):\n"
        "        pass\n"
        "    @classmethod\n"
        "    def load(cls, path):\n"
        "        return cls()\n"
    )
    cls = load_model_class(tmp_path)
    validate_model_class(cls)
    assert cls().predict([0, 0]) == [3.0, 3.0]

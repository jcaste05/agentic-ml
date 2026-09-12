"""The contract every agent-written ``model.py`` must satisfy.

A trial is a directory containing a ``model.py`` module that defines a class named ``Model``.
This module describes that interface, documents it for the agent prompt, and provides helpers
to dynamically import and validate a generated module.
"""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Protocol, runtime_checkable

MODEL_CLASS_NAME = "Model"
CONTRACT_VERSION = 1

#: Human-readable description of the contract, injected into the agent system prompt.
CONTRACT_DOC = """\
Each trial you create is a folder that MUST contain a file `model.py` defining a class named
`Model`. It MAY also contain a `helpers.py` file with supporting code that `model.py` can
import as `import helpers` (or `from helpers import ...`).

The `Model` class MUST implement exactly this interface:

    class Model:
        def __init__(self):
            # No required arguments. Configure hyper-parameters here.
            ...

        def fit(self, X, y):
            # X is a pandas DataFrame of features, y is a pandas Series (the target).
            # Train the whole pipeline: preprocessing, model and any post-processing.
            # Return self.
            ...

        def predict(self, X):
            # X is a pandas DataFrame with the same feature columns as in fit.
            # Return a 1D array-like of predictions.
            ...

        def save(self, path):
            # `path` is a directory this model fully owns. Persist everything needed to
            # restore a fitted model there (e.g. with joblib).
            ...

        @classmethod
        def load(cls, path):
            # Rebuild a fitted Model from the directory previously passed to save().
            # Return the restored Model instance.
            ...

Rules:
- Only `numpy`, `pandas` and `scikit-learn` are available. Do not use other third-party
  libraries and do not install packages.
- The code must be self-contained and reproducible: seed any randomness.
- `predict` after a save/load roundtrip MUST return the same values as before saving.
"""


@runtime_checkable
class TrialModel(Protocol):
    """Structural type describing a fitted trial model."""

    def fit(self, X: object, y: object) -> TrialModel: ...

    def predict(self, X: object) -> object: ...

    def save(self, path: str) -> None: ...

    @classmethod
    def load(cls, path: str) -> TrialModel: ...


class ContractError(Exception):
    """Raised when a generated ``model.py`` does not satisfy the contract."""


def validate_model_class(cls: type) -> None:
    """Validate that ``cls`` exposes the required methods."""
    required = ["fit", "predict", "save", "load"]
    missing = [name for name in required if not callable(getattr(cls, name, None))]
    if missing:
        raise ContractError(
            f"class '{MODEL_CLASS_NAME}' is missing required method(s): {', '.join(missing)}"
        )


def load_model_class(directory: str | Path) -> type:
    """Import ``directory/model.py`` and return its validated ``Model`` class.

    ``helpers.py`` (if present) is loaded first under the name ``helpers`` so that
    ``model.py`` can import it. This executes the generated code and is a trust boundary.
    """
    directory = Path(directory)
    model_file = directory / "model.py"
    if not model_file.exists():
        raise ContractError(f"no model.py found in {directory}")

    added_to_path = False
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))
        added_to_path = True
    try:
        helpers_file = directory / "helpers.py"
        if helpers_file.exists():
            helpers_spec = importlib.util.spec_from_file_location("helpers", helpers_file)
            assert helpers_spec is not None and helpers_spec.loader is not None
            helpers_module = importlib.util.module_from_spec(helpers_spec)
            sys.modules["helpers"] = helpers_module
            helpers_spec.loader.exec_module(helpers_module)

        module_name = f"agentic_ml_trial_{uuid.uuid4().hex}"
        spec = importlib.util.spec_from_file_location(module_name, model_file)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if added_to_path:
            try:
                sys.path.remove(str(directory))
            except ValueError:
                pass

    cls = getattr(module, MODEL_CLASS_NAME, None)
    if cls is None:
        raise ContractError(f"model.py must define a class named '{MODEL_CLASS_NAME}'")
    validate_model_class(cls)
    return cls

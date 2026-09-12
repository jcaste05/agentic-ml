"""The light, production-ready model exported from a research run.

:class:`AgenticModel` is intentionally free of any LLM / Strands dependency. It binds to the
code of a single trial (its ``model.py`` and optional ``helpers.py``) and delegates
``fit``/``predict``/``save``/``load`` to that trial's ``Model`` class.

Loading an :class:`AgenticModel` executes the trial's code. Treat saved artifacts with the
same trust as a pickle.
"""

from __future__ import annotations

import json
from pathlib import Path

from agentic_ml.core.contract import CONTRACT_VERSION, load_model_class

_MODEL_FILE = "model.py"
_HELPERS_FILE = "helpers.py"
_STATE_DIR = "state"
_MANIFEST_FILE = "manifest.json"


class NotFittedError(Exception):
    """Raised when ``predict`` or ``save`` is called before ``fit``."""


class AgenticModel:
    """Wraps a trial's ``Model`` implementation as a reusable estimator."""

    def __init__(
        self,
        model_py: str,
        helpers_py: str,
        model_cls: type,
        inner: object | None = None,
    ) -> None:
        self._model_py = model_py
        self._helpers_py = helpers_py
        self._model_cls = model_cls
        self._inner = inner

    @classmethod
    def from_trial(cls, trial_dir: str | Path) -> AgenticModel:
        """Build an (unfitted) model bound to the code in ``trial_dir``."""
        trial_dir = Path(trial_dir)
        model_py = (trial_dir / _MODEL_FILE).read_text()
        helpers_file = trial_dir / _HELPERS_FILE
        helpers_py = helpers_file.read_text() if helpers_file.exists() else ""
        model_cls = load_model_class(trial_dir)
        return cls(model_py, helpers_py, model_cls)

    @property
    def is_fitted(self) -> bool:
        return self._inner is not None

    def fit(self, X: object, y: object) -> AgenticModel:
        """Fit a fresh instance of the underlying trial model."""
        inner = self._model_cls()
        inner.fit(X, y)
        self._inner = inner
        return self

    def predict(self, X: object) -> object:
        if self._inner is None:
            raise NotFittedError("call fit(...) before predict(...)")
        return self._inner.predict(X)

    def save(self, path: str | Path) -> None:
        """Persist the trial code plus the fitted state to ``path``."""
        if self._inner is None:
            raise NotFittedError("call fit(...) before save(...)")
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        (path / _MODEL_FILE).write_text(self._model_py)
        if self._helpers_py.strip():
            (path / _HELPERS_FILE).write_text(self._helpers_py)
        state_dir = path / _STATE_DIR
        state_dir.mkdir(exist_ok=True)
        self._inner.save(str(state_dir))
        manifest = {
            "contract_version": CONTRACT_VERSION,
            "has_helpers": bool(self._helpers_py.strip()),
        }
        (path / _MANIFEST_FILE).write_text(json.dumps(manifest))

    @classmethod
    def load(cls, path: str | Path) -> AgenticModel:
        """Restore a fitted model previously written by :meth:`save`."""
        path = Path(path)
        model_cls = load_model_class(path)
        inner = model_cls.load(str(path / _STATE_DIR))
        model_py = (path / _MODEL_FILE).read_text()
        helpers_file = path / _HELPERS_FILE
        helpers_py = helpers_file.read_text() if helpers_file.exists() else ""
        return cls(model_py, helpers_py, model_cls, inner=inner)

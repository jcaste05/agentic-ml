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
    """Wraps a trial's ``Model`` implementation as a reusable, scikit-learn-like estimator.

    An :class:`AgenticModel` holds no model logic itself: it stores the trial's source code
    (``model.py`` / ``helpers.py``) and the ``Model`` class loaded from it, and forwards
    ``fit``/``predict``/``save``/``load`` to an instance of that class. This is what lets a
    research run export a plain, LLM-free object that still runs the exact code the agent
    wrote and evaluated.

    Attributes:
        is_fitted: Whether :meth:`fit` (or :meth:`load`) has produced a usable inner model.
    """

    def __init__(
        self,
        model_py: str,
        helpers_py: str,
        model_cls: type,
        inner: object | None = None,
    ) -> None:
        """Build a model wrapper. Prefer :meth:`from_trial` or :meth:`load` over this directly.

        Args:
            model_py: Source code of the trial's ``model.py``, kept so :meth:`save` can
                persist it alongside the fitted state.
            helpers_py: Source code of the trial's optional ``helpers.py`` (empty string if
                the trial has none).
            model_cls: The ``Model`` class loaded from ``model_py`` (see
                :func:`~agentic_ml.core.contract.load_model_class`), implementing the
                fit/predict/save/load contract checked in :mod:`agentic_ml.core.contract`.
            inner: An already-fitted instance of ``model_cls``, if any (set by :meth:`load`);
                ``None`` for a freshly built, unfitted model.
        """
        self._model_py = model_py
        self._helpers_py = helpers_py
        self._model_cls = model_cls
        self._inner = inner

    @classmethod
    def from_trial(cls, trial_dir: str | Path) -> AgenticModel:
        """Build an (unfitted) model bound to the code in ``trial_dir``.

        Reads ``model.py`` (and ``helpers.py`` if present) from ``trial_dir`` and loads the
        ``Model`` class from them, without fitting or restoring any state. Typically called
        via ``Researcher.export_model`` right after research finishes.

        Args:
            trial_dir: Directory containing a trial's ``model.py`` (and optional
                ``helpers.py``), as produced by :class:`~agentic_ml.core.workspace.Workspace`.

        Returns:
            A new, unfitted :class:`AgenticModel` bound to that trial's code.
        """
        trial_dir = Path(trial_dir)
        model_py = (trial_dir / _MODEL_FILE).read_text()
        helpers_file = trial_dir / _HELPERS_FILE
        helpers_py = helpers_file.read_text() if helpers_file.exists() else ""
        model_cls = load_model_class(trial_dir)
        return cls(model_py, helpers_py, model_cls)

    @property
    def is_fitted(self) -> bool:
        """Whether the model currently has a fitted (or loaded) inner instance."""
        return self._inner is not None

    def fit(self, X: object, y: object) -> AgenticModel:
        """Fit a fresh instance of the underlying trial model.

        Instantiates ``model_cls`` and calls its ``fit(X, y)``, replacing any previously
        fitted state. Safe to call more than once; each call starts from a fresh instance.

        Args:
            X: Feature data in whatever shape the trial's ``Model.fit`` expects (typically a
                ``pandas.DataFrame`` of raw feature columns).
            y: Target values aligned with ``X``.

        Returns:
            ``self``, now fitted, so calls can be chained (``AgenticModel(...).fit(X, y)``).
        """
        inner = self._model_cls()
        inner.fit(X, y)
        self._inner = inner
        return self

    def predict(self, X: object) -> object:
        """Predict on new data using the fitted inner model.

        Args:
            X: Feature data in the same shape accepted by :meth:`fit`.

        Returns:
            The trial's raw prediction output (typically a ``numpy`` array or ``pandas``
            ``Series``).

        Raises:
            NotFittedError: If called before :meth:`fit` or :meth:`load`.
        """
        if self._inner is None:
            raise NotFittedError("call fit(...) before predict(...)")
        return self._inner.predict(X)

    def save(self, path: str | Path) -> None:
        """Persist the trial code plus the fitted state to ``path``.

        Writes ``model.py`` (and ``helpers.py`` if non-empty), delegates saving the fitted
        state to the inner model's own ``save`` under a ``state/`` subdirectory, and writes a
        small ``manifest.json`` recording the contract version. The result is self-contained:
        :meth:`load` needs only this directory to reconstruct the model.

        Args:
            path: Directory to write to; created (with parents) if it does not exist.

        Raises:
            NotFittedError: If called before :meth:`fit`.
        """
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
        """Restore a fitted model previously written by :meth:`save`.

        Loads the ``Model`` class from the saved ``model.py``/``helpers.py``, then calls its
        ``load(state_dir)`` to restore the fitted inner instance. This executes the trial's
        saved code — treat saved artifacts with the same trust as a pickle (see the module
        docstring).

        Args:
            path: Directory previously written by :meth:`save`.

        Returns:
            A fitted :class:`AgenticModel` ready for :meth:`predict`.
        """
        path = Path(path)
        model_cls = load_model_class(path)
        inner = model_cls.load(str(path / _STATE_DIR))
        model_py = (path / _MODEL_FILE).read_text()
        helpers_file = path / _HELPERS_FILE
        helpers_py = helpers_file.read_text() if helpers_file.exists() else ""
        return cls(model_py, helpers_py, model_cls, inner=inner)

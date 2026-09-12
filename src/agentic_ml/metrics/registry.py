"""A small registry of named metrics with their optimization direction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _rmse(y_true: object, y_pred: object) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


@dataclass(frozen=True)
class Metric:
    """A named metric.

    ``greater_is_better`` drives best-trial selection: ``True`` means higher is better.
    """

    name: str
    fn: Callable[[object, object], float]
    greater_is_better: bool


_REGISTRY: dict[str, Metric] = {
    "rmse": Metric("rmse", _rmse, greater_is_better=False),
    "mae": Metric("mae", lambda t, p: float(mean_absolute_error(t, p)), greater_is_better=False),
    "r2": Metric("r2", lambda t, p: float(r2_score(t, p)), greater_is_better=True),
}


def get_metric(name: str) -> Metric:
    """Return the :class:`Metric` registered under ``name``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown metric '{name}'. Available: {', '.join(sorted(_REGISTRY))}"
        ) from None


def list_metrics() -> list[str]:
    """Return the names of all registered metrics."""
    return sorted(_REGISTRY)

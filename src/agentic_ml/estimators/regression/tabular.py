"""Tabular regression: task definition and a convenience researcher.

This is the reference implementation of the :class:`~agentic_ml.core.task.Task` extension
seam. Note how little is task-specific — prompt sections and metrics — while partitioning,
evaluation, tools and orchestration are all inherited.
"""

from __future__ import annotations

from agentic_ml.core.task import Task
from agentic_ml.research.researcher import Researcher


class TabularRegressionTask(Task):
    """A supervised regression problem on tabular data."""

    name = "tabular_regression"

    def role_description(self) -> str:
        return (
            "You are an expert ML engineer solving a supervised REGRESSION problem on tabular "
            "data. The target is a continuous numeric value. Build a full pipeline from raw "
            "features to numeric predictions."
        )

    def guidance(self) -> str:
        return (
            "Start simple (e.g. a linear model or a gradient-boosted tree baseline), then "
            "iterate. Consider: handling missing values, encoding categorical columns, scaling "
            "numeric features, non-linear models, regularization and target transformations. "
            "Always wrap preprocessing and the estimator together so that predict works on raw "
            "feature rows. Seed all randomness for reproducibility."
        )

    def default_metrics(self) -> list[str]:
        return ["rmse", "mae", "r2"]

    def primary_metric(self) -> str:
        return "rmse"


class TabularRegressionResearcher(Researcher):
    """A :class:`~agentic_ml.research.researcher.Researcher` preset for tabular regression."""

    def __init__(self, model: object | None = None) -> None:
        super().__init__(TabularRegressionTask(), model=model)

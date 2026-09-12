"""The :class:`Task` strategy: the single extension point for new ML problems.

A task encapsulates everything problem-specific: the agent prompt sections, the metrics and
their direction, how data partitions are built, and the (otherwise fixed) evaluation scheme.
Adding a new kind of problem means subclassing :class:`Task`; nothing else in ``core`` or
``research`` needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

from agentic_ml.core.contract import CONTRACT_DOC
from agentic_ml.core.evaluation import EvaluationResult, Partition, evaluate_trial
from agentic_ml.metrics.registry import get_metric

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


class Task(ABC):
    """Base class describing one machine-learning problem type."""

    #: Short, stable identifier stored in run metadata.
    name: str = "task"

    # --- prompt sections -------------------------------------------------------------
    @abstractmethod
    def role_description(self) -> str:
        """One or two sentences describing the agent's role for this task."""

    @abstractmethod
    def guidance(self) -> str:
        """Task-specific tips that steer the experimentation loop."""

    def model_contract_doc(self) -> str:
        """The interface the generated ``model.py`` must implement (usually the default)."""
        return CONTRACT_DOC

    # --- metrics ---------------------------------------------------------------------
    @abstractmethod
    def default_metrics(self) -> list[str]:
        """Metrics tracked by default when the user does not specify any."""

    @abstractmethod
    def primary_metric(self) -> str:
        """The metric used to select the best trial."""

    def maximize(self) -> bool:
        """Whether the primary metric is better when larger."""
        return get_metric(self.primary_metric()).greater_is_better

    # --- evaluation ------------------------------------------------------------------
    def build_partitions(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series,
        user_partitions: list[Partition] | None,
        n_splits: int,
        seed: int,
    ) -> list[Partition]:
        """Return the train/validation index splits used to score a trial.

        Defaults to shuffled K-Fold. Override for task-specific schemes (e.g. stratified
        folds for classification, or time-ordered splits for forecasting).
        """
        if user_partitions is not None:
            return [(list(train), list(val)) for train, val in user_partitions]
        from sklearn.model_selection import KFold

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return [(train.tolist(), val.tolist()) for train, val in splitter.split(x_data)]

    def evaluate_trial(self, trial_id: str, ctx: ResearchContext) -> EvaluationResult:
        """Score a trial with the fixed, task-owned evaluation scheme."""
        partitions = self.build_partitions(
            ctx.x_data, ctx.y_data, ctx.config.partitions, ctx.config.n_splits, ctx.config.seed
        )
        return evaluate_trial(
            trial_dir=ctx.workspace.trial_dir(trial_id),
            dataset_path=ctx.workspace.dataset_path,
            metrics=ctx.config.metrics,
            partitions=partitions,
            tolerance=ctx.config.tolerance,
            timeout=ctx.config.timeout,
        )

    # --- extensibility ---------------------------------------------------------------
    def extra_tools(self, ctx: ResearchContext) -> list:
        """Optional task-specific Strands tools, composed with the base tools."""
        return []

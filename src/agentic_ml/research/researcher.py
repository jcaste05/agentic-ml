"""The generic research orchestrator.

:class:`Researcher` ties a :class:`~agentic_ml.core.task.Task` to a Strands model and drives
the experimentation loop, then exports the winning trial as a light
:class:`~agentic_ml.core.model.AgenticModel`.

Strands is imported lazily inside :meth:`research`, so importing this module (and therefore a
concrete researcher such as ``TabularRegressionResearcher``) does not require the ``research``
extra. Only actually running :meth:`research` does.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from agentic_ml.core.engine import (
    Partition,
    ResearchConfig,
    ResearchContext,
    select_best,
    setup_research,
)
from agentic_ml.core.model import AgenticModel
from agentic_ml.core.task import Task
from agentic_ml.data.schema import DatasetSchema

SNAPSHOT_FILENAME = "snapshot.json"

logger = logging.getLogger(__name__)


class Researcher:
    """Run agent-driven research for a given task and export the best model.

    Attributes:
        task: The :class:`~agentic_ml.core.task.Task` defining the problem being solved
            (prompt sections, metrics, partitioning and evaluation scheme).
        model: The Strands model instance driving the agent, or ``None`` if this researcher
            was built without one — in that case :meth:`research` raises a
            :class:`ValueError` as soon as it is called.
        agent: The Strands agent built from ``task``/``model``, created lazily the first time
            :meth:`research` runs; ``None`` before that.
        context: The :class:`~agentic_ml.core.engine.ResearchContext` from the last
            :meth:`research` call; raises :class:`RuntimeError` if accessed before ``research``
            has run.
    """

    def __init__(self, task: Task, model: object | None = None) -> None:
        self.task = task
        self.model = model
        self._ctx: ResearchContext | None = None
        self.agent: object | None = None

    @property
    def context(self) -> ResearchContext:
        """The :class:`~agentic_ml.core.engine.ResearchContext` from the last research run."""
        if self._ctx is None:
            raise RuntimeError("no research has been run yet; call research(...) first")
        return self._ctx

    # ----------- Private Methods -----------

    def _kickoff_research_agent(self, config: ResearchConfig) -> Exception | None:
        from strands.types.exceptions import EventLoopException

        from agentic_ml.research.prompts import build_kickoff_prompt

        try:
            self.agent(build_kickoff_prompt(config))
        except EventLoopException as e:
            # Raised by Strands when the LLM provider call itself fails (e.g. a 500 from
            # the API); returned instead of raised so the session can still be saved.
            return RuntimeError(
                f"the LLM provider API failed with: {e.original_exception!r}. "
                "Check the provider's status or retry."
            )
        return None

    def _save_session(self, research_dir: str) -> str:
        snapshot = self.agent.take_snapshot(preset="session")
        snapshot_path = Path(research_dir) / SNAPSHOT_FILENAME
        snapshot_path.write_text(json.dumps(snapshot.to_dict()))
        return str(snapshot_path)

    def _load_session(self, research_dir: str, new_session: bool) -> str:
        from strands import Snapshot

        from agentic_ml.research.agent import build_research_agent

        self.agent = build_research_agent(self._ctx, self.model)
        snapshot_path = Path(research_dir) / SNAPSHOT_FILENAME
        if not new_session:
            snapshot = Snapshot.from_dict(json.loads(snapshot_path.read_text()))
            self.agent.load_snapshot(snapshot)
            return str(snapshot_path)
        return None

    # ----------- Public Methods -----------

    def research(
        self,
        data: pd.DataFrame,
        schema: DatasetSchema,
        *,
        metrics: list[str] | None = None,
        partitions: list[Partition] | None = None,
        n_splits: int = 5,
        seed: int = 0,
        tolerance: float = 1e-6,
        timeout: float = 120.0,
        iterations: int = 10,
        ideas: str | None = None,
        research_dir: str | None = None,
        sleep: float = 0.0,
        new_session: bool = True,
    ) -> ResearchContext:
        """Drive the research loop and return the resulting context.

        Args:
            data: The tabular dataset, including the target column.
            schema: Variable descriptions and the target column name.
            metrics: Metric names to track; defaults to the task's metrics.
            partitions: Optional explicit (train_idx, val_idx) splits; otherwise built by the
                task (K-Fold by default).
            iterations: Soft budget of experiment iterations offered to the agent.
            ideas: Natural-language ideas the user wants the agent to try.
            research_dir: Where to store trials and the leaderboard. Defaults to a
                timestamped directory under ``./agentic_ml_runs``.
            sleep: How long to wait between iterations, in seconds. This is useful
                to avoid RateLimit errors with the LLM provider.
            new_session: Whether to start a new research session or continue from an
                existing one.
        """
        if self.model is None:
            raise ValueError(
                "a Strands model instance is required to run research; pass one via "
                "Researcher(model=...) (install the 'research' extra)"
            )

        chosen_metrics = metrics or self.task.default_metrics()
        primary = self.task.primary_metric()
        if primary not in chosen_metrics:
            chosen_metrics = [primary, *chosen_metrics]

        config = ResearchConfig(
            metrics=chosen_metrics,
            primary_metric=primary,
            maximize=self.task.maximize(),
            n_splits=n_splits,
            seed=seed,
            tolerance=tolerance,
            timeout=timeout,
            iterations=iterations,
            ideas=ideas,
            partitions=partitions,
            sleep=sleep,
        )

        if research_dir is None:
            stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            research_dir = f"./agentic_ml_runs/{self.task.name}_{stamp}"

        ctx = setup_research(self.task, data, schema, config, research_dir)
        self._ctx = ctx

        snapshot_load_path = self._load_session(research_dir, new_session)
        logger.info("Research session snapshot loaded from: %s", snapshot_load_path)
        error = self._kickoff_research_agent(config)
        snapshot_save_path = self._save_session(research_dir)
        logger.info("Research session snapshot saved at: %s", snapshot_save_path)
        if error is not None:
            raise error

        return ctx

    def export_model(self, trial: str | None = None) -> AgenticModel:
        """Export a trial (the best by default) as a light production model."""
        ctx = self.context
        trial_id = trial or select_best(ctx)
        if trial_id is None:
            raise RuntimeError("no successful trial to export")
        return AgenticModel.from_trial(ctx.workspace.trial_dir(trial_id))

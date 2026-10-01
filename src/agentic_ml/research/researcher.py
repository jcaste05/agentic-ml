"""The generic research orchestrator.

:class:`Researcher` ties a :class:`~agentic_ml.core.task.Task` to a Strands model and drives
the experimentation loop, then exports the winning trial as a light
:class:`~agentic_ml.core.model.AgenticModel`.

Strands is imported lazily inside :meth:`research`, so importing this module (and therefore a
concrete researcher such as ``TabularRegressionResearcher``) does not require the ``research``
extra. Only actually running :meth:`research` does.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from agentic_ml.research.agent import AgentConfig

logger = logging.getLogger(__name__)


def _run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run ``coro`` to completion, even if called from an already-running event loop.

    Notebook kernels (Jupyter/IPython) run their own event loop, so a plain ``asyncio.run()``
    fails with "cannot be called from a running event loop". When that happens, run the
    coroutine on a separate thread with its own loop instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, coro).result()


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
            # the API); returned instead of raised so the caller sees a clearer error.
            return RuntimeError(
                f"the LLM provider API failed with: {e.original_exception!r}. "
                "Check the provider's status or retry."
            )
        return None

    def _load_session(self, new_session: bool) -> None:
        """Ensure ``self.agent`` is always rebuilt fresh against ``self._ctx`` for this call.

        Rebuilding from scratch (rather than reusing ``self.agent`` in-memory, even if one
        already exists on this instance) means the agent's tools always close over the current
        call's ``ResearchContext`` — with its own reset iteration budget — instead of the stale
        context from a previous call.

        - ``new_session=True``: clears the on-disk session cache first, so the new agent starts
          with an empty conversation.
        - ``new_session=False``: leaves the on-disk session cache alone. ``SnapshotSessionManager``
          persists every turn automatically, so the new agent's conversation is restored from
          that snapshot — whether it was written by an agent still alive on this instance or by
          an earlier process that has since restarted.
        """
        from agentic_ml.research.agent import build_research_agent, build_session_manager

        session_manager = build_session_manager(self._ctx)
        if new_session:
            _run_async(session_manager.delete_session())
            logger.info("Cleared previous session cache; starting a new research agent.")
        else:
            logger.info(
                "Restoring research agent session from: %s", self._ctx.workspace.session_dir
            )
        self.agent = build_research_agent(self._ctx, self.model, session_manager)

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
        agent_config: AgentConfig | None = None,
    ) -> ResearchContext:
        """Drive the research loop and return the resulting context.

        Args:
            data: The tabular dataset, including the target column.
            schema: Variable descriptions and the target column name.
            metrics: Metric names to track; defaults to the task's metrics.
            partitions: Optional explicit (train_idx, val_idx) splits; otherwise built by the
                task (K-Fold by default).
            iterations: Hard cap on the number of trials ``create_trial`` will actually create,
                enforced by ``create_trial_action`` (not just suggested to the agent via the
                kickoff prompt, which also mentions this number so the agent can plan ahead).
            ideas: Natural-language ideas the user wants the agent to try.
            research_dir: Where to store trials and the leaderboard. Defaults to a
                timestamped directory under ``./agentic_ml_runs``.
            sleep: How long to wait between iterations, in seconds. This is useful
                to avoid RateLimit errors with the LLM provider.
            new_session: If ``True``, clears any previous session cache under ``research_dir``
                and starts a brand-new agent. If ``False``, rebuilds the agent from scratch
                against this call's fresh context but restores its conversation from the
                on-disk session cache (whether written by an agent still alive on this instance
                or by an earlier process that has since restarted). Only meaningful when
                calling :meth:`research` again with the *same* ``research_dir`` as a previous
                call.
            agent_config: Tunes the Strands agent's conversation manager and context offloader
                (see :class:`~agentic_ml.research.agent.AgentConfig`); defaults to
                ``AgentConfig()`` (all field defaults) when not passed.
        """
        if self.model is None:
            raise ValueError(
                "a Strands model instance is required to run research; pass one via "
                "Researcher(model=...) (install the 'research' extra)"
            )

        from agentic_ml.research.agent import AgentConfig as _AgentConfig

        agent_config = agent_config or _AgentConfig()

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

        ctx = setup_research(self.task, data, schema, config, research_dir, agent_config)
        self._ctx = ctx

        self._load_session(new_session)
        error = self._kickoff_research_agent(config)
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

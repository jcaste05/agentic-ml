"""Research run configuration, shared context and the LLM-agnostic actions.

The *actions* here (``create_trial_action``, ``read_leaderboard_action``, ``select_best``)
contain all the real logic behind the agent tools, but import no LLM library. This keeps the
whole research pipeline testable without Strands or a live model, and lets the thin tool
wrappers in :mod:`agentic_ml.research.tools` simply forward to them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from agentic_ml.core.leaderboard import Leaderboard
from agentic_ml.core.workspace import Workspace
from agentic_ml.data.schema import DatasetSchema

if TYPE_CHECKING:
    from agentic_ml.core.task import Task

Partition = tuple[list[int], list[int]]


def _sanitize_description(description: str) -> str:
    """Remove newlines and `,` and truncate to 200 characters for the leaderboard."""
    return description.replace("\n", " ").replace(",", " ")[:200]


@dataclass
class ResearchConfig:
    """Everything that parameterizes a research run."""

    metrics: list[str]
    primary_metric: str
    maximize: bool
    n_splits: int = 5
    seed: int = 0
    tolerance: float = 1e-6
    timeout: float = 120.0
    iterations: int = 10
    ideas: str | None = None
    partitions: list[Partition] | None = None
    sleep: float = 0.0


@dataclass
class ResearchContext:
    """Mutable state shared between the researcher, the tools and the actions.

    Built once by :func:`setup_research` at the start of a run, then threaded through every
    tool call (:mod:`agentic_ml.research.tools`) and action in this module. Holding no LLM
    library reference is what keeps the actions unit-testable without Strands.

    Attributes:
        workspace: On-disk layout of the run (trial folders, dataset, leaderboard, run
            metadata); see :class:`~agentic_ml.core.workspace.Workspace`.
        task: The :class:`~agentic_ml.core.task.Task` being solved (partitioning, evaluation
            scheme and prompt sections).
        x_data: Feature columns only (target excluded), split out by :func:`setup_research`
            using ``schema``.
        y_data: The target column, row-aligned with ``x_data``.
        schema: The original :class:`~agentic_ml.data.schema.DatasetSchema`, kept so tools can
            re-describe the data to the agent.
        config: The :class:`ResearchConfig` for this run (metrics, splits, budget, etc.).
        leaderboard: The :class:`~agentic_ml.core.leaderboard.Leaderboard` where every trial's
            result is appended.
        profile_text: The aggregate, LLM-safe description of the dataset shown at kickoff (see
            :meth:`~agentic_ml.data.schema.DatasetSchema.describe_for_prompt`).
        history: Every trial summary recorded so far in this process (id, status, metrics,
            runtime, description). Kept in memory only — unlike ``leaderboard``, it is not
            persisted to disk and is lost if the process restarts.
    """

    workspace: Workspace
    task: Task
    x_data: pd.DataFrame
    y_data: pd.Series
    schema: DatasetSchema
    config: ResearchConfig
    leaderboard: Leaderboard
    profile_text: str = ""
    history: list[dict] = field(default_factory=list)


def setup_research(
    task: Task,
    data: pd.DataFrame,
    schema: DatasetSchema,
    config: ResearchConfig,
    research_dir: str,
) -> ResearchContext:
    """Create the workspace, persist the dataset and build a :class:`ResearchContext`."""
    schema.validate(data)
    x_data = data[schema.feature_names(data)].copy()
    y_data = data[schema.target].copy()

    workspace = Workspace(research_dir)
    workspace.create()
    workspace.save_dataset(x_data, y_data)

    leaderboard = Leaderboard(workspace.leaderboard_path, config.metrics)
    leaderboard.ensure()

    profile_text = schema.describe_for_prompt(data)
    workspace.save_run_metadata(
        {
            "task": task.name,
            "target": schema.target,
            "metrics": config.metrics,
            "primary_metric": config.primary_metric,
            "maximize": config.maximize,
            "n_splits": config.n_splits,
            "seed": config.seed,
            "iterations": config.iterations,
        }
    )

    return ResearchContext(
        workspace=workspace,
        task=task,
        x_data=x_data,
        y_data=y_data,
        schema=schema,
        config=config,
        leaderboard=leaderboard,
        profile_text=profile_text,
    )


def create_trial_action(
    ctx: ResearchContext,
    description: str,
    model_py: str,
    helpers_py: str = "",
) -> dict:
    """Write a new trial, evaluate it with the task's fixed scheme and record the result."""
    trial_id = ctx.workspace.next_trial_id()
    ctx.workspace.write_trial(trial_id, model_py, helpers_py)
    if description:
        ctx.workspace.write_description(trial_id, description)

    result = ctx.task.evaluate_trial(trial_id, ctx)
    sanitized_description = _sanitize_description(description)
    ctx.leaderboard.append(
        trial_id, result.status, result.metrics, result.runtime, sanitized_description
    )
    if result.status != "ok" and result.error:
        ctx.workspace.write_error(trial_id, result.error)

    summary = {
        "trial_id": trial_id,
        "status": result.status,
        "metrics": result.metrics,
        "runtime": result.runtime,
        "roundtrip_ok": result.roundtrip_ok,
        "error": result.error,
    }
    ctx.history.append({**summary, "description": sanitized_description})
    if ctx.config.sleep > 0:
        time.sleep(ctx.config.sleep)
    return summary


def read_leaderboard_action(ctx: ResearchContext, trial_id: str | None = None) -> str:
    """Return the leaderboard as CSV text for the agent to reason about."""
    frame = ctx.leaderboard.read()
    if trial_id is not None:
        frame = frame[frame["ID"] == trial_id]
    return frame.to_csv(index=False)


def select_best(ctx: ResearchContext) -> str | None:
    """Return the best successful trial id by the primary metric, or ``None``."""
    return ctx.leaderboard.best(ctx.config.primary_metric, ctx.config.maximize)

"""System and kickoff prompts for the research agent.

These are plain strings with no LLM dependency, so they can be unit-tested directly.
"""

from __future__ import annotations

from agentic_ml.core.engine import ResearchConfig
from agentic_ml.core.task import Task

_BASE_INSTRUCTIONS = """\
You are an autonomous machine-learning research agent. Your job is to find the best possible
model for the user's problem by proposing, implementing and evaluating concrete experiments
in a loop.

You improve a metric by writing REAL, self-contained Python code. You do not have direct
access to the raw data — you only see the aggregate schema below. Every experiment is scored
for you by a fixed evaluation pipeline that runs your code across data partitions and checks
that it survives a save/load roundtrip.

Workflow for every iteration:
1. Look at the schema and, when useful, call `read_leaderboard` to review what has already
   been tried (description column) and how it scored.
2. Form a hypothesis and implement it as a new trial by calling `create_trial` with a short
   `description` and the `model_py` (and optional `helpers_py`) source code.
3. Read the returned metrics. If a trial failed, inspect the error and fix it.
4. Iterate, trying to beat the best `{primary_metric}` so far. Vary preprocessing, models,
   feature engineering and hyper-parameters.
5. When you have exhausted your iteration budget or cannot improve further, call
   `finish_research`.

Available tools: `profile_dataset`, `read_leaderboard`, `read_leaderboard_row`,
    `create_trial`, `read_trial`, `finish_research`.
"""


def build_system_prompt(task: Task, schema_text: str, config: ResearchConfig) -> str:
    """Compose the full system prompt from the base template and task-specific sections."""
    sections = [
        _BASE_INSTRUCTIONS.format(primary_metric=config.primary_metric),
        f"## Task\n{task.role_description()}",
        f"## Guidance\n{task.guidance()}",
        f"## Model contract\n{task.model_contract_doc()}",
        f"## Dataset schema\n{schema_text}",
        (
            "## Objective\n"
            f"Primary metric: {config.primary_metric} "
            f"({'maximize' if config.maximize else 'minimize'}). "
            f"Also tracked: {', '.join(config.metrics)}."
        ),
    ]
    return "\n\n".join(sections)


def build_kickoff_prompt(config: ResearchConfig) -> str:
    """Build the first user message that starts the loop."""
    lines = [
        f"Run up to {config.iterations} experiment iterations to optimize {config.primary_metric}.",
    ]
    if config.ideas:
        lines.append(f"The user specifically wants you to explore these ideas:\n{config.ideas}")
    lines.append(
        "Start by proposing your first trial. If you have already tried something, "
        "continue from where you left off."
    )
    return "\n\n".join(lines)

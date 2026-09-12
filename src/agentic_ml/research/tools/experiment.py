"""The core ``create_trial`` tool: write and evaluate a new experiment."""

from __future__ import annotations

from typing import TYPE_CHECKING

from strands import tool

from agentic_ml.core.engine import create_trial_action

if TYPE_CHECKING:
    from agentic_ml.core.engine import ResearchContext


def make_create_trial(ctx: ResearchContext):
    """Build a ``create_trial`` tool bound to ``ctx``."""

    @tool
    def create_trial(description: str, model_py: str, helpers_py: str = "") -> dict:
        """Create and evaluate a new trial.

        Args:
            description: A short natural-language summary of the idea being tested.
            model_py: Source of a `model.py` defining a `Model` class per the contract.
            helpers_py: Optional source of a `helpers.py` module imported by `model.py`.

        Returns:
            A dict with the trial id, status ("ok"/"failed"), the metrics, the evaluation
            runtime, whether the save/load roundtrip passed, and any error message.
        """
        return create_trial_action(ctx, description, model_py, helpers_py)

    return create_trial


def make_read_trial(ctx: ResearchContext):
    """Build a ``read_trial`` tool bound to ``ctx``."""

    @tool
    def read_trial(trial_id: str) -> dict:
        """Read the `model.py` and `helpers.py`. This is useful for the agent to refresh
        the code for several scenarios:
            - Read the code of a previous trial to reuse it as a starting point if the
                new proposal is based on it.
            - Refresh the code of a previous trial to fix a bug.
            - Any other reason.
        Args:
            trial_id: The id of the trial to read.

        Returns:
            A dict with keys "model.py" and "helpers.py" containing the source code of
            each file ("helpers.py" is optional).
        """
        return ctx.workspace.read_trial_code(trial_id)

    return read_trial

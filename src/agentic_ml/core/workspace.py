"""On-disk layout of a research run.

A research directory looks like::

    <research_dir>/
        run.json            # run configuration (for reproducibility)
        leaderboard.csv     # metric table, the audit of every trial
        _dataset.joblib     # the research dataset, saved once
        trial_1/
            model.py
            helpers.py      # optional
            description.txt # the agent's natural-language description of the trial
            error.txt       # only present if the trial failed
        trial_2/
            ...
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import pandas as pd

_TRIAL_RE = re.compile(r"^trial_(\d+)$")


class Workspace:
    """Owns paths and file operations for a single research run."""

    def __init__(self, root: str | Path) -> None:
        # Resolved eagerly: trials are evaluated in a subprocess with a different cwd, so a
        # relative root would silently break dataset/trial lookups inside the sandbox.
        self.root = Path(root).resolve()

    def create(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def leaderboard_path(self) -> Path:
        return self.root / "leaderboard.csv"

    @property
    def dataset_path(self) -> Path:
        return self.root / "_dataset.joblib"

    def save_dataset(self, x_data: pd.DataFrame, y_data: pd.Series) -> None:
        """Persist the research dataset once so trials can reload it in the subprocess."""
        joblib.dump((x_data, y_data), self.dataset_path)

    def save_run_metadata(self, metadata: dict) -> None:
        (self.root / "run.json").write_text(json.dumps(metadata, indent=2, default=str))

    def next_trial_id(self) -> str:
        """Return the next unused ``trial_N`` id."""
        numbers = [
            int(match.group(1))
            for child in self.root.glob("trial_*")
            if child.is_dir() and (match := _TRIAL_RE.match(child.name))
        ]
        return f"trial_{max(numbers, default=0) + 1}"

    def trial_dir(self, trial_id: str) -> Path:
        return self.root / trial_id

    def write_trial(self, trial_id: str, model_py: str, helpers_py: str = "") -> Path:
        """Write the agent-provided code for a trial and return its directory."""
        directory = self.trial_dir(trial_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "model.py").write_text(model_py)
        if helpers_py.strip():
            (directory / "helpers.py").write_text(helpers_py)
        return directory

    def write_description(self, trial_id: str, description: str) -> None:
        (self.trial_dir(trial_id) / "description.txt").write_text(description)

    def write_error(self, trial_id: str, error: str) -> None:
        """Persist the failure reason so a failed trial can be debugged after the fact."""
        (self.trial_dir(trial_id) / "error.txt").write_text(error)

    def read_trial_code(self, trial_id: str) -> dict[str, str]:
        directory = self.trial_dir(trial_id)
        code = {"model.py": (directory / "model.py").read_text()}
        helpers = directory / "helpers.py"
        if helpers.exists():
            code["helpers.py"] = helpers.read_text()
        return code

    def copy_trial_artifacts(self, trial_id: str, destination: str | Path) -> Path:
        """Copy a trial's source files into ``destination`` (used when exporting a model)."""
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        for name, content in self.read_trial_code(trial_id).items():
            (destination / name).write_text(content)
        return destination

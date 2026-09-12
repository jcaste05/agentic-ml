"""The CSV leaderboard: the persistent, human-readable memory of every trial's metrics.

Columns are, in order: ``ID``, ``Status``, one column per requested metric, and ``runtime``.
Failed trials store ``NaN`` for every metric and for ``runtime``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMN = "ID"
STATUS_COLUMN = "Status"
RUNTIME_COLUMN = "runtime"
STATUS_OK = "ok"
STATUS_FAILED = "failed"
DESCRIPTION_COLUMN = "description"


class Leaderboard:
    """Read/append access to ``leaderboard.csv`` for a fixed set of metric columns."""

    def __init__(self, path: str | Path, metric_names: list[str]) -> None:
        self.path = Path(path)
        self.metric_names = list(metric_names)
        self.columns = [
            ID_COLUMN,
            STATUS_COLUMN,
            *self.metric_names,
            RUNTIME_COLUMN,
            DESCRIPTION_COLUMN,
        ]

    def ensure(self) -> None:
        """Create the CSV with its header if it does not exist yet."""
        if not self.path.exists():
            pd.DataFrame(columns=self.columns).to_csv(self.path, index=False)

    def append(
        self,
        trial_id: str,
        status: str,
        metrics: dict[str, float | None] | None,
        runtime: float | None,
        description: str = "",
    ) -> None:
        """Append a single trial result as a new row."""
        self.ensure()
        row: dict[str, object] = {
            ID_COLUMN: trial_id,
            STATUS_COLUMN: status,
            RUNTIME_COLUMN: runtime if runtime is not None else np.nan,
            DESCRIPTION_COLUMN: description,
        }
        for name in self.metric_names:
            value = metrics.get(name) if metrics else None
            row[name] = value if value is not None else np.nan

        frame = pd.read_csv(self.path)
        new_row = pd.DataFrame([row], columns=self.columns)
        frame = pd.concat([frame, new_row], ignore_index=True)
        frame.to_csv(self.path, index=False)

    def read(self) -> pd.DataFrame:
        self.ensure()
        return pd.read_csv(self.path)

    def best(self, metric: str, maximize: bool) -> str | None:
        """Return the ``ID`` of the best successful trial by ``metric``, or ``None``."""
        frame = self.read()
        ok = frame[frame[STATUS_COLUMN] == STATUS_OK].dropna(subset=[metric])
        if ok.empty:
            return None
        index = ok[metric].idxmax() if maximize else ok[metric].idxmin()
        return str(ok.loc[index, ID_COLUMN])

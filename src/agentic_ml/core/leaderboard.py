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
    """Read/append access to ``leaderboard.csv`` for a fixed set of metric columns.

    The leaderboard is the durable, human-readable memory of a research run: every call to
    :func:`~agentic_ml.core.engine.create_trial_action` appends one row here, and the agent
    reads it back (via the ``read_leaderboard`` / ``read_leaderboard_row`` tools) to decide
    what to try next. It is a thin wrapper around a CSV file rather than an in-memory
    structure, so it survives process restarts and can be inspected with any spreadsheet tool.

    Attributes:
        path: Filesystem path to ``leaderboard.csv``.
        metric_names: The metric columns tracked for this run, in the order given at
            construction (fixed for the run's lifetime).
        columns: Full column order written to the CSV: ``ID``, ``Status``, each of
            ``metric_names``, ``runtime``, ``description``.
    """

    def __init__(self, path: str | Path, metric_names: list[str]) -> None:
        """Store the leaderboard location and the fixed set of metric columns to track.

        Args:
            path: Where to read/write ``leaderboard.csv``. The file itself is not created
                until :meth:`ensure` or :meth:`append` is called.
            metric_names: Metric columns tracked for this run, in the order used for every
                row; must match the run's configured metrics.
        """
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
        """Create the CSV with its header if it does not exist yet; a no-op otherwise."""
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
        """Append a single trial result as a new row.

        Missing or ``None`` metric values (e.g. for a failed trial) are stored as ``NaN`` so
        the column stays numeric and easy to sort/aggregate.

        Args:
            trial_id: The trial's id (matches its folder name under the workspace).
            status: ``"ok"`` or ``"failed"`` (see the ``STATUS_*`` constants).
            metrics: Metric name to value mapping; entries missing from ``metric_names`` are
                ignored, missing/``None`` entries are stored as ``NaN``.
            runtime: Wall-clock evaluation time in seconds, or ``None`` if unavailable.
            description: Free-text note about the trial; the caller is expected to have
                already sanitized/truncated it.
        """
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
        """Return the whole leaderboard as a DataFrame, creating an empty one if needed."""
        self.ensure()
        return pd.read_csv(self.path)

    def best(self, metric: str, maximize: bool) -> str | None:
        """Return the ``ID`` of the best successful trial by ``metric``, or ``None``.

        Only rows with ``Status == "ok"`` and a non-``NaN`` value for ``metric`` are
        considered.

        Args:
            metric: Column name to rank by (must be one of ``metric_names``).
            maximize: Whether a larger value of ``metric`` is better.

        Returns:
            The winning trial's ``ID``, or ``None`` if no trial qualifies.
        """
        frame = self.read()
        ok = frame[frame[STATUS_COLUMN] == STATUS_OK].dropna(subset=[metric])
        if ok.empty:
            return None
        index = ok[metric].idxmax() if maximize else ok[metric].idxmin()
        return str(ok.loc[index, ID_COLUMN])

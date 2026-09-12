"""Description and aggregate profiling of a tabular dataset.

Only *aggregate* information (dtypes, missing counts, cardinality, target summary) is ever
exposed to the LLM. The raw data never leaves the machine through the prompt; it is only read
by the code executed inside the sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class DatasetSchema:
    """User-provided description of a dataset.

    ``variables`` maps a column name to a natural-language description. ``target`` is the name
    of the column to predict.
    """

    variables: dict[str, str]
    target: str

    def feature_names(self, data: pd.DataFrame) -> list[str]:
        """Return every column of ``data`` except the target."""
        return [column for column in data.columns if column != self.target]

    def validate(self, data: pd.DataFrame) -> None:
        """Check that the target and described variables exist in ``data``."""
        if self.target not in data.columns:
            raise ValueError(f"target column '{self.target}' is not present in the data")
        unknown = set(self.variables) - set(data.columns)
        if unknown:
            raise ValueError(f"described variables not found in the data: {sorted(unknown)}")

    def profile(self, data: pd.DataFrame) -> dict:
        """Compute an aggregate, LLM-safe profile of ``data``."""
        columns = {}
        for column in data.columns:
            series = data[column]
            info: dict[str, object] = {
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "n_unique": int(series.nunique(dropna=True)),
            }
            if pd.api.types.is_numeric_dtype(series):
                described = series.describe()
                info["min"] = float(described.get("min", float("nan")))
                info["mean"] = float(described.get("mean", float("nan")))
                info["max"] = float(described.get("max", float("nan")))
            columns[column] = info
        return {"n_rows": int(len(data)), "target": self.target, "columns": columns}

    def describe_for_prompt(self, data: pd.DataFrame) -> str:
        """Render a compact textual description of the schema and profile for the prompt."""
        profile = self.profile(data)
        lines = [
            f"Rows: {profile['n_rows']}",
            f"Target column: {self.target}",
            "Variables:",
        ]
        for column, info in profile["columns"].items():
            description = self.variables.get(column, "")
            role = "TARGET" if column == self.target else "feature"
            stats = f"dtype={info['dtype']}, missing={info['missing']}, unique={info['n_unique']}"
            if "mean" in info:
                stats += f", min={info['min']:.4g}, mean={info['mean']:.4g}, max={info['max']:.4g}"
            suffix = f" — {description}" if description else ""
            lines.append(f"  - {column} ({role}; {stats}){suffix}")
        return "\n".join(lines)

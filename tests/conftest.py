"""Shared fixtures and canned trial code for the test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from agentic_ml.data.schema import DatasetSchema

# A valid, deterministic trial model: a scaled linear regression pipeline.
VALID_MODEL_PY = """\
from pathlib import Path

import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class Model:
    def __init__(self):
        self.pipeline = Pipeline(
            [("scaler", StandardScaler()), ("model", LinearRegression())]
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def save(self, path):
        joblib.dump(self.pipeline, str(Path(path) / "pipeline.joblib"))

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.pipeline = joblib.load(str(Path(path) / "pipeline.joblib"))
        return obj
"""

# A model whose predictions are non-deterministic, so the save/load roundtrip fails.
ROUNDTRIP_FAIL_MODEL_PY = """\
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression


class Model:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X) + np.random.rand(len(X))

    def save(self, path):
        joblib.dump(self.model, str(Path(path) / "m.joblib"))

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.model = joblib.load(str(Path(path) / "m.joblib"))
        return obj
"""

# A model that raises during fit, producing a failed trial.
RAISING_MODEL_PY = """\
class Model:
    def fit(self, X, y):
        raise ValueError("intentional failure")

    def predict(self, X):
        return X.iloc[:, 0].to_numpy()

    def save(self, path):
        pass

    @classmethod
    def load(cls, path):
        return cls()
"""


@pytest.fixture
def regression_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 120
    area = rng.uniform(40, 200, n)
    rooms = rng.integers(1, 6, n).astype(float)
    price = 1000 * area + 5000 * rooms + rng.normal(0, 1000, n)
    return pd.DataFrame({"area": area, "rooms": rooms, "price": price})


@pytest.fixture
def regression_schema() -> DatasetSchema:
    return DatasetSchema(
        variables={
            "area": "House area in square meters",
            "rooms": "Number of rooms",
            "price": "Sale price (target)",
        },
        target="price",
    )

# agentic-ml

[![Docs](https://img.shields.io/badge/docs-online-blue)](https://jcaste05.github.io/docs/agentic-ml/)

An agentic framework that tackles a data scientist's machine learning problems with LLM
agents. Instead of a fixed AutoML search space, an agent **writes and runs real training
code** in an experimentation loop, keeps an auditable record of every trial, and exports a
lightweight, production-ready model.

> **Status: early prototype.** The first (and currently only) supported task is **tabular
> regression**. Agent-written trials may only use `numpy`, `pandas` and `scikit-learn` as
> ML dependencies — this keeps the evaluation environment predictable, but also means no
> deep learning / gradient-boosting libraries (e.g. XGBoost, LightGBM, PyTorch) are
> available yet. The sandbox that runs agent-written code is intentionally **naive**: see
> [Security note](#security-note) for exactly what it does and does not protect against.

## Table of Contents

- [Installation](#installation)
- [How it works](#how-it-works)
- [Public API](#public-api)
- [Security note](#security-note)
- [Development](#development)
- [License](#license)

## Installation

```bash
uv add agentic-ml              # base install: the light production model only
uv add "agentic-ml[research]"  # adds Strands, the agent runtime that drives research
```

`agentic-ml[research]` installs `strands-agents` itself but **not** a specific LLM provider.
Strands ships each provider as its own extra, so add whichever one matches your setup, e.g.:

```bash
uv add "strands-agents[gemini]"   # Google Gemini
uv add "strands-agents[openai]"   # OpenAI-compatible endpoints
```

See the [Strands Agents docs](https://strandsagents.com/docs/user-guide/quickstart/overview/)
for the full list of supported model providers and how to configure each one.

If you only need `AgenticModel` to serve an already-exported model in production, skip
`[research]` entirely: it depends solely on `numpy`, `pandas`, `scikit-learn` and `joblib`.

## How it works

The full workflow is demonstrated end-to-end in
[`docs/quick_example.ipynb`](docs/quick_example.ipynb) (tabular regression on the
scikit-learn `diabetes` dataset) — run it for a more detailed, runnable walkthrough than
the snippets below.

There are three objects worth knowing about:

- **[`Researcher`](src/agentic_ml/research/researcher.py)** (heavy, needs an LLM): a generic
  orchestrator that runs `research(...)`, driving a Strands agent that proposes, writes and
  evaluates trials. Each trial is a folder `trial_N/` containing the agent-written
  `model.py` (+ optional `helpers.py`). A **fixed, task-owned evaluation scheme** scores
  every trial across data partitions and checks a save/load roundtrip. All metrics are
  appended to a `leaderboard.csv` inside the research directory — that directory *is* the
  audit trail.
- **[`Task`](src/agentic_ml/core/task.py)** (the extension seam): an abstract strategy that
  encapsulates everything problem-specific — the agent prompt sections, the metrics and
  their direction, and how data partitions are built. `Researcher` itself is task-agnostic;
  it is adapted to a concrete ML problem simply by handing it a `Task` subclass. Adding a
  new kind of problem (classification, forecasting, ...) means subclassing `Task`; nothing
  else in `core` or `research` needs to change.
- **[`AgenticModel`](src/agentic_ml/core/model.py)** (light, no LLM dependency): the
  exported production model with `fit`, `predict`, `save` and `load`. It is bound to the
  code of a chosen trial (the best one by default).

Tabular regression, the only task implemented so far, is a thin example of this seam: see
[`TabularRegressionTask` / `TabularRegressionResearcher`](src/agentic_ml/estimators/regression/tabular.py) —
a `Task` subclass plus a `Researcher` preset that binds it, with no other custom logic.

```python
import pandas as pd
from strands.models.openai import OpenAIModel

from agentic_ml.data import DatasetSchema
from agentic_ml.estimators.regression import TabularRegressionResearcher

data = pd.read_csv("houses.csv")
schema = DatasetSchema(
    variables={
        "area": "House area in square meters",
        "rooms": "Number of rooms",
        "price": "Sale price in euros (target)",
    },
    target="price",
)

llm = OpenAIModel(model_id="gpt-4o", client_args={"api_key": "..."})
researcher = TabularRegressionResearcher(model=llm)
researcher.research(data, schema, metrics=["rmse", "mae", "r2"], iterations=10)

model = researcher.export_model()  # best trial by default
model.fit(data.drop(columns=["price"]), data["price"])
model.save("artifacts/price_model")
```

Later, in production (no `[research]` extra required):

```python
from agentic_ml import AgenticModel

model = AgenticModel.load("artifacts/price_model")
predictions = model.predict(new_data)
```

## Public API

### `Researcher` (`agentic_ml.research.researcher.Researcher`)

The generic research orchestrator; `TabularRegressionResearcher` etc. are thin presets
built on top of it.

- **Constructor:** `Researcher(task: Task, model: object | None = None)`
  - `task`: the `Task` strategy driving prompts, metrics and evaluation.
  - `model`: a Strands model instance (any OpenAI-compatible or provider-specific model),
    required before calling `research(...)`.
- **Attributes:** `task`, `model`, `agent` (the underlying Strands `Agent`, created on the
  first `research(...)` call).
- **`context` (property) → `ResearchContext`:** the state of the last research run
  (raises `RuntimeError` if `research(...)` hasn't been called yet). Notable fields:
  `workspace`, `task`, `x_data`, `y_data`, `schema`, `config`, `leaderboard`,
  `profile_text`.
- **`research(data, schema, *, metrics=None, partitions=None, n_splits=5, seed=0, tolerance=1e-6, timeout=120.0, iterations=10, ideas=None, research_dir=None, sleep=0.0, new_session=True) -> ResearchContext`:**
  drives the experimentation loop.
  - `data` / `schema`: the dataset and its `DatasetSchema` description.
  - `metrics` / `partitions` / `n_splits` / `seed` / `tolerance` / `timeout`: evaluation
    knobs, defaulting to the task's own scheme.
  - `iterations`: soft budget of experiment iterations offered to the agent.
  - `ideas`: natural-language guidance the user wants the agent to try.
  - `research_dir`: where trials and the leaderboard are stored; defaults to a timestamped
    directory under `./agentic_ml_runs`.
  - `sleep`: delay between iterations, useful to avoid provider rate limits.
  - `new_session`: start a fresh agent conversation, or resume the one saved in
    `research_dir` (`snapshot.json`).
- **`export_model(trial: str | None = None) -> AgenticModel`:** exports a trial (the best
  by primary metric, by default) as a production-ready `AgenticModel`.

### `AgenticModel` (`agentic_ml.core.model.AgenticModel`)

- **`AgenticModel.from_trial(trial_dir) -> AgenticModel`:** build an unfitted model bound
  to a trial's code.
- **`AgenticModel.load(path) -> AgenticModel`:** restore a fitted model previously written
  by `save(...)`.
- **`is_fitted` (property):** whether `fit(...)` has been called.
- **`fit(X, y) -> AgenticModel`**, **`predict(X)`**, **`save(path)`**: standard
  fit/predict/persist lifecycle, delegating to the bound trial's `Model` class.
- Raises `NotFittedError` if `predict(...)` or `save(...)` is called before `fit(...)`.

### `Task` (`agentic_ml.core.task.Task`)

Abstract base for new ML problem types. Subclasses implement `role_description()`,
`guidance()`, `default_metrics()` and `primary_metric()`; partitioning (K-Fold by default),
evaluation and Strands tool wiring are inherited and generally don't need overriding.

### Task presets (`agentic_ml.estimators.*`)

Each supported task ships a `Task` subclass plus a matching `Researcher` preset with no
extra API surface. Currently: `TabularRegressionTask` / `TabularRegressionResearcher`
(`agentic_ml.estimators.regression.tabular`) for supervised regression on tabular data.

## Security note

Research and `AgenticModel.load` **execute Python code produced by the agent** (or
previously saved by it). Every trial is run out-of-process by a naive sandbox
([`core/sandbox.py`](src/agentic_ml/core/sandbox.py)), which only provides:

- **A hard timeout** on the subprocess (`timeout` in `research(...)`), so a hung trial
  doesn't stall the whole run.
- **Environment variable filtering**: any variable whose name contains `KEY`, `TOKEN`,
  `SECRET`, `PASSWORD`, `PASSWD` or `CREDENTIAL` is stripped before launching the
  subprocess, to reduce accidental secret leakage into agent-written code.
- A prompt-level (not enforced) instruction telling the agent to only use `numpy`,
  `pandas` and `scikit-learn`.

It does **not** provide process isolation, filesystem or network sandboxing, or CPU/memory
limits — the subprocess runs as the same user, with the same filesystem and network access
as your own process. Treat agent-written code and saved artifacts with the same trust
level as a pickle: only run/load artifacts you produced or trust.

## Development

```bash
uv sync --all-extras
uv run ruff check .
uv run ruff format . --check --diff
uv run pytest
```

To preview the documentation locally:

```bash
uv run mkdocs serve
```

## License

Apache-2.0.
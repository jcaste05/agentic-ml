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
- [Links of interest](#links-of-interest)
- [Security note](#security-note)
- [Development](#development)
- [License](#license)

## Installation

Requires **Python 3.11** (this project currently pins `>=3.11,<3.12`).

```bash
uv add jcaste05-agentic-ml              # base install: the light production model only
uv add "jcaste05-agentic-ml[research]"  # adds Strands, the agent runtime that drives research
```

`jcaste05-agentic-ml[research]` installs `strands-agents` itself but **not** a specific LLM provider.
Strands ships each provider as its own extra, so you pick whichever one matches the LLM API you
want the research agent to use, e.g.:

```bash
uv add "strands-agents[gemini]"   # Google Gemini
uv add "strands-agents[openai]"   # OpenAI-compatible endpoints
```

See the [Strands Agents docs](https://strandsagents.com/docs/user-guide/quickstart/overview/)
for the full list of supported model providers and how to configure each one.

If you only need `AgenticModel` to serve an already-exported model in production, skip
`[research]` entirely: it depends solely on `numpy`, `pandas`, `scikit-learn` and `joblib`.

## Links of interest

- [Quick example](https://jcaste05.github.io/docs/agentic-ml/quick_example/) — a runnable
  end-to-end tutorial notebook (tabular regression on the scikit-learn `diabetes` dataset).
- [Documentation](https://jcaste05.github.io/docs/agentic-ml/) — architecture overview and
  full API reference generated from the source code's docstrings.

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
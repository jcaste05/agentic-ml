# agentic-ml

An agentic framework that tackles a data scientist's machine learning problems with LLM
agents. Instead of a fixed AutoML search space, an agent **writes and runs real training
code** in an experimentation loop, keeps an auditable record of every trial, and exports a
lightweight, production-ready model.

> **Status: early prototype.** The first (and currently only) supported task is **tabular
> regression**.
>
> **Important:** due to a name collision on PyPI, this package is published as
> `jcaste05-agentic-ml` (not `agentic-ml`). The import path is unaffected — it's still
> `import agentic_ml`. See the [Quick example](quick_example.ipynb) for install commands.

- [Quick example](quick_example.ipynb) — a runnable end-to-end notebook and installation guide.
- [Architecture](architecture.md) — how the source code is structured and how the research loop interacts with the production model.
- [Reference](reference/agentic_ml) — Library documentation generated from the source code's docstrings.
- [GitHub repository](https://github.com/jcaste05/agentic-ml).

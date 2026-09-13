# agentic-ml

An agentic framework that tackles a data scientist's machine learning problems with LLM
agents. Instead of a fixed AutoML search space, an agent **writes and runs real training
code** in an experimentation loop, keeps an auditable record of every trial, and exports a
lightweight, production-ready model.

> **Status: early prototype.** The first (and currently only) supported task is **tabular
> regression**.

- [Quick example](quick_example.ipynb) — a runnable end-to-end notebook.
- [Architecture](architecture.md) — how the source code is structured and how the research loop interacts with the production model.
- [Reference](reference/agentic_ml) — Library documentation generated from the source code's docstrings.
- [GitHub repository](https://github.com/jcaste05/agentic-ml).

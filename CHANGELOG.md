# Changelog
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## `[0.1.1]` - 2026-09-13
Update README.md
### Changed
- Update installation instructions in README.md to clarify Python version requirement and extras.

- Change `sync` to `lock` in GitHub Actions workflow for dependency management.


## `[0.1.0]` - 2026-09-12
Initial release
### Added

- `src/agentic_ml/core/`: the foundational package.
  - `contract.py`: the `Model` contract every trial must implement (`fit`/`predict`/`save`/`load`).
  - `sandbox.py`: a naive sandboxed subprocess executor with a hard timeout and secret-like
    environment variable filtering.
  - `workspace.py`: a research workspace that resolves paths absolutely so trial evaluation
    survives running in a subprocess with a different cwd.
  - `evaluation.py`: a generic evaluation runner that scores trials across data partitions
    and checks a save/load roundtrip.
  - `leaderboard.py`: a CSV-backed `Leaderboard` for tracking and ranking trials.
  - `task.py`: the `Task` strategy abstraction, the single extension seam for new ML problem
    types.
  - `engine.py`: the research context and trial actions (`create_trial_action`,
    `read_leaderboard_action`, ...) shared by the agent tools.
  - `model.py`: the light, LLM-independent `AgenticModel` production wrapper that loads and
    serves the best exported trial.
- `src/agentic_ml/data/`: dataset description utilities.
  - `schema.py`: `DatasetSchema`, mapping columns to natural-language descriptions and the
    target column; computes an aggregate, LLM-safe profile (dtypes, missing counts,
    cardinality, target summary) and renders it for the prompt — raw data never leaves the
    machine through the prompt.
- `src/agentic_ml/estimators/`: concrete task implementations.
  - `regression/tabular.py`: first task implementation, `TabularRegressionTask` and
    `TabularRegressionResearcher`.
- `src/agentic_ml/metrics/`: scoring utilities.
  - `registry.py`: a small named-metric registry (`rmse`, `mae`, `r2`) with optimization
    direction (`greater_is_better`) used to pick the best trial.
- `src/agentic_ml/research/`: the LLM-driven research loop.
  - `researcher.py`: the generic `Researcher` orchestrator that drives the agent loop,
    persists/resumes the agent's conversation snapshot per research directory, and exports
    the best trial.
  - `agent.py` / `prompts.py`: the Strands-based research agent and its system prompt.
  - `tools/`: agent tools — `profile_dataset` (data.py), `read_leaderboard` and
    `read_leaderboard_row` (leaderboard.py), `create_trial` and `read_trial`
    (experiment.py), and `finish_research` (control.py).
- Failed trials persist their error/traceback to `error.txt` in the trial folder, so a
  failing research run can be debugged after the fact instead of only in the live agent
  output.
- GitHub Actions workflow running `ruff` and `pytest` on pull requests.
- `docs/quick_example.ipynb`: an end-to-end tabular regression walkthrough on the
  scikit-learn `diabetes` dataset.

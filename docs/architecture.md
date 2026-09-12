# Architecture

`agentic-ml` separates *research* (finding a good model with an LLM agent) from the
*production model* (a light object you deploy). It is designed so that new machine-learning
tasks reuse everything except a small, well-defined strategy object.

## Layers

```
core/        LLM-agnostic building blocks (no Strands dependency)
  contract   The interface every agent-written model.py must implement
  model      AgenticModel: the light production wrapper (fit/predict/save/load)
  sandbox    Runs untrusted, generated code out-of-process with a timeout
  workspace  On-disk layout of a research run (trial folders, dataset, run.json)
  evaluation Generic scoring runner executed inside the sandbox
  leaderboard  CSV memory of every trial's metrics
  engine     Research config, shared context and the tool actions
  task       Task: the single extension point for new problems

research/    The agent layer (needs the `research` extra)
  prompts    System and kickoff prompts (plain strings)
  tools/     Thin Strands tools wrapping the engine actions
  agent      Builds the Strands agent
  researcher Researcher: orchestrates the loop and exports the best model

metrics/     Metric registry with optimization direction
data/        DatasetSchema and aggregate profiling

estimators/  Task implementations, organized by family
  regression/tabular.py   TabularRegressionTask + TabularRegressionResearcher
```

## The research loop

1. The agent inspects the schema (`profile_dataset`) and past results (`read_leaderboard` /
   `read_leaderboard_row`).
2. It writes a trial as `trial_N/model.py` (+ optional `helpers.py`) via `create_trial`, and
   can re-read its own code with `read_trial` while debugging a failure.
3. The task's fixed evaluation scheme scores the trial across partitions in a sandboxed
   subprocess and checks a save/load roundtrip.
4. The result is appended to `leaderboard.csv`.
5. Repeat until the iteration budget is exhausted, then `finish_research`.
6. `export_model()` binds an `AgenticModel` to the best trial's code.

`Researcher.research(...)` persists the agent's conversation as `snapshot.json` in the
research directory after every call, and can resume it (`new_session=False`) so a later run
continues the same conversation instead of starting cold.

## Adding a new task

Subclass `Task`, providing the prompt sections, metrics and (only if needed) a custom
partitioning or extra tools. Everything else — sandbox, workspace, evaluation, leaderboard,
base tools, agent and the production model — is reused unchanged.

## Trust boundary

Both research and `AgenticModel.load` execute agent-generated Python. The sandbox
(`core/sandbox.py`) is deliberately naive: it only enforces a hard timeout and strips
secret-like environment variables before launching the subprocess — there is no real
process isolation, filesystem/network sandboxing or resource limiting. Saved artifacts
should be trusted like pickles: only run/load what you produced or trust. Raw data is never
sent to the LLM — only aggregate profiles.

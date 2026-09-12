"""Generic trial evaluation.

This module has two roles:

* **Parent side** (:func:`evaluate_trial`): serialize an evaluation request, run the child in
  an isolated subprocess via :mod:`agentic_ml.core.sandbox`, and parse the result.
* **Child side** (``python -m agentic_ml.core.evaluation <in.json> <out.json>``): the actual,
  task-agnostic scoring loop that imports the trial's ``Model``, runs cross-validated
  ``fit``/``predict`` over the given partitions, computes metrics and checks the save/load
  roundtrip acceptance gate.

The parts that are task-specific (which metrics, how partitions are built, the roundtrip
tolerance) are provided by the caller, so the runner itself is reused by every task.
"""

from __future__ import annotations

import json
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from agentic_ml.core import sandbox

Partition = tuple[list[int], list[int]]


@dataclass
class EvaluationResult:
    """Aggregated outcome of evaluating a single trial."""

    status: str  # "ok" or "failed"
    metrics: dict[str, float | None]
    runtime: float | None
    roundtrip_ok: bool
    error: str | None = None
    per_fold: list[dict[str, float]] = field(default_factory=list)


def evaluate_trial(
    trial_dir: str | Path,
    dataset_path: str | Path,
    metrics: list[str],
    partitions: list[Partition],
    tolerance: float,
    timeout: float,
) -> EvaluationResult:
    """Evaluate the trial in ``trial_dir`` and return an :class:`EvaluationResult`."""
    failed_metrics: dict[str, float | None] = {name: None for name in metrics}
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        in_path = tmp_path / "in.json"
        out_path = tmp_path / "out.json"
        # The child runs with tmp_path as cwd, so paths must be absolute to resolve correctly.
        payload = {
            "trial_dir": str(Path(trial_dir).resolve()),
            "dataset_path": str(Path(dataset_path).resolve()),
            "metrics": list(metrics),
            "partitions": [[list(map(int, tr)), list(map(int, va))] for tr, va in partitions],
            "tolerance": tolerance,
        }
        in_path.write_text(json.dumps(payload))

        result = sandbox.run(
            [sys.executable, "-m", "agentic_ml.core.evaluation", str(in_path), str(out_path)],
            cwd=tmp_path,
            timeout=timeout,
        )

        if result.timed_out:
            return EvaluationResult(
                "failed", failed_metrics, None, False, error="evaluation timed out"
            )
        if not out_path.exists():
            detail = result.stderr.strip()[-2000:] or "no output produced by evaluation subprocess"
            return EvaluationResult("failed", failed_metrics, None, False, error=detail)

        data = json.loads(out_path.read_text())
        return EvaluationResult(
            status=data["status"],
            metrics=data.get("metrics", failed_metrics),
            runtime=data.get("runtime"),
            roundtrip_ok=data.get("roundtrip_ok", False),
            error=data.get("error"),
            per_fold=data.get("per_fold", []),
        )


def _run_child(payload: dict) -> dict:
    """The scoring loop that runs inside the sandboxed subprocess."""
    import time

    import joblib
    import numpy as np

    from agentic_ml.core.contract import load_model_class
    from agentic_ml.metrics.registry import get_metric

    metrics: list[str] = payload["metrics"]
    failed = {name: None for name in metrics}

    x_data, y_data = joblib.load(payload["dataset_path"])
    model_cls = load_model_class(Path(payload["trial_dir"]))
    partitions: list[Partition] = payload["partitions"]
    tolerance: float = payload["tolerance"]

    start = time.perf_counter()
    per_fold: list[dict[str, float]] = []
    roundtrip_ok = True

    for fold, (train_idx, val_idx) in enumerate(partitions):
        x_train, y_train = x_data.iloc[train_idx], y_data.iloc[train_idx]
        x_val, y_val = x_data.iloc[val_idx], y_data.iloc[val_idx]

        model = model_cls()
        model.fit(x_train, y_train)
        pred = np.asarray(model.predict(x_val), dtype=float)
        y_true = np.asarray(y_val, dtype=float)
        per_fold.append({name: float(get_metric(name).fn(y_true, pred)) for name in metrics})

        if fold == 0:
            with tempfile.TemporaryDirectory() as state_tmp:
                state_dir = Path(state_tmp) / "state"
                state_dir.mkdir()
                model.save(str(state_dir))
                reloaded = model_cls.load(str(state_dir))
                pred_reloaded = np.asarray(reloaded.predict(x_val), dtype=float)
                roundtrip_ok = bool(
                    np.allclose(pred, pred_reloaded, rtol=tolerance, atol=tolerance)
                )

    if not roundtrip_ok:
        return {
            "status": "failed",
            "metrics": failed,
            "runtime": None,
            "roundtrip_ok": False,
            "error": "predictions changed after a save/load roundtrip",
            "per_fold": per_fold,
        }

    runtime = time.perf_counter() - start
    aggregated = {
        name: float(np.mean([fold_metrics[name] for fold_metrics in per_fold])) for name in metrics
    }
    return {
        "status": "ok",
        "metrics": aggregated,
        "runtime": runtime,
        "roundtrip_ok": True,
        "per_fold": per_fold,
    }


def _main(argv: list[str]) -> int:
    import traceback

    in_path, out_path = Path(argv[1]), Path(argv[2])
    payload = json.loads(in_path.read_text())
    try:
        result = _run_child(payload)
    except Exception:  # noqa: BLE001 - any failure is a failed trial, reported structurally
        result = {
            "status": "failed",
            "metrics": {name: None for name in payload.get("metrics", [])},
            "runtime": None,
            "roundtrip_ok": False,
            "error": traceback.format_exc(),
            "per_fold": [],
        }
    out_path.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv))

"""Run untrusted, agent-generated code out-of-process.

This module is deliberately ML-agnostic: it only knows how to launch a Python subprocess with
a timeout, a clean-ish environment and captured output. The evaluation logic lives in
:mod:`agentic_ml.core.evaluation`, which is what gets executed here.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

_SECRET_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL")


def _sanitized_env(extra: dict[str, str] | None) -> dict[str, str]:
    """Copy the environment, dropping variables that look like secrets."""
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_HINTS)}
    if extra:
        env.update(extra)
    return env


@dataclass
class ExecResult:
    """Outcome of a subprocess run."""

    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration: float


def run(
    args: list[str],
    cwd: str | Path,
    timeout: float,
    env: dict[str, str] | None = None,
) -> ExecResult:
    """Run ``args`` in ``cwd`` with a hard ``timeout`` (seconds)."""
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            args,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_sanitized_env(env),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ExecResult(
            returncode=None,
            stdout=exc.stdout or "" if isinstance(exc.stdout, str) else "",
            stderr=exc.stderr or "" if isinstance(exc.stderr, str) else "",
            timed_out=True,
            duration=time.perf_counter() - start,
        )
    return ExecResult(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        timed_out=False,
        duration=time.perf_counter() - start,
    )

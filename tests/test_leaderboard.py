"""Tests for the CSV leaderboard."""

from __future__ import annotations

import math

from agentic_ml.core.leaderboard import Leaderboard


def test_columns_order(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.csv", ["rmse", "mae"])
    assert board.columns == ["ID", "Status", "rmse", "mae", "runtime", "description"]


def test_append_ok_and_failed_rows(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.csv", ["rmse", "mae"])
    board.append("trial_1", "ok", {"rmse": 2.0, "mae": 1.5}, runtime=0.3)
    board.append("trial_2", "failed", None, runtime=None)

    frame = board.read()
    assert list(frame["ID"]) == ["trial_1", "trial_2"]
    assert frame.loc[0, "Status"] == "ok"
    assert frame.loc[0, "rmse"] == 2.0
    assert frame.loc[1, "Status"] == "failed"
    assert math.isnan(frame.loc[1, "rmse"])
    assert math.isnan(frame.loc[1, "runtime"])


def test_best_selects_min_for_minimize(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.csv", ["rmse"])
    board.append("trial_1", "ok", {"rmse": 3.0}, runtime=0.1)
    board.append("trial_2", "ok", {"rmse": 1.0}, runtime=0.1)
    board.append("trial_3", "failed", None, runtime=None)
    assert board.best("rmse", maximize=False) == "trial_2"


def test_best_selects_max_for_maximize(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.csv", ["r2"])
    board.append("trial_1", "ok", {"r2": 0.5}, runtime=0.1)
    board.append("trial_2", "ok", {"r2": 0.9}, runtime=0.1)
    assert board.best("r2", maximize=True) == "trial_2"


def test_best_none_when_no_successful_trials(tmp_path):
    board = Leaderboard(tmp_path / "leaderboard.csv", ["rmse"])
    board.append("trial_1", "failed", None, runtime=None)
    assert board.best("rmse", maximize=False) is None

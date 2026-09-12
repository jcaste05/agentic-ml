"""Tests for the metric registry."""

from __future__ import annotations

import numpy as np
import pytest

from agentic_ml.metrics.registry import get_metric, list_metrics


def test_known_metrics_present():
    names = list_metrics()
    assert {"rmse", "mae", "r2"}.issubset(set(names))


def test_metric_directions():
    assert get_metric("rmse").greater_is_better is False
    assert get_metric("mae").greater_is_better is False
    assert get_metric("r2").greater_is_better is True


def test_rmse_value():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 5.0])
    assert get_metric("rmse").fn(y_true, y_pred) == pytest.approx((4 / 3) ** 0.5)


def test_unknown_metric_raises():
    with pytest.raises(KeyError):
        get_metric("does-not-exist")

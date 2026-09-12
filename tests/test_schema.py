"""Tests for the dataset schema."""

from __future__ import annotations

import pytest


def test_validate_accepts_matching_data(regression_data, regression_schema):
    regression_schema.validate(regression_data)


def test_feature_names_excludes_target(regression_data, regression_schema):
    assert regression_schema.feature_names(regression_data) == ["area", "rooms"]


def test_validate_missing_target_raises(regression_data, regression_schema):
    regression_schema.target = "not_a_column"
    with pytest.raises(ValueError):
        regression_schema.validate(regression_data)


def test_profile_is_aggregate(regression_data, regression_schema):
    profile = regression_schema.profile(regression_data)
    assert profile["n_rows"] == len(regression_data)
    assert set(profile["columns"]) == {"area", "rooms", "price"}
    assert "mean" in profile["columns"]["area"]


def test_describe_for_prompt_mentions_target_and_variables(regression_data, regression_schema):
    text = regression_schema.describe_for_prompt(regression_data)
    assert "TARGET" in text
    assert "House area" in text

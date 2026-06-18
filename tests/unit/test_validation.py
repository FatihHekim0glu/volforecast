"""Unit tests for the input-coercion and validation guardrails.

These cover the error and edge paths of :mod:`volforecast._validation`: the
shape and emptiness checks, coercion from plain Python containers and numpy
arrays, the NaN guards, index alignment, and the minimum-observation guard.
Every public compute function funnels its inputs through these helpers, so the
failure modes here are the library's first line of defence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast import (
    align_inner,
    ensure_dataframe,
    ensure_series,
    validate_min_obs,
)
from volforecast._exceptions import InsufficientDataError, ValidationError


@pytest.mark.unit
def test_ensure_series_coerces_list_to_float64() -> None:
    out = ensure_series([1, 2, 3], name="returns")
    assert isinstance(out, pd.Series)
    assert out.dtype == np.float64
    assert out.tolist() == [1.0, 2.0, 3.0]


@pytest.mark.unit
def test_ensure_series_copies_input_series() -> None:
    original = pd.Series([1.0, 2.0, 3.0])
    out = ensure_series(original)
    out.iloc[0] = 99.0
    assert original.iloc[0] == 1.0


@pytest.mark.unit
def test_ensure_series_rejects_2d_ndarray() -> None:
    with pytest.raises(ValidationError, match="1-dimensional"):
        ensure_series(np.zeros((2, 2)), name="returns")


@pytest.mark.unit
def test_ensure_series_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        ensure_series([])


@pytest.mark.unit
def test_ensure_series_allows_nan_when_requested() -> None:
    out = ensure_series([1.0, float("nan"), 3.0], allow_nan=True)
    assert bool(out.isna().any())


@pytest.mark.unit
def test_ensure_dataframe_coerces_mapping() -> None:
    out = ensure_dataframe({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    assert list(out.columns) == ["a", "b"]
    assert out.shape == (2, 2)
    assert (out.dtypes == np.float64).all()


@pytest.mark.unit
def test_ensure_dataframe_applies_columns_to_ndarray() -> None:
    out = ensure_dataframe(np.arange(6.0).reshape(3, 2), columns=["x", "y"])
    assert list(out.columns) == ["x", "y"]
    assert out.shape == (3, 2)


@pytest.mark.unit
def test_ensure_dataframe_rejects_1d_ndarray() -> None:
    with pytest.raises(ValidationError, match="2-dimensional"):
        ensure_dataframe(np.zeros(4))


@pytest.mark.unit
def test_ensure_dataframe_rejects_empty() -> None:
    with pytest.raises(ValidationError, match="at least one row"):
        ensure_dataframe(pd.DataFrame())


@pytest.mark.unit
def test_ensure_dataframe_rejects_nan_by_default() -> None:
    frame = pd.DataFrame({"a": [1.0, float("nan")]})
    with pytest.raises(ValidationError, match="NaN"):
        ensure_dataframe(frame)


@pytest.mark.unit
def test_align_inner_intersects_and_sorts() -> None:
    left = pd.DataFrame({"v": [1.0, 2.0, 3.0]}, index=[3, 1, 2])
    right = pd.DataFrame({"w": [4.0, 5.0]}, index=[2, 1])
    aligned_left, aligned_right = align_inner(left, right)
    assert list(aligned_left.index) == [1, 2]
    assert list(aligned_right.index) == [1, 2]


@pytest.mark.unit
def test_align_inner_rejects_disjoint_index() -> None:
    left = pd.DataFrame({"v": [1.0]}, index=[1])
    right = pd.DataFrame({"w": [2.0]}, index=[9])
    with pytest.raises(ValidationError, match="no common index"):
        align_inner(left, right)


@pytest.mark.unit
def test_validate_min_obs_passes_when_enough_rows() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    validate_min_obs(frame, 3)


@pytest.mark.unit
def test_validate_min_obs_raises_when_too_few_rows() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(InsufficientDataError, match="at least 5"):
        validate_min_obs(frame, 5, name="panel")

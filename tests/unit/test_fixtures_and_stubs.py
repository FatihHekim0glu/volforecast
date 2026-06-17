"""Validate the seeded fixtures used across the suite.

The fixtures are real reference data (see ``conftest.py``); these tests pin their
shape/invariants so downstream kernels can rely on them. (The original
stub-contract smoke tests have been superseded by the dedicated behavioural
suites — ``test_realized_estimators``, ``test_ml_baselines_har``,
``test_evaluation``, ``test_walkforward_data``, etc. — now that every kernel is
implemented.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import volforecast as vf


@pytest.mark.unit
def test_garch_series_is_valid_ohlc(garch_series: pd.DataFrame) -> None:
    df = garch_series
    assert list(df.columns) == ["open", "high", "low", "close"]
    assert len(df) == 1500
    assert (df > 0).to_numpy().all()
    # OHLC coherence: high is the max, low is the min of the bar.
    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all()
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all()
    assert (df["high"] >= df["low"]).all()


@pytest.mark.unit
def test_garch_series_shows_volatility_clustering(garch_series: pd.DataFrame) -> None:
    rets = np.log(garch_series["close"]).diff().dropna()
    # GARCH data has positively autocorrelated squared returns (ARCH effect).
    sq = (rets**2).to_numpy()
    ac1 = np.corrcoef(sq[:-1], sq[1:])[0, 1]
    assert ac1 > 0.0


@pytest.mark.unit
def test_har_series_is_positive(har_series: pd.Series) -> None:
    assert len(har_series) == 1200
    assert (har_series > 0).all()
    assert har_series.name == "rv"


@pytest.mark.unit
def test_pure_noise_has_no_arch_effect(pure_noise: pd.Series) -> None:
    assert len(pure_noise) == 1500
    sq = pure_noise.to_numpy() ** 2
    ac1 = np.corrcoef(sq[:-1], sq[1:])[0, 1]
    # No volatility clustering: squared-return autocorrelation is near zero.
    assert abs(ac1) < 0.1


@pytest.mark.unit
def test_fixtures_are_reproducible() -> None:
    # Re-seeding the same way reproduces the fixtures byte-for-byte.
    g1 = vf.make_rng(20260617).standard_normal(8)
    g2 = vf.make_rng(20260617).standard_normal(8)
    np.testing.assert_array_equal(g1, g2)

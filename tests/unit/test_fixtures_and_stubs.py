"""Validate the seeded fixtures and that every stub honours its NotImplemented contract.

The fixtures are real reference data (see ``conftest.py``); these tests pin their
shape/invariants so downstream kernels can rely on them. The stub-contract tests
document the public surface and will be replaced by behavioural tests as each
kernel is implemented.
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
    sq = (pure_noise.to_numpy() ** 2)
    ac1 = np.corrcoef(sq[:-1], sq[1:])[0, 1]
    # No volatility clustering: squared-return autocorrelation is near zero.
    assert abs(ac1) < 0.1


@pytest.mark.unit
def test_fixtures_are_reproducible() -> None:
    # Re-seeding the same way reproduces the fixtures byte-for-byte.
    g1 = vf.make_rng(20260617).standard_normal(8)
    g2 = vf.make_rng(20260617).standard_normal(8)
    np.testing.assert_array_equal(g1, g2)


# --- Stub-contract smoke tests (replaced by behaviour as kernels land) ------

@pytest.mark.unit
def test_realized_estimators_are_stubbed(garch_series: pd.DataFrame) -> None:
    with pytest.raises(NotImplementedError):
        vf.garman_klass_rv(garch_series)
    with pytest.raises(NotImplementedError):
        vf.parkinson_rv(garch_series)
    with pytest.raises(NotImplementedError):
        vf.close_to_close_rv(garch_series["close"])


@pytest.mark.unit
def test_forward_target_is_stubbed(har_series: pd.Series) -> None:
    with pytest.raises(NotImplementedError):
        vf.forward_rv_target(har_series, horizon=5)


@pytest.mark.unit
def test_garch_and_xgb_fitters_are_stubbed(
    garch_series: pd.DataFrame, har_series: pd.Series
) -> None:
    rets = vf.make_rng(1).standard_normal(64)
    with pytest.raises(NotImplementedError):
        vf.garch_11_log_likelihood(rets, 0.1, 0.05, 0.9)
    with pytest.raises(NotImplementedError):
        vf.fit_garch(pd.Series(rets))
    with pytest.raises(NotImplementedError):
        vf.fit_xgb(pd.DataFrame({"x": [1.0, 2.0]}), pd.Series([0.1, 0.2]))


@pytest.mark.unit
def test_evaluation_kernels_are_stubbed() -> None:
    a = np.array([0.1, 0.2, 0.3])
    b = np.array([0.11, 0.19, 0.31])
    with pytest.raises(NotImplementedError):
        vf.qlike(a, b)
    with pytest.raises(NotImplementedError):
        vf.mse(a, b)
    with pytest.raises(NotImplementedError):
        vf.derive_verdict({"garch": 0.5}, 0.5, {})


@pytest.mark.unit
def test_walk_forward_is_stubbed(garch_series: pd.DataFrame) -> None:
    with pytest.raises(NotImplementedError):
        vf.WalkForwardConfig(horizon=5)

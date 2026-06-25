"""Unit tests for the optional vol-targeting overlay (downstream P&L + DSR).

The overlay turns a volatility forecast into a vol-targeted position, charges a
per-side bps cost on the change in exposure, and reports a raw Sharpe AND a
Deflated Sharpe with the TRUE ``n_trials``. These pin its no-lookahead sizing,
cost accounting, validation guards, and the honest DSR deflation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast._exceptions import ValidationError
from volforecast.backtest.overlay import OverlayResult, vol_target_overlay


def _series(values: list[float], start: str = "2020-01-01") -> pd.Series:
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype="float64")


@pytest.mark.unit
def test_overlay_sizes_inversely_to_forecast_vol() -> None:
    returns = _series([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])
    # A low forecast vol => higher exposure (capped); high forecast => lower.
    vol_forecast = _series([0.05, 0.05, 0.20, 0.20, 0.10, 0.10])
    result = vol_target_overlay(
        returns, vol_forecast, target_vol=0.10, max_leverage=2.0, cost_bps=0.0
    )
    assert isinstance(result, OverlayResult)
    # exposure = clip(target/forecast, 0, max_leverage): 0.10/0.05=2.0 (capped),
    # 0.10/0.20=0.5, 0.10/0.10=1.0.
    np.testing.assert_allclose(
        result.exposure.to_numpy(), [2.0, 2.0, 0.5, 0.5, 1.0, 1.0], rtol=1e-12
    )


@pytest.mark.unit
def test_overlay_is_no_lookahead() -> None:
    """The position formed at t is applied to the t+1 return (shift by one)."""
    returns = _series([0.10, 0.20, 0.30, 0.40])
    vol_forecast = _series([0.10, 0.10, 0.10, 0.10])  # exposure == 1.0 each day
    result = vol_target_overlay(returns, vol_forecast, target_vol=0.10, cost_bps=0.0)
    # gross[0] == 0 (no prior position); gross[t] == returns[t] for t>=1.
    np.testing.assert_allclose(result.gross_returns.to_numpy(), [0.0, 0.20, 0.30, 0.40], rtol=1e-12)


@pytest.mark.unit
def test_overlay_charges_cost_on_turnover() -> None:
    returns = _series([0.01, 0.01, 0.01, 0.01])
    # Exposure flips 0 -> 2 -> 0 -> 2 (max turnover each step).
    vol_forecast = _series([0.05, 1.0, 0.05, 1.0])
    no_cost = vol_target_overlay(returns, vol_forecast, target_vol=0.10, cost_bps=0.0)
    with_cost = vol_target_overlay(returns, vol_forecast, target_vol=0.10, cost_bps=50.0)
    # The cost strictly lowers the net P&L.
    assert with_cost.net_returns.sum() < no_cost.net_returns.sum()
    assert with_cost.turnover > 0.0


@pytest.mark.unit
def test_overlay_deflated_sharpe_decreases_with_more_trials() -> None:
    rng = np.random.default_rng(0)
    returns = _series((rng.standard_normal(250) * 0.01).tolist())
    vol_forecast = _series((np.abs(rng.standard_normal(250)) * 0.1 + 0.05).tolist())
    one = vol_target_overlay(returns, vol_forecast, n_trials=1)
    many = vol_target_overlay(returns, vol_forecast, n_trials=50)
    # The Deflated Sharpe is non-increasing in the multiplicity count.
    assert many.deflated_sharpe <= one.deflated_sharpe
    assert many.n_trials == 50


@pytest.mark.unit
def test_overlay_uses_honest_trial_variance_not_unit_constant() -> None:
    """The DSR cross-trial variance V must be the honest per-obs-scale quantity.

    REGRESSION GUARD for the fabricated ``V = 1.0`` bug. With ``V`` hardcoded to
    ``1.0`` (in per-observation Sharpe units - an annualized SR of ~16) the
    expected-maximum benchmark was astronomically large, so the DSR was PINNED to
    zero for any ``n_trials > 1`` (over-deflation). The honest single-series
    fallback ``V = (1 + SR^2/2)/n_obs`` is a tiny, per-obs-scale quantity, so the
    deflation is meaningful but not fabricated.
    """
    rng = np.random.default_rng(1)
    returns = _series((rng.standard_normal(250) * 0.01).tolist())
    vol_forecast = _series((np.abs(rng.standard_normal(250)) * 0.1 + 0.05).tolist())

    result = vol_target_overlay(returns, vol_forecast, n_trials=50)
    # The honest V is the single-series proxy, NOT the absurd unit variance.
    v = result.meta["variance_of_trial_sharpes"]
    assert 0.0 < v < 0.1, v
    # On near-zero-Sharpe noise the deflated value is small but not the floored 0.0
    # that V = 1.0 produced at this n_trials; mostly we pin that V left unit-scale.
    assert result.deflated_sharpe >= 0.0


@pytest.mark.unit
def test_overlay_real_cross_trial_variance_from_trial_sharpes() -> None:
    """Supplied per-obs trial Sharpes drive the REAL cross-trial variance V."""
    rng = np.random.default_rng(2)
    returns = _series((rng.standard_normal(250) * 0.01).tolist())
    vol_forecast = _series((np.abs(rng.standard_normal(250)) * 0.1 + 0.05).tolist())

    # A dispersed grid of per-obs trial Sharpes => a non-trivial, REAL V that is
    # the sample variance (ddof=1) of those Sharpes.
    trial_sharpes = [0.02, -0.01, 0.05, 0.00, -0.03, 0.04]
    expected_v = float(np.var(np.asarray(trial_sharpes), ddof=1))
    result = vol_target_overlay(
        returns, vol_forecast, n_trials=len(trial_sharpes), trial_sharpes=trial_sharpes
    )
    assert result.meta["variance_of_trial_sharpes"] == pytest.approx(expected_v, rel=1e-12)


@pytest.mark.unit
def test_overlay_positive_control_detector_fires_on_real_skill() -> None:
    """POSITIVE CONTROL: a genuinely skilful series clears a high DSR.

    The honest-V fix must not *only* lower false positives - it must still let the
    detector fire when the overlay genuinely has skill. A strongly positive,
    low-noise net-return path (a large positive per-observation Sharpe) yields a
    Deflated Sharpe near 1.0 even after multiplicity deflation, whereas the
    near-zero-Sharpe noise series does not.
    """
    rng = np.random.default_rng(3)
    n = 500
    idx = pd.date_range("2020-01-01", periods=n, freq="B")

    # Genuine skill: a positive drift dominating the noise => large Sharpe.
    skilful_returns = pd.Series(rng.standard_normal(n) * 0.003 + 0.004, index=idx)
    flat_forecast = pd.Series(np.full(n, 0.08), index=idx)
    skilful = vol_target_overlay(
        skilful_returns, flat_forecast, cost_bps=0.0, n_trials=50
    )

    # No skill: zero-mean noise => near-zero Sharpe.
    noise_returns = pd.Series(rng.standard_normal(n) * 0.01, index=idx)
    noise = vol_target_overlay(noise_returns, flat_forecast, cost_bps=0.0, n_trials=50)

    # The detector fires for real skill (high DSR) and stays quiet on noise.
    assert skilful.deflated_sharpe > 0.99
    assert noise.deflated_sharpe < 0.9
    assert skilful.deflated_sharpe > noise.deflated_sharpe


@pytest.mark.unit
def test_overlay_to_dict_is_json_safe() -> None:
    import json

    returns = _series([0.01, -0.01, 0.02, 0.0, 0.01])
    vol_forecast = _series([0.1, 0.1, 0.1, 0.1, 0.1])
    result = vol_target_overlay(returns, vol_forecast, cost_bps=10.0, n_trials=3)
    payload = result.to_dict()
    json.dumps(payload)
    assert set(payload) >= {
        "net_returns",
        "gross_returns",
        "exposure",
        "sharpe",
        "deflated_sharpe",
        "turnover",
        "n_trials",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"target_vol": 0.0}, "target_vol"),
        ({"target_vol": -0.1}, "target_vol"),
        ({"max_leverage": -1.0}, "max_leverage"),
        ({"cost_bps": -1.0}, "cost_bps"),
        ({"n_trials": 0}, "n_trials"),
    ],
)
def test_overlay_rejects_bad_params(kwargs: dict[str, float], match: str) -> None:
    returns = _series([0.01, 0.02, 0.03])
    vol_forecast = _series([0.1, 0.1, 0.1])
    with pytest.raises(ValidationError, match=match):
        vol_target_overlay(returns, vol_forecast, **kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_overlay_requires_two_aligned_observations() -> None:
    returns = _series([0.01, 0.02, 0.03])
    # Forecast shares only one index label with returns after alignment.
    vol_forecast = pd.Series(
        [0.1], index=pd.date_range("2020-01-01", periods=1, freq="B"), dtype="float64"
    )
    with pytest.raises(ValidationError, match="two aligned"):
        vol_target_overlay(returns, vol_forecast)


@pytest.mark.unit
def test_overlay_zero_forecast_maps_to_zero_exposure() -> None:
    returns = _series([0.01, 0.02, 0.03, 0.04])
    vol_forecast = _series([0.0, 0.1, 0.0, 0.1])
    result = vol_target_overlay(returns, vol_forecast, target_vol=0.10, cost_bps=0.0)
    # A zero/negative forecast cannot size a position => zero exposure there.
    assert result.exposure.iloc[0] == 0.0
    assert result.exposure.iloc[2] == 0.0

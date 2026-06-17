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

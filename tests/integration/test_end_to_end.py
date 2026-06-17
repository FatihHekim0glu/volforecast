"""End-to-end walk-forward integration (filled in as the kernels land).

Runs the full synthetic-GARCH horse race through the public API and asserts the
response shape the FastAPI route depends on. Marked ``xfail`` until the pipeline
is implemented so the partition collects.
"""

from __future__ import annotations

import pandas as pd
import pytest

import volforecast as vf


@pytest.mark.integration
@pytest.mark.xfail(reason="end-to-end pipeline not yet implemented", strict=True)
def test_walk_forward_runs_on_synthetic_series(garch_series: pd.DataFrame) -> None:
    config = vf.WalkForwardConfig(horizon=5, models=("garch", "har_rv", "ewma", "xgboost", "rw"))
    result = vf.run_walk_forward(garch_series, config=config)
    assert result.n_folds > 0
    assert set(config.models) <= set(result.forecasts.columns)
    assert len(result.realized_vol) == len(result.forecasts)
    # Figures serialize to {data, layout}.
    fig = vf.rv_forecast_figure(result.realized_vol, result.forecasts)
    assert set(fig) >= {"data", "layout"}


@pytest.mark.integration
@pytest.mark.xfail(reason="generate_garch_ohlc not yet implemented", strict=True)
def test_synthetic_generator_round_trips() -> None:
    ohlc = vf.generate_garch_ohlc(n_obs=800, seed=7)
    assert list(ohlc.columns) == ["open", "high", "low", "close"]
    assert len(ohlc) == 800

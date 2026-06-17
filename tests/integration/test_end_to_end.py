"""End-to-end walk-forward integration on the synthetic GARCH series.

Runs the full synthetic-GARCH horse race through the public
:func:`volforecast.run_vol_forecast` entrypoint (the same function the FastAPI
route calls) and asserts the response shape the route depends on, plus the
headline honest-null guarantee: on GARCH-generated data ML is NOT crowned.

These use small, fast walk-forward configs (a wide ``step`` and the cheap
baseline/HAR set, with one GARCH/XGBoost arm) so the suite stays quick while
still exercising every fitter on the serve path.
"""

from __future__ import annotations

import json
import warnings

import pandas as pd
import pytest

import volforecast as vf


@pytest.mark.integration
def test_run_vol_forecast_summary_shape_and_honest_null(garch_series: pd.DataFrame) -> None:
    """The public entrypoint returns the documented summary and the honest null."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = vf.run_vol_forecast(
            garch_series,
            horizon=5,
            models=("garch", "har_rv", "ewma", "xgboost", "rw"),
            cost_bps=10.0,
            seed=7,
            train_window=300,
            step=40,
        )

    summary = run.summary.to_dict()
    expected_keys = {
        "qlike_by_model",
        "mse_by_model",
        "best_model",
        "best_model_class",
        "best_reference",
        "dm_pvalues",
        "spa_pvalue",
        "ml_beats_garch",
        "n_effective_trials",
        "data_source",
        "horizon",
        "n_folds",
    }
    assert expected_keys <= set(summary)
    assert summary["n_folds"] > 0
    assert summary["horizon"] == 5
    # The whole summary must be JSON-serializable (crosses the API boundary).
    json.dumps(summary)

    # HONEST NULL: on GARCH-generated data, ML must NOT be crowned the winner.
    assert summary["ml_beats_garch"] is False
    assert summary["best_model_class"] == "reference"
    assert summary["best_model"] in summary["qlike_by_model"]
    # The SPA composite null is not rejected on GARCH-true data.
    assert summary["spa_pvalue"] > 0.05


@pytest.mark.integration
def test_run_vol_forecast_builds_both_figures(garch_series: pd.DataFrame) -> None:
    """The figure helper assembles the forecast + QLIKE-bar figures as JSON dicts."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = vf.run_vol_forecast(
            garch_series,
            horizon=1,
            models=("har_rv", "ewma", "rw"),
            seed=7,
            train_window=252,
            step=50,
        )

    figures = vf.build_vol_forecast_figures(run)
    assert set(figures) == {"forecast_figure", "error_figure"}
    for fig in figures.values():
        assert set(fig) >= {"data", "layout"}
        # Figures are plain JSON-serializable dicts (no Plotly object leaks).
        json.dumps(fig)
    # The forecast figure has the truth line plus one trace per model.
    assert len(figures["forecast_figure"]["data"]) == 1 + len(run.result.forecasts.columns)


@pytest.mark.integration
def test_run_vol_forecast_accepts_a_close_series(garch_series: pd.DataFrame) -> None:
    """A bare close-price Series is widened to a degenerate OHLC bar (CTC RV)."""
    close = garch_series["close"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = vf.run_vol_forecast(
            close,
            horizon=1,
            models=("har_rv", "ewma", "rw"),
            seed=7,
            train_window=252,
            step=50,
        )
    assert run.summary.n_folds > 0
    # A close-only input forces the close-to-close RV proxy.
    assert run.meta["rv_estimator"] == "close_to_close"


@pytest.mark.integration
def test_walk_forward_runs_on_synthetic_series(garch_series: pd.DataFrame) -> None:
    config = vf.WalkForwardConfig(
        horizon=5,
        train_window=300,
        step=50,
        models=("har_rv", "ewma", "rw"),
    )
    result = vf.run_walk_forward(garch_series, config=config)
    assert result.n_folds > 0
    assert set(config.models) <= set(result.forecasts.columns)
    assert len(result.realized_vol) == len(result.forecasts)
    # Figures serialize to {data, layout}.
    fig = vf.rv_forecast_figure(result.realized_vol, result.forecasts)
    assert set(fig) >= {"data", "layout"}


@pytest.mark.integration
def test_synthetic_generator_round_trips() -> None:
    ohlc = vf.generate_garch_ohlc(n_obs=800, seed=7)
    assert list(ohlc.columns) == ["open", "high", "low", "close"]
    assert len(ohlc) == 800

"""Unit tests for the public ``run_vol_forecast`` entrypoint and figure helper.

These cover the cross-group wiring (walk-forward -> QLIKE/MSE -> SPA/DM ->
verdict -> summary) on small, fast configs, plus the input-coercion and
validation contracts. The headline honest-null guarantee is asserted directly on
the synthetic GARCH fixture (ML is never crowned).
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import volforecast as vf
from volforecast._exceptions import ValidationError
from volforecast.pipeline import (
    VolForecastRun,
    VolForecastSummary,
    _coerce_to_ohlc,
)


def _fast_run(garch_series: pd.DataFrame, **kwargs: object) -> VolForecastRun:
    params: dict[str, object] = {
        "horizon": 1,
        "models": ("har_rv", "ewma", "rw"),
        "seed": 7,
        "train_window": 252,
        "step": 50,
    }
    params.update(kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vf.run_vol_forecast(garch_series, **params)  # type: ignore[arg-type]


@pytest.mark.unit
def test_run_vol_forecast_returns_typed_bundle(garch_series: pd.DataFrame) -> None:
    run = _fast_run(garch_series)
    assert isinstance(run, VolForecastRun)
    assert isinstance(run.summary, VolForecastSummary)
    assert run.summary.n_folds > 0
    assert set(run.summary.qlike_by_model) <= {"har_rv", "ewma", "rw"}
    assert run.result.realized_vol.index.equals(run.result.forecasts.index)


@pytest.mark.unit
def test_run_vol_forecast_summary_is_json_safe(garch_series: pd.DataFrame) -> None:
    run = _fast_run(garch_series)
    payload = run.summary.to_dict()
    json.dumps(payload)
    assert payload["data_source"] == "synthetic"
    assert payload["best_model"] in payload["qlike_by_model"]
    # best_reference must itself be a scored model.
    assert payload["best_reference"] in payload["qlike_by_model"]


@pytest.mark.unit
def test_run_vol_forecast_honest_null_with_ml_arm(garch_series: pd.DataFrame) -> None:
    """Even with the XGBoost arm in the set, ML is not crowned on GARCH data."""
    run = _fast_run(
        garch_series,
        models=("garch", "har_rv", "xgboost", "rw"),
        train_window=300,
        step=60,
    )
    assert run.summary.ml_beats_garch is False
    assert run.summary.best_model_class == "reference"


@pytest.mark.unit
def test_run_vol_forecast_echoes_data_source(garch_series: pd.DataFrame) -> None:
    run = _fast_run(garch_series, data_source="polygon")
    assert run.summary.data_source == "polygon"
    assert run.summary.to_dict()["data_source"] == "polygon"


@pytest.mark.unit
def test_run_vol_forecast_rejects_negative_cost(garch_series: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="cost_bps"):
        vf.run_vol_forecast(garch_series, horizon=1, models=("rw",), cost_bps=-1.0)


@pytest.mark.unit
def test_run_vol_forecast_rejects_bad_data_type() -> None:
    with pytest.raises(ValidationError, match="DataFrame"):
        vf.run_vol_forecast([1, 2, 3], horizon=1)  # type: ignore[arg-type]


@pytest.mark.unit
def test_run_vol_forecast_requires_close_column() -> None:
    bad = pd.DataFrame(
        {"px": [1.0, 2.0, 3.0]}, index=pd.date_range("2020-01-01", periods=3, freq="B")
    )
    with pytest.raises(ValidationError, match="close"):
        vf.run_vol_forecast(bad, horizon=1, models=("rw",))


@pytest.mark.unit
def test_coerce_to_ohlc_widens_close_only_frame() -> None:
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    frame = pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0]}, index=idx)
    out = _coerce_to_ohlc(frame)
    assert list(out.columns) == ["open", "high", "low", "close"]
    # Degenerate bar: open == high == low == close.
    assert (out["open"] == out["close"]).all()
    assert (out["high"] == out["close"]).all()
    assert (out["low"] == out["close"]).all()


@pytest.mark.unit
def test_coerce_to_ohlc_passes_full_ohlc_through() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {
            "OPEN": [1.0, 2.0, 3.0],
            "HIGH": [1.5, 2.5, 3.5],
            "LOW": [0.5, 1.5, 2.5],
            "CLOSE": [1.2, 2.2, 3.2],
        },
        index=idx,
    )
    out = _coerce_to_ohlc(frame)
    assert list(out.columns) == ["open", "high", "low", "close"]
    np.testing.assert_allclose(out["high"].to_numpy(), [1.5, 2.5, 3.5])


@pytest.mark.unit
def test_coerce_to_ohlc_rejects_non_panel() -> None:
    with pytest.raises(ValidationError, match="DataFrame"):
        _coerce_to_ohlc("not a frame")  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_figures_highlights_best_model(garch_series: pd.DataFrame) -> None:
    run = _fast_run(garch_series)
    figures = vf.build_vol_forecast_figures(run)
    assert set(figures) == {"forecast_figure", "error_figure"}
    bars = figures["error_figure"]["data"][0]
    # One bar per scored model.
    assert len(bars["x"]) == len(run.summary.qlike_by_model)

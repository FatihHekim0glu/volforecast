"""Property + regression tests for the walk-forward engine and the data layer.

These cover the leakage-control heart of the project (the IMPLEMENT-group files
``walkforward/engine.py`` and ``data.py``):

- generator determinism, volatility clustering, OHLC coherence and validation;
- the loader degrades to the synthetic generator offline (no keys);
- ``log_returns`` is no-lookahead (log-then-diff, no ffill);
- the train/test slicer enforces the purge (embargo) and the warm-up;
- **future-perturbation invariance**: perturbing returns strictly after the
  forecast origin changes no earlier-fold forecast (the canonical leakage
  detector);
- **fit-on-train-only**: perturbing TEST-region rows leaves the fitted GARCH /
  XGBoost forecasts (which are refit per train fold) unchanged.
"""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import volforecast as vf
from volforecast.walkforward.engine import _train_test_slices

# ``arch``/``xgboost`` emit benign convergence/runtime warnings on short folds.
pytestmark = pytest.mark.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# data.py - synthetic GARCH(1,1) generator                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_generator_emits_coherent_ohlc() -> None:
    ohlc = vf.generate_garch_ohlc(n_obs=800, seed=7)
    assert list(ohlc.columns) == ["open", "high", "low", "close"]
    assert len(ohlc) == 800
    assert (ohlc.to_numpy() > 0.0).all()
    assert (ohlc["high"] >= ohlc[["open", "close"]].max(axis=1)).all()
    assert (ohlc["low"] <= ohlc[["open", "close"]].min(axis=1)).all()
    assert (ohlc["high"] >= ohlc["low"]).all()
    assert isinstance(ohlc.index, pd.DatetimeIndex)


@pytest.mark.unit
def test_generator_is_deterministic() -> None:
    a = vf.generate_garch_ohlc(n_obs=500, seed=11)
    b = vf.generate_garch_ohlc(n_obs=500, seed=11)
    pd.testing.assert_frame_equal(a, b)
    # A different seed yields a different path.
    c = vf.generate_garch_ohlc(n_obs=500, seed=12)
    assert not a["close"].equals(c["close"])


@pytest.mark.unit
def test_generator_shows_volatility_clustering() -> None:
    ohlc = vf.generate_garch_ohlc(n_obs=2000, seed=3, alpha=0.1, beta=0.88)
    rets = np.log(ohlc["close"]).diff().dropna().to_numpy()
    sq = rets**2
    # The hallmark of GARCH: positive autocorrelation in squared returns.
    ac1 = float(np.corrcoef(sq[:-1], sq[1:])[0, 1])
    assert ac1 > 0.05


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"n_obs": 1}, "n_obs"),
        ({"omega": 0.0}, "omega"),
        ({"alpha": -0.1}, "alpha"),
        ({"beta": -0.1}, "beta"),
        ({"alpha": 0.6, "beta": 0.6}, "stationarity"),
        ({"intraday_range_scale": 0.0}, "intraday_range_scale"),
    ],
)
def test_generator_rejects_bad_params(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(vf.ValidationError, match=match):
        vf.generate_garch_ohlc(**kwargs)  # type: ignore[arg-type]


@pytest.mark.unit
def test_generator_respects_start_anchor() -> None:
    ohlc = vf.generate_garch_ohlc(n_obs=10, seed=1, start=date(2021, 3, 1))
    assert ohlc.index[0] == pd.Timestamp("2021-03-01")
    # Business-day frequency: no weekends.
    assert (ohlc.index.dayofweek < 5).all()


# --------------------------------------------------------------------------- #
# data.py - loader + log_returns                                              #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_loader_falls_back_to_synthetic_offline() -> None:
    # No Polygon provider/key in this environment, so every preference that allows
    # synthetic must degrade to it and report honest provenance.
    frame, source = vf.get_ohlc("SPY", date(2020, 1, 1), date(2021, 1, 1), source_pref="auto")
    assert source == "synthetic"
    assert list(frame.columns) == ["open", "high", "low", "close"]
    assert len(frame) > 100

    _, source2 = vf.get_ohlc("AAPL", date(2019, 1, 1), date(2020, 1, 1), source_pref="polygon")
    assert source2 == "synthetic"  # polygon unavailable → synthetic fallback

    _, source3 = vf.get_ohlc("QQQ", date(2019, 1, 1), date(2020, 1, 1), source_pref="synthetic")
    assert source3 == "synthetic"


@pytest.mark.unit
def test_loader_is_seed_reproducible() -> None:
    a, _ = vf.get_ohlc("SPY", date(2020, 1, 1), date(2021, 1, 1), source_pref="synthetic", seed=5)
    b, _ = vf.get_ohlc("SPY", date(2020, 1, 1), date(2021, 1, 1), source_pref="synthetic", seed=5)
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("ticker", "start", "end", "pref", "match"),
    [
        ("", date(2020, 1, 1), date(2021, 1, 1), "auto", "ticker"),
        ("SPY", date(2021, 1, 1), date(2020, 1, 1), "auto", "after"),
        ("SPY", date(2020, 1, 1), date(2020, 1, 1), "auto", "after"),
        ("SPY", date(2020, 1, 1), date(2021, 1, 1), "bogus", "source_pref"),
    ],
)
def test_loader_rejects_bad_inputs(
    ticker: str, start: date, end: date, pref: str, match: str
) -> None:
    with pytest.raises(vf.ValidationError, match=match):
        vf.get_ohlc(ticker, start, end, source_pref=pref)  # type: ignore[arg-type]


@pytest.mark.unit
def test_log_returns_is_no_lookahead(garch_series: pd.DataFrame) -> None:
    close = garch_series["close"]
    lr = vf.log_returns(close)
    # Exactly one row (the leading NaN) is dropped; values match log-then-diff.
    expected = np.log(close.to_numpy())
    expected = pd.Series(expected, index=close.index).diff().dropna()
    np.testing.assert_allclose(lr.to_numpy(), expected.to_numpy(), rtol=1e-12)
    assert lr.index.equals(expected.index)


@pytest.mark.unit
def test_log_returns_rejects_non_positive() -> None:
    bad = pd.Series([100.0, -1.0, 101.0], index=pd.date_range("2020-01-01", periods=3, freq="B"))
    with pytest.raises(vf.ValidationError, match="non-positive"):
        vf.log_returns(bad)


# --------------------------------------------------------------------------- #
# engine - config validation + slicing/purge                                  #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_config_embargo_and_to_dict() -> None:
    cfg = vf.WalkForwardConfig(horizon=5, gap=2, train_window=300, step=4, anchored=False)
    assert cfg.embargo == 7  # gap + horizon
    d = cfg.to_dict()
    assert d["embargo"] == 7
    assert d["models"] == list(cfg.models)
    assert set(d) == {
        "horizon",
        "train_window",
        "gap",
        "step",
        "anchored",
        "models",
        "seed",
        "embargo",
    }


@pytest.mark.unit
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"horizon": 0}, "horizon"),
        ({"horizon": 5, "train_window": 1}, "train_window"),
        ({"horizon": 5, "gap": -1}, "gap"),
        ({"horizon": 5, "step": 0}, "step"),
        ({"horizon": 5, "seed": -1}, "seed"),
        ({"horizon": 5, "models": ()}, "non-empty"),
        ({"horizon": 5, "models": ("garch", "nope")}, "unknown model"),
    ],
)
def test_config_rejects_bad_params(kwargs: dict[str, object], match: str) -> None:
    with pytest.raises(vf.ValidationError, match=match):
        vf.WalkForwardConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.property
@given(
    horizon=st.sampled_from([1, 5, 22]),
    gap=st.integers(min_value=0, max_value=3),
    step=st.integers(min_value=1, max_value=15),
    anchored=st.booleans(),
)
@settings(max_examples=40, deadline=None)
def test_slices_enforce_purge_and_warmup(horizon: int, gap: int, step: int, anchored: bool) -> None:
    cfg = vf.WalkForwardConfig(
        horizon=horizon, gap=gap, step=step, train_window=120, anchored=anchored
    )
    slices = _train_test_slices(600, cfg)
    assert slices, "expected at least one fold for a 600-row series"
    for train_start, train_end, test_origin in slices:
        # Warm-up respected and slice well-formed.
        assert 0 <= train_start < train_end
        assert train_end >= cfg.train_window
        # The purge: the test origin is exactly ``embargo`` rows after the train
        # end, so the last train row's forward target cannot reach the origin.
        assert test_origin - train_end == cfg.embargo
        # The origin's full forward target window stays inside the series.
        assert test_origin + gap + horizon <= 599
        if anchored:
            assert train_start == 0
        else:
            assert train_end - train_start == cfg.train_window


@pytest.mark.unit
def test_run_walk_forward_too_short_raises() -> None:
    ohlc = vf.generate_garch_ohlc(n_obs=60, seed=1)
    cfg = vf.WalkForwardConfig(horizon=22, train_window=504)
    with pytest.raises(vf.InsufficientDataError):
        vf.run_walk_forward(ohlc, config=cfg)


@pytest.mark.unit
def test_run_walk_forward_rejects_bad_inputs() -> None:
    with pytest.raises(vf.ValidationError, match="DataFrame"):
        vf.run_walk_forward([1, 2, 3], config=vf.WalkForwardConfig(horizon=5))  # type: ignore[arg-type]
    ohlc = vf.generate_garch_ohlc(n_obs=400, seed=1)
    with pytest.raises(vf.ValidationError, match="rv_estimator"):
        vf.run_walk_forward(ohlc, config=vf.WalkForwardConfig(horizon=5), rv_estimator="bogus")
    no_close = ohlc.rename(columns={"close": "px"})
    with pytest.raises(vf.ValidationError, match="close"):
        vf.run_walk_forward(no_close, config=vf.WalkForwardConfig(horizon=5))


# --------------------------------------------------------------------------- #
# engine - leakage guards (the load-bearing properties)                       #
# --------------------------------------------------------------------------- #
def _baseline_config() -> vf.WalkForwardConfig:
    """A fast, baseline-only config (no GARCH/XGB) for the cheap leakage checks."""
    return vf.WalkForwardConfig(
        horizon=5, train_window=252, step=20, models=("har_rv", "ewma", "rw")
    )


@pytest.mark.property
def test_future_perturbation_invariance_baselines(garch_series: pd.DataFrame) -> None:
    """Perturbing the strictly-future tail changes no earlier-fold forecast."""
    cfg = _baseline_config()
    base = vf.run_walk_forward(garch_series, config=cfg)

    perturbed = garch_series.copy()
    perturbed.iloc[-10:] *= 1.5  # mutate only rows in the future of the early folds
    again = vf.run_walk_forward(perturbed, config=cfg)

    common = base.forecasts.index.intersection(again.forecasts.index)[:5]
    assert len(common) >= 3
    pd.testing.assert_frame_equal(base.forecasts.loc[common], again.forecasts.loc[common])
    pd.testing.assert_series_equal(base.realized_vol.loc[common], again.realized_vol.loc[common])


@pytest.mark.property
@pytest.mark.slow
def test_fit_on_train_only_garch_and_xgb(garch_series: pd.DataFrame) -> None:
    """Perturbing TEST-region rows leaves the refit GARCH/XGB train forecasts intact.

    This is the explicit fit-on-train-only guarantee: because every fitter is
    re-estimated inside its own train fold and never touches a row at/after the
    forecast origin, mutating data that lies strictly in the FUTURE of an early
    fold must not move that fold's GARCH or XGBoost forecast.
    """
    cfg = vf.WalkForwardConfig(
        horizon=5,
        train_window=252,
        step=30,
        models=("garch", "har_rv", "xgboost", "ewma", "rw"),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = vf.run_walk_forward(garch_series, config=cfg)

        perturbed = garch_series.copy()
        # Perturb the strictly-future tail (well after the first fold's origin).
        perturbed.iloc[-15:] *= 1.3
        again = vf.run_walk_forward(perturbed, config=cfg)

    common = base.forecasts.index.intersection(again.forecasts.index)[:3]
    assert len(common) >= 2
    # ALL models (including the stochastic-but-seeded XGBoost and the MLE GARCH)
    # must be byte-identical on the early, unperturbed folds.
    pd.testing.assert_frame_equal(base.forecasts.loc[common], again.forecasts.loc[common])


@pytest.mark.regression
def test_walk_forward_result_shape_and_to_dict(garch_series: pd.DataFrame) -> None:
    cfg = _baseline_config()
    result = vf.run_walk_forward(garch_series, config=cfg)
    assert result.n_folds > 0
    assert list(result.forecasts.columns) == list(cfg.models)
    assert len(result.realized_vol) == len(result.forecasts)
    assert result.realized_vol.index.equals(result.forecasts.index)
    assert (result.realized_vol.to_numpy() > 0.0).all()

    payload = result.to_dict()
    assert set(payload) == {"realized_vol", "forecasts", "config", "n_folds", "meta"}
    # ISO date keys and JSON-safe floats (no numpy types / NaN sentinels).
    first_key = next(iter(payload["realized_vol"]))
    assert "T" in first_key or "-" in first_key  # ISO timestamp string
    for col in cfg.models:
        for v in payload["forecasts"][col].values():
            assert v is None or isinstance(v, float)


@pytest.mark.unit
def test_generator_rejects_non_int_n_obs() -> None:
    with pytest.raises(vf.ValidationError, match="n_obs must be an int"):
        vf.generate_garch_ohlc(n_obs=10.0)  # type: ignore[arg-type]


@pytest.mark.unit
def test_generator_rejects_non_finite_param() -> None:
    with pytest.raises(vf.ValidationError, match="omega must be finite"):
        vf.generate_garch_ohlc(omega=float("nan"))


@pytest.mark.unit
def test_loader_rejects_non_date() -> None:
    with pytest.raises(vf.ValidationError, match="must be datetime"):
        vf.get_ohlc("SPY", "2020-01-01", date(2021, 1, 1))  # type: ignore[arg-type]


@pytest.mark.unit
def test_config_rejects_non_int_scalar() -> None:
    with pytest.raises(vf.ValidationError, match="horizon must be an int"):
        vf.WalkForwardConfig(horizon=5.0)  # type: ignore[arg-type]


@pytest.mark.regression
def test_xgboost_consumes_lagged_exog(garch_series: pd.DataFrame) -> None:
    """The engine threads an already-lagged exogenous feature into XGBoost.

    Exercises the ``exog=`` path: a caller-lagged column is reindexed to each
    train fold, joined into the HAR feature frame, and used at predict time.
    """
    rv = vf.realized_volatility(garch_series, estimator="garman_klass", window=1)
    # A trivially-lagged exogenous "vix-like" feature aligned to the OHLC index.
    exog = pd.DataFrame({"vix": rv.shift(1) * 100.0}, index=garch_series.index)
    cfg = vf.WalkForwardConfig(horizon=5, train_window=252, step=40, models=("xgboost", "rw"))
    result = vf.run_walk_forward(garch_series, config=cfg, exog=exog)
    assert result.n_folds > 0
    # XGBoost produced finite forecasts with the exogenous feature in the design.
    assert np.isfinite(result.forecasts["xgboost"].to_numpy()).any()


@pytest.mark.regression
def test_per_model_degrades_to_nan_not_abort() -> None:
    """A fold too short for GARCH yields a NaN for that model, not a hard failure.

    The earliest anchored folds have a warm-up below the GARCH minimum (50 obs),
    so the GARCH fitter raises ``InsufficientDataError`` on those folds; the engine
    must record a NaN for GARCH there (not abort the whole run) while the cheap
    random-walk baseline still forecasts - the graceful per-model degradation path
    (engine ``_forecast_fold`` except-clause).
    """
    ohlc = vf.generate_garch_ohlc(n_obs=200, seed=2)
    cfg = vf.WalkForwardConfig(horizon=1, train_window=20, step=10, gap=1, models=("garch", "rw"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = vf.run_walk_forward(ohlc, config=cfg)
    assert result.n_folds > 0
    garch = result.forecasts["garch"]
    # The early (warm-up < 50) folds degrade to NaN; once the anchored window
    # grows past 50 the GARCH fit succeeds - so we see BOTH NaN and finite rows.
    assert garch.isna().any()
    assert garch.notna().any()
    # The random-walk baseline never depends on the GARCH fit and always forecasts.
    assert result.forecasts["rw"].notna().any()


@pytest.mark.regression
def test_anchored_and_rolling_both_run(garch_series: pd.DataFrame) -> None:
    anchored = vf.run_walk_forward(
        garch_series,
        config=vf.WalkForwardConfig(
            horizon=5, train_window=300, step=25, anchored=True, models=("har_rv", "rw")
        ),
    )
    rolling = vf.run_walk_forward(
        garch_series,
        config=vf.WalkForwardConfig(
            horizon=5, train_window=300, step=25, anchored=False, models=("har_rv", "rw")
        ),
    )
    assert anchored.n_folds > 0
    assert rolling.n_folds > 0


@pytest.mark.regression
def test_parkinson_and_ctc_estimators_route(garch_series: pd.DataFrame) -> None:
    for est in ("close_to_close", "parkinson", "garman_klass"):
        result = vf.run_walk_forward(
            garch_series,
            config=vf.WalkForwardConfig(
                horizon=1, train_window=252, step=30, models=("rw", "ewma")
            ),
            rv_estimator=est,
        )
        assert result.n_folds > 0
        assert result.meta["rv_estimator"] == est

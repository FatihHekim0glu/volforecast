"""Behavioural tests for the ML/baseline/HAR-feature group.

Covers the kernels owned by this group:

- ``features/har.py``: HAR-RV component builder (lag-safety, exact values, the
  ``build_har_features`` feature/target bundle and exogenous join);
- ``baselines.py``: random-walk vol, EWMA/RiskMetrics (λ=0.94), HAR-RV (Corsi
  OLS) - correctness against hand references and positivity;
- ``ml/xgb.py``: XGBoost forecaster determinism, the feature contract, and the
  fit-on-train-only guards (validation/insufficient-data).

These tests assert behaviour directly (the stub-contract smoke tests in
``test_fixtures_and_stubs.py`` are retired as kernels land).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import volforecast as vf
from volforecast.features.har import HAR_COMPONENT_COLUMNS, build_har_features, har_components


def _rv_series(n: int, seed: int = 11) -> pd.Series:
    """A strictly-positive RV series on a business-day index."""
    gen = vf.make_rng(seed)
    values = np.exp(-4.0 + gen.standard_normal(n) * 0.3)
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=n, freq="B"), name="rv")


# --------------------------------------------------------------------------- #
# HAR feature builder                                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_har_components_columns_and_index() -> None:
    rv = _rv_series(60)
    hc = har_components(rv)
    assert list(hc.columns) == list(HAR_COMPONENT_COLUMNS)
    assert hc.index.equals(rv.index)


@pytest.mark.unit
def test_har_components_exact_lagged_values() -> None:
    # A ramp RV so the trailing means are easy to verify by hand.
    rv = pd.Series(
        np.arange(1.0, 31.0),
        index=pd.date_range("2020-01-01", periods=30, freq="B"),
        name="rv",
    )
    hc = har_components(rv)
    # rv_daily at row t == RV_{t-1}.
    assert hc["rv_daily"].iloc[5] == pytest.approx(5.0)
    # rv_weekly at row 6 == mean(RV_1..RV_5) == mean(2,3,4,5,6) == 4.0 (5-day mean
    # up to t-1, then shifted one more day).
    assert hc["rv_weekly"].iloc[6] == pytest.approx(4.0)
    # rv_monthly at row 22 == mean(RV_0..RV_21) == mean(1..22) == 11.5.
    assert hc["rv_monthly"].iloc[22] == pytest.approx(11.5)


@pytest.mark.unit
def test_har_components_warmup_is_nan() -> None:
    rv = _rv_series(40)
    hc = har_components(rv)
    # Daily needs 1 prior obs, weekly 5, monthly 22 - all then shifted by 1.
    assert np.isnan(hc["rv_daily"].iloc[0])
    assert hc["rv_weekly"].iloc[:5].isna().all()
    assert hc["rv_monthly"].iloc[:22].isna().all()
    # Past the warm-up everything is finite.
    assert hc.iloc[22:].notna().to_numpy().all()


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(t=st.integers(min_value=22, max_value=140))
def test_har_feature_lag_safety_no_lookahead(t: int) -> None:
    """A HAR feature at row t must be invariant to any change at rows >= t.

    This is the load-bearing leakage guarantee: perturbing RV from position ``t``
    onward (the feature's own present and future) must not move the feature value
    at ``t`` - it depends only on RV strictly before ``t``.
    """
    rv = _rv_series(160, seed=3)
    base = har_components(rv)

    perturbed = rv.copy()
    perturbed.iloc[t:] *= 5.0  # mutate the present and the future only
    after = har_components(perturbed)

    for col in HAR_COMPONENT_COLUMNS:
        np.testing.assert_allclose(
            base[col].iloc[:t].to_numpy(),
            after[col].iloc[:t].to_numpy(),
            equal_nan=True,
        )


@pytest.mark.unit
def test_build_har_features_is_nan_free_and_aligned() -> None:
    rv = _rv_series(120)
    target = rv.shift(-7)  # a stand-in forward target
    bundle = build_har_features(rv, target)
    assert bundle.feature_names == HAR_COMPONENT_COLUMNS
    assert not bundle.features.isna().to_numpy().any()
    assert not bundle.target.isna().any()
    assert bundle.features.index.equals(bundle.target.index)
    # The bundle serializes cleanly across the API boundary.
    payload = bundle.to_dict()
    assert set(payload) == {"features", "target", "feature_names"}


@pytest.mark.unit
def test_build_har_features_joins_lagged_exog() -> None:
    rv = _rv_series(120)
    target = rv.shift(-7)
    exog = pd.DataFrame({"vix": rv.shift(1) * 2.0})
    bundle = build_har_features(rv, target, exog=exog)
    assert bundle.feature_names == (*HAR_COMPONENT_COLUMNS, "vix")
    assert "vix" in bundle.features.columns


@pytest.mark.unit
def test_build_har_features_rejects_colliding_exog() -> None:
    rv = _rv_series(60)
    target = rv.shift(-7)
    bad = pd.DataFrame({"rv_daily": rv.shift(1)})
    with pytest.raises(vf.ValidationError):
        build_har_features(rv, target, exog=bad)


@pytest.mark.unit
def test_build_har_features_rejects_non_dataframe_exog() -> None:
    rv = _rv_series(60)
    target = rv.shift(-7)
    with pytest.raises(vf.ValidationError):
        build_har_features(rv, target, exog=[1, 2, 3])  # type: ignore[arg-type]


@pytest.mark.unit
def test_build_har_features_rejects_empty_exog() -> None:
    rv = _rv_series(60)
    target = rv.shift(-7)
    empty = pd.DataFrame(index=rv.index)
    with pytest.raises(vf.ValidationError):
        build_har_features(rv, target, exog=empty)


@pytest.mark.unit
def test_build_har_features_rejects_all_nan_target() -> None:
    rv = _rv_series(60)
    target = pd.Series(np.nan, index=rv.index)
    with pytest.raises(vf.ValidationError):
        build_har_features(rv, target)


# --------------------------------------------------------------------------- #
# Baselines: random-walk, EWMA, HAR-RV                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_random_walk_is_lagged_and_positive() -> None:
    rv = _rv_series(50)
    rw = vf.random_walk_vol_forecast(rv, horizon=5)
    # The forecast at t is RV_{t-1} (observable at t).
    assert rw.iloc[1] == pytest.approx(rv.iloc[0])
    assert np.isnan(rw.iloc[0])
    assert (rw.dropna() > 0).all()


@pytest.mark.unit
def test_random_walk_rejects_bad_horizon() -> None:
    with pytest.raises(vf.ValidationError):
        vf.random_walk_vol_forecast(_rv_series(10), horizon=0)


@pytest.mark.unit
def test_ewma_matches_riskmetrics_reference() -> None:
    returns = pd.Series(
        [0.01, -0.02, 0.015, -0.01, 0.02, -0.005, 0.012],
        index=pd.date_range("2020-01-01", periods=7, freq="B"),
    )
    lam = 0.94
    out = vf.ewma_vol_forecast(returns, horizon=1, lam=lam)

    # Hand-rolled RiskMetrics recursion seeded from sample (population) variance.
    sq = returns.to_numpy() ** 2
    sigma2, prev = [], float(np.var(returns.to_numpy()))
    for t in range(len(returns)):
        sigma2.append(prev)
        prev = (1.0 - lam) * sq[t] + lam * prev
    ref = np.sqrt(sigma2)
    np.testing.assert_allclose(out.to_numpy(), ref, rtol=1e-12, atol=1e-15)
    assert (out >= 0).all()


@pytest.mark.unit
def test_ewma_horizon_scales_by_sqrt_h() -> None:
    returns = pd.Series(
        vf.make_rng(5).standard_normal(40) * 0.01,
        index=pd.date_range("2020-01-01", periods=40, freq="B"),
    )
    h1 = vf.ewma_vol_forecast(returns, horizon=1)
    h9 = vf.ewma_vol_forecast(returns, horizon=9)
    np.testing.assert_allclose(h9.to_numpy(), h1.to_numpy() * 3.0, rtol=1e-12)


@pytest.mark.unit
def test_ewma_rejects_bad_lambda() -> None:
    returns = _rv_series(10)
    with pytest.raises(vf.ValidationError):
        vf.ewma_vol_forecast(returns, horizon=1, lam=1.0)
    with pytest.raises(vf.ValidationError):
        vf.ewma_vol_forecast(returns, horizon=1, lam=0.0)


@pytest.mark.unit
def test_ewma_rejects_bad_horizon() -> None:
    with pytest.raises(vf.ValidationError):
        vf.ewma_vol_forecast(_rv_series(10), horizon=0)


@pytest.mark.unit
def test_har_rv_predict_rejects_bad_input() -> None:
    model = vf.HARRVModel(
        intercept=0.0,
        beta_daily=1.0,
        beta_weekly=0.0,
        beta_monthly=0.0,
        n_train=10,
    )
    with pytest.raises(vf.ValidationError):
        model.predict([1.0, 2.0])  # type: ignore[arg-type]
    with pytest.raises(vf.ValidationError):
        model.predict(pd.DataFrame({"rv_daily": [0.1]}))  # missing weekly/monthly
    # A serializable summary of the fitted coefficients.
    assert set(model.to_dict()) == {
        "intercept",
        "beta_daily",
        "beta_weekly",
        "beta_monthly",
        "n_train",
    }


@pytest.mark.unit
def test_har_rv_ols_recovers_known_coefficients() -> None:
    gen = vf.make_rng(7)
    n = 400
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    daily = pd.Series(np.abs(gen.standard_normal(n)) + 0.1, index=idx)
    weekly = pd.Series(np.abs(gen.standard_normal(n)) + 0.1, index=idx)
    monthly = pd.Series(np.abs(gen.standard_normal(n)) + 0.1, index=idx)
    components = pd.DataFrame({"rv_daily": daily, "rv_weekly": weekly, "rv_monthly": monthly})
    target = 0.001 + 0.5 * daily + 0.3 * weekly + 0.2 * monthly

    model = vf.fit_har_rv(components, target)
    assert model.intercept == pytest.approx(0.001, abs=1e-9)
    assert model.beta_daily == pytest.approx(0.5, abs=1e-9)
    assert model.beta_weekly == pytest.approx(0.3, abs=1e-9)
    assert model.beta_monthly == pytest.approx(0.2, abs=1e-9)
    # The fitted model reproduces the (noise-free) target exactly.
    np.testing.assert_allclose(model.predict(components).to_numpy(), target.to_numpy(), atol=1e-9)
    assert model.n_train == n


@pytest.mark.unit
def test_har_rv_requires_columns_and_enough_obs() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    bad = pd.DataFrame({"rv_daily": [0.1, 0.2, 0.3]}, index=idx)
    with pytest.raises(vf.ValidationError):
        vf.fit_har_rv(bad, pd.Series([0.1, 0.2, 0.3], index=idx))

    good = pd.DataFrame(
        {"rv_daily": [0.1, 0.2, 0.3], "rv_weekly": [0.1, 0.2, 0.3], "rv_monthly": [0.1, 0.2, 0.3]},
        index=idx,
    )
    # 3 obs < 4 free coefficients (intercept + 3 betas) → InsufficientDataError.
    with pytest.raises(vf.InsufficientDataError):
        vf.fit_har_rv(good, pd.Series([0.1, 0.2, 0.3], index=idx))


@pytest.mark.unit
def test_har_rv_on_har_series_is_positive(har_series: pd.Series) -> None:
    """End-to-end on the HAR fixture: the OLS forecast is finite and positive-ish."""
    components = har_components(har_series).iloc[22:]
    target = har_series.shift(-5).reindex(components.index)
    joined = components.join(target.rename("y")).dropna()
    model = vf.fit_har_rv(joined[list(HAR_COMPONENT_COLUMNS)], joined["y"])
    pred = model.predict(joined[list(HAR_COMPONENT_COLUMNS)])
    assert np.isfinite(pred.to_numpy()).all()
    # HAR-RV tracks the positive RV level: most forecasts are positive.
    assert (pred > 0).mean() > 0.9


# --------------------------------------------------------------------------- #
# XGBoost forecaster                                                           #
# --------------------------------------------------------------------------- #
@pytest.mark.unit
def test_fit_xgb_is_deterministic_and_has_contract(har_series: pd.Series) -> None:
    feats = pd.DataFrame(
        {
            "rv_daily": har_series.shift(1),
            "rv_weekly": har_series.rolling(5).mean().shift(1),
            "rv_monthly": har_series.rolling(22).mean().shift(1),
        }
    ).dropna()
    target = har_series.reindex(feats.index)
    m1 = vf.fit_xgb(feats, target, seed=7)
    m2 = vf.fit_xgb(feats, target, seed=7)
    pd.testing.assert_series_equal(m1.predict(feats), m2.predict(feats))

    assert m1.feature_names == ("rv_daily", "rv_weekly", "rv_monthly")
    assert m1.n_train == len(feats)
    assert set(m1.importances) == set(m1.feature_names)
    # to_dict is JSON-safe and omits the opaque booster.
    payload = m1.to_dict()
    assert "booster" not in payload
    assert payload["seed"] == 7


@pytest.mark.unit
def test_fit_xgb_different_seed_metadata() -> None:
    feats = pd.DataFrame({"rv_daily": _rv_series(60).shift(1)}).dropna()
    target = _rv_series(60).reindex(feats.index)
    model = vf.fit_xgb(feats, target, seed=123)
    assert model.seed == 123


@pytest.mark.unit
def test_xgb_predict_requires_feature_columns(har_series: pd.Series) -> None:
    feats = pd.DataFrame({"rv_daily": har_series.shift(1)}).dropna()
    target = har_series.reindex(feats.index)
    model = vf.fit_xgb(feats, target, seed=7)
    with pytest.raises(vf.ValidationError):
        model.predict(pd.DataFrame({"wrong": [1.0, 2.0, 3.0]}))


@pytest.mark.unit
def test_xgb_ignores_extra_columns_and_normalizes_order(har_series: pd.Series) -> None:
    feats = pd.DataFrame(
        {
            "rv_daily": har_series.shift(1),
            "rv_weekly": har_series.rolling(5).mean().shift(1),
        }
    ).dropna()
    target = har_series.reindex(feats.index)
    model = vf.fit_xgb(feats, target, seed=7)
    # Reorder columns and add an extra one: prediction must be unaffected.
    scrambled = feats[["rv_weekly", "rv_daily"]].copy()
    scrambled["junk"] = 99.0
    pd.testing.assert_series_equal(model.predict(feats), model.predict(scrambled))


@pytest.mark.unit
def test_fit_xgb_rejects_too_few_rows() -> None:
    feats = pd.DataFrame({"x": [1.0, 2.0]}, index=pd.RangeIndex(2))
    target = pd.Series([0.1, 0.2], index=pd.RangeIndex(2))
    with pytest.raises(vf.InsufficientDataError):
        vf.fit_xgb(feats, target)


@pytest.mark.unit
def test_fit_xgb_rejects_misaligned_index() -> None:
    feats = pd.DataFrame({"x": np.arange(12, dtype="float64")}, index=pd.RangeIndex(12))
    target = pd.Series(np.arange(12, dtype="float64"), index=pd.RangeIndex(1, 13))
    with pytest.raises(vf.ValidationError):
        vf.fit_xgb(feats, target)


@pytest.mark.unit
def test_fit_xgb_rejects_nan() -> None:
    feats = pd.DataFrame({"x": [np.nan, *np.arange(11, dtype="float64")]})
    target = pd.Series(np.arange(12, dtype="float64"))
    with pytest.raises(vf.ValidationError):
        vf.fit_xgb(feats, target)


# --------------------------------------------------------------------------- #
# LSTM research-only guard                                                     #
# --------------------------------------------------------------------------- #
@pytest.mark.research
def test_lstm_module_is_import_pure_and_guarded() -> None:
    """Importing the LSTM module has no side effects; the serve path never sees it."""
    import volforecast.ml.lstm as lstm

    # Not re-exported from the top-level package (serve path stays TF-free).
    assert not hasattr(vf, "fit_lstm")
    assert not hasattr(vf, "LSTMForecaster")
    # The arm is still importable as a module.
    assert callable(lstm.fit_lstm)

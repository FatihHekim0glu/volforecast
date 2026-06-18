"""Unit and property tests for the realized-volatility estimators and forward target.

Covers (group ``realized``):

- RV estimator positivity, naming, warm-up NaN structure, and dispatch;
- scale behaviour - log-return RV is invariant to a multiplicative price rescale,
  while range-RV scales linearly when the *log range* is rescaled;
- determinism (same input → identical output);
- forward-target DISJOINTNESS - for a non-NaN target at ``t`` the aggregation
  window lives strictly in ``{> t + gap}`` while features observable at ``t`` live
  in ``{<= t}``, so the two index sets never overlap;
- input validation (bad window/horizon/gap, missing columns, non-positive prices,
  ``high < low``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import volforecast as vf
from volforecast._exceptions import ValidationError
from volforecast.realized.estimators import (
    _GK_CO_FACTOR,
    _PARKINSON_FACTOR,
)

# --------------------------------------------------------------------------- #
# close-to-close RV                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_close_to_close_window_one_is_abs_log_return(garch_series: pd.DataFrame) -> None:
    close = garch_series["close"]
    rv = vf.close_to_close_rv(close, window=1)
    expected = np.log(close).diff().abs()
    assert rv.name == "rv_cc"
    # First obs is a warm-up NaN (no prior close to difference against).
    assert np.isnan(rv.iloc[0])
    pd.testing.assert_series_equal(rv.dropna(), expected.dropna(), check_names=False, rtol=1e-12)


@pytest.mark.unit
def test_close_to_close_is_nonnegative_and_warmup_nan(garch_series: pd.DataFrame) -> None:
    rv = vf.close_to_close_rv(garch_series["close"], window=5)
    # window=5 plus the diff warm-up ⇒ first 5 entries NaN, rest finite & >= 0.
    assert rv.iloc[:5].isna().all()
    finite = rv.dropna()
    assert (finite >= 0.0).all()
    assert np.isfinite(finite.to_numpy()).all()


@pytest.mark.unit
def test_close_to_close_invariant_to_price_rescale(garch_series: pd.DataFrame) -> None:
    close = garch_series["close"]
    base = vf.close_to_close_rv(close, window=5)
    scaled = vf.close_to_close_rv(close * 37.0, window=5)
    # Log returns (and hence close-to-close RV) are invariant to a constant
    # multiplicative price scale.
    pd.testing.assert_series_equal(base, scaled, rtol=1e-12)


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(scale=st.floats(min_value=0.1, max_value=50.0, allow_nan=False, allow_infinity=False))
def test_close_to_close_scale_invariance_property(scale: float) -> None:
    close = pd.Series(
        100.0 * np.exp(np.cumsum(vf.make_rng(3).standard_normal(120) * 0.01)),
        index=pd.date_range("2020-01-01", periods=120, freq="B"),
    )
    base = vf.close_to_close_rv(close, window=3)
    scaled = vf.close_to_close_rv(close * scale, window=3)
    np.testing.assert_allclose(
        base.dropna().to_numpy(), scaled.dropna().to_numpy(), rtol=1e-9, atol=1e-12
    )


# --------------------------------------------------------------------------- #
# Parkinson / Garman-Klass range RV                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_parkinson_matches_closed_form(garch_series: pd.DataFrame) -> None:
    rv = vf.parkinson_rv(garch_series, window=1)
    log_hl = np.log(garch_series["high"] / garch_series["low"])
    expected = np.sqrt(_PARKINSON_FACTOR * log_hl**2)
    assert rv.name == "rv_parkinson"
    pd.testing.assert_series_equal(rv, expected, check_names=False, rtol=1e-12)


@pytest.mark.unit
def test_garman_klass_matches_closed_form(garch_series: pd.DataFrame) -> None:
    rv = vf.garman_klass_rv(garch_series, window=1)
    log_hl = np.log(garch_series["high"] / garch_series["low"])
    log_co = np.log(garch_series["close"] / garch_series["open"])
    daily = (0.5 * log_hl**2 - _GK_CO_FACTOR * log_co**2).clip(lower=0.0)
    expected = np.sqrt(daily)
    assert rv.name == "rv_garman_klass"
    pd.testing.assert_series_equal(rv, expected, check_names=False, rtol=1e-12)


@pytest.mark.unit
def test_range_estimators_positive_and_finite(garch_series: pd.DataFrame) -> None:
    for fn in (vf.parkinson_rv, vf.garman_klass_rv):
        rv = fn(garch_series, window=3).dropna()
        assert (rv >= 0.0).all()
        assert np.isfinite(rv.to_numpy()).all()


@pytest.mark.unit
def test_parkinson_case_insensitive_columns(garch_series: pd.DataFrame) -> None:
    upper = garch_series.rename(columns={c: c.upper() for c in garch_series.columns})
    rv_upper = vf.parkinson_rv(upper)
    rv_lower = vf.parkinson_rv(garch_series)
    pd.testing.assert_series_equal(rv_upper, rv_lower, rtol=1e-12)


@pytest.mark.property
@settings(max_examples=30, deadline=None)
@given(rescale=st.floats(min_value=1.001, max_value=3.0, allow_nan=False))
def test_parkinson_scales_with_log_range(rescale: float) -> None:
    """Widening every bar's log range by a constant factor scales RV by it."""
    gen = vf.make_rng(11)
    n = 80
    close = 100.0 * np.exp(np.cumsum(gen.standard_normal(n) * 0.01))
    open_ = close * np.exp(gen.standard_normal(n) * 0.002)
    base_lr = np.abs(gen.standard_normal(n)) * 0.01  # half-range in log space
    high = np.maximum(open_, close) * np.exp(base_lr)
    low = np.minimum(open_, close) * np.exp(-base_lr)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    base = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

    # Multiply the (log) high-low range by ``rescale`` about each bar's midpoint.
    mid = np.sqrt(high * low)
    half = 0.5 * np.log(high / low) * rescale
    wide = base.copy()
    wide["high"] = mid * np.exp(half)
    wide["low"] = mid * np.exp(-half)

    rv_base = vf.parkinson_rv(base).dropna()
    rv_wide = vf.parkinson_rv(wide).dropna()
    np.testing.assert_allclose(
        rv_wide.to_numpy(), rescale * rv_base.to_numpy(), rtol=1e-9, atol=1e-12
    )


@pytest.mark.unit
def test_estimators_are_deterministic(garch_series: pd.DataFrame) -> None:
    for fn in (vf.parkinson_rv, vf.garman_klass_rv):
        pd.testing.assert_series_equal(fn(garch_series, window=5), fn(garch_series, window=5))
    pd.testing.assert_series_equal(
        vf.close_to_close_rv(garch_series["close"], window=5),
        vf.close_to_close_rv(garch_series["close"], window=5),
    )


@pytest.mark.unit
def test_inputs_are_not_mutated(garch_series: pd.DataFrame) -> None:
    snapshot = garch_series.copy(deep=True)
    vf.garman_klass_rv(garch_series, window=3)
    vf.parkinson_rv(garch_series, window=3)
    vf.close_to_close_rv(garch_series["close"], window=3)
    pd.testing.assert_frame_equal(garch_series, snapshot)


# --------------------------------------------------------------------------- #
# realized_volatility dispatcher                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("name", ["close_to_close", "parkinson", "garman_klass"])
def test_dispatcher_matches_direct_call(garch_series: pd.DataFrame, name: str) -> None:
    via_dispatch = vf.realized_volatility(garch_series, estimator=name, window=2)
    direct = {
        "close_to_close": lambda: vf.close_to_close_rv(garch_series["close"], window=2),
        "parkinson": lambda: vf.parkinson_rv(garch_series, window=2),
        "garman_klass": lambda: vf.garman_klass_rv(garch_series, window=2),
    }[name]()
    pd.testing.assert_series_equal(via_dispatch, direct)


@pytest.mark.unit
def test_dispatcher_rejects_unknown_estimator(garch_series: pd.DataFrame) -> None:
    with pytest.raises(ValidationError, match="unknown estimator"):
        vf.realized_volatility(garch_series, estimator="bipower")


# --------------------------------------------------------------------------- #
# forward RV target - disjointness & correctness                              #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_forward_target_value_matches_future_window(har_series: pd.Series) -> None:
    horizon, gap = 5, 1
    target = vf.forward_rv_target(har_series, horizon=horizon, gap=gap)
    var = har_series.to_numpy() ** 2
    # Pick an interior position and recompute its window by hand.
    t = 100
    window = var[t + gap + 1 : t + gap + horizon + 1]
    assert len(window) == horizon
    assert target.iloc[t] == pytest.approx(float(np.sqrt(window.mean())), rel=1e-12)


@pytest.mark.unit
def test_forward_target_tail_is_nan(har_series: pd.Series) -> None:
    horizon, gap = 5, 1
    target = vf.forward_rv_target(har_series, horizon=horizon, gap=gap)
    # The last ``horizon + gap`` rows have an incomplete future window ⇒ NaN.
    assert target.iloc[-(horizon + gap) :].isna().all()
    assert target.iloc[: len(target) - (horizon + gap)].notna().all()
    assert target.name == "rv_target"
    assert target.index.equals(har_series.index)


@pytest.mark.property
@settings(max_examples=60, deadline=None)
@given(
    horizon=st.sampled_from([1, 5, 22]),
    gap=st.integers(min_value=0, max_value=4),
    n=st.integers(min_value=60, max_value=200),
)
def test_forward_target_disjointness(horizon: int, gap: int, n: int) -> None:
    """For every non-NaN target at t the window ⊂ {> t+gap}; features ⊆ {<= t}.

    This is the leakage guard: we reconstruct the exact window the target depends
    on (by perturbing one future position at a time) and assert every dependent
    position lies strictly after ``t + gap`` and strictly before ``t+gap+h+1``.
    """
    base_vals = np.abs(vf.make_rng(n).standard_normal(n)) + 0.05
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rv = pd.Series(base_vals, index=idx)
    target = vf.forward_rv_target(rv, horizon=horizon, gap=gap)

    finite_positions = np.flatnonzero(target.notna().to_numpy())
    for t in finite_positions:
        lo = t + gap + 1
        hi = t + gap + horizon  # inclusive
        # Window must be entirely strictly-future and in-bounds.
        assert lo > t + gap
        assert hi < n
        # Perturbing any position OUTSIDE [lo, hi] must NOT change target[t];
        # perturbing a position INSIDE MUST change it.
        for j in (lo - 1, lo, hi, hi + 1 if hi + 1 < n else hi):
            bumped = rv.copy()
            bumped.iloc[j] += 10.0
            new_t = vf.forward_rv_target(bumped, horizon=horizon, gap=gap).iloc[t]
            changed = not np.isclose(new_t, target.iloc[t])
            in_window = lo <= j <= hi
            assert changed == in_window


@pytest.mark.property
@settings(max_examples=40, deadline=None)
@given(horizon=st.sampled_from([1, 5, 22]), gap=st.integers(min_value=0, max_value=3))
def test_forward_target_never_includes_present_or_past(horizon: int, gap: int) -> None:
    """Perturbing rv at positions <= t + gap leaves target[t] unchanged."""
    n = 120
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    rv = pd.Series(np.abs(vf.make_rng(7).standard_normal(n)) + 0.1, index=idx)
    target = vf.forward_rv_target(rv, horizon=horizon, gap=gap)
    t = 50  # an interior position guaranteed to have a complete window for h<=22
    if np.isnan(target.iloc[t]):
        return
    for j in range(0, t + gap + 1):  # all of {<= t + gap}
        bumped = rv.copy()
        bumped.iloc[j] += 5.0
        assert np.isclose(
            vf.forward_rv_target(bumped, horizon=horizon, gap=gap).iloc[t],
            target.iloc[t],
        )


@pytest.mark.unit
def test_forward_target_gap_zero_starts_next_day(har_series: pd.Series) -> None:
    """With gap=0, horizon=1 the target at t is exactly rv at t+1."""
    target = vf.forward_rv_target(har_series, horizon=1, gap=0)
    t = 10
    assert target.iloc[t] == pytest.approx(har_series.iloc[t + 1], rel=1e-12)


@pytest.mark.unit
def test_forward_target_determinism(har_series: pd.Series) -> None:
    a = vf.forward_rv_target(har_series, horizon=5, gap=1)
    b = vf.forward_rv_target(har_series, horizon=5, gap=1)
    pd.testing.assert_series_equal(a, b)


# --------------------------------------------------------------------------- #
# validation                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1])
def test_window_must_be_positive(garch_series: pd.DataFrame, bad: int) -> None:
    with pytest.raises(ValidationError, match="window"):
        vf.parkinson_rv(garch_series, window=bad)
    with pytest.raises(ValidationError, match="window"):
        vf.close_to_close_rv(garch_series["close"], window=bad)


@pytest.mark.unit
def test_missing_ohlc_columns_raises(garch_series: pd.DataFrame) -> None:
    no_high = garch_series.drop(columns=["high"])
    with pytest.raises(ValidationError, match="missing required column"):
        vf.parkinson_rv(no_high)


@pytest.mark.unit
def test_non_positive_prices_raise() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    bad_close = pd.Series([100.0, 101.0, -1.0, 102.0, 103.0], index=idx)
    with pytest.raises(ValidationError, match="non-positive"):
        vf.close_to_close_rv(bad_close)


@pytest.mark.unit
def test_high_below_low_raises() -> None:
    idx = pd.date_range("2020-01-01", periods=3, freq="B")
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [100.0, 101.0, 99.0],  # last bar: high < low
            "low": [99.0, 100.0, 101.0],
            "close": [99.5, 100.5, 101.5],
        },
        index=idx,
    )
    with pytest.raises(ValidationError, match="high < low"):
        vf.parkinson_rv(frame)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("horizon", "gap"),
    [(0, 1), (-1, 1), (1, -1)],
)
def test_forward_target_rejects_bad_params(har_series: pd.Series, horizon: int, gap: int) -> None:
    with pytest.raises(ValidationError):
        vf.forward_rv_target(har_series, horizon=horizon, gap=gap)


@pytest.mark.unit
def test_window_must_be_int_type(garch_series: pd.DataFrame) -> None:
    # A float (or bool) window is rejected by the type guard, not silently coerced.
    with pytest.raises(ValidationError, match="window must be an int"):
        vf.parkinson_rv(garch_series, window=2.0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="window must be an int"):
        vf.close_to_close_rv(garch_series["close"], window=True)  # type: ignore[arg-type]


@pytest.mark.unit
def test_forward_target_horizon_and_gap_must_be_int(har_series: pd.Series) -> None:
    with pytest.raises(ValidationError, match="horizon must be an int"):
        vf.forward_rv_target(har_series, horizon=5.0)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="gap must be an int"):
        vf.forward_rv_target(har_series, horizon=5, gap=1.0)  # type: ignore[arg-type]


@pytest.mark.unit
def test_forward_target_nan_inside_window_yields_nan() -> None:
    """A NaN anywhere inside t's future window makes that target NaN, not partial."""
    idx = pd.date_range("2020-01-01", periods=12, freq="B")
    rv = pd.Series(np.arange(1, 13, dtype="float64"), index=idx)
    rv.iloc[6] = np.nan  # punch a hole
    target = vf.forward_rv_target(rv, horizon=3, gap=1)
    # t=2 has window positions (3,4,5,6] → includes index 6 (NaN) ⇒ NaN target.
    assert np.isnan(target.iloc[2])
    # t=0 has window (1,2,3,4] → no NaN ⇒ finite.
    assert np.isfinite(target.iloc[0])


@pytest.mark.unit
def test_dispatcher_close_to_close_needs_only_close() -> None:
    """The close_to_close dispatch path requires only a close column."""
    idx = pd.date_range("2020-01-01", periods=6, freq="B")
    frame = pd.DataFrame({"close": [100.0, 101.0, 102.0, 101.5, 103.0, 104.0]}, index=idx)
    rv = vf.realized_volatility(frame, estimator="close_to_close", window=1)
    pd.testing.assert_series_equal(rv, vf.close_to_close_rv(frame["close"], window=1))


@pytest.mark.unit
def test_forward_target_empty_series_raises() -> None:
    with pytest.raises(ValidationError, match="non-empty"):
        vf.forward_rv_target(pd.Series([], dtype="float64"), horizon=1, gap=0)

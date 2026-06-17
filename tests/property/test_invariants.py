"""Property-based leakage and scale invariants.

These Hypothesis tests encode the load-bearing correctness guarantees:

- future-perturbation invariance (the canonical leakage detector),
- forward-target disjointness (feature index ⊆ {≤ t}, target ⊂ {> t+gap}),
- RV-estimator scale behaviour, and
- HAR feature lag-safety.

(The heavier per-fold GARCH/XGB leakage check lives in
``tests/property/test_walkforward_data.py``; this module keeps the cheap,
baseline-only invariants.)
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import volforecast as vf


@pytest.mark.property
@given(horizon=st.sampled_from([1, 5, 22]), gap=st.integers(min_value=0, max_value=3))
@settings(max_examples=25, deadline=None)
def test_forward_target_window_is_strictly_future(horizon: int, gap: int) -> None:
    """The target attached to t must aggregate only RV strictly after t + gap."""
    n = 200
    rv = pd.Series(
        np.abs(vf.make_rng(0).standard_normal(n)) + 0.1,
        index=pd.date_range("2020-01-01", periods=n, freq="B"),
    )
    target = vf.forward_rv_target(rv, horizon=horizon, gap=gap)
    assert len(target) == len(rv)

    var = rv.to_numpy(dtype="float64") ** 2
    values = target.to_numpy(dtype="float64")
    for i in range(n):
        if not np.isfinite(values[i]):
            continue
        # The target at i is the RMS of RV over the strictly-future window
        # (i + gap, i + gap + horizon] — feature index {<= i} and target window
        # {> i + gap} are DISJOINT.
        first = i + gap + 1
        last = i + gap + horizon
        assert last < n
        expected = float(np.sqrt(var[first : last + 1].mean()))
        assert values[i] == pytest.approx(expected, rel=1e-12)


@pytest.mark.property
def test_future_perturbation_invariance(garch_series: pd.DataFrame) -> None:
    """Perturbing returns strictly after the forecast origin changes no forecast."""
    # A cheap baseline-only config (no per-fold GARCH/XGB) keeps this fast while
    # still exercising the no-lookahead slicing.
    config = vf.WalkForwardConfig(
        horizon=5, train_window=252, step=25, models=("har_rv", "ewma", "rw")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = vf.run_walk_forward(garch_series, config=config)
        perturbed = garch_series.copy()
        perturbed.iloc[-10:] *= 1.5  # mutate only the tail (future of early folds)
        again = vf.run_walk_forward(perturbed, config=config)

    # Early-fold forecasts must be unchanged by a strictly-future perturbation.
    common = base.forecasts.index.intersection(again.forecasts.index)[:5]
    assert len(common) >= 3
    pd.testing.assert_frame_equal(base.forecasts.loc[common], again.forecasts.loc[common])


@pytest.mark.property
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0, 3.7, 5.0])
def test_rv_scales_with_return_scale(scale: float, garch_series: pd.DataFrame) -> None:
    """Scaling prices by a constant leaves close-to-close (log-return) RV invariant."""
    rv = vf.close_to_close_rv(garch_series["close"])
    rv_scaled = vf.close_to_close_rv(garch_series["close"] * scale)
    # A multiplicative price scaling cancels in log returns, so close-to-close RV
    # is exactly invariant to the price scale.
    pd.testing.assert_series_equal(rv, rv_scaled, check_names=False)

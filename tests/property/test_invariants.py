"""Property-based leakage and scale invariants (filled in as kernels land).

These Hypothesis tests encode the load-bearing correctness guarantees:

- future-perturbation invariance (the canonical leakage detector),
- forward-target disjointness (feature index ⊆ {≤ t}, target ⊂ {> t+gap}),
- RV-estimator scale behaviour, and
- HAR feature lag-safety.

They are marked ``xfail`` until the kernels exist so the partition collects and
the intended invariants are explicit and reviewable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

import volforecast as vf


@pytest.mark.property
@pytest.mark.xfail(reason="forward_rv_target not yet implemented", strict=True)
@given(horizon=st.sampled_from([1, 5, 22]), gap=st.integers(min_value=0, max_value=3))
def test_forward_target_window_is_strictly_future(horizon: int, gap: int) -> None:
    """The target attached to t must aggregate only RV strictly after t + gap."""
    rv = pd.Series(
        np.abs(vf.make_rng(0).standard_normal(200)) + 0.1,
        index=pd.date_range("2020-01-01", periods=200, freq="B"),
    )
    target = vf.forward_rv_target(rv, horizon=horizon, gap=gap)
    # For a non-NaN target at position i, it must depend only on rv positions
    # > i + gap (asserted concretely once implemented).
    assert len(target) == len(rv)


@pytest.mark.property
@pytest.mark.xfail(reason="run_walk_forward not yet implemented", strict=True)
def test_future_perturbation_invariance(garch_series: pd.DataFrame) -> None:
    """Perturbing returns strictly after the forecast origin changes no forecast."""
    config = vf.WalkForwardConfig(horizon=5, train_window=252)
    base = vf.run_walk_forward(garch_series, config=config)
    perturbed = garch_series.copy()
    perturbed.iloc[-10:] *= 1.5  # mutate only the tail (future of early folds)
    again = vf.run_walk_forward(perturbed, config=config)
    # Early-fold forecasts must be unchanged by a strictly-future perturbation.
    common = base.forecasts.index.intersection(again.forecasts.index)[:5]
    pd.testing.assert_frame_equal(
        base.forecasts.loc[common], again.forecasts.loc[common]
    )


@pytest.mark.property
@pytest.mark.xfail(reason="realized_volatility not yet implemented", strict=True)
@given(scale=st.floats(min_value=0.5, max_value=5.0))
def test_rv_scales_with_return_scale(scale: float, garch_series: pd.DataFrame) -> None:
    """Scaling prices by a constant scales close-to-close RV by the same factor."""
    rv = vf.close_to_close_rv(garch_series["close"])
    rv_scaled = vf.close_to_close_rv(garch_series["close"] * scale)
    # Multiplicative price scaling leaves log-return RV invariant (a property the
    # implementation must honour); pinned concretely once implemented.
    assert len(rv) == len(rv_scaled)

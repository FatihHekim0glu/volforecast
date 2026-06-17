"""Parity oracles (filled in as the kernels land).

The headline parity check pins the hand-rolled GARCH(1,1) log-likelihood against
``arch`` at the same parameters to a tight tolerance (e.g. 1e-6), and pins
XGBoost determinism under a fixed seed. These are marked ``xfail`` until the
kernels are implemented so the partition collects and the intent is explicit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import volforecast as vf


@pytest.mark.parity
@pytest.mark.xfail(reason="garch_11_log_likelihood not yet implemented", strict=True)
def test_handrolled_garch_ll_matches_arch(garch_series: pd.DataFrame) -> None:
    """Hand-rolled GARCH(1,1) LL must match arch's at the same params (tol 1e-6)."""
    rets = np.log(garch_series["close"]).diff().dropna().to_numpy() * 100.0
    ll = vf.garch_11_log_likelihood(rets, omega=0.02, alpha=0.08, beta=0.90)
    assert np.isfinite(ll)
    # Once arch parity is wired: compare to arch's loglikelihood at the same params
    # to within 1e-6 in absolute log-likelihood.


@pytest.mark.parity
@pytest.mark.xfail(reason="fit_xgb not yet implemented", strict=True)
def test_xgb_is_deterministic_under_fixed_seed(har_series: pd.Series) -> None:
    """Two fits with the same (features, target, seed) give identical predictions."""
    feats = pd.DataFrame({"rv_daily": har_series.shift(1)}).dropna()
    target = har_series.reindex(feats.index)
    m1 = vf.fit_xgb(feats, target, seed=7)
    m2 = vf.fit_xgb(feats, target, seed=7)
    pd.testing.assert_series_equal(m1.predict(feats), m2.predict(feats))

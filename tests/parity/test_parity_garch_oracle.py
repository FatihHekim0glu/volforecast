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
def test_handrolled_garch_ll_finite(garch_series: pd.DataFrame) -> None:
    """Hand-rolled GARCH(1,1) LL is finite on the GARCH fixture."""
    rets = np.log(garch_series["close"]).diff().dropna().to_numpy() * 100.0
    ll = vf.garch_11_log_likelihood(rets, omega=0.02, alpha=0.08, beta=0.90)
    assert np.isfinite(ll)


@pytest.mark.parity
def test_handrolled_garch_ll_matches_arch(garch_series: pd.DataFrame) -> None:
    """Hand-rolled GARCH(1,1) LL must match arch's at the SAME params (tol 1e-6).

    We fit a zero-mean Gaussian GARCH(1,1) with ``arch`` (so the model has no mean
    term to subtract), then evaluate both ``arch``'s reported log-likelihood and
    the hand-rolled oracle at the fitted ``(omega, alpha, beta)``. They share the
    same backcast seed and recursion by construction, so they agree to ~1e-6.
    """
    arch_model = pytest.importorskip("arch").arch_model
    rets = np.log(garch_series["close"]).diff().dropna().to_numpy() * 100.0

    res = arch_model(rets, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
        disp="off", show_warning=False
    )
    omega = float(res.params["omega"])
    alpha = float(res.params["alpha[1]"])
    beta = float(res.params["beta[1]"])

    ll = vf.garch_11_log_likelihood(rets, omega=omega, alpha=alpha, beta=beta)
    assert ll == pytest.approx(float(res.loglikelihood), abs=1e-6)


@pytest.mark.parity
def test_xgb_is_deterministic_under_fixed_seed(har_series: pd.Series) -> None:
    """Two fits with the same (features, target, seed) give identical predictions."""
    feats = pd.DataFrame({"rv_daily": har_series.shift(1)}).dropna()
    target = har_series.reindex(feats.index)
    m1 = vf.fit_xgb(feats, target, seed=7)
    m2 = vf.fit_xgb(feats, target, seed=7)
    pd.testing.assert_series_equal(m1.predict(feats), m2.predict(feats))

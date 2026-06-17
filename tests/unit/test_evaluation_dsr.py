"""DSR / PSR guards for the optional vol-targeting overlay (evaluation group).

The Deflated Sharpe Ratio is reused verbatim from
:mod:`volforecast.evaluation.dsr`; the overlay must pass ``n_trials = #model
configs evaluated`` so that selecting the best config does not look better than
it is. These tests pin the honest behaviour the overlay relies on:

- the DSR is **monotonically non-increasing** in ``n_trials`` (more snooping ⇒
  a harder bar), so it can never reward a larger search grid;
- a single trial (``n_trials == 1``) reduces the DSR to the plain PSR vs zero;
- the ``n_trials`` / ``n_obs`` / variance guards reject nonsensical inputs.
"""

from __future__ import annotations

from itertools import pairwise

import pytest

from volforecast._exceptions import ValidationError
from volforecast.evaluation.dsr import (
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
)


@pytest.mark.unit
def test_dsr_is_non_increasing_in_n_trials() -> None:
    """More configurations tried (n_trials) can only lower the deflated Sharpe."""
    kwargs = {"n_obs": 1000, "variance_of_trial_sharpes": 0.01}
    values = [
        deflated_sharpe_ratio(0.10, n_trials=n, **kwargs)  # type: ignore[arg-type]
        for n in (1, 5, 20, 100, 500)
    ]
    for earlier, later in pairwise(values):
        assert later <= earlier + 1e-12


@pytest.mark.unit
def test_dsr_single_trial_equals_psr_vs_zero() -> None:
    """With one trial the DSR collapses to PSR against a zero benchmark."""
    dsr = deflated_sharpe_ratio(0.08, n_obs=750, n_trials=1, variance_of_trial_sharpes=0.02)
    psr = probabilistic_sharpe_ratio(0.08, n_obs=750, benchmark_sharpe=0.0)
    assert dsr == pytest.approx(psr, abs=1e-12)


@pytest.mark.unit
def test_dsr_zero_trial_variance_reduces_to_psr() -> None:
    """No dispersion across trials ⇒ no inflation, regardless of n_trials."""
    dsr = deflated_sharpe_ratio(0.05, n_obs=500, n_trials=50, variance_of_trial_sharpes=0.0)
    psr = probabilistic_sharpe_ratio(0.05, n_obs=500, benchmark_sharpe=0.0)
    assert dsr == pytest.approx(psr, abs=1e-12)


@pytest.mark.unit
def test_dsr_guards_reject_bad_n_trials_and_inputs() -> None:
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=100, n_trials=0, variance_of_trial_sharpes=0.01)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=1, n_trials=5, variance_of_trial_sharpes=0.01)
    with pytest.raises(ValidationError):
        deflated_sharpe_ratio(0.1, n_obs=100, n_trials=5, variance_of_trial_sharpes=-0.01)


@pytest.mark.unit
def test_psr_uses_full_kurtosis_term() -> None:
    """A Gaussian (kurtosis=3) is the calibration point for the PSR bracket."""
    # A clearly positive Sharpe over a long sample is highly probable.
    psr = probabilistic_sharpe_ratio(0.20, n_obs=2000, skew=0.0, kurtosis=3.0)
    assert 0.0 <= psr <= 1.0
    assert psr > 0.99

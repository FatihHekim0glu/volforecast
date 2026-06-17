"""Unit tests for the reused infra (rng, manifest, validation, constants, costs).

These exercise the copied-and-renamed HRP infrastructure that is fully
implemented (not stubbed), so the scaffold has real, passing coverage on its
foundation while the domain kernels are filled in.
"""

from __future__ import annotations

import numpy as np
import pytest

from volforecast import (
    EPS,
    RISKMETRICS_LAMBDA,
    SUPPORTED_HORIZONS,
    FixedBpsCost,
    RunManifest,
    config_hash,
    deflated_sharpe_ratio,
    ensure_series,
    make_rng,
    probabilistic_sharpe_ratio,
    spawn_substreams,
)
from volforecast._exceptions import ValidationError


@pytest.mark.unit
def test_make_rng_is_deterministic() -> None:
    a = make_rng(7).standard_normal(16)
    b = make_rng(7).standard_normal(16)
    np.testing.assert_array_equal(a, b)


@pytest.mark.unit
def test_make_rng_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_rng(-1)


@pytest.mark.unit
def test_spawn_substreams_are_independent() -> None:
    subs = spawn_substreams(7, 3)
    assert len(subs) == 3
    draws = [s.standard_normal(8) for s in subs]
    # Distinct substreams produce distinct draws.
    assert not np.array_equal(draws[0], draws[1])
    assert not np.array_equal(draws[1], draws[2])


@pytest.mark.unit
def test_config_hash_is_order_invariant() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


@pytest.mark.unit
def test_run_manifest_round_trips() -> None:
    man = RunManifest.capture({"horizon": 5}, seed=7)
    d = man.to_dict()
    assert d["seed"] == 7
    assert set(d) >= {"git_sha", "dirty", "config_hash", "seed"}


@pytest.mark.unit
def test_ensure_series_rejects_nan() -> None:
    with pytest.raises(ValidationError, match="NaN"):
        ensure_series([1.0, float("nan"), 3.0])


@pytest.mark.unit
def test_fixed_bps_cost_is_linear_in_turnover() -> None:
    cost = FixedBpsCost(bps=10.0)
    assert cost.cost(0.0) == 0.0
    assert cost.cost(1.0) == pytest.approx(10.0 / 10_000.0)
    assert cost.cost(2.0) == pytest.approx(2.0 * 10.0 / 10_000.0)


@pytest.mark.unit
def test_fixed_bps_cost_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        FixedBpsCost(bps=-1.0)


@pytest.mark.unit
def test_psr_and_dsr_are_probabilities() -> None:
    psr = probabilistic_sharpe_ratio(0.1, n_obs=250)
    assert 0.0 <= psr <= 1.0
    dsr = deflated_sharpe_ratio(
        0.1, n_obs=250, n_trials=10, variance_of_trial_sharpes=0.01
    )
    assert 0.0 <= dsr <= 1.0


@pytest.mark.unit
def test_dsr_is_non_increasing_in_n_trials() -> None:
    kwargs = {"n_obs": 250, "variance_of_trial_sharpes": 0.01}
    low = deflated_sharpe_ratio(0.2, n_trials=2, **kwargs)
    high = deflated_sharpe_ratio(0.2, n_trials=100, **kwargs)
    assert high <= low


@pytest.mark.unit
def test_domain_constants() -> None:
    assert RISKMETRICS_LAMBDA == 0.94
    assert SUPPORTED_HORIZONS == (1, 5, 22)
    assert 0.0 < EPS < 1e-6

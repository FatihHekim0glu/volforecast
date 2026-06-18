"""GARCH-family model tests: oracle parity, fit-on-train, and forecasting.

These cover the ``garch/models.py`` group end to end:

- The hand-rolled GARCH(1,1) log-likelihood matches ``arch`` at shared params to
  a pinned tolerance (the parity oracle) and respects an explicit backcast.
- ``fit_garch`` fits GARCH / EGARCH / GJR with normal and Student-t innovations
  on a TRAIN slice only and reports finite, well-formed results.
- ``forecast_garch_vol`` returns a strictly positive, finite h-ahead RV forecast
  on the unscaled return scale for every supported horizon and kind.
- Input validation rejects bad kinds/dists/horizons/degenerate series, and the
  fit never touches the held-out fold (fit-on-train-only).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import volforecast as vf
from volforecast._exceptions import (
    InsufficientDataError,
    ValidationError,
)
from volforecast.garch.models import (
    GARCH_KINDS,
    GARCHFit,
    _backcast,
    fit_garch,
    forecast_garch_vol,
    garch_11_log_likelihood,
)


def _log_returns(close: pd.Series) -> pd.Series:
    """Daily log returns of a close-price series (NaN dropped)."""
    return np.log(close).diff().dropna()


# --------------------------------------------------------------------------- #
# Hand-rolled GARCH(1,1) log-likelihood - the parity oracle                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
def test_oracle_matches_arch_zero_mean(garch_series: pd.DataFrame) -> None:
    """Oracle LL == arch LL at the same params for a zero-mean fit (tol 1e-6)."""
    arch_model = pytest.importorskip("arch").arch_model
    rets = _log_returns(garch_series["close"]).to_numpy() * 100.0
    res = arch_model(rets, mean="Zero", vol="GARCH", p=1, q=1, dist="normal", rescale=False).fit(
        disp="off", show_warning=False
    )
    ll = garch_11_log_likelihood(
        rets,
        omega=float(res.params["omega"]),
        alpha=float(res.params["alpha[1]"]),
        beta=float(res.params["beta[1]"]),
    )
    assert ll == pytest.approx(float(res.loglikelihood), abs=1e-6)


@pytest.mark.parity
def test_oracle_explicit_backcast_path(garch_series: pd.DataFrame) -> None:
    """An explicit backcast is honored and differs from the default seed path."""
    rets = _log_returns(garch_series["close"]).to_numpy() * 100.0
    default_ll = garch_11_log_likelihood(rets, omega=0.02, alpha=0.08, beta=0.90)
    explicit_ll = garch_11_log_likelihood(
        rets, omega=0.02, alpha=0.08, beta=0.90, backcast=rets.var()
    )
    assert np.isfinite(default_ll)
    assert np.isfinite(explicit_ll)
    assert default_ll != explicit_ll


@pytest.mark.parity
def test_backcast_is_arch_compatible_ewma() -> None:
    """The internal backcast equals arch's 0.94-EWMA of squared returns."""
    rng = np.random.default_rng(0)
    rets = rng.standard_normal(200).astype("float64")
    tau = min(75, rets.shape[0])
    w = 0.94 ** np.arange(tau)
    w = w / w.sum()
    expected = float(np.sum((rets[:tau] ** 2) * w))
    assert _backcast(rets) == pytest.approx(expected, rel=0, abs=1e-15)


@pytest.mark.parity
@pytest.mark.parametrize(
    ("kwargs", "exc"),
    [
        ({"omega": 0.0, "alpha": 0.1, "beta": 0.8}, ValidationError),
        ({"omega": -1.0, "alpha": 0.1, "beta": 0.8}, ValidationError),
        ({"omega": 0.1, "alpha": -0.1, "beta": 0.8}, ValidationError),
        ({"omega": 0.1, "alpha": 0.1, "beta": -0.8}, ValidationError),
        ({"omega": np.inf, "alpha": 0.1, "beta": 0.8}, ValidationError),
    ],
)
def test_oracle_rejects_bad_params(kwargs: dict[str, float], exc: type[Exception]) -> None:
    """Out-of-domain GARCH parameters raise ValidationError."""
    rets = np.random.default_rng(1).standard_normal(100).astype("float64")
    with pytest.raises(exc):
        garch_11_log_likelihood(rets, **kwargs)


@pytest.mark.parity
def test_oracle_rejects_empty_and_nonfinite() -> None:
    """Empty and non-finite return arrays raise ValidationError."""
    with pytest.raises(ValidationError):
        garch_11_log_likelihood(np.array([], dtype="float64"), 0.1, 0.1, 0.8)
    bad = np.array([0.1, np.nan, 0.2], dtype="float64")
    with pytest.raises(ValidationError):
        garch_11_log_likelihood(bad, 0.1, 0.1, 0.8)


@pytest.mark.parity
def test_oracle_rejects_bad_backcast() -> None:
    """A non-positive explicit backcast raises ValidationError."""
    rets = np.random.default_rng(2).standard_normal(50).astype("float64")
    with pytest.raises(ValidationError):
        garch_11_log_likelihood(rets, 0.1, 0.1, 0.8, backcast=0.0)


@pytest.mark.parity
def test_oracle_exposed_on_top_level_api(garch_series: pd.DataFrame) -> None:
    """The oracle is reachable via the curated package API."""
    rets = _log_returns(garch_series["close"]).to_numpy() * 100.0
    assert np.isfinite(vf.garch_11_log_likelihood(rets, 0.02, 0.08, 0.90))


# --------------------------------------------------------------------------- #
# fit_garch - fit-on-train-only across the family                              #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@pytest.mark.parametrize("kind", GARCH_KINDS)
@pytest.mark.parametrize("dist", ["normal", "t"])
def test_fit_garch_family(garch_series: pd.DataFrame, kind: str, dist: str) -> None:
    """Every (kind, dist) fits and yields a finite LL and a populated arch result."""
    rets = _log_returns(garch_series["close"])
    fit = fit_garch(rets, kind=kind, dist=dist)
    assert isinstance(fit, GARCHFit)
    assert fit.kind == kind
    assert fit.dist == dist
    assert fit.n_train == len(rets)
    assert np.isfinite(fit.loglikelihood)
    assert fit.params  # non-empty mapping
    assert "_arch_result" in fit.meta


@pytest.mark.parity
def test_fit_garch_to_dict_drops_private_meta(garch_series: pd.DataFrame) -> None:
    """``to_dict`` is JSON-clean: no private ``_arch_result`` leaks into it."""
    import json

    fit = fit_garch(_log_returns(garch_series["close"]))
    payload = fit.to_dict()
    assert "_arch_result" not in payload["meta"]
    assert payload["meta"]["scale"] == 100.0
    # Round-trips through JSON without error.
    assert json.loads(json.dumps(payload))["kind"] == "garch"


@pytest.mark.parity
def test_fit_on_train_only_ignores_future(garch_series: pd.DataFrame) -> None:
    """Mutating the held-out (future) fold does not change the train fit at all."""
    rets = _log_returns(garch_series["close"])
    split = 1000
    train = rets.iloc[:split]
    fit_a = fit_garch(train)

    # Corrupt the future fold wildly; the train fit must be byte-identical.
    poisoned = rets.copy()
    poisoned.iloc[split:] *= 100.0
    fit_b = fit_garch(poisoned.iloc[:split])

    assert fit_a.loglikelihood == pytest.approx(fit_b.loglikelihood, abs=1e-9)
    assert fit_a.params == pytest.approx(fit_b.params)


@pytest.mark.parity
def test_fit_garch_rejects_bad_kind_and_dist(garch_series: pd.DataFrame) -> None:
    """Unsupported selectors raise ValidationError before any arch import."""
    rets = _log_returns(garch_series["close"])
    with pytest.raises(ValidationError):
        fit_garch(rets, kind="figarch")
    with pytest.raises(ValidationError):
        fit_garch(rets, dist="cauchy")


@pytest.mark.parity
def test_fit_garch_rejects_bad_scale_and_short_series() -> None:
    """A non-positive scale and an under-length train fold are rejected."""
    rets = pd.Series(np.random.default_rng(3).standard_normal(200))
    with pytest.raises(ValidationError):
        fit_garch(rets, scale=0.0)
    with pytest.raises(InsufficientDataError):
        fit_garch(rets.iloc[:10])


@pytest.mark.parity
def test_fit_garch_rejects_zero_variance() -> None:
    """A constant return series is not GARCH-identified and is rejected."""
    flat = pd.Series(np.zeros(120, dtype="float64"))
    with pytest.raises(ValidationError):
        fit_garch(flat)


# --------------------------------------------------------------------------- #
# forecast_garch_vol - positive, finite, unscaled, every horizon and kind      #
# --------------------------------------------------------------------------- #
@pytest.mark.parity
@pytest.mark.parametrize("kind", GARCH_KINDS)
@pytest.mark.parametrize("horizon", [1, 5, 22])
def test_forecast_positive_and_finite(garch_series: pd.DataFrame, kind: str, horizon: int) -> None:
    """h-ahead RV forecasts are strictly positive and finite for all kinds."""
    rets = _log_returns(garch_series["close"])
    fit = fit_garch(rets, kind=kind)
    rv = forecast_garch_vol(fit, rets, horizon=horizon)
    assert np.isfinite(rv)
    assert rv > 0.0


@pytest.mark.parity
def test_forecast_scales_with_horizon(garch_series: pd.DataFrame) -> None:
    """Longer windows accumulate more variance, so RV(22) > RV(1)."""
    rets = _log_returns(garch_series["close"])
    fit = fit_garch(rets, kind="garch")
    rv1 = forecast_garch_vol(fit, rets, horizon=1)
    rv22 = forecast_garch_vol(fit, rets, horizon=22)
    assert rv22 > rv1


@pytest.mark.parity
def test_forecast_matches_arch_analytic(garch_series: pd.DataFrame) -> None:
    """The aggregated forecast equals arch's analytic path, unscaled (tol 1e-9)."""
    rets = _log_returns(garch_series["close"])
    fit = fit_garch(rets, kind="garch")
    res = fit.meta["_arch_result"]
    fc = res.forecast(horizon=5, reindex=False, method="analytic")
    expected = float(np.sqrt(fc.variance.to_numpy().sum())) / 100.0
    assert forecast_garch_vol(fit, rets, horizon=5) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parity
def test_forecast_rejects_bad_horizon(garch_series: pd.DataFrame) -> None:
    """A horizon below 1 raises ValidationError."""
    rets = _log_returns(garch_series["close"])
    fit = fit_garch(rets)
    with pytest.raises(ValidationError):
        forecast_garch_vol(fit, rets, horizon=0)


@pytest.mark.parity
def test_forecast_requires_arch_result() -> None:
    """A GARCHFit without its arch result cannot forecast (clear error)."""
    fit = GARCHFit(
        kind="garch",
        params={"omega": 0.1, "alpha[1]": 0.1, "beta[1]": 0.8},
        loglikelihood=-1.0,
        dist="normal",
        n_train=100,
        meta={"scale": 100.0},
    )
    with pytest.raises(ValidationError):
        forecast_garch_vol(fit, pd.Series([0.01, -0.02, 0.0]), horizon=1)

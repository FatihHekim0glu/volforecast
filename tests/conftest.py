"""Shared, seeded test fixtures.

Every fixture is deterministic (driven by :func:`volforecast._rng.make_rng`) and
returns pandas objects, so tests across the suite share identical synthetic data
with known structure:

- ``garch_series`` — a GARCH(1,1)-like OHLC frame with realistic volatility
  clustering (the honest-null default; GARCH is the true model here, so ML must
  not reliably beat it).
- ``har_series`` — a realized-volatility series with HAR-style daily/weekly/
  monthly persistence, for exercising the HAR feature builder and baseline.
- ``pure_noise`` — i.i.d. Gaussian returns with constant volatility (the no-ARCH
  null: there is no volatility structure to forecast).

These fixtures are built self-contained (not via the library generator under
test) so they remain valid reference inputs while the kernels are implemented.
Importing this module has no side effects beyond fixture registration.
"""

from __future__ import annotations

import os

# Pin BLAS / OpenMP / XGBoost thread pools to a single thread BEFORE numpy (and,
# transitively, arch/scipy/xgboost) import. The per-fold GARCH/XGBoost fits in
# the walk-forward suite otherwise oversubscribe every core (hundreds of % CPU)
# and run far slower than a single-threaded fit on these short folds — pinning
# here keeps the suite both fast and bit-reproducible.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "XGBOOST_NTHREAD",
):
    os.environ.setdefault(_var, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from volforecast._rng import make_rng  # noqa: E402

_SEED = 20260617


def _business_index(n: int, start: str = "2016-01-04") -> pd.DatetimeIndex:
    """Return an ``n``-length business-day index anchored at ``start``."""
    return pd.date_range(start=start, periods=n, freq="B")


def _garch_returns(
    n: int,
    gen: np.random.Generator,
    *,
    omega: float = 2.0e-6,
    alpha: float = 0.08,
    beta: float = 0.90,
    mu: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Simulate GARCH(1,1) log returns and the conditional vol path.

    Returns ``(returns, sigma)`` where ``sigma`` is the per-day conditional
    volatility used to synthesize a coherent intraday range in ``garch_series``.
    """
    eps = gen.standard_normal(n)
    sigma2 = np.empty(n, dtype="float64")
    returns = np.empty(n, dtype="float64")
    sigma2[0] = omega / max(1.0 - alpha - beta, 1e-6)  # unconditional variance
    returns[0] = mu + np.sqrt(sigma2[0]) * eps[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * (returns[t - 1] - mu) ** 2 + beta * sigma2[t - 1]
        returns[t] = mu + np.sqrt(sigma2[t]) * eps[t]
    return returns, np.sqrt(sigma2)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded PCG64 generator shared by tests that need raw randomness."""
    return make_rng(_SEED)


@pytest.fixture
def garch_series() -> pd.DataFrame:
    """A GARCH(1,1)-like OHLC frame with volatility clustering.

    Shape ``(1500, 4)`` with columns ``open, high, low, close`` on a business-day
    index. Close prices follow a GARCH(1,1) log-return path; the intraday range is
    drawn consistently with each day's conditional volatility, so Parkinson /
    Garman-Klass RV is well-defined and ``high >= max(open, close)`` and
    ``low <= min(open, close)`` hold by construction.
    """
    gen = make_rng(_SEED)
    n = 1500
    returns, sigma = _garch_returns(n, gen)

    close = 100.0 * np.exp(np.cumsum(returns))
    prev_close = np.empty(n, dtype="float64")
    prev_close[0] = 100.0
    prev_close[1:] = close[:-1]
    # Open near the previous close with a small overnight gap.
    open_ = prev_close * np.exp(0.25 * sigma * gen.standard_normal(n))

    hi_lo_base = np.maximum(open_, close)
    lo_hi_base = np.minimum(open_, close)
    # Intraday excursions scaled by the day's conditional vol (non-negative).
    up = np.abs(gen.standard_normal(n)) * sigma
    down = np.abs(gen.standard_normal(n)) * sigma
    high = hi_lo_base * np.exp(up)
    low = lo_hi_base * np.exp(-down)

    frame = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=_business_index(n),
    ).astype("float64")
    return frame


@pytest.fixture
def har_series() -> pd.Series:
    """A realized-volatility series with HAR-style multi-scale persistence.

    Shape ``(1200,)`` on a business-day index. The log-RV process mixes a slow
    monthly component, a faster weekly component, and daily noise, so the trailing
    daily/weekly/monthly HAR averages are genuinely informative (the structure
    HAR-RV and XGBoost are meant to pick up). Values are strictly positive.
    """
    gen = make_rng(_SEED + 1)
    n = 1200
    monthly = np.cumsum(gen.standard_normal(n) * 0.01)
    weekly = pd.Series(gen.standard_normal(n) * 0.05).rolling(5, min_periods=1).mean().to_numpy()
    daily = gen.standard_normal(n) * 0.10
    log_rv = -4.0 + 0.6 * monthly + 0.3 * weekly + daily
    rv = np.exp(log_rv)
    return pd.Series(rv, index=_business_index(n), name="rv", dtype="float64")


@pytest.fixture
def pure_noise() -> pd.Series:
    """An i.i.d. Gaussian return series with constant volatility (no-ARCH null).

    Shape ``(1500,)`` on a business-day index. There is no volatility clustering,
    so no model should be able to forecast volatility better than the constant
    baseline — the null where even GARCH has nothing to find.
    """
    gen = make_rng(_SEED + 2)
    n = 1500
    returns = gen.standard_normal(n) * 0.01
    return pd.Series(returns, index=_business_index(n), name="returns", dtype="float64")

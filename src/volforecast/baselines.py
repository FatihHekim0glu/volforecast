"""Baseline volatility forecasters — the real bars GARCH/ML must clear.

Three honest baselines, in increasing sophistication:

- **random-walk vol** (``rw``): the naive ``RV_t`` carried forward as the forecast
  of ``RV_{t+h}`` (a martingale in RV);
- **EWMA / RiskMetrics** (``ewma``): the exponentially-weighted variance with the
  RiskMetrics decay ``lambda = 0.94``;
- **HAR-RV** (``har_rv``, Corsi 2009): an OLS regression of forward RV on the
  lagged daily/weekly/monthly RV components — the hardest baseline to beat.

Each forecaster is FIT on a train fold only (HAR-RV's OLS coefficients are
estimated in-sample; EWMA seeds its variance from the train history) and then
produces a one-shot forecast for the test fold. Importing this module has no
side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class HARRVModel:
    """A fitted HAR-RV (Corsi 2009) linear model.

    Attributes
    ----------
    intercept:
        The fitted OLS intercept.
    beta_daily, beta_weekly, beta_monthly:
        The fitted coefficients on the lagged daily/weekly/monthly RV components.
    n_train:
        The number of in-sample observations the model was fit on.
    """

    intercept: float
    beta_daily: float
    beta_weekly: float
    beta_monthly: float
    n_train: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of the fitted coefficients."""
        return {
            "intercept": float(self.intercept),
            "beta_daily": float(self.beta_daily),
            "beta_weekly": float(self.beta_weekly),
            "beta_monthly": float(self.beta_monthly),
            "n_train": int(self.n_train),
        }

    def predict(self, har_components: pd.DataFrame) -> pd.Series:
        """Forecast forward RV from a frame of lagged HAR components.

        Parameters
        ----------
        har_components:
            A frame with ``rv_daily``, ``rv_weekly``, ``rv_monthly`` columns
            (already lagged), as produced by
            :func:`volforecast.features.har.har_components`.

        Returns
        -------
        pandas.Series
            The HAR-RV point forecast aligned to ``har_components.index``.

        Raises
        ------
        ValidationError
            If required columns are missing.
        """
        raise NotImplementedError


def random_walk_vol_forecast(rv: pd.Series, *, horizon: int) -> pd.Series:
    r"""Random-walk volatility forecast: ``RV_t`` predicts ``RV_{t+h}``.

    The martingale-in-RV baseline carries today's realized volatility forward
    unchanged as the forecast of the ``horizon``-day-ahead RV. There is nothing
    to fit; the forecast is simply ``rv`` shifted so it is observable at the
    forecast origin.

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.
    horizon:
        The forecast horizon in trading days (``>= 1``), used only for
        validation/labelling (the RW forecast is horizon-invariant in level).

    Returns
    -------
    pandas.Series
        The random-walk forecast aligned to ``rv.index``.

    Raises
    ------
    ValidationError
        If ``rv`` is empty or ``horizon < 1``.
    """
    raise NotImplementedError


def ewma_vol_forecast(
    returns: pd.Series,
    *,
    horizon: int,
    lam: float = 0.94,
) -> pd.Series:
    r"""EWMA / RiskMetrics volatility forecast (decay ``lam``).

    Runs the RiskMetrics variance recursion

    .. math::

        \sigma^2_t = (1 - \lambda)\, r_{t-1}^2 + \lambda\, \sigma^2_{t-1},

    seeded from the sample variance of the training history, and reports
    ``sqrt(sigma2_t)`` scaled to the ``horizon`` (square-root-of-time). The
    recursion uses only past squared returns, so the forecast at ``t`` is
    observable at ``t``.

    Parameters
    ----------
    returns:
        A per-day (log) return series indexed by date.
    horizon:
        The forecast horizon in trading days (``>= 1``); the EWMA daily vol is
        scaled by ``sqrt(horizon)``.
    lam:
        The EWMA decay in ``(0, 1)`` (RiskMetrics default ``0.94``).

    Returns
    -------
    pandas.Series
        The EWMA volatility forecast aligned to ``returns.index``.

    Raises
    ------
    ValidationError
        If ``returns`` is empty, ``horizon < 1``, or ``lam`` is outside ``(0, 1)``.
    """
    raise NotImplementedError


def fit_har_rv(har_components: pd.DataFrame, target: pd.Series) -> HARRVModel:
    """Fit a HAR-RV model by OLS on a TRAIN fold only.

    Regresses the forward RV ``target`` on the lagged daily/weekly/monthly RV
    components via ordinary least squares. The fit is in-sample only; the
    walk-forward engine refits this on every train fold, never on the full
    series.

    Parameters
    ----------
    har_components:
        A frame with ``rv_daily``, ``rv_weekly``, ``rv_monthly`` columns
        (already lagged), aligned to ``target``.
    target:
        The aligned forward RV target for the train fold.

    Returns
    -------
    HARRVModel
        The fitted, frozen HAR-RV model.

    Raises
    ------
    ValidationError
        If the inputs are misaligned, contain NaN, or required columns are
        missing.
    InsufficientDataError
        If there are fewer observations than free coefficients (rank-deficient
        design).
    """
    raise NotImplementedError

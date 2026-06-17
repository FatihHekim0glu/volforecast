"""Realized-volatility estimators and the forward RV-window target builder.

This module computes daily realized-volatility (RV) proxies from an OHLC frame
(Parkinson and Garman-Klass range estimators) and from a close-to-close return
series, plus the *forward* RV target used by every model: the realized volatility
over a strictly future window ``(t + gap, t + gap + h]``. The explicit ``gap``
between the feature timestamp and the target window is the leakage guard — a
feature observed at ``t`` may never see a return inside its own target window.

All estimators are scale-aware (RV scales linearly with the return scale) and
contain no fit/network/randomness. Importing this module has no side effects.
"""

from __future__ import annotations

import pandas as pd

#: The OHLC columns every range-based estimator requires (lower-cased on input).
OHLC_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")


def close_to_close_rv(close: pd.Series, *, window: int = 1) -> pd.Series:
    r"""Close-to-close realized volatility over a trailing ``window``.

    Computes log returns ``r_t = ln(C_t / C_{t-1})`` and returns the rolling
    root-mean-square return over ``window`` observations (the simplest RV proxy):

    .. math::

        \mathrm{RV}^{cc}_t = \sqrt{\frac{1}{\text{window}}
                              \sum_{i=0}^{\text{window}-1} r_{t-i}^2}.

    With ``window=1`` this is ``|r_t|`` (absolute log return), the squared form of
    which is the canonical noisy one-day variance proxy.

    Parameters
    ----------
    close:
        A 1-D series of close prices indexed by date.
    window:
        Number of trailing observations to aggregate (``>= 1``).

    Returns
    -------
    pandas.Series
        The close-to-close RV series, named ``"rv_cc"``, NaN over the warm-up.

    Raises
    ------
    ValidationError
        If ``close`` is empty, contains non-positive prices, or ``window < 1``.
    """
    raise NotImplementedError


def parkinson_rv(ohlc: pd.DataFrame, *, window: int = 1) -> pd.Series:
    r"""Parkinson (1980) high-low range realized volatility.

    The Parkinson estimator uses the daily high-low range and is ~5x more
    efficient than close-to-close under a zero-drift GBM:

    .. math::

        \mathrm{RV}^{park}_t = \sqrt{\frac{1}{4\ln 2}\,
                               \frac{1}{\text{window}}
                               \sum_{i=0}^{\text{window}-1}
                               \left(\ln \frac{H_{t-i}}{L_{t-i}}\right)^2}.

    Parameters
    ----------
    ohlc:
        A frame with ``high`` and ``low`` columns (case-insensitive) indexed by
        date.
    window:
        Number of trailing observations to aggregate (``>= 1``).

    Returns
    -------
    pandas.Series
        The Parkinson RV series, named ``"rv_parkinson"``.

    Raises
    ------
    ValidationError
        If required columns are missing, prices are non-positive, any
        ``high < low``, or ``window < 1``.
    """
    raise NotImplementedError


def garman_klass_rv(ohlc: pd.DataFrame, *, window: int = 1) -> pd.Series:
    r"""Garman-Klass (1980) OHLC realized volatility.

    Uses the full open-high-low-close bar and is more efficient still than
    Parkinson:

    .. math::

        \mathrm{RV}^{gk}_t = \sqrt{\frac{1}{\text{window}}
                             \sum_{i=0}^{\text{window}-1}
                             \left[ \tfrac{1}{2}\!\left(\ln\tfrac{H}{L}\right)^2
                             - (2\ln 2 - 1)\!\left(\ln\tfrac{C}{O}\right)^2
                             \right]_{t-i}}.

    Parameters
    ----------
    ohlc:
        A frame with ``open``, ``high``, ``low``, ``close`` columns
        (case-insensitive) indexed by date.
    window:
        Number of trailing observations to aggregate (``>= 1``).

    Returns
    -------
    pandas.Series
        The Garman-Klass RV series, named ``"rv_garman_klass"``.

    Raises
    ------
    ValidationError
        If required columns are missing, prices are non-positive, any
        ``high < low``, or ``window < 1``.
    """
    raise NotImplementedError


def realized_volatility(
    ohlc: pd.DataFrame,
    *,
    estimator: str = "garman_klass",
    window: int = 1,
) -> pd.Series:
    """Dispatch to a named RV estimator.

    Parameters
    ----------
    ohlc:
        An OHLC frame (case-insensitive columns) indexed by date.
    estimator:
        One of ``"close_to_close"``, ``"parkinson"``, ``"garman_klass"``.
    window:
        Trailing aggregation window (``>= 1``).

    Returns
    -------
    pandas.Series
        The chosen RV series.

    Raises
    ------
    ValidationError
        If ``estimator`` is unknown or the inputs are malformed.
    """
    raise NotImplementedError


def forward_rv_target(
    rv: pd.Series,
    *,
    horizon: int,
    gap: int = 1,
) -> pd.Series:
    r"""Build the strictly-forward realized-volatility target with an explicit gap.

    The target attached to timestamp ``t`` is the (annualization-free) realized
    volatility aggregated over the FUTURE window ``(t + gap, t + gap + horizon]``:

    .. math::

        y_t = \sqrt{\frac{1}{\text{horizon}}
              \sum_{k=1}^{\text{horizon}} \mathrm{RV}_{\,t + \text{gap} + k}^2 }.

    LEAKAGE GUARD: the window starts strictly after ``t + gap``, so for any model
    whose features are observable at ``t`` the feature index ``{<= t}`` and the
    target window ``{> t + gap}`` are DISJOINT. The trailing ``horizon + gap``
    rows have an incomplete future window and are returned as NaN (callers drop
    them). A unit test asserts this disjointness directly.

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.
    horizon:
        The forward window length in trading days (``>= 1``).
    gap:
        The number of days skipped between the feature timestamp ``t`` and the
        first day of the target window (``>= 0``; default ``1`` so the target
        never includes the same-day return).

    Returns
    -------
    pandas.Series
        The forward RV target aligned to ``rv.index``, named ``"rv_target"``,
        NaN where the future window is incomplete.

    Raises
    ------
    ValidationError
        If ``horizon < 1``, ``gap < 0``, or ``rv`` is empty.
    """
    raise NotImplementedError

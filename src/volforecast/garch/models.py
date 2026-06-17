"""GARCH-family volatility models via ``arch`` plus a hand-rolled parity oracle.

The serve path fits a GARCH-family model PER TRAIN FOLD with the ``arch`` package
(GARCH(1,1), EGARCH, GJR-GARCH, optional Student-t innovations) and produces a
multi-step-ahead volatility forecast for the test fold. A self-contained,
NumPy-only GARCH(1,1) log-likelihood (:func:`garch_11_log_likelihood`) is the
PARITY ORACLE: the parity test pins it against ``arch`` to a tight tolerance, so
we trust the third-party fit without trusting it blindly.

LAZY IMPORT: ``arch`` is imported INSIDE the fitting functions, never at module
import time, so ``import volforecast`` pulls in no heavy compiled extension and
the module stays import-pure. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

#: The GARCH-family specifications supported on the serve path.
GARCH_KINDS: tuple[str, ...] = ("garch", "egarch", "gjr")


@dataclass(frozen=True, slots=True)
class GARCHFit:
    """Immutable result of fitting a GARCH-family model on a train fold.

    Attributes
    ----------
    kind:
        The specification fitted, one of :data:`GARCH_KINDS`.
    params:
        The fitted parameter mapping (e.g. ``{"omega":..., "alpha[1]":...,
        "beta[1]":...}``) as reported by ``arch``.
    loglikelihood:
        The maximized log-likelihood of the fit.
    dist:
        The innovation distribution, ``"normal"`` or ``"t"``.
    n_train:
        The number of in-sample observations the model was fit on.
    converged:
        Whether the optimizer reported convergence.
    """

    kind: str
    params: dict[str, float]
    loglikelihood: float
    dist: str
    n_train: int
    converged: bool = True
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this fit."""
        return {
            "kind": str(self.kind),
            "params": {str(k): float(v) for k, v in self.params.items()},
            "loglikelihood": float(self.loglikelihood),
            "dist": str(self.dist),
            "n_train": int(self.n_train),
            "converged": bool(self.converged),
            "meta": dict(self.meta),
        }


def fit_garch(
    returns: pd.Series,
    *,
    kind: str = "garch",
    dist: str = "normal",
    scale: float = 100.0,
) -> GARCHFit:
    r"""Fit a GARCH-family model on a TRAIN fold via ``arch`` (lazy import).

    Estimates one of GARCH(1,1), EGARCH(1,1,1), or GJR-GARCH(1,1,1) with normal
    or Student-t innovations by maximum likelihood. Returns are internally scaled
    by ``scale`` (``arch`` is numerically happiest on percent-returns) and the
    reported parameters are on that scale; the forecaster undoes the scaling.

    FIT-ON-TRAIN-ONLY: this is called once per walk-forward train fold; it never
    sees the test fold. ``arch`` is imported inside this function.

    Parameters
    ----------
    returns:
        A per-day (log or simple) return series for the train fold.
    kind:
        One of :data:`GARCH_KINDS` (``"garch"``, ``"egarch"``, ``"gjr"``).
    dist:
        Innovation distribution, ``"normal"`` or ``"t"``.
    scale:
        Multiplicative scale applied to returns before fitting (default ``100``).

    Returns
    -------
    GARCHFit
        The frozen fit bundle.

    Raises
    ------
    ValidationError
        If ``kind``/``dist`` is unsupported or ``returns`` is malformed.
    InsufficientDataError
        If the train fold is too short to fit the requested specification.
    ConvergenceError
        If the optimizer fails to produce a finite log-likelihood.
    """
    raise NotImplementedError


def forecast_garch_vol(
    fit: GARCHFit,
    returns: pd.Series,
    *,
    horizon: int,
    scale: float = 100.0,
) -> float:
    r"""Produce an ``horizon``-day-ahead volatility forecast from a fitted model.

    Builds the analytic multi-step variance forecast implied by ``fit`` and the
    most recent conditional variance (re-filtered from ``returns`` using the
    fitted parameters), aggregates it over the ``horizon`` window, and returns the
    forecast realized volatility on the ORIGINAL (unscaled) return scale.

    Parameters
    ----------
    fit:
        A :class:`GARCHFit` from :func:`fit_garch`.
    returns:
        The return series up to the forecast origin (train fold, possibly
        extended with observed test-fold returns up to ``t`` — never future).
    horizon:
        The forecast horizon in trading days (``>= 1``).
    scale:
        The same scale passed to :func:`fit_garch` (to undo the scaling).

    Returns
    -------
    float
        The ``horizon``-day-ahead realized-volatility forecast (unscaled).

    Raises
    ------
    ValidationError
        If ``horizon < 1`` or the inputs are inconsistent with ``fit``.
    """
    raise NotImplementedError


def garch_11_log_likelihood(
    returns: NDArray[np.float64],
    omega: float,
    alpha: float,
    beta: float,
    *,
    backcast: float | None = None,
) -> float:
    r"""Hand-rolled Gaussian GARCH(1,1) log-likelihood (the PARITY ORACLE).

    Filters the conditional-variance recursion

    .. math::

        \sigma^2_t = \omega + \alpha\, r_{t-1}^2 + \beta\, \sigma^2_{t-1},

    seeded by ``backcast`` (the sample variance of ``returns`` when ``None``), and
    returns the Gaussian quasi-log-likelihood

    .. math::

        \ell = -\tfrac{1}{2} \sum_{t} \left[
                 \ln(2\pi) + \ln \sigma^2_t + \frac{r_t^2}{\sigma^2_t}\right].

    This NumPy-only implementation is independent of ``arch`` so the parity test
    can pin ``arch``'s optimum against it to a tight tolerance (e.g. ``1e-6`` in
    log-likelihood at the same parameters). It performs NO optimization.

    Parameters
    ----------
    returns:
        A 1-D float array of returns (already on the desired scale).
    omega, alpha, beta:
        The GARCH(1,1) parameters; require ``omega > 0``, ``alpha >= 0``,
        ``beta >= 0``.
    backcast:
        Initial variance seed; defaults to the sample variance of ``returns``.

    Returns
    -------
    float
        The Gaussian GARCH(1,1) log-likelihood at the given parameters.

    Raises
    ------
    ValidationError
        If ``returns`` is empty/degenerate or the parameters are out of domain.
    """
    raise NotImplementedError

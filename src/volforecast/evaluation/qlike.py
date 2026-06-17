"""Volatility-forecast loss functions: QLIKE and MSE on realized variance.

QLIKE is the robust-to-noisy-proxy loss of choice for volatility forecasting
(Patton 2011): it is robust to the fact that realized variance is only a noisy
proxy of the latent variance, which biases MSE-style losses. We report BOTH so a
reader can see that the ranking is not an artefact of the loss choice.

All losses operate on realized VARIANCE (squared volatility) and are pure,
NaN-aware reductions. Importing this module has no side effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def qlike_loss_series(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> NDArray[np.float64]:
    r"""Per-observation QLIKE loss between realized and forecast variance.

    .. math::

        \mathrm{QLIKE}_t = \frac{\sigma^2_t}{h_t} - \ln \frac{\sigma^2_t}{h_t} - 1,

    where :math:`\sigma^2_t` is the realized-variance proxy and :math:`h_t` the
    forecast variance. The loss is non-negative, minimized at
    :math:`h_t = \sigma^2_t`, and (unlike MSE) robust to noise in the proxy.

    Parameters
    ----------
    realized_var:
        The realized-variance proxy series/array (``> 0``).
    forecast_var:
        The forecast-variance series/array (``> 0``), aligned to
        ``realized_var``.

    Returns
    -------
    numpy.ndarray
        The per-observation QLIKE loss (1-D float array).

    Raises
    ------
    ValidationError
        If lengths differ, or any variance is non-positive after the EPS floor.
    """
    raise NotImplementedError


def qlike(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> float:
    """Mean QLIKE loss (lower is better).

    Parameters
    ----------
    realized_var, forecast_var:
        Aligned realized- and forecast-variance series/arrays (``> 0``).

    Returns
    -------
    float
        The mean QLIKE over all aligned observations.

    Raises
    ------
    ValidationError
        If lengths differ or variances are non-positive.
    """
    raise NotImplementedError


def mse(
    realized_var: pd.Series | NDArray[np.float64],
    forecast_var: pd.Series | NDArray[np.float64],
) -> float:
    r"""Mean squared error on realized variance (lower is better).

    .. math::

        \mathrm{MSE} = \frac{1}{T}\sum_t \left(\sigma^2_t - h_t\right)^2.

    Reported alongside QLIKE as a robustness cross-check on the model ranking.

    Parameters
    ----------
    realized_var, forecast_var:
        Aligned realized- and forecast-variance series/arrays.

    Returns
    -------
    float
        The mean squared error over all aligned observations.

    Raises
    ------
    ValidationError
        If lengths differ.
    """
    raise NotImplementedError

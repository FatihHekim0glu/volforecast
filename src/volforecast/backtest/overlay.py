"""Optional downstream vol-targeting overlay with a Deflated Sharpe guard.

A volatility forecast is only useful if it improves a decision. The overlay
turns each model's RV forecast into a daily position that targets a constant
annualized volatility (``target_vol``): scale exposure DOWN when the forecast is
high and UP when it is low, capped by a leverage limit, and charge a per-side
basis-point cost on the day-to-day change in exposure.

HONESTY REQUIREMENT: the resulting P&L Sharpe is then DEFLATED with the TRUE
number of trials (``n_trials`` = the number of model configurations whose
overlays were evaluated) via :func:`volforecast.evaluation.dsr.deflated_sharpe_ratio`.
No raw-Sharpe profit claim is made; the overlay is a sensitivity check, not the
headline. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class OverlayResult:
    """Immutable result of a vol-targeting overlay backtest for one model.

    Attributes
    ----------
    net_returns:
        The after-cost daily P&L series of the vol-targeted position.
    gross_returns:
        The before-cost daily P&L series.
    exposure:
        The daily position scale (leverage) applied (``shift``-safe).
    sharpe:
        The annualized net Sharpe ratio (raw, pre-deflation).
    deflated_sharpe:
        The Deflated Sharpe Ratio with the TRUE ``n_trials`` (the honest figure).
    turnover:
        The mean absolute day-to-day change in exposure.
    n_trials:
        The multiplicity count used to deflate the Sharpe.
    """

    net_returns: pd.Series
    gross_returns: pd.Series
    exposure: pd.Series
    sharpe: float
    deflated_sharpe: float
    turnover: float
    n_trials: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "net_returns": {str(k): _safe_float(v) for k, v in self.net_returns.items()},
            "gross_returns": {str(k): _safe_float(v) for k, v in self.gross_returns.items()},
            "exposure": {str(k): _safe_float(v) for k, v in self.exposure.items()},
            "sharpe": _safe_float(self.sharpe),
            "deflated_sharpe": _safe_float(self.deflated_sharpe),
            "turnover": _safe_float(self.turnover),
            "n_trials": int(self.n_trials),
            "meta": dict(self.meta),
        }


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def vol_target_overlay(
    returns: pd.Series,
    vol_forecast: pd.Series,
    *,
    target_vol: float = 0.10,
    max_leverage: float = 2.0,
    cost_bps: float = 10.0,
    n_trials: int = 1,
) -> OverlayResult:
    r"""Run a vol-targeting overlay driven by a one-step volatility forecast.

    The position at day ``t`` is

    .. math::

        w_t = \mathrm{clip}\!\left(\frac{\text{target\_vol}}{\widehat{\sigma}_{t}},\,
              0,\, \text{max\_leverage}\right),

    applied to the NEXT day's return via ``shift(1)`` (no lookahead: the forecast
    available at ``t`` sizes the ``t+1`` position). A per-side ``cost_bps`` charge
    is levied on ``|w_t - w_{t-1}|``. The net Sharpe is reported AND deflated with
    ``n_trials`` so no overfit profit claim survives.

    Parameters
    ----------
    returns:
        The per-day return series of the underlying.
    vol_forecast:
        The per-day annualized-volatility forecast aligned to ``returns``
        (already observable at the position date).
    target_vol:
        The annualized volatility target (e.g. ``0.10`` = 10%).
    max_leverage:
        The cap on position size (``>= 0``).
    cost_bps:
        Per-side transaction cost in basis points (``>= 0``).
    n_trials:
        The TRUE number of model-overlay configurations evaluated, for the
        Deflated Sharpe (``>= 1``).

    Returns
    -------
    OverlayResult
        The net/gross P&L, exposure path, raw and deflated Sharpe, and turnover.

    Raises
    ------
    ValidationError
        If inputs are misaligned, ``target_vol <= 0``, ``max_leverage < 0``,
        ``cost_bps < 0``, or ``n_trials < 1``.
    """
    raise NotImplementedError

"""Plotly figure builders (lazy plotly).

Each builder returns a plain ``dict`` shaped ``{"data": [...], "layout": {...}}``
— the same JSON shape the FastAPI layer serializes and the Next.js
``PlotlyChart`` component renders — so figures cross the API boundary with no
Plotly object leaking through. Plotly is an OPTIONAL dependency (the ``viz``
extra) imported lazily inside each builder; importing this module has no side
effects and does not require Plotly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# quantcore-candidate: mirrors hrp / markowitz / pairs-trading plots.py
# ({data, layout} figure shape).

#: A Plotly figure serialized as a plain mapping with ``data`` and ``layout`` keys.
FigureDict = dict[str, Any]


def _jsonify(value: Any) -> Any:
    """Recursively convert numpy/pandas scalars and arrays to native Python types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp | pd.Period):
        return value.isoformat() if hasattr(value, "isoformat") else str(value)
    return value


def rv_forecast_figure(
    realized_vol: pd.Series,
    forecasts: pd.DataFrame,
    *,
    title: str = "Realized volatility: actual vs model forecasts",
) -> FigureDict:
    """Build the RV-actual-vs-forecasts line figure.

    Plots the realized forward volatility as a reference line plus one line per
    model column in ``forecasts``, all on a shared date axis.

    LAZY IMPORT: ``plotly`` is imported inside this function.

    Parameters
    ----------
    realized_vol:
        The realized forward-volatility series (the truth) indexed by date.
    forecasts:
        A ``(T, M)`` frame of per-model forecasts aligned to ``realized_vol``.
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    ValidationError
        If ``realized_vol`` and ``forecasts`` cannot be aligned.
    """
    raise NotImplementedError


def qlike_bar_figure(
    qlike_by_model: dict[str, float],
    *,
    best_model: str | None = None,
    title: str = "Out-of-sample QLIKE by model (lower is better)",
) -> FigureDict:
    """Build the QLIKE-by-model bar figure (the headline error chart).

    One bar per model, sorted ascending (best first); the ``best_model`` bar is
    highlighted so the honest ranking is obvious at a glance.

    LAZY IMPORT: ``plotly`` is imported inside this function.

    Parameters
    ----------
    qlike_by_model:
        Mapping ``{model_label: mean_OOS_QLIKE}``.
    best_model:
        The label to highlight (defaults to the argmin of ``qlike_by_model``).
    title:
        The figure title.

    Returns
    -------
    FigureDict
        A ``{"data": [...], "layout": {...}}`` mapping.

    Raises
    ------
    ValidationError
        If ``qlike_by_model`` is empty.
    """
    raise NotImplementedError

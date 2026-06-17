"""HAR-RV feature builder (Corsi 2009).

The Heterogeneous Autoregressive model of Realized Volatility decomposes the
RV process into trailing daily, weekly, and monthly average-RV components. These
same components are the regressors for the HAR-RV baseline and the engineered
features for the XGBoost model.

LAG SAFETY: every component is a trailing average of PAST RV and is then
``.shift()``-lagged so that the feature row at timestamp ``t`` uses only RV
observed at or before ``t`` — never ``t``'s own forward target. A property test
asserts this lag-safety. Importing this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True, slots=True)
class HARFeatures:
    """Immutable bundle of HAR-RV feature components and the aligned target.

    Attributes
    ----------
    features:
        The ``(T, k)`` feature frame with columns ``rv_daily``, ``rv_weekly``,
        ``rv_monthly`` (plus any exogenous columns such as ``vix``), all
        lagged so they are observable at the row timestamp.
    target:
        The aligned forward RV target series (NaN-dropped jointly with
        ``features``).
    feature_names:
        The ordered feature column names (the model's input contract).
    """

    features: pd.DataFrame
    target: pd.Series
    feature_names: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this bundle.

        Frames/series are rendered with ISO-formatted date keys so the bundle
        crosses the API boundary cleanly.
        """
        return {
            "features": {
                str(idx): {str(c): _safe_float(v) for c, v in row.items()}
                for idx, row in self.features.iterrows()
            },
            "target": {str(k): _safe_float(v) for k, v in self.target.items()},
            "feature_names": list(self.feature_names),
        }


def _safe_float(value: object) -> float | None:
    """Coerce ``value`` to a finite float, mapping NaN/Inf/None to ``None``."""
    import numpy as np

    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def har_components(rv: pd.Series) -> pd.DataFrame:
    r"""Build the three lagged HAR-RV components from a daily RV series.

    Returns a frame with columns

    - ``rv_daily``  : ``RV_{t-1}`` (yesterday's RV),
    - ``rv_weekly`` : the trailing 5-day mean of RV, lagged one day,
    - ``rv_monthly``: the trailing 22-day mean of RV, lagged one day,

    each ``.shift()``-lagged so the row at ``t`` contains only information
    available strictly before ``t`` (Corsi's cascade with no contemporaneous RV).

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.

    Returns
    -------
    pandas.DataFrame
        The lagged HAR component frame aligned to ``rv.index`` (leading rows NaN
        over the 22-day warm-up).

    Raises
    ------
    ValidationError
        If ``rv`` is empty.
    """
    raise NotImplementedError


def build_har_features(
    rv: pd.Series,
    target: pd.Series,
    *,
    exog: pd.DataFrame | None = None,
) -> HARFeatures:
    """Assemble the lagged HAR feature frame and align it to the forward target.

    Joins the lagged HAR components (and any exogenous, already-lagged columns
    such as a VIX level) with the forward RV ``target``, then drops rows with any
    NaN so the feature/target pair is complete and disjoint in time.

    Parameters
    ----------
    rv:
        A per-day realized-volatility series indexed by date.
    target:
        The forward RV target (e.g. from
        :func:`volforecast.realized.estimators.forward_rv_target`).
    exog:
        Optional exogenous feature frame (e.g. ``{"vix": ...}``). MUST already be
        lagged by the caller; this function does not re-lag exogenous inputs.

    Returns
    -------
    HARFeatures
        The aligned, NaN-free feature/target bundle.

    Raises
    ------
    ValidationError
        If ``rv``/``target`` cannot be aligned or the joined frame is empty.
    """
    raise NotImplementedError

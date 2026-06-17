"""Anchored/expanding walk-forward engine with strict fit-on-train-only.

This is the leakage-control heart of the project and the explicit fix for the
Stock-Price-Forecast anti-pattern (fitting a scaler/model on the FULL series and
then "evaluating" out-of-sample). Here EVERY estimator — the RV scaler, the GARCH
parameters, the HAR-RV OLS, and the XGBoost booster — is fit INSIDE each
walk-forward TRAIN fold and only then asked to forecast the disjoint TEST fold.

Guards enforced and property-tested:

- **Forward-only targets** with an explicit ``gap`` (the target window is strictly
  ``> t + gap``); a **purge** removes the boundary rows whose target window
  overlaps the test fold, and an **embargo** sized to the horizon ``h`` removes
  the rows just after the train fold.
- **Future-perturbation invariance**: perturbing returns strictly after the
  forecast origin must not change a fold's forecasts (the canonical leakage
  detector), asserted by a property test.

The engine collects, per test point, the realized RV and each model's forecast,
so the evaluation layer can compute QLIKE/MSE and run DM/SPA. Importing this
module has no side effects (``arch``/``xgboost`` are imported lazily by the
fitters this engine calls).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: The default model set fitted on the serve path (LSTM excluded — research only).
DEFAULT_MODELS: tuple[str, ...] = ("garch", "egarch", "har_rv", "ewma", "xgboost", "rw")


@dataclass(frozen=True, slots=True)
class WalkForwardConfig:
    """Immutable configuration of a walk-forward run.

    Attributes
    ----------
    horizon:
        The forecast horizon ``h`` in trading days (one of 1, 5, 22).
    train_window:
        The minimum train-fold length in trading days (the warm-up).
    gap:
        The gap between the feature timestamp and the target window (``>= 0``).
    step:
        The number of test points advanced between refits (``>= 1``).
    anchored:
        If ``True``, the train fold expands (anchored at the start); else it
        rolls with fixed ``train_window`` length.
    models:
        The model labels to fit each fold (subset of :data:`DEFAULT_MODELS`).
    seed:
        Master seed for any stochastic fitter (XGBoost determinism).
    """

    horizon: int
    train_window: int = 504
    gap: int = 1
    step: int = 1
    anchored: bool = True
    models: tuple[str, ...] = DEFAULT_MODELS
    seed: int = 7

    def __post_init__(self) -> None:
        """Validate the configuration scalars.

        Raises
        ------
        ValidationError
            If ``horizon < 1``, ``train_window < 2``, ``gap < 0``, ``step < 1``,
            or ``models`` is empty / contains an unknown label.
        """
        raise NotImplementedError

    @property
    def embargo(self) -> int:
        """The embargo length (sized to the horizon): ``gap + horizon``."""
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this configuration."""
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class WalkForwardResult:
    """Immutable result of a walk-forward run: aligned actuals and forecasts.

    Attributes
    ----------
    realized_vol:
        The realized forward volatility at each evaluated test origin (the truth).
    forecasts:
        A ``(n_test, n_models)`` frame of per-model forward-vol forecasts aligned
        to ``realized_vol.index``.
    config:
        The :class:`WalkForwardConfig` used.
    n_folds:
        The number of train/test folds evaluated.
    """

    realized_vol: pd.Series
    forecasts: pd.DataFrame
    config: WalkForwardConfig
    n_folds: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result.

        Series/frames are rendered with ISO-formatted date keys and NaN scrubbed
        to ``None`` so the result crosses the API boundary cleanly.
        """
        raise NotImplementedError


def _train_test_slices(
    n_obs: int,
    config: WalkForwardConfig,
) -> list[tuple[int, int, int]]:
    """Enumerate ``(train_start, train_end, test_origin)`` index triples.

    Each triple defines a fold: the train fold is ``[train_start, train_end)``,
    and the test origin is ``train_end + embargo`` (the first row whose features
    are observable and whose forward target is strictly disjoint from training).
    ``train_start`` is ``0`` when ``anchored`` else ``train_end - train_window``.

    Parameters
    ----------
    n_obs:
        The total number of observations available.
    config:
        The walk-forward configuration.

    Returns
    -------
    list[tuple[int, int, int]]
        The ordered fold triples (possibly empty if the series is too short).
    """
    raise NotImplementedError


def run_walk_forward(
    ohlc: pd.DataFrame,
    *,
    config: WalkForwardConfig,
    rv_estimator: str = "garman_klass",
    exog: pd.DataFrame | None = None,
) -> WalkForwardResult:
    """Run the leakage-guarded walk-forward forecast over an OHLC series.

    For each fold the engine: (1) computes the daily RV proxy and the forward RV
    target with the configured ``gap``; (2) fits EVERY model in ``config.models``
    on the TRAIN fold ONLY (GARCH via ``arch``, HAR-RV by OLS, XGBoost with a
    fixed seed, EWMA/RW closed-form); (3) forecasts the ``horizon``-ahead vol at
    the test origin; (4) records the realized forward vol and each forecast. No
    estimator ever touches data at or after its forecast origin.

    Parameters
    ----------
    ohlc:
        A wide OHLC frame (``open/high/low/close``, case-insensitive) indexed by
        date.
    config:
        The :class:`WalkForwardConfig`.
    rv_estimator:
        The RV proxy used for features AND target, one of ``"close_to_close"``,
        ``"parkinson"``, ``"garman_klass"``.
    exog:
        Optional exogenous, ALREADY-LAGGED features (e.g. a VIX level) aligned to
        ``ohlc.index``.

    Returns
    -------
    WalkForwardResult
        The aligned realized-vol series and per-model forecast frame.

    Raises
    ------
    ValidationError
        If ``ohlc`` is malformed or ``rv_estimator`` is unknown.
    InsufficientDataError
        If the series is too short for even one train/test fold under ``config``.
    """
    raise NotImplementedError

"""XGBoost volatility forecaster on HAR / lagged-RV / VIX features.

The container's ML arm: gradient-boosted trees regressing the forward RV target
on the lagged HAR components (and any exogenous features such as a VIX level).
The model is FIT PER TRAIN FOLD with a fixed seed for determinism, and the same
feature contract (:class:`volforecast.features.har.HARFeatures`) feeds both this
model and the HAR-RV baseline so the comparison is apples-to-apples.

LAZY IMPORT: ``xgboost`` is imported INSIDE the fitting function, never at module
import time, so ``import volforecast`` stays light and import-pure. Importing
this module has no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

#: Default XGBoost hyper-parameters (small, regularized — RV data is short and
#: noisy, so we keep the model deliberately modest to honour the null).
DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
}


@dataclass(frozen=True, slots=True)
class XGBForecaster:
    """A fitted XGBoost RV forecaster (opaque booster + feature contract).

    The fitted booster is held in ``meta["booster"]`` (not serialized by
    :meth:`to_dict`); ``to_dict`` exposes only the JSON-safe metadata
    (hyper-parameters, feature names, seed, train size, feature importances).

    Attributes
    ----------
    feature_names:
        The ordered feature columns the booster expects at predict time.
    params:
        The hyper-parameters used for the fit.
    seed:
        The RNG seed fixed for determinism.
    n_train:
        The number of in-sample observations the model was fit on.
    importances:
        Per-feature gain importances (``{}`` until populated by the fitter).
    """

    feature_names: tuple[str, ...]
    params: dict[str, Any]
    seed: int
    n_train: int
    importances: dict[str, float] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (booster excluded)."""
        return {
            "feature_names": list(self.feature_names),
            "params": dict(self.params),
            "seed": int(self.seed),
            "n_train": int(self.n_train),
            "importances": {str(k): float(v) for k, v in self.importances.items()},
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Forecast forward RV from a feature frame.

        Parameters
        ----------
        features:
            A frame whose columns are a superset of ``feature_names`` (extra
            columns ignored, order normalized to the contract).

        Returns
        -------
        pandas.Series
            The XGBoost point forecast aligned to ``features.index``.

        Raises
        ------
        ValidationError
            If any required feature column is missing.
        """
        raise NotImplementedError


def fit_xgb(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    params: dict[str, Any] | None = None,
    seed: int = 7,
) -> XGBForecaster:
    """Fit an XGBoost RV forecaster on a TRAIN fold only (deterministic).

    FIT-ON-TRAIN-ONLY: called once per walk-forward train fold; it never sees the
    test fold. The seed is fixed and single-threaded determinism is enforced so a
    fixed ``(features, target, params, seed)`` yields byte-identical predictions
    (pinned by the parity suite). ``xgboost`` is imported inside this function.

    Parameters
    ----------
    features:
        The lagged feature frame for the train fold (no NaN).
    target:
        The aligned forward RV target for the train fold (no NaN).
    params:
        XGBoost hyper-parameters; defaults to :data:`DEFAULT_XGB_PARAMS`.
    seed:
        The RNG seed fixed for determinism.

    Returns
    -------
    XGBForecaster
        The fitted, frozen forecaster (booster in ``meta["booster"]``).

    Raises
    ------
    ValidationError
        If ``features``/``target`` are misaligned or contain NaN.
    InsufficientDataError
        If the train fold is too small to fit a tree model.
    """
    raise NotImplementedError

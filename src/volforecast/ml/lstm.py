"""RESEARCH-ONLY LSTM volatility forecaster (NOT served, NOT vendored).

This arm exists purely to document that we tried a deep-learning forecaster and
that — per Hansen & Lunde (2005) and our own honest-null discipline — it rarely
justifies its compute against a well-specified GARCH(1,1)/HAR-RV. It is gated
behind the ``[research]`` extra and a LAZY TensorFlow import, and is NEVER
imported on the serve path: the top-level :mod:`volforecast` package does not
re-export anything from this module, and the FastAPI router must not import it.

CONTAINER GUARANTEE: TensorFlow is in ``[research]`` only, so the lean ``[data]``
container cannot import this module's body. Calling any function here without
TensorFlow installed raises a clear, catchable error rather than crashing the
process. Importing THIS module has no side effects (TF is imported lazily, inside
the functions, behind a guard).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


def _require_tensorflow() -> Any:
    """Lazily import TensorFlow, or raise a clear, catchable error.

    Returns
    -------
    module
        The imported ``tensorflow`` module.

    Raises
    ------
    VolForecastError
        If TensorFlow is not installed (i.e. the ``[research]`` extra is absent,
        as in the lean serve container). The message points at the extra.
    """
    raise NotImplementedError


@dataclass(frozen=True, slots=True)
class LSTMForecaster:
    """A fitted research-only LSTM RV forecaster (opaque Keras model).

    Attributes
    ----------
    feature_names:
        The ordered feature columns the model expects.
    lookback:
        The sequence length (number of trailing timesteps) fed to the LSTM.
    seed:
        The RNG seed fixed for (best-effort) determinism.
    n_train:
        The number of in-sample sequences the model was fit on.
    """

    feature_names: tuple[str, ...]
    lookback: int
    seed: int
    n_train: int
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` (Keras model excluded)."""
        return {
            "feature_names": list(self.feature_names),
            "lookback": int(self.lookback),
            "seed": int(self.seed),
            "n_train": int(self.n_train),
        }

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Forecast forward RV from a feature frame (research-only).

        Raises
        ------
        VolForecastError
            If TensorFlow is unavailable (lean container).
        """
        raise NotImplementedError


def fit_lstm(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    lookback: int = 22,
    seed: int = 7,
    epochs: int = 50,
) -> LSTMForecaster:
    """Fit a research-only LSTM on a TRAIN fold (lazy TensorFlow, NOT served).

    FIT-ON-TRAIN-ONLY and research-only: this is never called on the serve path.
    It builds length-``lookback`` sequences from ``features``, fits a small LSTM
    regressor, and returns an :class:`LSTMForecaster`.

    Parameters
    ----------
    features:
        The lagged feature frame for the train fold (no NaN).
    target:
        The aligned forward RV target for the train fold (no NaN).
    lookback:
        Sequence length in timesteps.
    seed:
        The RNG seed fixed for best-effort determinism.
    epochs:
        Training epochs.

    Returns
    -------
    LSTMForecaster
        The fitted, frozen forecaster.

    Raises
    ------
    VolForecastError
        If TensorFlow is unavailable (the lean ``[data]`` container).
    ValidationError
        If ``features``/``target`` are misaligned or too short for ``lookback``.
    """
    raise NotImplementedError

"""Golden honest-null regression.

On the synthetic GARCH series, GARCH is the true model, so the pure verdict MUST
report ``ml_beats_garch=False`` and pick a GARCH/HAR-RV reference as
``best_model``. This is the project's headline guarantee, pinned here against
both the full :func:`volforecast.run_vol_forecast` pipeline and the pure
:func:`volforecast.derive_verdict` contract.
"""

from __future__ import annotations

import warnings

import pandas as pd
import pytest

import volforecast as vf
from volforecast.evaluation.verdict import REFERENCE_MODELS


@pytest.mark.regression
def test_ml_does_not_beat_garch_on_garch_data(garch_series: pd.DataFrame) -> None:
    """ML must not be crowned the winner on GARCH-generated data (the honest null)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run = vf.run_vol_forecast(
            garch_series,
            horizon=5,
            models=("garch", "har_rv", "ewma", "xgboost", "rw"),
            seed=7,
            train_window=300,
            step=40,
        )
    summary = run.summary
    # The encoded honest null: ML never clears the SPA/DM bar on GARCH-true data.
    assert summary.ml_beats_garch is False
    assert summary.best_model_class == "reference"
    assert summary.best_model in REFERENCE_MODELS


@pytest.mark.regression
def test_verdict_requires_significance_to_crown_ml() -> None:
    """A lower ML point QLIKE alone must NOT yield ml_beats_garch=True."""
    qlike_by_model = {"garch": 0.50, "har_rv": 0.49, "xgboost": 0.48}
    # XGBoost has the lowest QLIKE but the SPA/DM gates are insignificant.
    verdict = vf.derive_verdict(
        qlike_by_model,
        spa_pvalue=0.40,  # not significant
        dm_pvalues_vs_best={"xgboost": 0.40},
    )
    assert verdict.ml_beats_garch is False

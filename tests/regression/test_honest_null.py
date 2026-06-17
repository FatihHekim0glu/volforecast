"""Golden honest-null regression (filled in as the kernels land).

On the synthetic GARCH series, GARCH is the true model, so the pure verdict MUST
report ``ml_beats_garch=False`` and pick a GARCH/HAR-RV reference as
``best_model``. This is the project's headline guarantee, pinned here. The
behavioural assertion is ``xfail`` until the pipeline exists; the pure-verdict
contract below is asserted directly once ``derive_verdict`` is implemented.
"""

from __future__ import annotations

import pandas as pd
import pytest

import volforecast as vf


@pytest.mark.regression
@pytest.mark.xfail(reason="end-to-end pipeline not yet implemented", strict=True)
def test_ml_does_not_beat_garch_on_garch_data(garch_series: pd.DataFrame) -> None:
    """ML must not be crowned the winner on GARCH-generated data (the honest null)."""
    config = vf.WalkForwardConfig(horizon=5)
    result = vf.run_walk_forward(garch_series, config=config)
    qlike_by_model = {m: vf.qlike(result.realized_vol, result.forecasts[m]) for m in result.forecasts}
    # ... compute SPA + DM, then:
    verdict = vf.derive_verdict(qlike_by_model, spa_pvalue=0.5, dm_pvalues_vs_best={})
    assert verdict.ml_beats_garch is False


@pytest.mark.regression
@pytest.mark.xfail(reason="derive_verdict not yet implemented", strict=True)
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

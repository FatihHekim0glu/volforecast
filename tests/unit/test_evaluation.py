"""Tests for the honest-statistics layer: QLIKE/MSE, DM/SPA, and the verdict.

Covers the evaluation group's contract:

- **QLIKE/MSE correctness** on closed-form fixtures (zero at a perfect forecast,
  the exact ``q - ln q - 1`` value, NaN-awareness, and the validation guards).
- **Diebold-Mariano** statistic/p-value parity against a SciPy Student-t
  reference, sign/favoured logic, and the HAC long-run variance.
- **Hansen SPA** controls data snooping: on a no-edge model set the composite
  null is NOT rejected (large p-value), while a genuinely dominating model
  produces a small consistent p-value.
- **Verdict truth table**: ``best_model`` is the strict QLIKE argmin with a
  reference-favouring tie-break, and ``ml_beats_garch`` is ``True`` only when an
  ML model wins AND both the SPA and DM gates clear.
- **Honest-null guard**: on the GARCH fixture, point QLIKE alone never crowns ML.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t as scipy_t

from volforecast._exceptions import ValidationError
from volforecast.evaluation.qlike import mse, qlike, qlike_loss_series
from volforecast.evaluation.tests import (
    DMResult,
    SPAResult,
    _betainc,
    _fallback_spa,
    _student_t_sf_two_sided,
    diebold_mariano,
    hansen_spa,
    newey_west_lrv,
)
from volforecast.evaluation.verdict import (
    ML_MODELS,
    REFERENCE_MODELS,
    BestModelClass,
    Verdict,
    derive_verdict,
)

# --------------------------------------------------------------------------- #
# QLIKE / MSE                                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_qlike_is_zero_at_perfect_forecast() -> None:
    rv = np.array([0.10, 0.20, 0.30, 0.25])
    assert qlike(rv, rv) == pytest.approx(0.0, abs=1e-12)
    assert mse(rv, rv) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_qlike_loss_series_matches_closed_form() -> None:
    # QLIKE_t = sigma2/h - ln(sigma2/h) - 1 ; here sigma2=2, h=1 => 2 - ln2 - 1.
    loss = qlike_loss_series(np.array([2.0]), np.array([1.0]))
    assert loss[0] == pytest.approx(2.0 - math.log(2.0) - 1.0, rel=1e-12)


@pytest.mark.unit
def test_qlike_is_non_negative_and_penalizes_misspecification() -> None:
    rng = np.random.default_rng(0)
    rv = np.abs(rng.standard_normal(200)) * 0.01 + 1e-4
    loss_exact = qlike_loss_series(rv, rv)
    assert np.all(loss_exact >= -1e-12)
    # Doubling or halving the forecast strictly increases the mean loss.
    assert qlike(rv, rv * 2.0) > qlike(rv, rv)
    assert qlike(rv, rv * 0.5) > qlike(rv, rv)


@pytest.mark.unit
def test_qlike_accepts_series_and_array_interchangeably() -> None:
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    rv = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5], index=idx)
    fc = pd.Series([0.11, 0.19, 0.31, 0.39, 0.52], index=idx)
    from_series = qlike(rv, fc)
    from_array = qlike(
        np.asarray(rv.to_numpy(), dtype=np.float64),
        np.asarray(fc.to_numpy(), dtype=np.float64),
    )
    assert from_series == pytest.approx(from_array, rel=1e-12)


@pytest.mark.unit
def test_qlike_is_nan_aware() -> None:
    rv = np.array([0.1, np.nan, 0.3])
    fc = np.array([0.1, 0.2, 0.3])
    # The NaN row drops out of the mean; the two finite rows are perfect => 0.
    assert qlike(rv, fc) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_qlike_floors_zero_variance_at_eps() -> None:
    # A zero forecast variance is floored at EPS rather than producing inf.
    loss = qlike_loss_series(np.array([1.0]), np.array([0.0]))
    assert np.isfinite(loss[0])


@pytest.mark.unit
def test_qlike_rejects_length_mismatch() -> None:
    with pytest.raises(ValidationError):
        qlike(np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(ValidationError):
        mse(np.array([0.1, 0.2]), np.array([0.1]))


@pytest.mark.unit
def test_qlike_rejects_negative_variance() -> None:
    with pytest.raises(ValidationError):
        qlike_loss_series(np.array([-1.0]), np.array([1.0]))


@pytest.mark.unit
def test_qlike_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        qlike(np.array([]), np.array([]))


# --------------------------------------------------------------------------- #
# Newey-West long-run variance                                                #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_newey_west_lrv_recovers_variance_for_iid() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2000)
    # For (near-)iid data the LRV is close to gamma0 = the sample variance.
    assert newey_west_lrv(x) == pytest.approx(float(np.var(x)), rel=0.25)


@pytest.mark.unit
def test_newey_west_lrv_inflates_with_positive_autocorrelation() -> None:
    rng = np.random.default_rng(2)
    eps = rng.standard_normal(3000)
    # AR(1) with phi=0.7 has a long-run variance far above gamma0.
    x = np.empty_like(eps)
    x[0] = eps[0]
    for t in range(1, len(eps)):
        x[t] = 0.7 * x[t - 1] + eps[t]
    lrv = newey_west_lrv(x)
    gamma0 = float(np.var(x))
    assert lrv > gamma0


@pytest.mark.unit
def test_newey_west_lrv_is_non_negative_with_fixed_lag() -> None:
    rng = np.random.default_rng(3)
    x = rng.standard_normal(100)
    assert newey_west_lrv(x, lag=5) >= 0.0
    assert newey_west_lrv(x, lag=0) == pytest.approx(float(np.var(x)), rel=1e-12)


@pytest.mark.unit
def test_newey_west_lrv_validates_inputs() -> None:
    with pytest.raises(ValidationError):
        newey_west_lrv(np.array([1.0]))  # < 2 finite obs
    with pytest.raises(ValidationError):
        newey_west_lrv(np.array([1.0, 2.0, 3.0]), lag=-1)


# --------------------------------------------------------------------------- #
# Diebold-Mariano                                                             #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_dm_pvalue_matches_scipy_student_t() -> None:
    rng = np.random.default_rng(4)
    loss_a = np.abs(rng.standard_normal(300)) * 0.5
    # Model B is systematically worse by a small, noisy margin.
    loss_b = loss_a + 0.03 + rng.standard_normal(300) * 0.02
    result = diebold_mariano(loss_a, loss_b, label_a="a", label_b="b")
    expected_p = 2.0 * float(scipy_t.sf(abs(result.statistic), result.n_obs - 1))
    assert result.p_value == pytest.approx(expected_p, abs=1e-10)
    assert isinstance(result, DMResult)


@pytest.mark.unit
def test_dm_favours_the_lower_loss_model() -> None:
    rng = np.random.default_rng(5)
    loss_a = np.abs(rng.standard_normal(400)) * 0.4
    loss_b = loss_a + 0.05  # B is strictly worse
    result = diebold_mariano(loss_a, loss_b, label_a="garch", label_b="xgboost")
    assert result.mean_loss_diff < 0.0  # A better
    assert result.favored == "garch"
    assert result.p_value < 0.05  # the gap is detectable


@pytest.mark.unit
def test_dm_is_symmetric_in_statistic_sign() -> None:
    rng = np.random.default_rng(6)
    la = np.abs(rng.standard_normal(250)) * 0.3
    lb = la + 0.04 + rng.standard_normal(250) * 0.01
    r_ab = diebold_mariano(la, lb)
    r_ba = diebold_mariano(lb, la)
    assert r_ab.statistic == pytest.approx(-r_ba.statistic, rel=1e-12)
    assert r_ab.p_value == pytest.approx(r_ba.p_value, rel=1e-12)


@pytest.mark.unit
def test_dm_equal_losses_give_insignificant_pvalue() -> None:
    x = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    result = diebold_mariano(x, x)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_dm_to_dict_is_json_safe() -> None:
    r = diebold_mariano(np.array([0.1, 0.3, 0.2]), np.array([0.2, 0.1, 0.4]))
    d = r.to_dict()
    assert set(d) == {"statistic", "p_value", "mean_loss_diff", "n_obs", "favored"}
    assert isinstance(d["n_obs"], int)
    assert isinstance(d["favored"], str)


@pytest.mark.unit
def test_dm_validates_inputs() -> None:
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.1, 0.2]), np.array([0.1]))
    with pytest.raises(ValidationError):
        diebold_mariano(np.array([0.1]), np.array([0.2]))  # < 2 obs


@pytest.mark.unit
def test_dm_handles_constant_loss_differential() -> None:
    # A constant (non-zero) loss differential has zero long-run variance: the
    # test is undefined in t-units, so we report an insignificant p-value.
    loss_a = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    loss_b = np.array([0.10, 0.10, 0.10, 0.10, 0.10])
    result = diebold_mariano(loss_a, loss_b)
    assert result.statistic == pytest.approx(0.0, abs=1e-12)
    assert result.p_value == pytest.approx(1.0, abs=1e-12)
    assert result.favored == "b"  # B has the lower (constant) loss


@pytest.mark.unit
def test_dm_drops_nan_loss_differentials() -> None:
    rng = np.random.default_rng(7)
    la = np.abs(rng.standard_normal(50)) * 0.3
    lb = la + 0.05
    la[3] = np.nan  # this row drops out of the differential
    result = diebold_mariano(la, lb)
    assert result.n_obs == 49


@pytest.mark.unit
def test_student_t_tail_matches_scipy_across_grid() -> None:
    for df in (2, 5, 10, 30, 100):
        for stat in (0.0, 0.5, 1.0, 2.0, 3.0):
            mine = _student_t_sf_two_sided(stat, df)
            ref = 1.0 if stat == 0.0 else 2.0 * float(scipy_t.sf(stat, df))
            assert mine == pytest.approx(ref, abs=1e-10)


@pytest.mark.unit
def test_betainc_endpoints() -> None:
    assert _betainc(2.0, 3.0, 0.0) == 0.0
    assert _betainc(2.0, 3.0, 1.0) == 1.0


# --------------------------------------------------------------------------- #
# Hansen SPA - the data-snooping guard                                        #
# --------------------------------------------------------------------------- #


def _spa_panel(n: int, edge: float, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Build a loss panel where every candidate beats the benchmark by ``edge``.

    With ``edge == 0`` no candidate has a real advantage (the SPA null holds).
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="B")
    bench = np.abs(rng.standard_normal(n)) * 0.3 + 0.5
    cols = {}
    for k in range(4):
        cols[f"m{k}"] = bench - edge + rng.standard_normal(n) * 0.02
    return pd.DataFrame(cols, index=idx), pd.Series(bench, index=idx, name="bench")


@pytest.mark.unit
def test_spa_does_not_reject_under_the_null() -> None:
    losses, bench = _spa_panel(400, edge=0.0, seed=10)
    result = hansen_spa(losses, bench, n_boot=399, seed=7)
    # No candidate truly beats the benchmark => the consistent p-value is large.
    assert result.p_value_consistent > 0.10
    assert isinstance(result, SPAResult)
    assert result.n_models == 4


@pytest.mark.unit
def test_spa_rejects_when_a_model_dominates() -> None:
    losses, bench = _spa_panel(500, edge=0.08, seed=11)
    result = hansen_spa(losses, bench, n_boot=399, seed=7)
    # Every candidate genuinely beats the benchmark => the null is rejected.
    assert result.p_value_consistent < 0.05


@pytest.mark.unit
def test_spa_is_deterministic_under_fixed_seed() -> None:
    losses, bench = _spa_panel(300, edge=0.02, seed=12)
    r1 = hansen_spa(losses, bench, n_boot=199, seed=7)
    r2 = hansen_spa(losses, bench, n_boot=199, seed=7)
    assert r1.p_value_consistent == pytest.approx(r2.p_value_consistent, abs=1e-12)
    assert r1.p_value_lower == pytest.approx(r2.p_value_lower, abs=1e-12)
    assert r1.p_value_upper == pytest.approx(r2.p_value_upper, abs=1e-12)


@pytest.mark.unit
def test_spa_to_dict_is_json_safe() -> None:
    losses, bench = _spa_panel(200, edge=0.0, seed=13)
    d = hansen_spa(losses, bench, n_boot=99, seed=7).to_dict()
    assert set(d) == {
        "p_value_consistent",
        "p_value_lower",
        "p_value_upper",
        "best_model",
        "n_models",
        "n_boot",
    }


@pytest.mark.unit
def test_fallback_spa_matches_arch_on_null_and_dominating() -> None:
    """The no-arch container path (``_fallback_spa``) agrees with ``arch`` SPA.

    The deployed container ships without ``arch``, so the self-contained
    stationary-bootstrap fallback is what runs in production. It must reproduce
    the arch verdict: a large p-value under the null and a small one when a model
    truly dominates.
    """
    from volforecast._rng import make_rng

    for edge, dominates in ((0.0, False), (0.10, True)):
        losses, bench = _spa_panel(600, edge=edge, seed=30)
        excess = bench.to_numpy()[:, None] - losses.to_numpy()
        fallback = _fallback_spa(excess, list(losses.columns), 599, make_rng(7))
        arch_result = hansen_spa(losses, bench, n_boot=599, seed=7)
        assert isinstance(fallback, SPAResult)
        assert fallback.n_models == 4
        if dominates:
            # Both the container fallback and arch reject the no-edge null.
            assert fallback.p_value_consistent < 0.05
            assert arch_result.p_value_consistent < 0.05
        else:
            # Neither path crowns a snooped winner under the null.
            assert fallback.p_value_consistent > 0.05
            assert arch_result.p_value_consistent > 0.05


@pytest.mark.unit
def test_fallback_spa_handles_zero_variance_column() -> None:
    """A constant (zero-variance) candidate column does not crash the fallback.

    The studentized statistic for a zero-variance column is ``-inf`` (never the
    max), so such a column is harmlessly ignored by the bootstrap.
    """
    from volforecast._rng import make_rng

    rng = np.random.default_rng(21)
    n = 200
    bench = np.abs(rng.standard_normal(n)) * 0.3 + 0.5
    losses = np.column_stack(
        [
            bench + rng.standard_normal(n) * 0.02,  # ordinary
            bench,  # excess identically zero ⇒ zero-variance column
        ]
    )
    excess = bench[:, None] - losses
    result = _fallback_spa(excess, ["ordinary", "flat"], 199, make_rng(7))
    assert 0.0 <= result.p_value_consistent <= 1.0


@pytest.mark.unit
def test_spa_validates_inputs() -> None:
    losses, bench = _spa_panel(50, edge=0.0, seed=14)
    with pytest.raises(ValidationError):
        hansen_spa(losses, bench, n_boot=0)
    with pytest.raises(ValidationError):
        hansen_spa(pd.DataFrame(index=bench.index), bench, n_boot=99)
    with pytest.raises(ValidationError):
        # Too few aligned observations.
        hansen_spa(losses.iloc[:4], bench.iloc[:4], n_boot=99)
    with pytest.raises(ValidationError):
        # losses must be a DataFrame.
        hansen_spa(losses["m0"], bench, n_boot=99)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        # benchmark must be a Series.
        hansen_spa(losses, losses, n_boot=99)  # type: ignore[arg-type]


@pytest.mark.unit
def test_spa_drops_nan_rows_before_bootstrapping() -> None:
    """Rows with any NaN (in a model or the benchmark) are dropped, not crashed."""
    losses, bench = _spa_panel(300, edge=0.0, seed=40)
    losses.iloc[5, 0] = np.nan  # one model NaN
    bench.iloc[9] = np.nan  # one benchmark NaN
    result = hansen_spa(losses, bench, n_boot=199, seed=7)
    assert 0.0 <= result.p_value_consistent <= 1.0


# --------------------------------------------------------------------------- #
# Verdict truth table                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_model_partition_is_disjoint_and_labels_known() -> None:
    assert ML_MODELS.isdisjoint(REFERENCE_MODELS)
    assert "xgboost" in ML_MODELS
    assert "garch" in REFERENCE_MODELS and "har_rv" in REFERENCE_MODELS


@pytest.mark.unit
def test_verdict_best_model_is_strict_qlike_argmin() -> None:
    v = derive_verdict(
        {"garch": 0.50, "har_rv": 0.47, "xgboost": 0.52},
        spa_pvalue=0.30,
        dm_pvalues_vs_best={},
    )
    assert v.best_model == "har_rv"
    assert v.best_model_class is BestModelClass.REFERENCE
    assert v.ml_beats_garch is False
    assert isinstance(v, Verdict)


@pytest.mark.unit
def test_verdict_tie_breaks_in_favour_of_reference() -> None:
    # ML and a reference tie exactly: the reference wins the tie (conservative).
    v = derive_verdict(
        {"xgboost": 0.40, "garch": 0.40, "har_rv": 0.45},
        spa_pvalue=0.01,
        dm_pvalues_vs_best={"xgboost": 0.01},
    )
    assert v.best_model == "garch"
    assert v.best_model_class is BestModelClass.REFERENCE
    assert v.ml_beats_garch is False


@pytest.mark.unit
def test_verdict_crowns_ml_only_when_all_gates_clear() -> None:
    v = derive_verdict(
        {"garch": 0.50, "har_rv": 0.49, "xgboost": 0.40},
        spa_pvalue=0.01,  # significant
        dm_pvalues_vs_best={"xgboost": 0.01},  # significant
    )
    assert v.best_model == "xgboost"
    assert v.best_model_class is BestModelClass.ML
    assert v.ml_beats_garch is True
    assert v.dm_pvalue_vs_best_reference == pytest.approx(0.01)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("spa_p", "dm_p", "expected"),
    [
        (0.40, 0.40, False),  # neither gate clears
        (0.01, 0.40, False),  # SPA only
        (0.40, 0.01, False),  # DM only
        (0.01, 0.01, True),  # both clear
    ],
)
def test_verdict_gate_truth_table(spa_p: float, dm_p: float, expected: bool) -> None:
    # XGBoost has the strictly lowest QLIKE; only the significance gates decide.
    v = derive_verdict(
        {"garch": 0.50, "har_rv": 0.49, "xgboost": 0.45},
        spa_pvalue=spa_p,
        dm_pvalues_vs_best={"xgboost": dm_p},
    )
    assert v.best_model == "xgboost"
    assert v.ml_beats_garch is expected


@pytest.mark.unit
def test_verdict_boundary_alpha_is_strict() -> None:
    # p exactly equal to alpha does NOT clear the (strict ``<``) gate.
    v = derive_verdict(
        {"garch": 0.5, "xgboost": 0.4},
        spa_pvalue=0.05,
        dm_pvalues_vs_best={"xgboost": 0.01},
        alpha=0.05,
    )
    assert v.ml_beats_garch is False


@pytest.mark.unit
def test_verdict_reference_winner_has_no_dm_pvalue() -> None:
    v = derive_verdict(
        {"garch": 0.40, "xgboost": 0.50},
        spa_pvalue=0.01,
        dm_pvalues_vs_best={"xgboost": 0.01},
    )
    assert v.best_model == "garch"
    assert v.dm_pvalue_vs_best_reference is None
    assert v.ml_beats_garch is False


@pytest.mark.unit
def test_verdict_to_dict_round_trips() -> None:
    v = derive_verdict(
        {"garch": 0.5, "xgboost": 0.4},
        spa_pvalue=0.02,
        dm_pvalues_vs_best={"xgboost": 0.03},
    )
    d = v.to_dict()
    assert d["best_model"] == "xgboost"
    assert d["best_model_class"] == "ml"
    assert d["ml_beats_garch"] is True
    assert d["spa_pvalue"] == pytest.approx(0.02)
    assert d["dm_pvalue_vs_best_reference"] == pytest.approx(0.03)


@pytest.mark.unit
def test_verdict_validates_inputs() -> None:
    with pytest.raises(ValidationError):
        derive_verdict({}, spa_pvalue=0.5, dm_pvalues_vs_best={})
    with pytest.raises(ValidationError):
        derive_verdict({"garch": math.nan}, spa_pvalue=0.5, dm_pvalues_vs_best={})
    with pytest.raises(ValidationError):
        derive_verdict({"garch": 0.5}, spa_pvalue=1.5, dm_pvalues_vs_best={})
    with pytest.raises(ValidationError):
        derive_verdict({"garch": 0.5}, spa_pvalue=0.5, dm_pvalues_vs_best={"xgboost": 2.0})


@pytest.mark.unit
def test_verdict_raises_when_ml_winner_missing_dm_pvalue() -> None:
    # XGBoost wins on QLIKE but its DM p-value was not supplied.
    with pytest.raises(ValidationError):
        derive_verdict(
            {"garch": 0.5, "xgboost": 0.4},
            spa_pvalue=0.01,
            dm_pvalues_vs_best={},  # missing xgboost
        )


# --------------------------------------------------------------------------- #
# Honest-null guard - the headline guarantee on GARCH data                    #
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_honest_null_qlike_alone_never_crowns_ml() -> None:
    """A marginally lower ML QLIKE with insignificant gates stays a non-winner.

    This is the encoded honest null: on GARCH-generated data the SPA/DM gates do
    not clear, so even when XGBoost edges out on point QLIKE the verdict reports
    ``ml_beats_garch=False`` by construction.
    """
    # XGBoost wins by a hair on QLIKE but the snooping-robust gates are large.
    qlike_by_model = {"garch": 0.500, "har_rv": 0.495, "egarch": 0.498, "xgboost": 0.494}
    verdict = derive_verdict(
        qlike_by_model,
        spa_pvalue=0.62,  # composite null not rejected
        dm_pvalues_vs_best={"xgboost": 0.55},  # pairwise gap insignificant
    )
    assert verdict.best_model == "xgboost"
    assert verdict.ml_beats_garch is False


@pytest.mark.unit
def test_honest_null_on_garch_losses_does_not_reject_spa(garch_series: pd.DataFrame) -> None:
    """SPA over GARCH-data QLIKE losses does not crown a snooped ML winner.

    Build QLIKE losses for a realized-variance proxy where every model is a noisy
    copy of the benchmark (no model has a real edge - the GARCH-true regime). The
    SPA composite null must NOT be rejected.
    """
    rets = np.log(garch_series["close"]).diff().dropna()
    realized_var = (rets**2).to_numpy()
    n = realized_var.shape[0]
    idx = garch_series.index[1 : 1 + n]
    rng = np.random.default_rng(99)

    # Benchmark = GARCH-like forecast (close to the truth); ML = same plus noise,
    # i.e. no genuine improvement. Losses are per-observation QLIKE.
    bench_fc = realized_var * np.exp(rng.standard_normal(n) * 0.05)
    bench_loss = pd.Series(qlike_loss_series(realized_var, bench_fc), index=idx)
    loss_cols = {}
    for name in ("xgboost", "egarch", "har_rv"):
        fc = realized_var * np.exp(rng.standard_normal(n) * 0.05)
        loss_cols[name] = qlike_loss_series(realized_var, fc)
    losses = pd.DataFrame(loss_cols, index=idx)

    spa = hansen_spa(losses, bench_loss, n_boot=299, seed=7)
    assert spa.p_value_consistent > 0.05

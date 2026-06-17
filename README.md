# volforecast — GARCH vs ML for volatility forecasting

[![CI](https://github.com/FatihHekim0glu/volforecast/actions/workflows/ci.yml/badge.svg)](https://github.com/FatihHekim0glu/volforecast/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

Forecast the **h-day-ahead realized volatility** of a stock index and honestly
test whether **XGBoost** (or a research-only **LSTM**) beats a well-specified
**GARCH(1,1) / HAR-RV** out-of-sample — judged on **QLIKE** plus
**Diebold-Mariano** and **Hansen-SPA** significance.

> **Honest headline (the null-to-modest finding).** GARCH(1,1) and HAR-RV are
> **hard to beat**. Across the model set, XGBoost wins only *marginally* on
> out-of-sample QLIKE — if at all — and the LSTM rarely justifies its compute.
> Hansen-SPA does **not** crown ML a significant winner. This reproduces
> Hansen & Lunde (2005). **No profit is claimed.** The shipped default runs on a
> synthetic GARCH(1,1)-like series, so the null holds *by construction*; real
> data is available via Polygon / `--data`.

The `best_model` and `ml_beats_garch` outputs are a **pure function** of OOS
QLIKE plus DM/SPA significance — never a narrative choice. `ml_beats_garch` is
`True` only when an ML model has the strictly lowest QLIKE **and** beats the best
GARCH/HAR-RV reference by an SPA- and DM-significant margin.

## Install

```bash
uv venv
# Lean serve stack: GARCH (arch) + XGBoost only — NO TensorFlow.
uv pip install -e ".[data,viz,dev]"
```

The research-only LSTM arm lives behind the `[research]` extra (TensorFlow) and
is **never** imported on the serve path or in the API container.

## Quickstart

```bash
# Honest horse race on the synthetic GARCH default (no data key needed):
uv run volforecast run SPY --horizon 5
```

```python
import volforecast as vf

ohlc = vf.generate_garch_ohlc(n_obs=1500, seed=7)          # synthetic, seeded
config = vf.WalkForwardConfig(horizon=5)                    # fit-on-train-only
result = vf.run_walk_forward(ohlc, config=config)           # GARCH/HAR/XGB/...
verdict = vf.derive_verdict(...)                            # pure, honest verdict
```

`import volforecast` pulls in **no** `arch` / `xgboost` / TensorFlow — every
heavy fitter is imported lazily inside its function, and the package is
import-pure.

## Method

- **Target.** h-day-ahead realized volatility over a strictly *forward* window
  `(t+gap, t+gap+h]`, horizons `h ∈ {1, 5, 22}`. RV estimators: Parkinson,
  Garman-Klass (from OHLC), close-to-close.
- **Baselines.** Random-walk vol, EWMA/RiskMetrics (λ=0.94), HAR-RV (Corsi 2009).
- **GARCH family.** GARCH(1,1), EGARCH, GJR-GARCH, Student-t innovations via
  `arch`; a hand-rolled GARCH(1,1) log-likelihood is the parity oracle.
- **ML.** XGBoost on HAR / lagged-RV / VIX features. (LSTM optional, research-only.)
- **Evaluation.** QLIKE (robust to the noisy RV proxy) + MSE; Diebold-Mariano
  pairwise; Hansen-SPA across the whole set (snooping control). Optional
  vol-targeting overlay with a Deflated/Probabilistic Sharpe (`n_trials` = number
  of model configs evaluated).
- **Leakage control.** Anchored/expanding walk-forward with the scaler, GARCH
  params, HAR-RV OLS, and XGBoost booster all fit **inside each train fold only**;
  forward-only targets with an explicit `gap`; purge + embargo sized to `h`.

## Validation

| Check | Reference | Status |
| --- | --- | --- |
| GARCH(1,1) log-likelihood vs `arch` | hand-rolled oracle, tol 1e-6 | _pending impl_ |
| QLIKE / Diebold-Mariano | reference values | _pending impl_ |
| XGBoost determinism (fixed seed) | byte-identical predictions | _pending impl_ |
| Forward-target disjointness | feature index ⊆ {≤ t}, target ⊂ {> t+gap} | _pending impl_ |
| Future-perturbation invariance | property test | _pending impl_ |
| Golden best-model on synthetic GARCH | honest null (ML does not beat GARCH) | _pending impl_ |

_(This is a scaffold: the table is populated as the kernels land.)_

## Limitations

- **Survivorship bias: N/A.** A single index series is forecast, not a
  cross-section selected on survival, so survivorship bias does not apply.
- **Synthetic default.** The key-free default is a GARCH(1,1)-generated series, on
  which GARCH is the true model — the null is honest *by construction*. Real data
  (Polygon) may shift the margins but, per Hansen & Lunde (2005), GARCH/HAR-RV
  remain hard to beat.
- **LSTM is research-only.** It is excluded from the container and the serve path;
  any LSTM result is illustrative, not a shipped capability.
- **No profit claim.** The optional overlay's Sharpe is *deflated* with the true
  `n_trials`; the project's headline is forecast accuracy, not P&L.

## References

- Hansen, P. R., & Lunde, A. (2005). *A forecast comparison of volatility models:
  does anything beat a GARCH(1,1)?* Journal of Applied Econometrics, 20(7).
- Corsi, F. (2009). *A simple approximate long-memory model of realized
  volatility.* Journal of Financial Econometrics, 7(2). (HAR-RV)
- Hansen, P. R. (2005). *A test for superior predictive ability.* JBES, 23(4). (SPA)
- Diebold, F. X., & Mariano, R. S. (1995). *Comparing predictive accuracy.* JBES,
  13(3). (DM)
- Patton, A. J. (2011). *Volatility forecast comparison using imperfect volatility
  proxies.* Journal of Econometrics, 160(1). (QLIKE robustness)
- Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* Journal
  of Portfolio Management, 40(5). (DSR/PSR)

## License

[MIT](LICENSE) © FatihHekim0glu

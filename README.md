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

## Results on the synthetic default

These are the **actual** numbers from the shipped key-free default — a seeded
GARCH(1,1)-like OHLC series (`generate_garch_ohlc(n_obs=1500, seed=7)`),
Garman-Klass RV, anchored walk-forward (`train_window=504`, `gap=1`). Reproduce
them with the [Reproduce](#reproduce) block; OOS **QLIKE** (lower is better):

| horizon | garch | egarch | har_rv | ewma | xgboost | rw | best_model | ML beats GARCH? | SPA *p* |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: |
| **h=1**  | 1.798 | 1.811 | **0.529** | 1.677 | 0.565 | 1.905 | `har_rv` | **NO** | 0.503 |
| **h=5**  | 0.171 | **0.168** | 0.178 | 0.198 | 0.197 | 1.820 | `egarch` | **NO** | 0.897 |
| **h=22** | 0.931 | 0.940 | **0.200** | 1.002 | 0.213 | 2.083 | `har_rv` | **NO** | 0.884 |

Read this honestly. A **GARCH/HAR-RV reference wins at every horizon**, never the
ML arm, so `ml_beats_garch=False` throughout and `n_effective_trials=6`. XGBoost
is *competitive* — a close second at h=1 (0.565 vs 0.529) and h=22 (0.213 vs
0.200) — which is exactly the "marginal, if at all" ML story. At h=1 its pairwise
Diebold-Mariano *p* against the best reference is 0.015, yet it is **not** crowned
the winner: it does not have the lowest QLIKE *and* the Hansen-SPA composite null
(*p*=0.50) is not rejected. The SPA gate (which controls the snooping across all
six configs) is the discipline that keeps a lucky pairwise margin from minting a
false "ML wins". This is Hansen & Lunde (2005) reproduced by construction.

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
run = vf.run_vol_forecast(ohlc, horizon=5)                  # the public entrypoint
print(run.summary.best_model, run.summary.ml_beats_garch)   # honest verdict
figs = vf.build_vol_forecast_figures(run)                   # {forecast,error}_figure
```

`run_vol_forecast` is the single, import-pure entrypoint the CLI and the FastAPI
route both call: it runs the leakage-guarded walk-forward (GARCH/EGARCH/HAR-RV/
EWMA/XGBoost/RW), scores OOS QLIKE/MSE, runs Hansen-SPA + Diebold-Mariano, and
returns the pure `best_model` / `ml_beats_garch` verdict. The research-only LSTM
is never reachable from it, so the serve path can never import TensorFlow.

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

The compute core is validated oracle → test: each numeric claim is pinned to an
**independent reference** at a fixed tolerance, then locked by a test. The suite
is **270 passed** (3 research/LSTM tests deselected — the serve path is green
without TensorFlow), **coverage 91.9 %** (gate `fail_under=85`), ruff + strict
mypy clean.

| Check | Independent oracle | Tolerance | Test |
| --- | --- | --- | --- |
| GARCH(1,1) log-likelihood | hand-rolled LL vs `arch` at identical params (matched `0.94`-decay EWMA backcast over 75 obs) | `abs=1e-6` | `tests/parity` |
| GARCH forecast aggregation | hand-rolled path vs `arch` analytic forecast | `abs=1e-9` | `tests/parity` |
| QLIKE / Diebold-Mariano | closed-form / SciPy Student-t reference | exact / `1e-12` | `tests/unit`, `tests/parity` |
| XGBoost determinism (fixed seed) | byte-identical predictions across re-fits | exact | `tests/parity` |
| Forward-target disjointness | feature index ⊆ {≤ t}, target window ⊂ {> t+gap}, disjoint | exact (set algebra) | `tests/property` |
| Future-perturbation invariance | perturb returns after the forecast origin → forecasts unchanged | exact | `tests/property` |
| Fit-on-train-only (scaler/GARCH/HAR/XGB) | future perturbation leaves every train-fold fit intact | exact | `tests/property` |
| Golden best-model on synthetic GARCH | honest null: a reference wins, `ml_beats_garch=False` at h∈{1,5,22} | locked | `tests/regression`, `tests/integration` |
| Import purity | `import volforecast` triggers no `arch`/`xgboost`/TF import, no I/O | subprocess | `tests/regression` |

## Reproduce

```bash
git clone https://github.com/FatihHekim0glu/volforecast && cd volforecast
uv venv && uv pip install -e ".[data,viz,dev]"

# 1) Full quality gate (ruff + strict mypy + pytest-cov >= 85, NO TensorFlow):
uv run ruff check .
uv run mypy src
uv run pytest -m "not research" --cov=volforecast      # 270 passed, ~92% cov

# 2) Regenerate the synthetic results table above (seed=7, byte-stable):
uv run python - <<'PY'
import volforecast as vf
ohlc = vf.generate_garch_ohlc(n_obs=1500, seed=7)
for h in (1, 5, 22):
    s = vf.run_vol_forecast(ohlc, horizon=h, seed=7).summary
    q = {k: round(v, 3) for k, v in s.qlike_by_model.items()}
    print(f"h={h:>2}  best={s.best_model:<7} ml_beats_garch={s.ml_beats_garch} "
          f"SPA_p={s.spa_pvalue:.3f}  QLIKE={q}")
PY
```

Same seed → byte-identical QLIKE, `best_model`, and SPA *p*-values; the table is
locked by `tests/regression`. The research-only LSTM (the `[research]` extra) is
the only thing TensorFlow gates, and it never appears on this path.

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

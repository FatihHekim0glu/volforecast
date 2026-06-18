# Design

This document explains how `volforecast` is put together: the layering, the data
flow through a single walk-forward fold, the leakage invariants the compute core
guarantees, and the testing strategy that keeps the honest headline honest. For
*why* individual contested choices were made, see the numbered ADRs in
[`docs/decisions/`](decisions/).

## Goals and non-goals

**Goals**

- A pure, typed (`mypy --strict`, `py.typed`), side-effect-free compute core that
  can be audited line by line and vendored into a backend without dragging UI,
  network, or TensorFlow dependencies along.
- A faithful GARCH family via `arch`, with a hand-rolled GARCH(1,1)
  log-likelihood parity-tested to `1e-6` against `arch`
  ([ADR-0003](decisions/0003-garch-arch-parity.md)).
- A **leakage-proof** horse race: scaler, GARCH params, HAR-RV OLS, and the
  XGBoost booster are all fit **inside each train fold only**
  ([ADR-0001](decisions/0001-fit-on-train-only.md)), with forward-only RV targets
  separated by an explicit gap ([ADR-0002](decisions/0002-forward-rv-target-gap.md)).
- A statistically defensible verdict (`best_model` / `ml_beats_garch`) that
  survives multiplicity correction (Hansen-SPA) and is *mechanically* prevented
  from over-claiming ([ADR-0004](decisions/0004-honest-garch-hard-to-beat.md)).

**Non-goals**

- Beating GARCH(1,1)/HAR-RV. The honest finding is that, on OOS QLIKE with SPA
  control, ML does not, by a significant margin
  ([ADR-0004](decisions/0004-honest-garch-hard-to-beat.md)). No profit is claimed.
- A live trading system. The optional vol-targeting overlay exists only to attach
  a *deflated* Sharpe to the forecasts; it is not a strategy.
- Shipping the LSTM. It is research-only and never on the serve path or in the
  container ([ADR-0005](decisions/0005-lstm-research-only-no-tf-container.md)).

## Layered architecture

The package is strictly layered; each layer imports only from the ones below it.
`src/` has **zero import-time side effects**, guarded by a subprocess
import-purity test, and `import volforecast` pulls in **no** `arch` / `xgboost` /
TensorFlow; every heavy fitter is imported lazily inside its function.

```
        cli.py (Typer)        plots.py (Plotly, lazy)        api/ (FastAPI)
             |                       |                            |
  ┌──────────┴────────────────────────┴────────────────────────────┘
  │                          pipeline.py
  │     run_vol_forecast  ·  build_vol_forecast_figures  ·  Summary/Run
  │     (the ONE import-pure entrypoint the CLI and the API both call)
  ├──────────────────────────────────────────────────────────────────
  │                          evaluation/
  │   qlike.py · tests.py (DM · Hansen-SPA · HAC) · dsr.py · verdict.py
  │   (QLIKE/MSE · pairwise DM · SPA over the set · pure best_model deriver)
  ├──────────────────────────────────────────────────────────────────
  │                          walkforward/engine.py
  │   anchored/expanding · purge + embargo sized to h · FIT-ON-TRAIN-ONLY
  ├──────────────────────────────────────────────────────────────────
  │   garch/models.py     ml/xgb.py          baselines.py     backtest/
  │   (arch + oracle)     (XGBoost, lazy)    (RW · EWMA ·      overlay.py
  │   ml/lstm.py = research-only, NEVER served   HAR-RV/Corsi)  costs.py
  ├──────────────────────────────────────────────────────────────────
  │   realized/estimators.py        features/har.py        data.py
  │   (Parkinson · Garman-Klass ·   (HAR daily/weekly/      (synthetic GARCH
  │    close-to-close · forward      monthly RV, .shift-     generator +
  │    RV target with gap)           lagged)                 Polygon loader)
  ├──────────────────────────────────────────────────────────────────
  │   foundation (no internal deps): _validation · _constants · _typing
  │   _exceptions · _manifest (BLAKE2b config hash) · _rng (seeded PCG64)
  └──────────────────────────────────────────────────────────────────
```

### Foundation (`_*.py`)

Copied verbatim from the HRP infra and renamed `hrp` → `volforecast`.
`_constants.py` is the single source of truth (`TRADING_DAYS=252`,
`RISKMETRICS_LAMBDA=0.94`, `SUPPORTED_HORIZONS={1,5,22}`, the HAR windows).
`_validation.py` holds the input guards; `_rng.py` provides seeded PCG64
substreams and `_manifest.py` the `RunManifest` whose BLAKE2b config-hash makes a
whole run reproducible: the same seed yields byte-identical QLIKE, verdict, and
SPA *p*-value.

### `realized/` and `features/`

`estimators.py` builds the three RV proxies (Parkinson, Garman-Klass from OHLC,
close-to-close) and, critically, `forward_rv_target`, the strictly-forward
target over `(t+gap, t+gap+h]` ([ADR-0002](decisions/0002-forward-rv-target-gap.md)).
`features/har.py` builds the Corsi (2009) HAR-RV design (daily/weekly/monthly RV
components), every column `.shift()`-lagged so no feature can see its own bar.

### `garch/`, `ml/`, `baselines.py`

`garch/models.py` wraps `arch` for GARCH(1,1)/EGARCH/GJR with normal or Student-t
innovations, and carries a **hand-rolled GARCH(1,1) log-likelihood** as the parity
oracle: it mirrors `arch`'s `0.94`-decay EWMA backcast over 75 observations so the
two LLs agree to `1e-6` ([ADR-0003](decisions/0003-garch-arch-parity.md)).
`ml/xgb.py` is the XGBoost forecaster on HAR/lagged-RV/VIX features (single-thread,
fixed-seed deterministic). `ml/lstm.py` is the research-only LSTM behind a lazy
TensorFlow import, **never** re-exported from `__init__` and **never** reachable
from `pipeline.run_vol_forecast`
([ADR-0005](decisions/0005-lstm-research-only-no-tf-container.md)). `baselines.py`
holds the real bars to beat: random-walk vol, EWMA/RiskMetrics (λ=0.94), and the
HAR-RV OLS.

### `walkforward/engine.py`

The leakage-control heart. Anchored (expanding) or rolling folds, a **purge** that
drops boundary rows whose target window overlaps the test fold, and an **embargo
sized to `h`**. Every estimator is fit **inside the train fold only**
([ADR-0001](decisions/0001-fit-on-train-only.md)), the explicit fix for the
"fit-on-the-full-series then evaluate OOS" anti-pattern.

### `evaluation/`

`qlike.py` computes QLIKE (robust to the noisy RV proxy, Patton 2011) and MSE.
`tests.py` holds pairwise Diebold-Mariano (1995) with HAC/Newey-West long-run
variance and the Hansen-SPA (2005) consistent *p*-value over the **whole** model
set (the snooping control). `verdict.py` is a **pure function** mapping
`(qlike_by_model, spa_pvalue, dm_pvalues)` → `best_model` / `ml_beats_garch`
([ADR-0004](decisions/0004-honest-garch-hard-to-beat.md)). `dsr.py` (reused from
HRP) attaches a Deflated/Probabilistic Sharpe to the optional overlay only, with
the true `n_trials`.

## Data flow through one walk-forward fold

```
train fold (≤ t)  ──►  realized_volatility (RV proxy)  ──►  HAR features (.shift-lagged)
   (OHLC)               │                                          │
                        ├─► GARCH/EGARCH:  fit arch on train returns ─► σ̂_{t+h}
                        ├─► HAR-RV:        fit OLS on train RV       ─► RV̂_{t+h}
                        ├─► EWMA λ=0.94:   recursion on train RV     ─► RV̂_{t+h}
                        ├─► XGBoost:       fit booster on train (X,y)─► RV̂_{t+h}
                        └─► RW:            last train RV             ─► RV̂_{t+h}
                        │
                        ▼  EVERY fit uses train-fold data ONLY (ADR-0001)
   purge + embargo sized to h  ·  forward target on (t+gap, t+gap+h]  (ADR-0002)
                        │
   test fold (> t+gap)  ──►  realized forward RV  vs  each model's RV̂  ─► QLIKE/MSE
                        │
                        ▼ (aggregate the per-point losses across all folds)
   Hansen-SPA over the set  ·  pairwise Diebold-Mariano vs best reference
                        │
                        ▼
        verdict.py  ──►  best_model · ml_beats_garch  (pure-derived; ADR-0004)
```

The benchmark in DM/SPA is the **best GARCH/HAR-RV reference**, so a challenger
must clear a *well-specified* bar, not a straw man.

## Key invariants

The compute core guarantees, and tests enforce:

1. **Forward-target disjointness.** For any origin `t`, the feature index ⊆ `{≤ t}`
   and the target window ⊂ `{> t + gap}`; the two sets are disjoint (set-algebra
   property test).
2. **Fit-on-train-only.** Perturbing returns strictly after a fold's forecast
   origin leaves that fold's scaler, GARCH params, HAR OLS, XGB booster, and
   forecast unchanged (future-perturbation-invariance property test).
3. **GARCH parity.** The hand-rolled GARCH(1,1) LL equals `arch`'s at identical
   params to `1e-6`; the aggregated forecast matches `arch`'s analytic path.
4. **Scale behaviour.** RV estimators scale linearly with the return scale; QLIKE
   is invariant to a common rescaling of forecast and target.
5. **Determinism.** Same seed / `RunManifest` → byte-identical QLIKE, verdict, and
   SPA *p*-value (XGBoost and the SPA bootstrap are seeded; BLAS/OMP pinned to one
   thread in `conftest`).
6. **Verdict safety.** `ml_beats_garch` cannot be `True` unless an ML model has the
   strictly lowest QLIKE **and** clears both the SPA composite-null gate and the
   pairwise DM gate (truth-table unit-tested; [ADR-0004](decisions/0004-honest-garch-hard-to-beat.md)).
7. **Import purity.** Importing any `src/volforecast` module triggers no I/O, no
   network, no `arch`/`xgboost`/TensorFlow import, no RNG draw (subprocess test).
8. **No-TF serve path.** `run_vol_forecast` and everything it transitively imports
   never touch TensorFlow; the LSTM is unreachable from it
   ([ADR-0005](decisions/0005-lstm-research-only-no-tf-container.md)).

## Testing strategy

Tests are partitioned by intent under `tests/` (markers in `pyproject.toml`);
seeded `conftest` fixtures (`garch_series`, `har_series`, `pure_noise`) give every
layer deterministic, adversarial inputs.

- **`unit/`**: isolated kernels: RV estimators, the HAR builder, QLIKE/MSE, the
  verdict truth table.
- **`parity/`**: golden checks vs independent references: hand-rolled GARCH(1,1)
  LL vs `arch` at `1e-6`, the forecast path at `1e-9`, DM vs a closed form,
  XGBoost determinism (byte-identical).
- **`property/`** (Hypothesis): the invariants above: forward-target
  disjointness, future-perturbation invariance, RV scale behaviour, HAR lag-safety.
- **`regression/`**: the honest null, locked: a GARCH/HAR-RV reference wins on
  `garch_series` with `ml_beats_garch=False` at every horizon; no-lookahead
  walk-forward; the import-purity subprocess test.
- **`integration/`**: end-to-end `run_vol_forecast` on the synthetic GARCH series.

The serve-path suite is **284 passed** with the 4 research/LSTM tests **deselected**
(`-m "not research"`), coverage **94.9 %** (gate 85). TensorFlow is required only
for the deselected LSTM tests.

## Backend & frontend boundary

The compute core is decoupled from delivery. The backend vendors
`volforecast[data]` (`arch` + `xgboost`, **not** `[research]`) under
`api/lib/volforecast/` and exposes `POST /tools/volforecast/run`, fitting
GARCH+HAR+XGBoost **per request** (fast on a short index series, no pre-trained
artifact). The LSTM arm is not vendored and is not importable on the serve path; a
Polygon-provider failure degrades to the synthetic generator
(`data_source: polygon|synthetic`) rather than hard-failing. The response returns
summary scalars plus Plotly `{data, layout}` figures; the frontend surfaces the
pure-derived verdict and a prominent **"ML beats GARCH: NO"** badge as the first
thing a visitor reads.

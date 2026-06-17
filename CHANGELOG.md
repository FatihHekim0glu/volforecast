# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [0.1.0] - 2026-06-17

Initial public release: an import-pure, typed library that forecasts h-day-ahead
realized volatility and honestly tests whether ML beats a well-specified
GARCH(1,1)/HAR-RV out-of-sample. The honest, null-to-modest headline (Hansen &
Lunde 2005) holds by construction on the synthetic GARCH default.

### Added

- **Public entrypoint.** `run_vol_forecast` — the single, import-pure horse-race
  entrypoint the CLI and the FastAPI route both call. It accepts an OHLC frame (or
  a close-price series), runs the leakage-guarded walk-forward
  (GARCH/EGARCH/HAR-RV/EWMA/XGBoost/RW), scores OOS QLIKE/MSE, runs Hansen-SPA +
  pairwise Diebold-Mariano against the best GARCH/HAR-RV reference, and returns the
  pure `best_model` / `ml_beats_garch` verdict (`VolForecastSummary` /
  `VolForecastRun`). The research-only LSTM is never reachable from it, so the
  serve path never imports TensorFlow. `build_vol_forecast_figures` assembles the
  `forecast_figure` (RV actual vs model forecasts) and `error_figure` (OOS QLIKE by
  model) as plain JSON dicts.
- **Realized vol & features.** Parkinson, Garman-Klass, and close-to-close RV
  estimators plus the strictly-forward `forward_rv_target` with an explicit `gap`
  (`realized/estimators.py`); the `.shift()`-lagged Corsi (2009) HAR-RV feature
  builder (`features/har.py`).
- **Models.** GARCH(1,1)/EGARCH/GJR with normal/Student-t innovations via `arch`,
  plus a hand-rolled GARCH(1,1) log-likelihood parity oracle matched to `arch`'s
  backcast (`garch/models.py`); the XGBoost forecaster (`ml/xgb.py`); and the
  random-walk / EWMA(λ=0.94) / HAR-RV baselines (`baselines.py`). The research-only
  LSTM (`ml/lstm.py`, lazy TensorFlow) is never served or re-exported.
- **Leakage-proof walk-forward.** The anchored/expanding, fit-on-train-only engine
  with purge + embargo sized to `h` (`walkforward/engine.py`).
- **Evaluation.** QLIKE/MSE losses (`evaluation/qlike.py`); Diebold-Mariano,
  Hansen-SPA, and HAC inference (`evaluation/tests.py`); the pure `best_model` /
  `ml_beats_garch` verdict (`evaluation/verdict.py`); and the optional
  vol-targeting overlay with an honest Deflated-Sharpe deflation by the true
  `n_trials` (`backtest/overlay.py`).
- **Data & I/O.** Synthetic GARCH(1,1)-like OHLC generator + real loader
  (`data.py`), lazy Plotly figure builders (`plots.py`), and the Typer CLI
  (`cli.py`, delegating to `run_vol_forecast`).
- **Reused infra (renamed from HRP).** Foundation helpers `_constants`, `_typing`,
  `_exceptions`, `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), `_rng` (seeded PCG64 + substreams), `py.typed`; the
  Deflated/Probabilistic Sharpe (`evaluation/dsr.py`); and the fixed-bps cost model
  (`backtest/costs.py`).
- **Tests & CI.** Partitioned suite (`unit/parity/property/regression/integration`)
  with seeded conftest fixtures (`garch_series`, `har_series`, `pure_noise`),
  including the GARCH(1,1)↔`arch` parity (`1e-6`), future-perturbation invariance,
  forward-target disjointness, import-purity, and the locked honest-null regression.
  270 serve-path tests pass (3 research/LSTM deselected), coverage ~92%. CI runs
  ruff + strict mypy + pytest-cov ≥ 85 on Python 3.11/3.12/3.13, plus the
  `no-ai-attribution` guard.

### Docs

- `README` with the honest null-to-modest headline, the actual synthetic-default
  QLIKE table across h∈{1,5,22}, an oracle→test Validation table, a Reproduce
  block, Limitations, and references.
- `docs/DESIGN.md` and ADRs `0001`–`0005` (fit-on-train-only, forward-RV-target
  gap, GARCH↔`arch` parity, honest GARCH-hard-to-beat verdict, LSTM
  research-only/no-TF container).
- `CITATION.cff`.

[Unreleased]: https://github.com/FatihHekim0glu/volforecast/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/volforecast/releases/tag/v0.1.0

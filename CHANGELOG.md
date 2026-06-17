# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `run_vol_forecast` — the public, import-pure horse-race entrypoint the CLI and
  the FastAPI route both call. It accepts an OHLC frame (or a close-price series),
  runs the leakage-guarded walk-forward (GARCH/EGARCH/HAR-RV/EWMA/XGBoost/RW),
  scores OOS QLIKE/MSE, runs Hansen-SPA + pairwise Diebold-Mariano against the
  best GARCH/HAR-RV reference, and returns the pure `best_model` /
  `ml_beats_garch` verdict (`VolForecastSummary`/`VolForecastRun`). The
  research-only LSTM is never reachable from it, so the serve path never imports
  TensorFlow.
- `build_vol_forecast_figures` — assembles the `forecast_figure` (RV actual vs
  model forecasts) and `error_figure` (OOS QLIKE by model) as plain JSON dicts.

### Changed

- Implemented every previously-stubbed kernel (realized-vol estimators, HAR
  features, GARCH family + hand-rolled parity oracle, XGBoost, baselines,
  QLIKE/MSE, DM/SPA/HAC, the pure verdict, the vol-targeting overlay, and the
  fit-on-train-only walk-forward engine) and wired them into one coherent library.
- `cli.run` now delegates to `run_vol_forecast` so the CLI and the API share a
  single pipeline.
- `vol_target_overlay` is now implemented (vol-targeted position sizing with a
  per-side bps cost and an honest Deflated-Sharpe deflation by the true
  `n_trials`).
- The walk-forward suite pins BLAS/OpenMP/XGBoost to a single thread (via
  `conftest`) for reproducibility and speed; replaced the stale stub-contract and
  `xfail` scaffold tests with behavioural unit/integration/regression tests
  (including the honest-null guarantee on `garch_series`).

## [0.1.0] - 2026-06-17

### Added

- Initial package skeleton (src-layout, import name `volforecast`).
- Core helpers copied from the HRP infra and renamed: `_constants`, `_typing`,
  `_exceptions`, `_validation`, `_manifest` (`RunManifest` with BLAKE2b
  config-hash), and `_rng` (seeded PCG64 generator + substream spawning),
  plus `py.typed`.
- Stub signatures with full contracts for: realized-volatility estimators
  (`realized/estimators.py`), the HAR-RV feature builder (`features/har.py`),
  the GARCH family with a hand-rolled GARCH(1,1) log-likelihood parity oracle
  (`garch/models.py`), the XGBoost forecaster (`ml/xgb.py`) and the
  research-only LSTM (`ml/lstm.py`, never served), the random-walk / EWMA /
  HAR-RV baselines (`baselines.py`), QLIKE/MSE losses (`evaluation/qlike.py`),
  Diebold-Mariano / Hansen-SPA / HAC inference (`evaluation/tests.py`), the
  pure `best_model` / `ml_beats_garch` verdict (`evaluation/verdict.py`), the
  vol-targeting overlay with a Deflated Sharpe guard (`backtest/overlay.py`),
  and the fit-on-train-only walk-forward engine (`walkforward/engine.py`).
- Synthetic GARCH(1,1)-like OHLC generator + real loader (`data.py`), lazy
  Plotly figure builders (`plots.py`), and the Typer CLI stub (`cli.py`).
- Reused infra: Deflated/Probabilistic Sharpe (`evaluation/dsr.py`) and the
  fixed-bps cost model (`backtest/costs.py`).
- Partitioned test suite (`unit/parity/property/regression/integration`) with
  seeded conftest fixtures (`garch_series`, `har_series`, `pure_noise`).
- CI (ruff + strict mypy + pytest-cov ≥ 85, Python 3.11/3.12/3.13) and the
  `no-ai-attribution` guard.

[Unreleased]: https://github.com/FatihHekim0glu/volforecast/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/FatihHekim0glu/volforecast/releases/tag/v0.1.0

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

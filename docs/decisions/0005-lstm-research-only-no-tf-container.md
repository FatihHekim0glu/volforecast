# ADR-0005: The LSTM is research-only, no TensorFlow on the serve path or in the container

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** volforecast maintainers
- **Related:** [ADR-0004](0004-honest-garch-hard-to-beat.md) (honest verdict)

## Context

A "GARCH vs ML" study is expected to include a deep-learning arm, so we implement
an LSTM volatility forecaster for completeness. But TensorFlow is a very heavy
dependency (hundreds of MB, slow cold start, frequent ABI/CUDA friction) and the
honest finding is that the LSTM **rarely justifies its compute**: on OOS QLIKE it
does not clear the GARCH/HAR-RV bar by a SPA-significant margin
([ADR-0004](0004-honest-garch-hard-to-beat.md)). Vendoring TensorFlow into the
hosted API container would bloat the image and slow every request to ship a model
that, by the project's own honest verdict, does not win, while also widening the
import-purity and supply-chain surface.

## Decision

The LSTM is **research-only** and is **never** on the serve path or in the
container:

- It lives behind a **`[research]` extra** (`tensorflow>=2.16`); the lean
  serve-path stack is `[data]` = `arch` + `xgboost` + numpy/pandas/scipy/
  statsmodels only. There is **no `[all]`** extra that could pull TF in transitively.
- `ml/lstm.py` imports TensorFlow **lazily** inside its functions and is **not**
  re-exported from `volforecast/__init__.py` (only `XGBForecaster`/`fit_xgb` are).
- The public entrypoint `pipeline.run_vol_forecast`, which the CLI **and** the
  FastAPI route both call, uses the default served set
  (`garch/egarch/har_rv/ewma/xgboost/rw`) and **never** routes to the LSTM, so the
  serve path cannot transitively import TensorFlow.
- The backend vendors `volforecast[data]` (not `[research]`) under
  `api/lib/volforecast/`; the LSTM module is not vendored.
- LSTM tests are marked `research` and **deselected** on the serve-path suite
  (`-m "not research"`), so CI and coverage are green **without** TensorFlow
  installed (284 passed, 4 deselected).

## Consequences

- **Positive.** The hosted container stays lean (GARCH + XGBoost only), cold starts
  are fast, and the import-purity guarantee (`import volforecast` pulls in no TF)
  holds.
- **Positive.** The serve path is reproducible and CI runs without the heaviest,
  most fragile dependency in the stack.
- **Positive / honest.** We do not pay a large compute and image-size cost to ship a
  model that the verdict does not crown; any LSTM result is illustrative research,
  not a shipped capability.
- **Cost.** Anyone wanting the LSTM must opt in via `pip install -e ".[research]"`
  and run it offline; it is a second-class, behind-the-glass arm by design.
- **Risk addressed.** "TensorFlow leaks onto the serve path / into the container"
  is prevented structurally (lazy import + no re-export + `not research` CI gate +
  `[data]`-only vendoring), not just by convention.

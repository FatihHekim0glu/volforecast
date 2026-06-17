# ADR-0001: Fit every estimator inside the train fold only

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** volforecast maintainers
- **Related:** [ADR-0002](0002-forward-rv-target-gap.md) (forward target gap)

## Context

The single most common — and most invisible — error in published
"ML-beats-classical" volatility-forecasting results is **full-sample leakage**:
fitting a feature scaler, the GARCH parameters, or the ML model on the *entire*
series and only afterward slicing an "out-of-sample" tail to score. Because the
in-sample fit has already seen the test period's level and scale, the OOS metric
is optimistically biased, and the bias flatters the most flexible model (the ML
arm) the most. This is the explicit anti-pattern the sibling Stock-Price-Forecast
project fell into, and it is the top risk for this repo: get it wrong and the
honest null silently becomes a fake "ML wins".

Every estimator in the horse race has a fittable state that can leak:

- the RV/feature **scaler**,
- the **GARCH/EGARCH/GJR** parameters (via `arch`),
- the **HAR-RV OLS** coefficients (Corsi 2009),
- the **XGBoost** booster.

## Decision

**No estimator is ever fit on the full series.** In `walkforward/engine.py`, every
fit happens **inside the current train fold only**, and the resulting frozen state
is then asked to forecast the disjoint test fold:

- The walk-forward is anchored (expanding) or rolling; for each fold we slice the
  train window, fit the scaler, GARCH, HAR OLS, and XGB on **that slice**, and
  forecast forward.
- Returns are computed with `pct_change(fill_method=None)` (no forward-fill before
  differencing) and features are `.shift()`-lagged, so no row can see its own bar.
- A **purge** drops boundary rows whose target window overlaps the test fold and an
  **embargo sized to `h`** removes the rows just after the train fold
  ([ADR-0002](0002-forward-rv-target-gap.md)).

The property test is the canonical leakage detector: **future-perturbation
invariance**. Perturbing the returns strictly *after* a fold's forecast origin must
leave that fold's scaler, GARCH params, HAR coefficients, XGB booster, and final
forecast bit-for-bit unchanged. If any fitter had peeked at future data, the
perturbation would move its output and the test would fail.

## Consequences

- **Positive.** The OOS QLIKE is an honest estimate of generalization error; the
  verdict in [ADR-0004](0004-honest-garch-hard-to-beat.md) rests on numbers that
  cannot be inflated by leakage.
- **Positive.** The guard is *mechanical* (a property test), not a code-review
  promise, so it cannot silently regress.
- **Cost.** Per-fold refitting (especially `arch` MLE and XGBoost) is much slower
  than one full-sample fit. We mitigate with per-request fitting on a short index
  series and single-threaded BLAS/OMP/XGBoost pins for reproducibility; the cost is
  accepted as the price of correctness.
- **Risk addressed.** "Fit on the full series, then evaluate OOS" — the headline
  leakage failure mode — is rejected and continuously tested against.

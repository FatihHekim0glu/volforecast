# ADR-0003: A hand-rolled GARCH(1,1) log-likelihood as the `arch` parity oracle

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** volforecast maintainers
- **Related:** [ADR-0004](0004-honest-garch-hard-to-beat.md) (honest verdict)

## Context

GARCH(1,1) is the *bar to beat* in this project (Hansen & Lunde 2005), so the
correctness of the GARCH numbers is load-bearing: if the reference were subtly
mis-fit, any "ML wins / loses" verdict would be meaningless. We fit the GARCH
family with the well-tested `arch` library rather than re-deriving a maximum-
likelihood optimizer, but "we used a library" is not evidence that we are driving
it correctly (right variance recursion, right backcast seed, right scaling).

We need an *independent* check that the model we think we are fitting is the model
`arch` actually fits, without taking on a second heavy dependency or re-
implementing the MLE.

## Decision

Ship a **hand-rolled GARCH(1,1) log-likelihood** (`garch_11_log_likelihood` in
`garch/models.py`) as a parity oracle, and pin it against `arch` at the **same
parameters** to a tight tolerance:

- The oracle reproduces `arch`'s variance recursion exactly, including the
  **backcast seed**: an EWMA of the first `min(75, n)` squared residuals with
  `0.94 ** k` decay weights (`_BACKCAST_TAU = 75`, `_BACKCAST_DECAY = 0.94`). We
  mirror this so the recursion starts from the identical `σ²_0`.
- Returns are scaled to percent style (`scale = 100`) before fitting, matching how
  `arch` is numerically happiest and how we drive it.
- The parity test (`tests/parity`) asserts the oracle LL equals
  `res.loglikelihood` at the fitted params to **`abs=1e-6`**, and a second test
  asserts the aggregated multi-step forecast matches `arch`'s analytic path to
  **`abs=1e-9`**.

The oracle is a *verification artifact*, not the production fitter: the serve path
uses `arch`'s optimizer; the oracle exists to prove we are reading and forecasting
from it correctly.

## Consequences

- **Positive.** The GARCH reference is independently validated, so the honest null
  rests on a *well-specified* GARCH, not a mis-driven one.
- **Positive.** No second numerical dependency: the oracle is ~one function and is
  exercised only in tests.
- **Cost.** The oracle must track `arch`'s conventions (backcast window, decay,
  scaling). If `arch` changed its default backcast, the `1e-6` test would fail
  loudly (which is the point) and the oracle would be updated to match. That
  coupling is stated here so it is not a surprise.
- **Risk addressed.** "The benchmark is silently wrong", which would invalidate
  every comparison, is closed by an independent, pinned cross-check.

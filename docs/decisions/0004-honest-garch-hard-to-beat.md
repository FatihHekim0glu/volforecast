# ADR-0004: The verdict is a pure function — GARCH/HAR-RV is hard to beat

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** volforecast maintainers
- **Related:** [ADR-0001](0001-fit-on-train-only.md), [ADR-0003](0003-garch-arch-parity.md)

## Context

The headline of a "GARCH vs ML" tool is exactly the thing a motivated author bends:
it is tempting to declare the shiny model the winner on a lower point metric, a
single horizon, or a cherry-picked window. Hansen & Lunde (2005) — "does anything
beat a GARCH(1,1)?" — found that, judged honestly out-of-sample, very little does.
Our synthetic default is GARCH-generated, so GARCH is the *true* model and ML
cannot reliably beat it; the null holds **by construction**. The risk is not the
finding, it is letting narrative override the evidence.

A single low QLIKE is not enough to crown ML: with six model configs evaluated, the
best point score is partly luck (data-snooping). We need a verdict that (a) is
computed only from the evidence and (b) is hard against snooping.

## Decision

`best_model` and `ml_beats_garch` are a **pure function** of the out-of-sample
evidence — never an editorial choice (`derive_verdict` in `evaluation/verdict.py`,
truth-table unit-tested):

1. `best_model` = the strict QLIKE argmin over the full set. Ties break
   **conservatively toward the REFERENCE family** (GARCH/EGARCH/GJR/HAR-RV/EWMA/RW).
2. `ml_beats_garch` is `True` **iff all** of:
   - (a) `best_model` is an ML model (`xgboost`/`lstm`), **and**
   - (b) the **Hansen-SPA** consistent *p*-value over the whole set clears
     significance (`< alpha`, default `0.05`) — the composite "no model beats the
     reference benchmark" null is rejected, controlling the snooping across all
     configs, **and**
   - (c) the **pairwise Diebold-Mariano** *p*-value of `best_model` vs the best
     GARCH/HAR-RV reference is significant (`< alpha`).

   If any condition fails, `ml_beats_garch = False`.

The benchmark for DM/SPA is the **best** reference, so ML must clear a
well-specified bar ([ADR-0003](0003-garch-arch-parity.md)), and the loss is QLIKE
(robust to the noisy RV proxy, Patton 2011). Crucially, condition (b) means a lucky
*pairwise* DM margin alone — which can occur, e.g. XGBoost at h=1 in the synthetic
default has DM *p*≈0.015 — does **not** mint a win: it is neither the QLIKE argmin
nor SPA-significant (SPA *p*≈0.50), so the function returns `False`.

## Consequences

- **Positive.** The verdict cannot over-claim: the SPA gate makes "ML wins" survive
  a multiple-comparisons correction, not just one favorable pairwise test.
- **Positive.** The honest null is *encoded*, not asserted in prose — on
  GARCH-generated data a reference wins at every horizon
  (`har_rv`/`egarch`) and `ml_beats_garch=False`, locked by `tests/regression`.
- **Positive / honest framing.** XGBoost is competitive (a close second at h=1 and
  h=22), which is the accurate "marginal, if at all" ML story — surfaced, not
  hidden, and still not crowned.
- **Cost.** A genuinely superior ML model on real data must clear a deliberately
  high, snooping-robust bar before the tool will say so. We accept the higher Type
  II risk to keep Type I (false "ML wins") near zero.
- **Risk addressed.** "Narrative crowns ML on a lower point metric" is rejected: the
  verdict is mechanical, multiplicity-corrected, and tested.

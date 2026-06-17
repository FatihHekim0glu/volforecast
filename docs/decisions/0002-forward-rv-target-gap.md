# ADR-0002: Forward-only RV target with an explicit gap

- **Status:** Accepted
- **Date:** 2026-06-17
- **Deciders:** volforecast maintainers
- **Related:** [ADR-0001](0001-fit-on-train-only.md) (fit-on-train-only)

## Context

We forecast **realized volatility**, which is itself an aggregate over a *window*
of days, not a point. That makes the target temporally fat: the RV "label" at
origin `t` summarizes several future days. If the target window is allowed to
touch — or overlap — the bars that produced the features, the model is implicitly
scored on data it could see, a subtle form of look-ahead that is easy to introduce
and hard to spot. A horizon-`h` RV target built naively (e.g. a centered or a
`(t, t+h]` window with `gap=0` and features drawn from the same bars) leaks.

The leakage surface is the boundary between the feature timestamp and the start of
the target window. We need that boundary to be explicit and tunable, not implicit
in array slicing.

## Decision

The target is the **strictly forward** realized volatility aggregated over the
future window `(t + gap, t + gap + h]`, with an **explicit `gap` parameter**
(default `gap=1`):

```
RV_target(t) = sqrt( sum_{k=1}^{h} RV^2_{ t + gap + k } )
```

(`forward_rv_target` in `realized/estimators.py`.) Two guarantees follow and are
property-tested:

- **Disjointness.** For any origin `t`, the feature index ⊆ `{≤ t}` and the target
  window ⊂ `{> t + gap}`; the two sets are **disjoint**. A property test asserts
  this with set algebra across horizons and gaps.
- **Trailing trim.** The last `h + gap` origins have no complete forward window and
  are dropped, rather than padded with partial or forward-filled values.

The `gap` is wired through the walk-forward purge/embargo
([ADR-0001](0001-fit-on-train-only.md)): the embargo is sized to `h` and the purge
removes any train row whose target window would reach into the test fold, so the
forward-target discipline holds *across* folds, not just within a single label.

## Consequences

- **Positive.** The one place leakage could hide in an RV problem — the
  feature/target boundary — is named (`gap`), defaulted conservatively, and proven
  disjoint by a test.
- **Positive.** Horizons `h ∈ {1, 5, 22}` and the gap compose cleanly with the
  purge/embargo, so changing `h` cannot reintroduce overlap.
- **Cost.** We discard `h + gap` origins at the tail and one gap day at each
  origin, trading a little usable data for an unambiguous, non-overlapping label.
- **Risk addressed.** "RV target window overlaps the feature window" — silent
  look-ahead specific to volatility targets — is rejected by construction.

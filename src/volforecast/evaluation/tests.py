"""Forecast-comparison inference: Diebold-Mariano, Hansen SPA, and HAC SEs.

Two complementary tools control for the fact that forecast losses are serially
correlated and that comparing many models inflates the chance of a spurious
"winner":

- **Diebold-Mariano** (1995): a PAIRWISE test of equal predictive accuracy on the
  loss differential, with a HAC (Newey-West / Bartlett) long-run variance so the
  test is valid under autocorrelated losses (Harvey-Leybourne-Newbold small-sample
  correction applied).
- **Hansen SPA** (2005): a test of the COMPOSITE null that NO model in the
  candidate set beats the benchmark, via a studentized-maximum statistic and a
  stationary bootstrap — this is the multiple-testing guard that stops us crowning
  ML a winner by data snooping. If ``arch`` is installed its
  :class:`arch.bootstrap.SPA` is used; otherwise a self-contained fallback runs.

Importing this module has no side effects (``arch`` is imported lazily inside the
SPA function). HAC variance is implemented in NumPy with the Andrews (1991)
automatic lag.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DMResult:
    """Immutable result of a pairwise Diebold-Mariano test.

    Attributes
    ----------
    statistic:
        The (HLN small-sample-corrected) DM test statistic.
    p_value:
        The two-sided p-value (Student-t reference, ``T - 1`` df).
    mean_loss_diff:
        The mean loss differential ``L(model_a) - L(model_b)`` (negative ⇒ A
        better).
    n_obs:
        The number of aligned loss observations.
    favored:
        The label of the model with the lower mean loss (``"a"`` or ``"b"``).
    """

    statistic: float
    p_value: float
    mean_loss_diff: float
    n_obs: int
    favored: str

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "statistic": float(self.statistic),
            "p_value": float(self.p_value),
            "mean_loss_diff": float(self.mean_loss_diff),
            "n_obs": int(self.n_obs),
            "favored": str(self.favored),
        }


@dataclass(frozen=True, slots=True)
class SPAResult:
    """Immutable result of a Hansen (2005) SPA test.

    Attributes
    ----------
    p_value_consistent, p_value_lower, p_value_upper:
        Hansen's three SPA p-values for the composite null that no candidate
        beats the benchmark; ``consistent`` is the headline value.
    best_model:
        The label of the candidate with the lowest mean loss vs the benchmark.
    n_models:
        The number of candidate models tested.
    n_boot:
        The number of bootstrap replicates.
    """

    p_value_consistent: float
    p_value_lower: float
    p_value_upper: float
    best_model: str
    n_models: int
    n_boot: int

    def to_dict(self) -> dict[str, Any]:
        """Return a plain, JSON-serializable ``dict`` of this result."""
        return {
            "p_value_consistent": float(self.p_value_consistent),
            "p_value_lower": float(self.p_value_lower),
            "p_value_upper": float(self.p_value_upper),
            "best_model": str(self.best_model),
            "n_models": int(self.n_models),
            "n_boot": int(self.n_boot),
        }


def newey_west_lrv(x: NDArray[np.float64], *, lag: int | None = None) -> float:
    r"""Newey-West (1987) Bartlett long-run variance of a 1-D series.

    .. math::

        \widehat{\omega} = \gamma_0 + 2\sum_{h=1}^{L}
                            \left(1 - \tfrac{h}{L+1}\right)\gamma_h,

    with the Andrews (1991) automatic lag ``L = ceil(4 (T/100)^{2/9})`` when
    ``lag`` is ``None``. Used as the denominator of the DM statistic so it is
    valid under autocorrelated loss differentials.

    Parameters
    ----------
    x:
        A 1-D float array (e.g. the loss differential).
    lag:
        Bartlett truncation lag; ``None`` selects the Andrews rule.

    Returns
    -------
    float
        The (non-negative) long-run variance estimate.

    Raises
    ------
    ValidationError
        If ``x`` has fewer than two finite observations or ``lag < 0``.
    """
    raise NotImplementedError


def diebold_mariano(
    loss_a: pd.Series | NDArray[np.float64],
    loss_b: pd.Series | NDArray[np.float64],
    *,
    label_a: str = "a",
    label_b: str = "b",
) -> DMResult:
    r"""Pairwise Diebold-Mariano test of equal predictive accuracy.

    Tests :math:`H_0:\ \mathbb{E}[d_t] = 0` for the loss differential
    :math:`d_t = L^{a}_t - L^{b}_t`, using the HAC long-run variance from
    :func:`newey_west_lrv` and the Harvey-Leybourne-Newbold (1997) small-sample
    correction. A negative ``mean_loss_diff`` favours model A.

    Parameters
    ----------
    loss_a, loss_b:
        Aligned per-observation loss series/arrays for the two models (e.g. from
        :func:`volforecast.evaluation.qlike.qlike_loss_series`).
    label_a, label_b:
        Human-readable labels recorded in the result's ``favored`` field.

    Returns
    -------
    DMResult
        The test statistic, two-sided p-value, mean loss differential, and the
        favoured model.

    Raises
    ------
    ValidationError
        If the two loss series differ in length or have fewer than two
        observations.
    """
    raise NotImplementedError


def hansen_spa(
    losses: pd.DataFrame,
    benchmark_loss: pd.Series,
    *,
    n_boot: int = 999,
    seed: int = 7,
) -> SPAResult:
    """Hansen (2005) Superior Predictive Ability test over the model set.

    Tests the composite null that no candidate model has lower expected loss than
    the benchmark, controlling for data snooping across the whole set. Works in
    LOSS space: the loss differential is ``benchmark_loss - losses[model]`` so a
    positive mean means the model beats the benchmark. Uses ``arch.bootstrap.SPA``
    when available (lazy import), else a self-contained stationary-bootstrap
    fallback seeded from :func:`volforecast._rng.make_rng`.

    Parameters
    ----------
    losses:
        A ``(T, M)`` frame of per-observation losses, one column per candidate
        model.
    benchmark_loss:
        The length-``T`` benchmark per-observation loss (e.g. GARCH(1,1) or
        HAR-RV), aligned to ``losses``.
    n_boot:
        Number of bootstrap replicates (``> 0``).
    seed:
        Master seed for the bootstrap RNG.

    Returns
    -------
    SPAResult
        The consistent/lower/upper SPA p-values and the best candidate label.

    Raises
    ------
    ValidationError
        If ``losses`` is empty, misaligned, or ``n_boot <= 0``, or there are too
        few aligned observations.
    """
    raise NotImplementedError

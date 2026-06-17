"""Command-line interface (Typer).

A thin orchestration layer over the compute library: load (or synthesize) OHLC,
run the leakage-guarded walk-forward GARCH-vs-ML horse race, evaluate with
QLIKE/DM/SPA, and print the honest verdict (``best_model``, ``ml_beats_garch``).
Typer and the heavy fitters are imported lazily inside :func:`build_app`, so
importing this module has no side effects (no command registration or I/O at
import time). The module-level ``app`` is a lazily-built singleton consumed by
the ``volforecast`` console-script entry point.

Importing this module has no side effects.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import typer


def build_app() -> typer.Typer:
    """Construct and return the Typer application.

    Registers the CLI commands (``run`` and ``demo``) on a fresh ``typer.Typer``
    instance. Typer is imported lazily inside this function so that importing
    :mod:`volforecast.cli` does not import Typer or register any commands.

    Returns
    -------
    typer.Typer
        The configured Typer application.
    """
    raise NotImplementedError


def run(
    ticker: str = "SPY",
    *,
    start: str = "2015-01-01",
    end: str = "2023-12-31",
    horizon: int = 5,
    models: list[str] | None = None,
    rv_estimator: str = "garman_klass",
    cost_bps: float = 10.0,
    data_source_pref: str = "auto",
    seed: int = 7,
) -> dict[str, object]:
    """Run the end-to-end horse race and return the JSON-safe summary.

    Loads OHLC for ``ticker`` (degrading to the synthetic GARCH generator when no
    data key is present), runs the walk-forward forecast for the configured
    ``models`` at ``horizon``, evaluates QLIKE/MSE + Diebold-Mariano + Hansen-SPA,
    and derives the honest verdict. This is the shared engine behind both the CLI
    ``run`` command and the FastAPI route, so the two never diverge.

    Parameters
    ----------
    ticker:
        The symbol to forecast (default ``"SPY"``).
    start, end:
        Inclusive date range (``YYYY-MM-DD``).
    horizon:
        Forecast horizon in trading days (1, 5, or 22).
    models:
        Model labels to evaluate; ``None`` uses the default served set
        (GARCH/EGARCH/HAR-RV/EWMA/XGBoost/RW — never the research-only LSTM).
    rv_estimator:
        RV proxy for features and target.
    cost_bps:
        Per-side cost in basis points for the optional overlay.
    data_source_pref:
        ``"auto"``/``"polygon"``/``"synthetic"``.
    seed:
        Master seed.

    Returns
    -------
    dict[str, object]
        A JSON-serializable summary: ``qlike_by_model``, ``mse_by_model``,
        ``best_model``, ``dm_pvalues``, ``spa_pvalue``, ``ml_beats_garch``,
        ``n_effective_trials``, ``data_source``.

    Raises
    ------
    ValidationError
        If any argument is out of domain (unknown horizon/model, bad dates).
    """
    raise NotImplementedError

"""Unit tests for the Plotly figure builders and the Typer CLI.

Covers:

- ``volforecast.plots`` - both figure builders (realized-vol actual-vs-forecasts
  line chart and the QLIKE-by-model bar). Every builder must return a plain
  ``{"data", "layout"}`` mapping whose contents are JSON-serializable (no
  numpy/pandas/Plotly object leaks across the API boundary), and we assert real
  numerical structure (trace count, alignment, NaN -> None, best-bar highlight,
  ascending sort) rather than merely "it runs".
- ``volforecast.cli`` - ``build_app`` wiring (``run``/``demo`` registered, fresh
  instance per call, help output), plus a guarded tiny synthetic forecast smoke
  run that exercises the shared :func:`volforecast.cli.run` orchestration once
  the walk-forward arm is implemented (skipped while it is still a stub so this
  group's suite stays green in isolation).

All inputs are synthetic/seeded; nothing touches the network.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from volforecast import plots
from volforecast._exceptions import ValidationError

pytestmark = pytest.mark.unit


def _assert_figure_dict(fig: object) -> dict:
    """Assert ``fig`` is a ``{"data", "layout"}`` mapping with JSON-safe contents.

    Returns the figure so callers can make further structural assertions. The
    ``json.dumps`` round-trip is the load-bearing check: it fails loudly if any
    numpy scalar/array, pandas object, or Plotly graph-object leaked through.
    """
    assert isinstance(fig, dict)
    assert set(fig) == {"data", "layout"}
    assert isinstance(fig["data"], list)
    assert isinstance(fig["layout"], dict)
    encoded = json.dumps(fig)  # raises if anything non-JSON leaked through
    assert json.loads(encoded) == fig
    return fig


# --------------------------------------------------------------------------- #
# Fixtures (seeded, self-contained - independent of the stubbed kernels)       #
# --------------------------------------------------------------------------- #
@pytest.fixture
def rv_panel() -> tuple[pd.Series, pd.DataFrame]:
    """A small realized-vol series and an aligned per-model forecast frame."""
    index = pd.date_range("2021-01-01", periods=6, freq="B")
    realized = pd.Series([0.10, 0.12, 0.11, 0.13, 0.09, 0.14], index=index, name="rv_target")
    forecasts = pd.DataFrame(
        {
            "garch": [0.10, 0.11, 0.12, 0.12, 0.10, 0.13],
            "har_rv": [0.11, 0.12, 0.11, 0.13, 0.10, 0.13],
            "xgboost": [0.15, 0.14, 0.13, 0.14, 0.12, 0.11],
        },
        index=index,
    )
    return realized, forecasts


# --------------------------------------------------------------------------- #
# rv_forecast_figure                                                           #
# --------------------------------------------------------------------------- #
def test_rv_forecast_figure_one_trace_for_truth_plus_each_model(
    rv_panel: tuple[pd.Series, pd.DataFrame],
) -> None:
    """The figure has a realized-vol line plus exactly one line per model column."""
    realized, forecasts = rv_panel
    fig = _assert_figure_dict(plots.rv_forecast_figure(realized, forecasts))

    # One reference line ("realized vol") + one per model column.
    assert len(fig["data"]) == 1 + forecasts.shape[1]
    names = [t["name"] for t in fig["data"]]
    assert names[0] == "realized vol"
    assert set(names[1:]) == set(forecasts.columns)
    for trace in fig["data"]:
        assert trace["type"] == "scatter"
        assert trace["mode"] == "lines"


def test_rv_forecast_figure_values_and_iso_x_axis(
    rv_panel: tuple[pd.Series, pd.DataFrame],
) -> None:
    """Truth and each model's y-values match the inputs; the x-axis is ISO strings."""
    realized, forecasts = rv_panel
    fig = plots.rv_forecast_figure(realized, forecasts)

    truth = fig["data"][0]
    np.testing.assert_allclose(np.asarray(truth["y"]), realized.to_numpy())
    # Datetime index serialized to ISO strings - no Timestamp leaked.
    assert truth["x"][0] == realized.index[0].isoformat()
    assert all(isinstance(v, str) for v in truth["x"])

    xgb = next(t for t in fig["data"] if t["name"] == "xgboost")
    np.testing.assert_allclose(np.asarray(xgb["y"]), forecasts["xgboost"].to_numpy())


def test_rv_forecast_figure_nan_becomes_none(
    rv_panel: tuple[pd.Series, pd.DataFrame],
) -> None:
    """A NaN in the realized series serializes to JSON ``null`` (not numpy NaN)."""
    realized, forecasts = rv_panel
    realized = realized.copy()
    realized.iloc[2] = np.nan

    fig = _assert_figure_dict(plots.rv_forecast_figure(realized, forecasts))
    truth = fig["data"][0]
    assert truth["y"][2] is None
    # The surrounding finite values are untouched.
    assert truth["y"][0] == pytest.approx(0.10)


def test_rv_forecast_figure_inner_aligns_on_common_index() -> None:
    """Forecasts covering only part of the realized index are inner-aligned."""
    full = pd.date_range("2021-01-01", periods=6, freq="B")
    realized = pd.Series(np.linspace(0.1, 0.2, 6), index=full)
    # Forecasts only cover the last four dates.
    forecasts = pd.DataFrame({"garch": np.linspace(0.1, 0.18, 4)}, index=full[2:])

    fig = plots.rv_forecast_figure(realized, forecasts)
    truth = fig["data"][0]
    # Both lines are restricted to the four common dates.
    assert len(truth["x"]) == 4
    assert truth["x"][0] == full[2].isoformat()


def test_rv_forecast_figure_custom_title(
    rv_panel: tuple[pd.Series, pd.DataFrame],
) -> None:
    """The custom title propagates into the layout."""
    realized, forecasts = rv_panel
    fig = plots.rv_forecast_figure(realized, forecasts, title="custom RV")
    assert fig["layout"]["title"] == {"text": "custom RV"}


def test_rv_forecast_figure_disjoint_index_raises() -> None:
    """No shared index between truth and forecasts raises ``ValidationError``."""
    realized = pd.Series([0.1, 0.2], index=pd.date_range("2021-01-01", periods=2, freq="B"))
    forecasts = pd.DataFrame(
        {"garch": [0.1, 0.2]}, index=pd.date_range("2022-01-01", periods=2, freq="B")
    )
    with pytest.raises(ValidationError):
        plots.rv_forecast_figure(realized, forecasts)


# --------------------------------------------------------------------------- #
# qlike_bar_figure                                                             #
# --------------------------------------------------------------------------- #
def test_qlike_bar_figure_sorted_ascending_best_first() -> None:
    """Bars are sorted ascending so the lowest-QLIKE model is leftmost."""
    qlike_by_model = {"xgboost": 0.70, "garch": 0.50, "har_rv": 0.45, "rw": 0.90}
    fig = _assert_figure_dict(plots.qlike_bar_figure(qlike_by_model))

    trace = fig["data"][0]
    assert trace["type"] == "bar"
    # Ascending by QLIKE: har_rv (0.45) < garch (0.50) < xgboost (0.70) < rw (0.90).
    assert trace["x"] == ["har_rv", "garch", "xgboost", "rw"]
    np.testing.assert_allclose(np.asarray(trace["y"]), [0.45, 0.50, 0.70, 0.90])


def test_qlike_bar_figure_default_highlight_is_argmin() -> None:
    """With no explicit best_model, the argmin (leftmost) bar is highlighted."""
    fig = plots.qlike_bar_figure({"garch": 0.5, "xgboost": 0.7, "har_rv": 0.45})
    colors = fig["data"][0]["marker"]["color"]
    # Exactly one highlighted bar, and it is the leftmost (lowest-QLIKE) one.
    assert colors[0] != colors[1]
    assert colors.count(colors[0]) == 1


def test_qlike_bar_figure_explicit_best_model_highlight() -> None:
    """An explicit ``best_model`` is the highlighted (distinct-colour) bar."""
    qlike_by_model = {"garch": 0.40, "xgboost": 0.70, "har_rv": 0.45}
    fig = plots.qlike_bar_figure(qlike_by_model, best_model="garch")
    trace = fig["data"][0]
    highlight_idx = trace["x"].index("garch")
    colors = trace["marker"]["color"]
    # The garch bar gets a unique colour; the others share the muted colour.
    assert colors.count(colors[highlight_idx]) == 1


def test_qlike_bar_figure_nan_qlike_sorts_last_and_serializes_none() -> None:
    """A non-finite QLIKE sorts to the far right and serializes to ``null``."""
    qlike_by_model = {"garch": 0.5, "broken": float("nan"), "har_rv": 0.4}
    fig = _assert_figure_dict(plots.qlike_bar_figure(qlike_by_model))
    trace = fig["data"][0]
    # The broken model with NaN QLIKE is pushed to the last position.
    assert trace["x"][-1] == "broken"
    assert trace["y"][-1] is None


def test_qlike_bar_figure_custom_title() -> None:
    """The custom title propagates into the layout."""
    fig = plots.qlike_bar_figure({"garch": 0.5}, title="custom bar")
    assert fig["layout"]["title"] == {"text": "custom bar"}


def test_qlike_bar_figure_empty_raises() -> None:
    """An empty QLIKE mapping raises ``ValidationError``."""
    with pytest.raises(ValidationError):
        plots.qlike_bar_figure({})


# --------------------------------------------------------------------------- #
# CLI: structure / help (works regardless of kernel implementation state)      #
# --------------------------------------------------------------------------- #
def test_cli_help_lists_run_and_demo() -> None:
    """``--help`` exits 0 and lists both the ``run`` and ``demo`` commands."""
    import typer
    from typer.testing import CliRunner

    from volforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["--help"])
    assert result.exit_code == 0, result.output
    assert "run" in result.output
    assert "demo" in result.output
    assert isinstance(build_app(), typer.Typer)


def test_cli_run_help_shows_options() -> None:
    """``run --help`` documents the headline options (horizon, models, source)."""
    from typer.testing import CliRunner

    from volforecast.cli import build_app

    result = CliRunner().invoke(build_app(), ["run", "--help"])
    assert result.exit_code == 0, result.output
    out = result.output.lower()
    assert "horizon" in out
    assert "models" in out


def test_cli_no_args_shows_help() -> None:
    """Invoking with no arguments prints help (no_args_is_help) and lists demo."""
    from typer.testing import CliRunner

    from volforecast.cli import build_app

    result = CliRunner().invoke(build_app(), [])
    # no_args_is_help exits with Typer's usage code (2) and lists the commands.
    assert result.exit_code == 2
    assert "demo" in result.output
    assert "run" in result.output


def test_build_app_is_isolated_instance() -> None:
    """build_app returns a fresh Typer app each call (no shared mutable state)."""
    import typer

    from volforecast.cli import build_app

    app_a = build_app()
    app_b = build_app()
    assert isinstance(app_a, typer.Typer)
    assert app_a is not app_b


def test_cli_run_rejects_bad_date_range() -> None:
    """``run`` validates the date range before any heavy compute (end <= start)."""
    from volforecast.cli import run

    with pytest.raises(ValidationError):
        run("SYN", start="2020-01-01", end="2019-01-01", data_source_pref="synthetic")


# --------------------------------------------------------------------------- #
# CLI: tiny synthetic forecast smoke run (guarded on the walk-forward arm)     #
# --------------------------------------------------------------------------- #
def _walk_forward_ready() -> bool:
    """True once the (other-group) walk-forward kernel is implemented.

    The CLI ``run`` orchestration depends on ``run_walk_forward``; while that
    arm is still a stub (raises ``NotImplementedError``) the end-to-end smoke run
    cannot pass. We probe it on a trivial frame so this group's suite stays green
    in isolation and turns the smoke run on automatically once the kernel lands.
    """
    from volforecast.data import generate_garch_ohlc
    from volforecast.walkforward.engine import WalkForwardConfig, run_walk_forward

    ohlc = generate_garch_ohlc(n_obs=40, seed=7)
    config = WalkForwardConfig(horizon=1, train_window=10, models=("rw",), seed=7)
    try:
        run_walk_forward(ohlc, config=config, rv_estimator="close_to_close")
    except NotImplementedError:
        return False
    except Exception:
        return True
    return True


def test_cli_demo_synthetic_smoke_run() -> None:
    """The ``demo`` command runs the full synthetic horse race offline and exits 0.

    Skipped while the walk-forward kernel is still a stub (this group only owns
    ``plots``/``cli``); it activates automatically once that arm is implemented.
    """
    if not _walk_forward_ready():
        pytest.skip("walk-forward kernel not yet implemented (other group)")

    from typer.testing import CliRunner

    from volforecast.cli import build_app

    result = CliRunner().invoke(
        build_app(), ["demo", "--horizon", "5", "--rv-estimator", "close_to_close"]
    )
    assert result.exit_code == 0, result.output
    assert "volforecast GARCH-vs-ML walk-forward horse race" in result.stdout
    assert "data source        : synthetic" in result.stdout
    assert "ML beats GARCH" in result.stdout


def test_cli_run_synthetic_summary_shape() -> None:
    """``run`` returns the documented JSON-safe summary keys on synthetic data.

    Guarded on the walk-forward kernel (see :func:`_walk_forward_ready`).
    """
    if not _walk_forward_ready():
        pytest.skip("walk-forward kernel not yet implemented (other group)")

    from volforecast.cli import run

    summary = run(
        "SYN",
        start="2016-01-04",
        end="2020-12-31",
        horizon=5,
        models=["har_rv", "ewma", "rw", "xgboost"],
        rv_estimator="close_to_close",
        data_source_pref="synthetic",
        seed=7,
    )
    expected_keys = {
        "qlike_by_model",
        "mse_by_model",
        "best_model",
        "best_model_class",
        "dm_pvalues",
        "spa_pvalue",
        "ml_beats_garch",
        "n_effective_trials",
        "data_source",
    }
    assert expected_keys <= set(summary)
    assert summary["data_source"] == "synthetic"
    # The whole summary must be JSON-serializable (crosses the API boundary).
    json.dumps(summary)
    # Honest null by construction on GARCH-generated data: ML must NOT be crowned.
    assert summary["ml_beats_garch"] is False


# --------------------------------------------------------------------------- #
# _jsonify helper (the JSON-safety boundary shared by both figure builders)    #
# --------------------------------------------------------------------------- #
def test_jsonify_coerces_numpy_and_pandas_to_native() -> None:
    """Nested numpy/pandas containers become plain JSON-serializable Python types."""
    value = {
        "scalar": np.float64(1.5),
        "array": np.array([1.0, 2.0]),
        "ints": (np.int64(3), 4),
        "ts": pd.Timestamp("2021-01-01"),
        "period": pd.Period("2021-01", freq="M"),
    }
    out = plots._jsonify(value)
    # The whole structure round-trips through JSON unchanged (no numpy/pandas leak).
    assert json.loads(json.dumps(out)) == out
    assert out["scalar"] == 1.5 and isinstance(out["scalar"], float)
    assert out["array"] == [1.0, 2.0]
    assert out["ints"] == [3, 4]
    assert out["ts"].startswith("2021-01-01")
    assert out["period"].startswith("2021-01")


def test_jsonify_maps_non_finite_floats_to_none() -> None:
    """NaN/Inf floats map to ``None`` so the figure stays JSON-serializable."""
    assert plots._jsonify(float("nan")) is None
    assert plots._jsonify(float("inf")) is None
    assert plots._jsonify(1.25) == 1.25


# --------------------------------------------------------------------------- #
# CLI: ``run`` subcommand body (exercises the Typer-registered run command)    #
# --------------------------------------------------------------------------- #
def test_cli_run_command_bad_dates_exits_nonzero() -> None:
    """The ``run`` subcommand reports a bad date range as a non-zero exit (no traceback).

    This drives the registered ``run`` command body and the shared error-reporting
    path without paying for a heavy fit (the date check fails fast).
    """
    from typer.testing import CliRunner

    from volforecast.cli import build_app

    result = CliRunner().invoke(
        build_app(),
        ["run", "SYN", "--start", "2020-01-01", "--end", "2019-01-01"],
    )
    assert result.exit_code == 1
    assert "error:" in result.output


def test_cli_run_command_synthetic_exits_zero() -> None:
    """The ``run`` subcommand runs a tiny synthetic horse race and exits 0.

    Guarded on the walk-forward kernel (see :func:`_walk_forward_ready`); covers
    the success branch of the registered ``run`` command end to end.
    """
    if not _walk_forward_ready():
        pytest.skip("walk-forward kernel not yet implemented (other group)")

    from typer.testing import CliRunner

    from volforecast.cli import build_app

    result = CliRunner().invoke(
        build_app(),
        [
            "run",
            "SYN",
            "--start",
            "2016-01-04",
            "--end",
            "2020-12-31",
            "--horizon",
            "5",
            "--rv-estimator",
            "close_to_close",
            "--data-source-pref",
            "synthetic",
            "--models",
            "har_rv",
            "--models",
            "rw",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "best model" in result.stdout

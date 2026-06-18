"""Import-purity and public-API surface guards.

These guard the two load-bearing invariants of the scaffold:

1. ``import volforecast`` has ZERO import-time side effects and pulls in NONE of
   the heavy fitters (``arch``, ``xgboost``, ``tensorflow``) - they are imported
   lazily inside their functions only.
2. The research-only LSTM is NOT reachable from the top-level package or the
   ``volforecast.ml`` subpackage (the container guarantee).
"""

from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.unit
def test_import_volforecast_pulls_no_heavy_deps() -> None:
    """A fresh interpreter importing volforecast must not import arch/xgboost/TF."""
    code = (
        "import sys; import volforecast; "
        "heavy = [m for m in ('arch', 'xgboost', 'tensorflow') if m in sys.modules]; "
        "assert not heavy, heavy; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


@pytest.mark.unit
def test_public_api_is_importable() -> None:
    """Every name in ``volforecast.__all__`` resolves on the package."""
    import volforecast

    missing = [name for name in volforecast.__all__ if not hasattr(volforecast, name)]
    assert not missing, missing
    assert volforecast.__version__ == "0.1.0"


@pytest.mark.unit
def test_lstm_not_reexported_on_serve_path() -> None:
    """The research-only LSTM is not on the top-level or ``ml`` public surface."""
    import volforecast
    import volforecast.ml as ml

    assert not hasattr(volforecast, "fit_lstm")
    assert not hasattr(volforecast, "LSTMForecaster")
    assert "fit_lstm" not in ml.__all__
    assert "LSTMForecaster" not in ml.__all__


@pytest.mark.unit
def test_importing_lstm_module_has_no_side_effects() -> None:
    """Importing the LSTM module itself is pure (TF only loaded lazily, in funcs)."""
    code = (
        "import sys; import volforecast.ml.lstm as lstm; "
        "assert 'tensorflow' not in sys.modules; print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"

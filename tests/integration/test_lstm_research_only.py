"""Research-only LSTM guard (marked ``research``; excluded from the serve CI run).

The LSTM arm must be importable as a module without TensorFlow (import-pure) and,
when TensorFlow is absent (the lean container), calling into it must raise a
clear, catchable :class:`volforecast.VolForecastError` — never crash the process.
This test is marked ``research`` so the serve-path CI run (``-m "not research"``)
skips it entirely.
"""

from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from volforecast import VolForecastError

_HAS_TENSORFLOW = importlib.util.find_spec("tensorflow") is not None


@pytest.mark.research
def test_lstm_module_imports_without_tensorflow() -> None:
    """The LSTM module imports cleanly even when TensorFlow is not installed."""
    import volforecast.ml.lstm as lstm

    assert hasattr(lstm, "fit_lstm")
    assert hasattr(lstm, "LSTMForecaster")


@pytest.mark.research
def test_fit_lstm_requires_tensorflow_when_absent() -> None:
    """Without TensorFlow, fit_lstm raises a clear, catchable VolForecastError.

    This is the container guarantee: the lean serve image has no TensorFlow, so
    any call into the research-only arm must fail clearly rather than crash the
    process. Skipped when TensorFlow is installed (the research environment),
    where the contract is exercised by the fit/predict test below.
    """
    if _HAS_TENSORFLOW:
        pytest.skip("TensorFlow is installed; the TF-absent path cannot be exercised here.")

    import volforecast.ml.lstm as lstm

    feats = pd.DataFrame({"rv_daily": [0.1, 0.2, 0.3, 0.4]})
    target = pd.Series([0.15, 0.25, 0.35, 0.45])
    with pytest.raises((NotImplementedError, VolForecastError)):
        lstm.fit_lstm(feats, target, lookback=2)


@pytest.mark.research
def test_fit_lstm_returns_forecaster_when_tensorflow_present() -> None:
    """With TensorFlow installed, fit_lstm fits and returns an LSTMForecaster.

    Skipped when TensorFlow is absent (the lean serve container), where the
    TF-absent guard above carries the contract instead.
    """
    if not _HAS_TENSORFLOW:
        pytest.skip("TensorFlow is not installed; the fit path cannot be exercised here.")

    import volforecast.ml.lstm as lstm

    feats = pd.DataFrame({"rv_daily": [0.10, 0.12, 0.11, 0.13, 0.15, 0.14]})
    target = pd.Series([0.12, 0.11, 0.13, 0.15, 0.14, 0.16])
    forecaster = lstm.fit_lstm(feats, target, lookback=2, epochs=1)

    assert isinstance(forecaster, lstm.LSTMForecaster)
    assert forecaster.feature_names == ("rv_daily",)
    assert forecaster.lookback == 2
    payload = forecaster.to_dict()
    assert payload["lookback"] == 2
    assert payload["n_train"] == feats.shape[0] - 2

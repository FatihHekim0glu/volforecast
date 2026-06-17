"""Research-only LSTM guard (marked ``research``; excluded from the serve CI run).

The LSTM arm must be importable as a module without TensorFlow (import-pure) and,
when TensorFlow is absent (the lean container), calling into it must raise a
clear, catchable :class:`volforecast.VolForecastError` — never crash the process.
This test is marked ``research`` so the serve-path CI run (``-m "not research"``)
skips it entirely.
"""

from __future__ import annotations

import pandas as pd
import pytest

from volforecast import VolForecastError


@pytest.mark.research
def test_lstm_module_imports_without_tensorflow() -> None:
    """The LSTM module imports cleanly even when TensorFlow is not installed."""
    import volforecast.ml.lstm as lstm

    assert hasattr(lstm, "fit_lstm")
    assert hasattr(lstm, "LSTMForecaster")


@pytest.mark.research
def test_fit_lstm_is_stubbed_or_requires_tensorflow() -> None:
    """Calling fit_lstm raises NotImplementedError (stub) or a clear VolForecastError."""
    import volforecast.ml.lstm as lstm

    feats = pd.DataFrame({"rv_daily": [0.1, 0.2, 0.3, 0.4]})
    target = pd.Series([0.15, 0.25, 0.35, 0.45])
    with pytest.raises((NotImplementedError, VolForecastError)):
        lstm.fit_lstm(feats, target, lookback=2)

import importlib
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPTIMIZER_API_KEY", "test-key")

    def fake_load(tickers=None, start=None, end=None):
        # Like yfinance, return the tickers in alphabetical order
        rng = np.random.default_rng(1)
        fake = pd.DataFrame(rng.normal(0, 0.01, (500, len(tickers))), columns=sorted(tickers))
        return fake, fake.mean(), fake.cov()

    import api
    importlib.reload(api)
    with patch.object(api, "load_returns", side_effect=fake_load):
        api.limiter.reset()
        yield TestClient(api.app)

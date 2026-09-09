import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from fastapi.testclient import TestClient

TICKERS = ["CBA.AX", "BHP.AX", "CSL.AX"]
PAYLOAD = {"tickers": TICKERS, "start": "2021-01-01", "end": "2025-01-01"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("OPTIMIZER_API_KEY", "test-key")
    np.random.seed(1)
    fake = pd.DataFrame(np.random.randn(500, 3) * 0.01, columns=TICKERS)

    def fake_load(tickers=None, start=None, end=None):
        return fake, fake.mean(), fake.cov()

    import importlib
    import api
    importlib.reload(api)
    with patch.object(api, "load_returns", side_effect=fake_load):
        api.limiter.reset()
        yield TestClient(api.app)


def test_rejects_missing_api_key(client):
    assert client.post("/optimize", json=PAYLOAD).status_code == 401


def test_rejects_wrong_api_key(client):
    r = client.post("/optimize", json=PAYLOAD, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_accepts_valid_api_key(client):
    r = client.post("/optimize", json=PAYLOAD, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    assert "max_sharpe" in r.json()


def test_rejects_malformed_ticker(client):
    bad = dict(PAYLOAD, tickers=["CBA.AX", "'; DROP TABLE--"])
    r = client.post("/optimize", json=bad, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_rejects_too_many_tickers(client):
    bad = dict(PAYLOAD, tickers=[f"A{i}.AX" for i in range(11)])
    r = client.post("/optimize", json=bad, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_rejects_inverted_date_range(client):
    bad = dict(PAYLOAD, start="2025-01-01", end="2021-01-01")
    r = client.post("/optimize", json=bad, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_rejects_duplicate_tickers(client):
    bad = dict(PAYLOAD, tickers=["CBA.AX", "CBA.AX"])
    r = client.post("/optimize", json=bad, headers={"X-API-Key": "test-key"})
    assert r.status_code == 422


def test_rate_limit_blocks_after_ten(client):
    codes = [
        client.post("/optimize", json=PAYLOAD, headers={"X-API-Key": "test-key"}).status_code
        for _ in range(11)
    ]
    assert codes.count(200) == 10
    assert codes[-1] == 429

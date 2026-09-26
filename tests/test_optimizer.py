from unittest.mock import patch

import numpy as np
import pandas as pd

from src.optimizer import monte_carlo_search, optimize_portfolios

PAYLOAD = {"tickers": ["CBA.AX", "BHP.AX", "CSL.AX"], "start": "2021-01-01", "end": "2025-01-01"}


def make_returns():
    # BHP.AX clearly has the best returns and the other two lose money.
    # Columns are in alphabetical order, the way yfinance returns them.
    rng = np.random.default_rng(0)
    drift = {"BHP.AX": 0.002, "CBA.AX": -0.001, "CSL.AX": -0.001}
    return pd.DataFrame({t: d + rng.normal(0, 0.01, 500) for t, d in drift.items()})


def test_api_labels_weights_with_the_right_tickers(client):
    # The request lists CBA.AX first, but the data comes back sorted,
    # so the weights must be labelled from the data, not the request
    import api
    returns = make_returns()
    with patch.object(api, "load_returns", return_value=(returns, returns.mean(), returns.cov())):
        r = client.post("/optimize", json=PAYLOAD, headers={"X-API-Key": "test-key"})
    weights = r.json()["max_sharpe"]["weights"]
    assert max(weights, key=weights.get) == "BHP.AX"


def test_weights_sum_to_one_and_are_not_negative():
    returns = make_returns()
    results = optimize_portfolios(returns.mean(), returns.cov())
    for name in ("max_sharpe", "min_variance"):
        weights = np.array(list(results[name]["weights"].values()))
        assert abs(weights.sum() - 1) < 1e-6
        assert (weights >= 0).all()


def test_optimizer_beats_the_random_search():
    returns = make_returns()
    results = optimize_portfolios(returns.mean(), returns.cov())
    assert results["max_sharpe"]["sharpe"] >= results["monte_carlo"]["sharpe"] - 1e-9


def test_monte_carlo_search_is_reproducible():
    returns = make_returns()
    first = monte_carlo_search(returns.mean(), returns.cov(), num_portfolios=500)
    second = monte_carlo_search(returns.mean(), returns.cov(), num_portfolios=500)
    assert first == second

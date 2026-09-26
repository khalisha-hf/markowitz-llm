import numpy as np
import pandas as pd

from src.backtest import evaluate, run_backtest, split_returns


def make_returns():
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2021-01-01", "2025-12-31")
    return pd.DataFrame(rng.normal(0.0005, 0.01, (len(dates), 3)), index=dates,
                        columns=["BHP.AX", "CBA.AX", "CSL.AX"])


def test_split_has_no_overlap_and_no_gap():
    returns = make_returns()
    train, test = split_returns(returns, "2024-01-01")
    assert train.index.max() < pd.Timestamp("2024-01-01") <= test.index.min()
    assert len(train) + len(test) == len(returns)


def test_total_return_and_drawdown_on_known_returns():
    # Up 10%, then down 10%: ends at 0.99 of the start, 10% below its peak
    r = evaluate(np.array([1.0]), pd.DataFrame({"A": [0.10, -0.10]}))
    assert abs(r["total_return"] - (-0.01)) < 1e-12
    assert abs(r["max_drawdown"] - (-0.10)) < 1e-12


def test_max_drawdown_counts_a_loss_on_the_first_day():
    r = evaluate(np.array([1.0]), pd.DataFrame({"A": [-0.10, 0.01, 0.01]}))
    assert abs(r["max_drawdown"] - (-0.10)) < 1e-12


def test_weights_only_use_the_training_period():
    # Changing prices in the test period must not change the fitted weights
    returns = make_returns()
    changed = returns.copy()
    changed.loc["2024-01-01":, "CSL.AX"] += 0.05
    _, weights, _, _ = run_backtest(returns, "2024-01-01")
    _, weights_changed, _, _ = run_backtest(changed, "2024-01-01")
    for name in weights:
        assert np.allclose(weights[name], weights_changed[name])

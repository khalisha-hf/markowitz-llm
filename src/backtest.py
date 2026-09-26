import numpy as np
import pandas as pd

from src.optimizer import (
    TRADING_DAYS, RISK_FREE_RATE,
    max_sharpe_portfolio, min_variance_portfolio,
)


def split_returns(returns, split_date):
    """Split into a fitting period and an unseen testing period."""
    train = returns[returns.index < split_date]
    test = returns[returns.index >= split_date]
    return train, test


def evaluate(weights, test_returns):
    """Apply fixed weights to unseen returns and measure the result."""
    daily = test_returns.values @ weights
    total_return = float(np.prod(1 + daily) - 1)
    years = len(daily) / TRADING_DAYS
    annual_return = float((1 + total_return) ** (1 / years) - 1)
    annual_vol = float(np.std(daily, ddof=1) * np.sqrt(TRADING_DAYS))
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol

    equity = np.cumprod(1 + daily)
    running_peak = np.maximum.accumulate(equity)
    max_drawdown = float(np.min(equity / running_peak) - 1)

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
    }


def run_backtest(returns, split_date):
    train, test = split_returns(returns, split_date)
    mean_returns = train.mean()
    cov_matrix = train.cov()
    n = len(mean_returns)

    strategies = {
        "Max Sharpe (optimised)": max_sharpe_portfolio(mean_returns, cov_matrix),
        "Min Variance (optimised)": min_variance_portfolio(mean_returns, cov_matrix),
        "Equal Weight (benchmark)": np.array([1 / n] * n),
    }

    results = {name: evaluate(w, test) for name, w in strategies.items()}
    return results, strategies, len(train), len(test)

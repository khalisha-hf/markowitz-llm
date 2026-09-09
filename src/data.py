import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WES.AX', 'MQG.AX']


def load_returns(tickers=DEFAULT_TICKERS, start="2021-01-01", end="2025-12-31"):
    data = yf.download(tickers, start=start, end=end)
    close_prices = data['Close']
    returns = close_prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix

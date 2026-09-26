import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ['CBA.AX', 'BHP.AX', 'CSL.AX', 'WES.AX', 'MQG.AX']
# Given to the LLM so it doesn't guess company names from ticker codes
COMPANY_NAMES = {
    'CBA.AX': 'Commonwealth Bank of Australia',
    'BHP.AX': 'BHP Group',
    'CSL.AX': 'CSL Limited',
    'WES.AX': 'Wesfarmers',
    'MQG.AX': 'Macquarie Group',
}
DEFAULT_START = "2021-01-01"
DEFAULT_END = "2025-12-31"


def load_returns(tickers=DEFAULT_TICKERS, start=DEFAULT_START, end=DEFAULT_END):
    data = yf.download(tickers, start=start, end=end)
    close_prices = data['Close']
    returns = close_prices.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return returns, mean_returns, cov_matrix

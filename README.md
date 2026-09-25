# Markowitz LLM Project

A personal project exploring portfolio optimization using Modern Portfolio Theory (Markowitz), applied to ASX stocks, with plans to add a Large Language Model (LLM) layer to explain the results in plain language.

This project is still a work in progress.

## About

The idea behind Modern Portfolio Theory is simple: for any level of risk, there is a mix of investments that gives the best possible return. This project uses a Monte Carlo simulation to test thousands of random portfolio combinations and find that mix, then builds what is called an "efficient frontier" from the results. It also finds the best portfolios directly with optimizers (scipy and cvxpy) and compares the two approaches.

The next step is to add an LLM on top of this, so the numbers and charts can be explained in plain language, not just shown as raw data. The exact scope of the LLM part is still being decided.

## What it does right now

- Downloads daily price data for a set of ASX stocks (currently CBA, BHP, CSL, WES, and MQG) using Yahoo Finance
- Calculates daily returns, average returns, and the covariance between stocks
- Runs a Monte Carlo simulation of 10,000 random portfolios
- Calculates return, risk (volatility), and Sharpe ratio for each portfolio
- Plots the efficient frontier
- Identifies two key portfolios:
  - The portfolio with the highest Sharpe ratio (best risk-adjusted return)
  - The portfolio with the lowest risk (minimum variance)
- Finds the same two portfolios directly with optimizers (scipy's SLSQP for the highest Sharpe ratio, cvxpy for minimum variance) and compares the Sharpe ratio and speed with the Monte Carlo search
- Serves the optimizers through a FastAPI endpoint (`POST /optimize`) with API key authentication, rate limiting (10 requests per minute) and input validation
- Runs security scans (Bandit, pip-audit) and the tests automatically with GitHub Actions

## Results

![Monte Carlo simulation of 10,000 ASX portfolios](images/efficient_frontier.png)

Each dot is one randomly weighted portfolio of the five stocks, shaded by its Sharpe ratio (yellow is higher). The red star is the portfolio with the highest Sharpe ratio and the blue star is the one with the lowest risk. The notebook prints the return, volatility, Sharpe ratio and stock weights for both.

The results use daily prices from January 2021 to December 2025 and assume a 4% risk-free rate.

## Tech stack

- Python
- pandas, numpy
- yfinance (market data)
- matplotlib (charts)
- scipy, cvxpy (optimization)
- FastAPI, slowapi (API and rate limiting)
- pytest, Bandit, pip-audit, GitHub Actions (tests and security checks)
- Jupyter

## Getting started

You need Python 3.11 or newer and an internet connection (prices are downloaded from Yahoo Finance).

```bash
git clone https://github.com/khalisha-hf/markowitz-llm.git
cd markowitz-llm
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Notebook

```bash
pip install jupyter
jupyter notebook notebooks/01_eda.ipynb
```

Then run all the cells (or open the notebook in VS Code and pick the `venv` kernel). The last cell saves the chart to `images/efficient_frontier.png`.

### Compare Monte Carlo with the optimizers

```bash
python run_comparison.py
```

### API

Create a `.env` file in the project folder with a key of your choice (`.env` is in `.gitignore`, so it won't be committed):

```
OPTIMIZER_API_KEY=choose-a-secret-key
```

Start the server:

```bash
uvicorn api:app --reload
```

Then send a request, or try it in the browser at http://127.0.0.1:8000/docs:

```bash
curl -X POST http://127.0.0.1:8000/optimize \
  -H "X-API-Key: choose-a-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["CBA.AX", "BHP.AX", "CSL.AX"], "start": "2021-01-01", "end": "2025-12-31"}'
```

### Tests

```bash
python -m pytest tests/ -v
```

## Project structure

```
notebooks/
  01_eda.ipynb              # Exploratory data analysis and Monte Carlo simulation
src/
  data.py                   # Downloads prices and calculates returns
  optimizer.py              # Monte Carlo search and direct optimizers
tests/
  test_security.py          # API tests: API key, input validation, rate limiting
images/
  efficient_frontier.png    # Chart shown above (created by the notebook)
.github/workflows/
  security.yml              # Runs Bandit, pip-audit and the tests on GitHub
api.py                      # FastAPI app for the optimizers
run_comparison.py           # Compares Monte Carlo with the optimizers
conftest.py                 # Lets pytest import from the project root
requirements.txt            # Python packages the project needs
```

## Next steps

- Decide the exact role of the LLM (for example, explaining portfolio results, or answering questions about them)
- Add more stocks or let the stock list be configurable
- Clean up the notebook into reusable functions/scripts

## Status

Work in progress. Built as a personal learning project.

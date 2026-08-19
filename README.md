# Markowitz LLM Project

A personal project exploring portfolio optimization using Modern Portfolio Theory (Markowitz), applied to ASX stocks, with plans to add a Large Language Model (LLM) layer to explain the results in plain language.

This project is still a work in progress.

## About

The idea behind Modern Portfolio Theory is simple: for any level of risk, there is a mix of investments that gives the best possible return. This project uses a Monte Carlo simulation to test thousands of random portfolio combinations and find that mix, then builds what is called an "efficient frontier" from the results.

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

## Tech stack

- Python
- pandas, numpy
- yfinance (market data)
- matplotlib (charts)
- scikit-learn, scipy, cvxpy (for future optimization work)

## Project structure

```
01_eda.ipynb   # Exploratory data analysis and Monte Carlo simulation
```

## Next steps

- Decide the exact role of the LLM (for example, explaining portfolio results, or answering questions about them)
- Expand beyond Monte Carlo to direct optimization methods (using scipy/cvxpy)
- Add more stocks or let the stock list be configurable
- Clean up the notebook into reusable functions/scripts

## Status

Work in progress. Built as a personal learning project.

import time
from src.data import load_returns
from src.optimizer import (
    portfolio_performance, max_sharpe_portfolio,
    min_variance_portfolio, monte_carlo_search
)


def main():
    print("Downloading price data...")
    returns, mean_returns, cov_matrix = load_returns()

    t0 = time.time()
    mc_perf = monte_carlo_search(mean_returns, cov_matrix)
    mc_time = time.time() - t0

    t0 = time.time()
    opt_weights = max_sharpe_portfolio(mean_returns, cov_matrix)
    opt_perf = portfolio_performance(opt_weights, mean_returns, cov_matrix)
    opt_time = time.time() - t0

    t0 = time.time()
    mv_weights = min_variance_portfolio(mean_returns, cov_matrix)
    mv_perf = portfolio_performance(mv_weights, mean_returns, cov_matrix)
    mv_time = time.time() - t0

    print(f"\nMonte Carlo (10,000 samples): Sharpe={mc_perf[2]:.4f}, time={mc_time:.4f}s")
    print(f"SLSQP max-Sharpe optimum:      Sharpe={opt_perf[2]:.4f}, time={opt_time:.4f}s")
    print(f"cvxpy min-variance optimum:    Vol={mv_perf[1]:.4f}, time={mv_time:.4f}s")
    print(f"\nSharpe improvement over Monte Carlo: {(opt_perf[2]-mc_perf[2])/mc_perf[2]*100:.2f}%")
    print(f"Speed: optimizer is {mc_time/opt_time:.1f}x faster than the Monte Carlo search")


if __name__ == "__main__":
    main()

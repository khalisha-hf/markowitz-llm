import numpy as np
import scipy.optimize as sco
import cvxpy as cp

RISK_FREE_RATE = 0.04
TRADING_DAYS = 252


def portfolio_performance(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights) * TRADING_DAYS
    vol = np.sqrt(weights.T @ (cov_matrix * TRADING_DAYS) @ weights)
    sharpe = (ret - RISK_FREE_RATE) / vol
    return ret, vol, sharpe


def max_sharpe_portfolio(mean_returns, cov_matrix):
    """Not a convex problem in the weights, so this uses SLSQP
    (a local-optimum solver), not cvxpy."""
    n = len(mean_returns)

    def neg_sharpe(weights):
        _, _, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        return -sharpe

    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.array([1 / n] * n)

    result = sco.minimize(
        neg_sharpe, init_guess, method='SLSQP',
        bounds=bounds, constraints=constraints
    )
    return result.x


def min_variance_portfolio(mean_returns, cov_matrix):
    """This one is a true convex quadratic program, so cvxpy
    guarantees a global optimum, unlike the Monte Carlo search."""
    n = len(mean_returns)
    annual_cov = cov_matrix.values * TRADING_DAYS

    w = cp.Variable(n)
    risk = cp.quad_form(w, annual_cov)
    constraints = [cp.sum(w) == 1, w >= 0]

    problem = cp.Problem(cp.Minimize(risk), constraints)
    problem.solve()
    return w.value


def monte_carlo_search(mean_returns, cov_matrix, num_portfolios=10000):
    """Your existing approach, wrapped so it can be timed and
    compared against the optimizer functions above."""
    n = len(mean_returns)
    best_sharpe = -np.inf
    best_perf = None

    for _ in range(num_portfolios):
        weights = np.random.random(n)
        weights /= weights.sum()
        ret, vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_perf = (ret, vol, sharpe)
    return best_perf

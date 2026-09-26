from src.data import load_returns
from src.backtest import run_backtest

SPLIT_DATE = "2024-01-01"


def main():
    print("Downloading price data...")
    returns, _, _ = load_returns()

    results, strategies, n_train, n_test = run_backtest(returns, SPLIT_DATE)

    print(f"\nFitted on {n_train} trading days before {SPLIT_DATE}")
    print(f"Tested on {n_test} unseen trading days from {SPLIT_DATE}\n")

    header = f"{'Strategy':<26}{'Return':>9}{'Vol':>9}{'Sharpe':>9}{'MaxDD':>9}"
    print(header)
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<26}"
              f"{r['annual_return']*100:8.2f}%"
              f"{r['annual_volatility']*100:8.2f}%"
              f"{r['sharpe']:9.3f}"
              f"{r['max_drawdown']*100:8.2f}%")

    print("\nWeights fitted on the training period:")
    for name, w in strategies.items():
        pairs = ", ".join(f"{t}={x:.1%}" for t, x in zip(returns.columns, w))
        print(f"  {name}: {pairs}")


if __name__ == "__main__":
    main()

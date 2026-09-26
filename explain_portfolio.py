from dotenv import load_dotenv

from src.data import DEFAULT_END, DEFAULT_START, load_returns
from src.explain import ExplainerUnavailable, explain_results
from src.optimizer import optimize_portfolios


def main():
    load_dotenv()

    print("Downloading price data...")
    _, mean_returns, cov_matrix = load_returns()
    results = optimize_portfolios(mean_returns, cov_matrix)

    print("Asking the local model to explain the results...\n")
    try:
        explanation = explain_results(results, DEFAULT_START, DEFAULT_END)
    except ExplainerUnavailable as e:
        raise SystemExit(str(e))

    print(explanation.text)
    if explanation.unsupported_numbers:
        print("\nWarning: these numbers don't match the optimizer's results: "
              + ", ".join(explanation.unsupported_numbers))


if __name__ == "__main__":
    main()

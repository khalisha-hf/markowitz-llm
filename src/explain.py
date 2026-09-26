"""Plain-English explanations of the optimizer results from a local LLM (Ollama).

The model only explains numbers the optimizer has already calculated. Any
number in its answer that isn't in the data it was given gets flagged, so
made-up or miscalculated figures are easy to spot.
"""
import os
import re
from dataclasses import dataclass, field

import httpx

from src.data import COMPANY_NAMES
from src.optimizer import NUM_PORTFOLIOS, RISK_FREE_RATE

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
TIMEOUT_SECONDS = 120

SYSTEM_PROMPT = (
    "You explain portfolio optimization results to someone with no finance background. "
    "Only use the numbers you are given and copy them exactly as written. "
    "Only use the company names you are given; if a stock has no name, use its ticker code. "
    "Do not calculate new numbers, add facts about the companies, predict future "
    "returns or give investment advice. Keep it under 200 words."
)

PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
DECIMAL = re.compile(r"(?<![\d.])\d+\.\d+(?!\.?\d)")


class ExplainerUnavailable(Exception):
    """Raised when the local model can't be reached or gives no answer."""


@dataclass
class Explanation:
    text: str
    unsupported_numbers: list = field(default_factory=list)


def _pct(x):
    return f"{x * 100:.1f}%"


def build_prompt(results, start, end):
    ms, mv, mc = results["max_sharpe"], results["min_variance"], results["monte_carlo"]

    def describe(portfolio):
        # Largest weight first, so the model doesn't have to rank them itself
        ranked = sorted(portfolio["weights"].items(), key=lambda tw: tw[1], reverse=True)
        weights = ", ".join(f"{t} {_pct(w)}" for t, w in ranked)
        return [
            f"- Expected annual return: {_pct(portfolio['return'])}",
            f"- Return above the risk-free rate: {_pct(portfolio['return'] - RISK_FREE_RATE)}",
            f"- Annual volatility (risk): {_pct(portfolio['volatility'])}",
            f"- Sharpe ratio: {portfolio['sharpe']:.2f}",
            f"- Weights, largest first: {weights}",
        ]

    def name(ticker):
        return f"{ticker} ({COMPANY_NAMES[ticker]})" if ticker in COMPANY_NAMES else ticker

    return "\n".join([
        f"Stocks: {', '.join(name(t) for t in ms['weights'])}",
        f"Data: daily prices from {start} to {end}",
        f"Risk-free rate: {_pct(RISK_FREE_RATE)}",
        "",
        "Highest Sharpe ratio portfolio (best return for its risk, found with an optimizer):",
        *describe(ms),
        "",
        "Lowest risk portfolio (minimum variance, found with an optimizer):",
        *describe(mv),
        "",
        f"Best of {NUM_PORTFOLIOS:,} random portfolios (Monte Carlo search): "
        f"Sharpe ratio {mc['sharpe']:.2f}. Its weights are not given, so don't describe its holdings.",
        "",
        "The optimizer calculates the exact best portfolio for this data. The random "
        "search only tries random mixes, so its best Sharpe ratio can get close to the "
        "optimizer's but can't beat it.",
        "If you mention what the Sharpe ratio means, say it is the return above the "
        "risk-free rate for each unit of risk. Don't turn it into a percentage.",
        "",
        "Explain what each portfolio means and which stocks it holds most of, using the "
        "weights in the order listed. Compare the optimizer's Sharpe ratio with the "
        "random search's. Mention that the results are based on past prices.",
    ])


def _is_rounded_from(written, allowed):
    """True if some allowed value rounds to the number as written,
    e.g. '57' or '57.4' for 57.4, but not '57.0'."""
    places = len(written.split(".")[1]) if "." in written else 0
    return any(abs(round(a, places) - float(written)) < 1e-9 for a in allowed)


def find_unsupported_numbers(text, source):
    """Returns the percentages and decimals in text that don't appear in source.

    A number may be rounded (57% for 57.4%), and 100% is always allowed."""
    allowed_percents = [float(p) for p in PERCENT.findall(source)] + [100.0]
    allowed_decimals = [float(d) for d in DECIMAL.findall(PERCENT.sub(" ", source))]

    unsupported = []
    for match in PERCENT.finditer(text):
        if not _is_rounded_from(match.group(1), allowed_percents):
            unsupported.append(match.group(0))
    for match in DECIMAL.finditer(PERCENT.sub(" ", text)):
        if not _is_rounded_from(match.group(0), allowed_decimals):
            unsupported.append(match.group(0))
    return unsupported


def explain_results(results, start, end):
    """Asks the local Ollama model to explain the results in plain English."""
    url = os.environ.get("OLLAMA_URL", DEFAULT_OLLAMA_URL).rstrip("/") + "/api/chat"
    model = os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
    prompt = build_prompt(results, start, end)

    try:
        response = httpx.post(url, timeout=TIMEOUT_SECONDS, json={
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.2},
        })
        response.raise_for_status()
        text = response.json()["message"]["content"].strip()
    except (httpx.HTTPError, KeyError, ValueError) as e:
        raise ExplainerUnavailable(
            f"Could not get an explanation from Ollama at {url} ({e}). "
            f"Is Ollama running, and have you run `ollama pull {model}`?"
        ) from e

    if not text:
        raise ExplainerUnavailable("The model returned an empty explanation")
    return Explanation(text, find_unsupported_numbers(text, prompt))

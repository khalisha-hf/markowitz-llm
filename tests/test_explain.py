from unittest.mock import patch

import httpx
import pytest

from src.explain import (
    Explanation, ExplainerUnavailable, build_prompt, explain_results, find_unsupported_numbers
)

RESULTS = {
    "monte_carlo": {"return": 0.165, "volatility": 0.1553, "sharpe": 0.8047},
    "max_sharpe": {
        "return": 0.1694, "volatility": 0.1577, "sharpe": 0.8203,
        "weights": {"BHP.AX": 0.182, "CBA.AX": 0.574, "CSL.AX": 0.0, "WES.AX": 0.244},
    },
    "min_variance": {
        "return": 0.1053, "volatility": 0.1405, "sharpe": 0.4645,
        "weights": {"BHP.AX": 0.226, "CBA.AX": 0.253, "CSL.AX": 0.259, "WES.AX": 0.262},
    },
}
PAYLOAD = {"tickers": ["CBA.AX", "BHP.AX", "CSL.AX"], "start": "2021-01-01", "end": "2025-01-01"}


def ollama_reply(text):
    return httpx.Response(200, json={"message": {"role": "assistant", "content": text}},
                          request=httpx.Request("POST", "http://localhost:11434/api/chat"))


def test_prompt_contains_the_results():
    prompt = build_prompt(RESULTS, "2021-01-01", "2025-12-31")
    for expected in ["CBA.AX 57.4%", "16.9%", "15.8%", "0.82", "0.80", "4.0%", "10,000", "2021-01-01"]:
        assert expected in prompt


def test_prompt_gives_company_names_and_excess_return():
    prompt = build_prompt(RESULTS, "2021-01-01", "2025-12-31")
    assert "WES.AX (Wesfarmers)" in prompt
    assert "Return above the risk-free rate: 12.9%" in prompt


def test_prompt_uses_ticker_only_when_the_name_is_unknown():
    results = {**RESULTS, "max_sharpe": {**RESULTS["max_sharpe"], "weights": {"XYZ.AX": 1.0}}}
    assert "Stocks: XYZ.AX\n" in build_prompt(results, "2021-01-01", "2025-12-31")


def test_prompt_lists_weights_largest_first():
    prompt = build_prompt(RESULTS, "2021-01-01", "2025-12-31")
    assert "CBA.AX 57.4%, WES.AX 24.4%, BHP.AX 18.2%, CSL.AX 0.0%" in prompt
    assert "WES.AX 26.2%, CSL.AX 25.9%, CBA.AX 25.3%, BHP.AX 22.6%" in prompt


def test_flags_numbers_that_are_not_in_the_results():
    prompt = build_prompt(RESULTS, "2021-01-01", "2025-12-31")
    text = ("This portfolio returned 16.9% a year with a Sharpe ratio of 0.82 (roughly 0.8). "
            "About 57% is in CBA.AX, and it could return 25.0% next year with a Sharpe of 1.3.")
    # Rounded real numbers are fine; 25.0% and 1.3 aren't in the results
    assert find_unsupported_numbers(text, prompt) == ["25.0%", "1.3"]


def test_sends_the_prompt_to_ollama():
    with patch("src.explain.httpx.post", return_value=ollama_reply("Plain English.")) as post:
        explanation = explain_results(RESULTS, "2021-01-01", "2025-12-31")

    assert explanation == Explanation("Plain English.", [])
    url, payload = post.call_args.args[0], post.call_args.kwargs["json"]
    assert url == "http://localhost:11434/api/chat"
    assert payload["model"] == "llama3.2"
    assert payload["stream"] is False
    assert payload["messages"][1]["content"] == build_prompt(RESULTS, "2021-01-01", "2025-12-31")


def test_model_and_url_can_be_changed(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "mistral")
    monkeypatch.setenv("OLLAMA_URL", "http://other-host:11434/")
    with patch("src.explain.httpx.post", return_value=ollama_reply("Hi.")) as post:
        explain_results(RESULTS, "2021-01-01", "2025-12-31")
    assert post.call_args.args[0] == "http://other-host:11434/api/chat"
    assert post.call_args.kwargs["json"]["model"] == "mistral"


def test_raises_when_ollama_is_not_running():
    with patch("src.explain.httpx.post", side_effect=httpx.ConnectError("connection refused")):
        with pytest.raises(ExplainerUnavailable, match="ollama pull"):
            explain_results(RESULTS, "2021-01-01", "2025-12-31")


def test_explain_endpoint_needs_an_api_key(client):
    assert client.post("/explain", json=PAYLOAD).status_code == 401


def test_explain_endpoint_returns_results_and_explanation(client):
    import api
    with patch.object(api, "explain_results", return_value=Explanation("Plain English.", ["25.0%"])):
        r = client.post("/explain", json=PAYLOAD, headers={"X-API-Key": "test-key"})
    assert r.status_code == 200
    body = r.json()
    assert body["explanation"] == "Plain English."
    assert body["unsupported_numbers"] == ["25.0%"]
    assert "max_sharpe" in body


def test_explain_endpoint_returns_503_when_the_model_is_down(client):
    import api
    with patch.object(api, "explain_results", side_effect=ExplainerUnavailable("down")):
        r = client.post("/explain", json=PAYLOAD, headers={"X-API-Key": "test-key"})
    assert r.status_code == 503

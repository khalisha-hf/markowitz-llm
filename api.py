import os
import re
import secrets
from datetime import date
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.data import load_returns
from src.explain import ExplainerUnavailable, explain_results
from src.optimizer import optimize_portfolios

load_dotenv()

API_KEY = os.environ.get("OPTIMIZER_API_KEY")
if not API_KEY:
    raise RuntimeError("OPTIMIZER_API_KEY environment variable is not set")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    # compare_digest takes the same time whether or not the key matches,
    # so the key can't be guessed one character at a time from response times
    if not key or not secrets.compare_digest(key.encode(), API_KEY.encode()):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Markowitz Portfolio Optimizer")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

TICKER_PATTERN = re.compile(r"^[A-Z0-9]{1,6}\.AX$")


class OptimizeRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=2, max_length=10)
    start: date
    end: date

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, tickers):
        for t in tickers:
            if not TICKER_PATTERN.match(t):
                raise ValueError(f"Invalid ticker format: {t!r}. Expected e.g. 'CBA.AX'")
        if len(set(tickers)) != len(tickers):
            raise ValueError("Duplicate tickers are not allowed")
        return tickers

    @field_validator("end")
    @classmethod
    def validate_date_range(cls, end, info):
        start = info.data.get("start")
        if start and end <= start:
            raise ValueError("end date must be after start date")
        if start and (end - start).days > 365 * 10:
            raise ValueError("Date range cannot exceed 10 years")
        return end


def run_optimization(body: OptimizeRequest):
    try:
        returns, mean_returns, cov_matrix = load_returns(
            tickers=body.tickers,
            start=str(body.start),
            end=str(body.end),
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not load data: {e}")

    if mean_returns.isnull().any():
        raise HTTPException(status_code=422, detail="One or more tickers returned no data")

    return optimize_portfolios(mean_returns, cov_matrix)


@app.post("/optimize")
@limiter.limit("10/minute")
def optimize(request: Request, body: OptimizeRequest, api_key: str = Security(verify_api_key)):
    return {"tickers": body.tickers, **run_optimization(body)}


@app.post("/explain")
@limiter.limit("5/minute")
def explain(request: Request, body: OptimizeRequest, api_key: str = Security(verify_api_key)):
    results = run_optimization(body)
    try:
        explanation = explain_results(results, start=str(body.start), end=str(body.end))
    except ExplainerUnavailable:
        raise HTTPException(status_code=503, detail="The explanation model is not available right now")

    return {
        "tickers": body.tickers,
        **results,
        "explanation": explanation.text,
        "unsupported_numbers": explanation.unsupported_numbers,
    }

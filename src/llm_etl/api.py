"""Frontend lane: a natural-language query endpoint over the pipeline output.

The LLM is used to *compile* a natural-language question into a small, safe
query plan (not to render UI, and not per-request over raw data). The plan is
executed deterministically with pandas. The static ``frontend/index.html`` page
calls this endpoint.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from .config import get_llm
from .llm import LLM

_NL_SYSTEM = """You translate a natural-language question about a tabular
dataset into a JSON query plan (this is the nl-query compiler). Respond with
ONLY JSON, one of:
  {"op":"count"}
  {"op":"avg","col":"<column>"}
  {"op":"top","n":<int>,"by":"<column>"}
  {"op":"group","by":"<column>","agg":"count"}
  {"op":"head","n":<int>}
"""

_FRONTEND = Path(__file__).resolve().parents[2] / "frontend" / "index.html"


class QueryRequest(BaseModel):
    question: str


def _records(frame: pd.DataFrame) -> list[dict]:
    """DataFrame -> JSON-safe records (NaN/NaT -> None)."""
    return frame.astype(object).where(pd.notnull(frame), None).to_dict(orient="records")


def execute_plan(df: pd.DataFrame, plan: dict) -> dict:
    """Run a validated query plan against the dataframe. Deterministic."""
    op = plan.get("op")
    if op == "count":
        return {"answer": int(len(df)), "plan": plan}
    if op == "avg":
        col = plan.get("col")
        if col not in df.columns:
            return {"error": f"unknown column {col!r}", "plan": plan}
        return {"answer": float(pd.to_numeric(df[col], errors="coerce").mean()), "plan": plan}
    if op == "top":
        by = plan.get("by")
        n = int(plan.get("n", 5))
        if by not in df.columns:
            return {"error": f"unknown column {by!r}", "plan": plan}
        rows = df.sort_values(by, ascending=False).head(n)
        return {"answer": _records(rows), "plan": plan}
    if op == "group":
        by = plan.get("by")
        if by not in df.columns:
            return {"error": f"unknown column {by!r}", "plan": plan}
        counts = df.groupby(by).size().sort_values(ascending=False)
        return {"answer": counts.to_dict(), "plan": plan}
    n = int(plan.get("n", 5))
    return {"answer": _records(df.head(n)), "plan": plan}


def create_app(df: pd.DataFrame, *, llm: LLM | None = None) -> FastAPI:
    app = FastAPI(title="LLM-ETL NL Query API")
    llm = llm or get_llm()

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "rows": int(len(df)), "columns": list(df.columns)}

    @app.post("/query")
    def query(req: QueryRequest) -> JSONResponse:
        raw = llm.complete(_NL_SYSTEM, req.question)
        try:
            plan = json.loads(raw)
        except json.JSONDecodeError:
            return JSONResponse({"error": "could not parse query plan", "raw": raw}, 400)
        return JSONResponse(execute_plan(df, plan))

    @app.get("/")
    def index():
        if _FRONTEND.exists():
            return FileResponse(_FRONTEND)
        return JSONResponse({"message": "frontend not found"}, 404)

    return app

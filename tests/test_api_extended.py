import pandas as pd
from fastapi.testclient import TestClient

from llm_etl import create_app


def _df():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carlos", "Dana"],
            "country": ["US", "US", "MX", "US"],
            "amount": [120.5, 89.0, 240.0, 15.75],
        }
    )


def _ask(q):
    return TestClient(create_app(_df())).post("/query", json={"question": q}).json()


def test_sum_query():
    r = _ask("What is the total amount?")
    assert round(r["answer"], 2) == 465.25
    assert r["plan"]["op"] == "sum"


def test_max_query():
    r = _ask("Which customer has the maximum amount?")
    assert r["answer"]["name"] == "Carlos"
    assert r["plan"]["which"] == "max"


def test_min_query():
    r = _ask("Show the lowest amount")
    assert r["answer"]["name"] == "Dana"


def test_filter_query():
    r = _ask("List customers in US")
    assert r["count"] == 3
    assert {row["country"] for row in r["answer"]} == {"US"}

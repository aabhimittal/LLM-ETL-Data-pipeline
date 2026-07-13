import pandas as pd
from fastapi.testclient import TestClient

from llm_etl import create_app, execute_plan


def _df():
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carlos", "Dana"],
            "country": ["US", "US", "MX", "US"],
            "amount": [120.5, 89.0, 240.0, 15.75],
        }
    )


def test_health():
    client = TestClient(create_app(_df()))
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["rows"] == 4


def test_count_query():
    client = TestClient(create_app(_df()))
    r = client.post("/query", json={"question": "How many customers are there?"}).json()
    assert r["answer"] == 4


def test_avg_query():
    client = TestClient(create_app(_df()))
    r = client.post("/query", json={"question": "What is the average amount?"}).json()
    assert round(r["answer"], 2) == 116.31


def test_top_query():
    client = TestClient(create_app(_df()))
    r = client.post("/query", json={"question": "Show me the top 2 by amount"}).json()
    assert [row["name"] for row in r["answer"]] == ["Carlos", "Alice"]


def test_group_query():
    client = TestClient(create_app(_df()))
    r = client.post("/query", json={"question": "Group customers by country"}).json()
    assert r["answer"]["US"] == 3


def test_execute_plan_unknown_column():
    out = execute_plan(_df(), {"op": "avg", "col": "nope"})
    assert "error" in out

"""End-to-end demo.

Run:  python examples/run_demo.py

Shows all three lanes with the offline provider (no AWS needed):
  1. LLM-as-compiler ETL  (rules -> generated pandas -> deterministic run)
  2. Runtime batch extraction (unstructured text -> structured JSON)
  3. Natural-language query API over the transformed data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from llm_etl import (  # noqa: E402
    Pipeline,
    create_app,
    extract_batch,
    extract_csv,
    load_csv,
)

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "data"


def hr(title: str) -> None:
    print("\n" + "=" * 72 + f"\n{title}\n" + "=" * 72)


def demo_etl() -> pd.DataFrame:
    hr("1) LLM-AS-COMPILER ETL  (rules compiled once, then run deterministically)")
    raw = extract_csv(HERE / "customers_raw.csv")
    print("RAW input:")
    print(raw.to_string(index=False))

    rules = [
        "Normalize the phone column to E.164 format (default country code 1).",
        "Split full_name into first_name and last_name.",
        "Normalize email to lowercase and trim whitespace.",
        "Uppercase the country column.",
    ]
    pipe = Pipeline(rules=rules).compile(raw.head(3))

    print("\nGENERATED pandas code (one LLM call per rule, cached thereafter):")
    for rule, code in zip(rules, pipe.generated_code):
        print(f"\n# rule: {rule}\n{code}")

    out = pipe.run(raw)
    print("\nTRANSFORMED output:")
    print(out.to_string(index=False))

    dest = load_csv(out, OUT / "customers_clean.csv")
    print(f"\nLoaded -> {dest}")
    return out


def demo_runtime() -> None:
    hr("2) RUNTIME BATCH EXTRACTION  (unstructured text -> structured JSON)")
    notes = [
        "Payment of $120.50 received from Alice Johnson on 2026-01-04, thanks!",
        "Refund requested by Bob Smith - item arrived broken and late. $89 USD.",
        "Carlos Mendez paid 240.00 on 2026-02-11. Great service.",
    ]
    schema = {
        "type": "object",
        "properties": {
            "customer": {"type": "string"},
            "amount": {"type": "number"},
            "currency": {"type": "string"},
            "date": {"type": "string"},
            "sentiment": {"type": "string"},
        },
    }
    for note, rec in zip(notes, extract_batch(notes, schema)):
        print(f"\nTEXT : {note}\nJSON : {rec}")


def demo_api(df: pd.DataFrame) -> None:
    hr("3) NATURAL-LANGUAGE QUERY API  (NL question -> query plan -> pandas)")
    client = TestClient(create_app(df))
    print("GET /health ->", client.get("/health").json())
    for q in [
        "How many customers are there?",
        "What is the average amount?",
        "Show me the top 3 by amount",
        "Group customers by country",
    ]:
        r = client.post("/query", json={"question": q}).json()
        print(f"\nQ: {q}\nA: {r}")


if __name__ == "__main__":
    clean = demo_etl()
    demo_runtime()
    demo_api(clean)
    hr("DEMO COMPLETE")

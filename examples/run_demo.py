"""End-to-end demo.

Run:  python examples/run_demo.py

Shows every lane with the offline provider (no AWS needed):
  1. LLM-as-compiler ETL  (rules -> generated pandas -> deterministic run)
  2. Deterministic join enrichment (multi-table)
  3. Runtime extraction: synchronous + Bedrock Batch Inference lifecycle
  4. Natural-language query API over the transformed data
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from llm_etl import (  # noqa: E402
    OfflineBatchBackend,
    Pipeline,
    create_app,
    extract_batch,
    extract_csv,
    join_lookup,
    load_csv,
    run_batch,
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
        "Convert currency amounts to USD.",
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


def demo_join(clean: pd.DataFrame) -> pd.DataFrame:
    hr("2) DETERMINISTIC JOIN ENRICHMENT  (multi-table -> code, not per-row LLM)")
    regions = pd.DataFrame(
        {"country": ["US", "MX", "CZ"], "region": ["North America", "North America", "Europe"]}
    )
    print("Lookup / dimension table:")
    print(regions.to_string(index=False))
    enriched = join_lookup(clean, regions, on="country")
    print("\nEnriched (region joined on country):")
    print(enriched[["first_name", "country", "region", "amount_usd"]].to_string(index=False))
    return enriched


def demo_runtime() -> None:
    hr("3a) RUNTIME EXTRACTION - SYNCHRONOUS  (unstructured text -> structured JSON)")
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
    return notes, schema


def demo_batch(notes: list[str], schema: dict) -> None:
    hr("3b) RUNTIME EXTRACTION - BEDROCK BATCH INFERENCE  (async job lifecycle)")
    backend = OfflineBatchBackend()
    job_id = backend.submit(notes, schema)
    print(f"submit()  -> job_id = {job_id}")
    print(f"status()  -> {backend.status(job_id)}")
    print("results() ->")
    for rec in backend.results(job_id):
        print(f"  {rec}")
    # convenience one-shot equivalent:
    same = run_batch(notes, schema)
    print(f"\nrun_batch() one-shot returned {len(same)} records (same as above).")


def demo_api(df: pd.DataFrame) -> None:
    hr("4) NATURAL-LANGUAGE QUERY API  (NL question -> query plan -> pandas)")
    client = TestClient(create_app(df))
    print("GET /health ->", client.get("/health").json())
    for q in [
        "How many customers are there?",
        "What is the average amount in USD?",
        "What is the total amount_usd?",
        "Which customer has the maximum amount_usd?",
        "Show me the top 3 by amount_usd",
        "List customers in US",
        "Group customers by country",
    ]:
        r = client.post("/query", json={"question": q}).json()
        print(f"\nQ: {q}\nA: {r}")


if __name__ == "__main__":
    clean = demo_etl()
    enriched = demo_join(clean)
    notes, schema = demo_runtime()
    demo_batch(notes, schema)
    demo_api(enriched)
    hr("DEMO COMPLETE")

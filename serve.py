"""Uvicorn entrypoint for the natural-language query frontend.

Run:  uvicorn serve:app --reload   ->   http://localhost:8000

Loads the demo CSV, applies the demo rules once, and serves the query API +
static frontend over the cleaned data.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from llm_etl import Pipeline, create_app, extract_csv  # noqa: E402

_RAW = Path(__file__).resolve().parent / "examples" / "customers_raw.csv"

_df = Pipeline(
    rules=[
        "Normalize the phone column to E.164 format.",
        "Split full_name into first_name and last_name.",
        "Normalize email to lowercase and trim whitespace.",
        "Uppercase the country column.",
    ]
).run(extract_csv(_RAW))

app = create_app(_df)

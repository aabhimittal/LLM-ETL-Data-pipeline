# LLM-ETL-Data-pipeline

Use **AWS Bedrock LLMs (Claude) as a *compiler* for ETL — not as a per-row
runtime.** Describe your extract/transform/load manipulation rules in plain
English; the LLM generates deterministic, cached, testable pandas code **once**;
you run it at scale with **zero LLM calls, no rate limits, and full
determinism**.

> **Why not just call an LLM on every row?** Because Bedrock has hard RPM/TPM
> quotas and per-token cost — putting it in the hot path makes rate-limiting and
> cost *worse*. This repo shows the state-of-the-art pattern that avoids that
> entirely. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## What's inside — three lanes

| Lane | Module | LLM used for | LLM at scale? |
|------|--------|--------------|---------------|
| **Compiler ETL** | `compiler.py`, `runner.py` | Turn English rule → pandas code (once, cached) | ❌ never |
| **Runtime extraction** | `runtime_llm.py` | Unstructured text → structured JSON (batched) | ⚠️ only for text that can't be ruled |
| **NL query frontend** | `api.py`, `frontend/` | Question → query plan (pandas executes it) | ❌ never per-row |

## Quickstart (no AWS account needed)

```bash
pip install -r requirements.txt

python examples/run_demo.py     # end-to-end demo, all three lanes
python -m pytest                # 17 tests, all green
```

The repo ships an **offline provider** so everything runs, tests, and demos
deterministically with no credentials. Switch to real Bedrock with one env var:

```bash
export LLM_ETL_PROVIDER=bedrock
export AWS_REGION=us-east-1
export LLM_ETL_MODEL_ID=us.anthropic.claude-sonnet-5-20250929-v1:0
# normal AWS credentials in the environment
```

No other code changes — the whole codebase depends only on a tiny `LLM`
protocol (`src/llm_etl/llm.py`).

## Example: compile English rules into an ETL pipeline

```python
from llm_etl import Pipeline, extract_csv

df = extract_csv("examples/customers_raw.csv")

pipe = Pipeline(rules=[
    "Normalize the phone column to E.164 format (default country code 1).",
    "Split full_name into first_name and last_name.",
    "Normalize email to lowercase and trim whitespace.",
    "Uppercase the country column.",
])

clean = pipe.run(df)          # each rule compiled once, then pure pandas
print(pipe.generated_code)    # inspect / diff / version the generated code
```

Raw → clean (real output from the demo):

```
Alice Johnson |  Alice.Johnson@Example.COM  | (415) 555-0100 | us   ─▶
Alice | Johnson | alice.johnson@example.com | +14155550100 | US
```

## Example: runtime extraction of unstructured text

```python
from llm_etl import extract_batch

schema = {"type": "object", "properties": {
    "customer": {"type": "string"}, "amount": {"type": "number"},
    "date": {"type": "string"}, "sentiment": {"type": "string"}}}

extract_batch(
    ["Payment of $120.50 received from Alice Johnson on 2026-01-04, thanks!"],
    schema,
)
# [{'amount': 120.5, 'customer': 'Alice Johnson', 'date': '2026-01-04',
#   'sentiment': 'positive', ...}]
```

## Example: natural-language query frontend

```bash
uvicorn serve:app --reload      # then open http://localhost:8000
```

```
Q: "Show me the top 3 by amount"   →  [{Emil 320.20}, {Carlos 240.00}, {Alice 120.50}]
Q: "Group customers by country"    →  {"US": 3, "CZ": 1, "MX": 1}
```

The LLM compiles the question into a small JSON query plan; pandas runs it. The
model never renders the page and never sees the rows on each request.

## Layout

```
src/llm_etl/
  llm.py          # LLM protocol + BedrockLLM (real) + OfflineLLM (deterministic)
  config.py       # provider selection via env var
  compiler.py     # rule -> pandas code, validated & cached  (the core idea)
  sandbox.py      # restricted execution of generated code
  runner.py       # extract / transform (Pipeline) / load
  runtime_llm.py  # unstructured text -> structured JSON (batch)
  api.py          # FastAPI NL-query endpoint
frontend/index.html   # minimal static UI for the query endpoint
examples/             # demo data + run_demo.py
tests/                # pytest suite (17 tests)
docs/                 # ARCHITECTURE.md, DEMO_RESULTS.md
```

## Results

See [`docs/DEMO_RESULTS.md`](docs/DEMO_RESULTS.md) for the full captured output
of the demo and the test suite (17 passed).

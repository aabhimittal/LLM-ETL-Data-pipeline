"""Live Bedrock integration test.

Skipped unless ``LLM_ETL_PROVIDER=bedrock`` (and working AWS credentials) are
present, so CI/offline runs stay green. Run it against a real account with:

    LLM_ETL_PROVIDER=bedrock AWS_REGION=us-east-1 \
        python -m pytest tests/test_bedrock_integration.py -v

It exercises the exact same code paths as the offline suite - the only thing
that changes is the provider behind the ``LLM`` protocol.
"""

import os

import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("LLM_ETL_PROVIDER", "").lower() != "bedrock",
    reason="set LLM_ETL_PROVIDER=bedrock (with AWS creds) to run live Bedrock tests",
)


def test_bedrock_compiles_and_runs_a_rule():
    from llm_etl import compile_rule

    df = pd.DataFrame({"phone": ["(415) 555-0100"], "email": ["A@B.COM"]})
    cr = compile_rule("Normalize the phone column to E.164 format.", df)
    out = cr(df)
    assert out["phone"].iloc[0].startswith("+")


def test_bedrock_structured_extraction():
    from llm_etl import extract_one

    schema = {
        "type": "object",
        "properties": {"amount": {"type": "number"}, "customer": {"type": "string"}},
    }
    rec = extract_one("Payment of $120.50 from Alice Johnson.", schema)
    assert rec["amount"] == pytest.approx(120.5, rel=0.01)

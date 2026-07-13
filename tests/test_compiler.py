import tempfile

import pytest

from llm_etl import compile_rule
from llm_etl.sandbox import compile_transform


def test_phone_rule_produces_e164(raw_df):
    with tempfile.TemporaryDirectory() as d:
        cr = compile_rule(
            "Normalize the phone column to E.164 format.", raw_df, cache_dir=d
        )
        out = cr(raw_df)
    assert out["phone"].tolist() == ["+14155550100", "+14155550111", "+525512345678"]


def test_split_name_rule(raw_df):
    with tempfile.TemporaryDirectory() as d:
        cr = compile_rule("Split full_name into first_name and last_name.", raw_df, cache_dir=d)
        out = cr(raw_df)
    assert out["first_name"].tolist() == ["Alice", "Bob", "Carlos"]
    assert out["last_name"].tolist() == ["Johnson", "Smith", "Mendez"]


def test_email_normalization(raw_df):
    with tempfile.TemporaryDirectory() as d:
        cr = compile_rule("Normalize email to lowercase and trim.", raw_df, cache_dir=d)
        out = cr(raw_df)
    assert out["email"].tolist() == [
        "alice.johnson@example.com",
        "bob@example.com",
        "carlos@example.mx",
    ]


def test_generated_code_is_cached(raw_df):
    with tempfile.TemporaryDirectory() as d:
        cr1 = compile_rule("Uppercase the country column.", raw_df, cache_dir=d)
        # second call must read from cache and yield identical code
        cr2 = compile_rule("Uppercase the country column.", raw_df, cache_dir=d)
    assert cr1.code == cr2.code
    assert cr1(raw_df)["country"].tolist() == ["US", "US", "MX"]


def test_sandbox_blocks_forbidden_tokens():
    with pytest.raises(ValueError):
        compile_transform("def transform(df):\n    import os\n    return df")
    with pytest.raises(ValueError):
        compile_transform("def transform(df):\n    open('/etc/passwd')\n    return df")

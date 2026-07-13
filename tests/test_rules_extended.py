import tempfile

import pandas as pd
import pytest

from llm_etl import compile_rule, join_lookup


@pytest.fixture
def dup_df():
    return pd.DataFrame(
        {
            "email": ["a@x.com", "a@x.com", "b@x.com"],
            "amount": ["120.5", "120.5", "bad"],
            "currency": ["USD", "USD", "EUR"],
            "country": ["US", "US", "CZ"],
        }
    )


def test_dedup_rule(dup_df):
    with tempfile.TemporaryDirectory() as d:
        cr = compile_rule("Drop duplicate rows on email.", dup_df, cache_dir=d)
        out = cr(dup_df)
    assert len(out) == 2
    assert out["email"].tolist() == ["a@x.com", "b@x.com"]


def test_numeric_coercion_rule(dup_df):
    with tempfile.TemporaryDirectory() as d:
        cr = compile_rule("Convert the amount column to numeric.", dup_df, cache_dir=d)
        out = cr(dup_df)
    assert out["amount"].tolist()[:2] == [120.5, 120.5]
    assert pd.isna(out["amount"].iloc[2])  # "bad" -> NaN


def test_currency_conversion_rule(dup_df):
    with tempfile.TemporaryDirectory() as d:
        cr = compile_rule("Convert currency amounts to USD.", dup_df, cache_dir=d)
        out = cr(dup_df)
    # 120.5 USD -> 120.5 ; row 3 amount is "bad" -> NaN
    assert out["amount_usd"].iloc[0] == 120.5
    assert pd.isna(out["amount_usd"].iloc[2])


def test_join_lookup_enrichment():
    df = pd.DataFrame({"country": ["US", "MX", "CZ"], "amount": [1, 2, 3]})
    dim = pd.DataFrame(
        {"country": ["US", "MX", "CZ"], "region": ["NA", "NA", "EU"]}
    )
    out = join_lookup(df, dim, on="country")
    assert out["region"].tolist() == ["NA", "NA", "EU"]


def test_join_lookup_bad_key():
    df = pd.DataFrame({"country": ["US"]})
    dim = pd.DataFrame({"region": ["NA"]})
    with pytest.raises(KeyError):
        join_lookup(df, dim, on="country")

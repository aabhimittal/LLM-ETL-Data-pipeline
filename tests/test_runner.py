import tempfile
from pathlib import Path

from llm_etl import Pipeline, run_pipeline, extract_csv


def test_pipeline_chains_rules(raw_df):
    pipe = Pipeline(
        rules=[
            "Normalize the phone column to E.164.",
            "Split full_name into first_name and last_name.",
            "Uppercase the country column.",
        ]
    )
    out = pipe.run(raw_df)
    assert list(out["first_name"]) == ["Alice", "Bob", "Carlos"]
    assert out["phone"].iloc[0] == "+14155550100"
    assert out["country"].iloc[0] == "US"


def test_pipeline_compiles_each_rule_once(raw_df):
    pipe = Pipeline(rules=["Uppercase the country column."]).compile(raw_df)
    assert len(pipe.generated_code) == 1
    # running again does not recompile
    pipe.run(raw_df)
    assert len(pipe.generated_code) == 1


def test_run_pipeline_end_to_end_csv():
    src = Path(__file__).resolve().parents[1] / "examples" / "customers_raw.csv"
    with tempfile.TemporaryDirectory() as d:
        dest = Path(d) / "out.csv"
        out = run_pipeline(
            src,
            ["Normalize email to lowercase and trim.", "Uppercase the country column."],
            dest=dest,
        )
        assert dest.exists()
        reloaded = extract_csv(dest)
    assert reloaded["country"].tolist() == ["US", "US", "MX", "US", "CZ"]
    assert out["email"].iloc[0] == "alice.johnson@example.com"

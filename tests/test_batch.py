import tempfile

from llm_etl import run_batch, OfflineBatchBackend, build_record

SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "amount": {"type": "number"},
        "date": {"type": "string"},
        "sentiment": {"type": "string"},
    },
}

NOTES = [
    "Payment of $120.50 received from Alice Johnson on 2026-01-04, thanks!",
    "Refund requested by Bob Smith - broken and late. $89 USD.",
    "Carlos Mendez paid 240.00 on 2026-02-11. Great service.",
]


def test_offline_batch_lifecycle():
    with tempfile.TemporaryDirectory() as d:
        backend = OfflineBatchBackend(work_dir=d)
        job_id = backend.submit(NOTES, SCHEMA)
        assert job_id.startswith("offline-")
        assert backend.status(job_id) == "COMPLETED"
        results = backend.results(job_id)
    assert len(results) == 3
    assert results[0]["customer"] == "Alice Johnson"
    assert results[1]["sentiment"] == "negative"


def test_run_batch_preserves_order():
    with tempfile.TemporaryDirectory() as d:
        out = run_batch(NOTES, SCHEMA, backend=OfflineBatchBackend(work_dir=d))
    assert [r["amount"] for r in out] == [120.5, 89.0, 240.0]


def test_build_record_shape():
    rec = build_record("r0", "hello", SCHEMA)
    assert rec["recordId"] == "r0"
    mi = rec["modelInput"]
    assert mi["anthropic_version"] == "bedrock-2023-05-31"
    assert mi["tool_choice"] == {"type": "tool", "name": "emit"}
    assert mi["tools"][0]["input_schema"] == SCHEMA

from llm_etl import extract_batch, extract_one

SCHEMA = {
    "type": "object",
    "properties": {
        "customer": {"type": "string"},
        "amount": {"type": "number"},
        "currency": {"type": "string"},
        "date": {"type": "string"},
        "sentiment": {"type": "string"},
    },
}


def test_extract_one_structured():
    rec = extract_one(
        "Payment of $120.50 received from Alice Johnson on 2026-01-04, thanks!", SCHEMA
    )
    assert rec["amount"] == 120.50
    assert rec["customer"] == "Alice Johnson"
    assert rec["date"] == "2026-01-04"
    assert rec["currency"] == "USD"
    assert rec["sentiment"] == "positive"


def test_extract_negative_sentiment():
    rec = extract_one("Refund requested - item arrived broken and late. $89 USD.", SCHEMA)
    assert rec["sentiment"] == "negative"
    assert rec["amount"] == 89.0


def test_extract_batch_isolates_errors():
    texts = ["Paid 240.00 by Carlos Mendez on 2026-02-11. Great.", ""]
    out = extract_batch(texts, SCHEMA)
    assert len(out) == 2
    assert out[0]["customer"] == "Carlos Mendez"

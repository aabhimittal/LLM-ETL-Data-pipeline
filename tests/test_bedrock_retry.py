"""Unit-test the Bedrock throttling/backoff path with a fake client (no AWS)."""

import pytest

boto3 = pytest.importorskip("boto3")
from botocore.exceptions import ClientError  # noqa: E402

from llm_etl.llm import BedrockLLM  # noqa: E402


class _FakeClient:
    def __init__(self, fail_times: int):
        self.calls = 0
        self.fail_times = fail_times

    def converse(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ClientError(
                {"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
                "Converse",
            )
        return {"output": {"message": {"content": [{"text": "def transform(df):\n    return df"}]}}}


def test_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)  # don't actually wait
    fake = _FakeClient(fail_times=2)
    llm = BedrockLLM(max_retries=5, _client=fake)
    out = llm.complete("system", "user")
    assert "transform" in out
    assert fake.calls == 3  # 2 throttles + 1 success


def test_raises_after_max_retries(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda *_: None)
    fake = _FakeClient(fail_times=99)
    llm = BedrockLLM(max_retries=2, _client=fake)
    with pytest.raises(ClientError):
        llm.complete("system", "user")
    assert fake.calls == 3  # initial + 2 retries

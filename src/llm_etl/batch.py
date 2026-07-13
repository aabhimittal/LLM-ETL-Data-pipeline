"""Bedrock Batch Inference backend for the runtime extraction lane.

At volume you should NOT loop synchronous Converse calls (online RPM/TPM quota,
per-call latency). Bedrock **Batch Inference** runs a large JSONL of records as
one asynchronous job against a separate, higher batch quota - typically ~50%
cheaper. This module wraps that lifecycle behind a small backend interface:

    submit(records) -> job_id
    status(job_id)  -> "SUBMITTED" | "IN_PROGRESS" | "COMPLETED" | "FAILED"
    results(job_id) -> list[dict]

Two backends:

* ``OfflineBatchBackend`` - simulates the whole submit/poll/results lifecycle on
  local disk using :class:`~llm_etl.llm.OfflineLLM`, so it is fully testable with
  no AWS. It even writes/reads JSONL the same way the real one uses S3.
* ``BedrockBatchBackend`` - real implementation: writes input JSONL to S3, calls
  ``bedrock.create_model_invocation_job``, and parses the output JSONL. Wired and
  structurally correct; requires AWS credentials + S3 to actually run.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from .config import get_llm
from .llm import OfflineLLM


def build_record(record_id: str, text: str, schema: dict) -> dict:
    """One line of a Bedrock batch JSONL, native (Anthropic) invoke format.

    Uses tool-use to force structured output conforming to ``schema``.
    """
    return {
        "recordId": record_id,
        "modelInput": {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "system": "Extract the requested fields from the text. Only use information present.",
            "messages": [{"role": "user", "content": text}],
            "tools": [
                {
                    "name": "emit",
                    "description": "Emit the extracted structured record.",
                    "input_schema": schema,
                }
            ],
            "tool_choice": {"type": "tool", "name": "emit"},
        },
    }


class BatchBackend(Protocol):
    def submit(self, texts: list[str], schema: dict) -> str: ...
    def status(self, job_id: str) -> str: ...
    def results(self, job_id: str) -> list[dict]: ...


# --------------------------------------------------------------------------- #
# Offline backend - simulates the async lifecycle on local disk
# --------------------------------------------------------------------------- #
@dataclass
class OfflineBatchBackend:
    """Testable, no-AWS simulation of Bedrock Batch Inference."""

    work_dir: str | Path = ".batch_jobs"
    _llm: OfflineLLM = field(default_factory=OfflineLLM)

    def _dir(self, job_id: str) -> Path:
        d = Path(self.work_dir) / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def submit(self, texts: list[str], schema: dict) -> str:
        job_id = f"offline-{uuid.uuid4().hex[:12]}"
        d = self._dir(job_id)
        # write the input manifest exactly like the real S3 input JSONL
        with (d / "input.jsonl").open("w") as fh:
            for i, text in enumerate(texts):
                fh.write(json.dumps(build_record(f"r{i}", text, schema)) + "\n")
        # "run" the job now, but expose it through the async status/results API
        tool = {"name": "emit", "inputSchema": {"json": schema}}
        with (d / "output.jsonl").open("w") as fh:
            for i, text in enumerate(texts):
                out = json.loads(self._llm.complete("", text, tool=tool))
                fh.write(json.dumps({"recordId": f"r{i}", "modelOutput": out}) + "\n")
        return job_id

    def status(self, job_id: str) -> str:
        return "COMPLETED" if (Path(self.work_dir) / job_id / "output.jsonl").exists() else "FAILED"

    def results(self, job_id: str) -> list[dict]:
        path = Path(self.work_dir) / job_id / "output.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        rows.sort(key=lambda r: int(r["recordId"][1:]))  # restore input order
        return [r["modelOutput"] for r in rows]


# --------------------------------------------------------------------------- #
# Real Bedrock backend
# --------------------------------------------------------------------------- #
@dataclass
class BedrockBatchBackend:
    """Real Bedrock Batch Inference via ``create_model_invocation_job``.

    Requires an S3 bucket (``s3_uri``, e.g. ``s3://my-bucket/llm-etl``) and an
    IAM ``role_arn`` Bedrock can assume to read/write it.
    """

    s3_uri: str
    role_arn: str
    model_id: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    region: str = "us-east-1"
    _bedrock: Any = field(default=None, repr=False)
    _s3: Any = field(default=None, repr=False)

    def _clients(self):
        if self._bedrock is None:
            import boto3

            self._bedrock = boto3.client("bedrock", region_name=self.region)
            self._s3 = boto3.client("s3", region_name=self.region)
        return self._bedrock, self._s3

    @staticmethod
    def _split_uri(uri: str) -> tuple[str, str]:
        bucket, _, key = uri.removeprefix("s3://").partition("/")
        return bucket, key.rstrip("/")

    def submit(self, texts: list[str], schema: dict) -> str:
        bedrock, s3 = self._clients()
        bucket, prefix = self._split_uri(self.s3_uri)
        job = f"llm-etl-{uuid.uuid4().hex[:8]}"
        body = "\n".join(
            json.dumps(build_record(f"r{i}", t, schema)) for i, t in enumerate(texts)
        )
        s3.put_object(Bucket=bucket, Key=f"{prefix}/{job}/input.jsonl", Body=body.encode())
        resp = bedrock.create_model_invocation_job(
            jobName=job,
            roleArn=self.role_arn,
            modelId=self.model_id,
            inputDataConfig={
                "s3InputDataConfig": {"s3Uri": f"s3://{bucket}/{prefix}/{job}/input.jsonl"}
            },
            outputDataConfig={
                "s3OutputDataConfig": {"s3Uri": f"s3://{bucket}/{prefix}/{job}/output/"}
            },
        )
        return resp["jobArn"]

    def status(self, job_id: str) -> str:
        bedrock, _ = self._clients()
        return bedrock.get_model_invocation_job(jobIdentifier=job_id)["status"]

    def wait(self, job_id: str, *, poll_seconds: int = 30, timeout: int = 3600) -> str:
        deadline = time.time() + timeout
        while time.time() < deadline:
            st = self.status(job_id)
            if st in ("Completed", "Failed", "Stopped", "Expired"):
                return st
            time.sleep(poll_seconds)
        raise TimeoutError(f"batch job {job_id} did not finish within {timeout}s")

    def results(self, job_id: str) -> list[dict]:
        _, s3 = self._clients()
        bucket, prefix = self._split_uri(self.s3_uri)
        job = job_id.rsplit("/", 1)[-1]
        listing = s3.list_objects_v2(Bucket=bucket, Prefix=f"{prefix}/{job}/output/")
        rows: list[dict] = []
        for obj in listing.get("Contents", []):
            if not obj["Key"].endswith(".jsonl.out"):
                continue
            body = s3.get_object(Bucket=bucket, Key=obj["Key"])["Body"].read().decode()
            for line in body.splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: int(r["recordId"][1:]))
        return [_parse_model_output(r) for r in rows]


def _parse_model_output(row: dict) -> dict:
    """Pull the tool-use input out of a batch modelOutput record."""
    out = row.get("modelOutput", row)
    for block in out.get("content", []):
        if block.get("type") == "tool_use":
            return block["input"]
    return out


def run_batch(
    texts: list[str], schema: dict, *, backend: BatchBackend | None = None
) -> list[dict]:
    """Submit a batch and block until results are ready (offline by default)."""
    backend = backend or OfflineBatchBackend()
    job_id = backend.submit(texts, schema)
    if backend.status(job_id) != "COMPLETED":
        # real backend: wait for terminal state
        wait = getattr(backend, "wait", None)
        if callable(wait):
            wait(job_id)
    return backend.results(job_id)

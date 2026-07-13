"""Runtime LLM lane - the narrow escape hatch for genuinely unstructured data.

Use this ONLY for records that cannot be compiled to deterministic rules:
free-text notes, emails, scanned document text, etc. It extracts structured
JSON via tool-use (structured output) and processes records in **batches** to
stay under Bedrock's online rate limits and to amortize cost. For very high
volume, back this with Bedrock Batch Inference (async, ~50% cheaper).
"""

from __future__ import annotations

import json
from typing import Iterable

from .config import get_llm
from .llm import LLM

_SYSTEM = "Extract the requested fields from the text. Only use information present."


def _tool_spec(schema: dict) -> dict:
    return {
        "name": "emit",
        "description": "Emit the extracted structured record.",
        "inputSchema": {"json": schema},
    }


def extract_one(text: str, schema: dict, *, llm: LLM | None = None) -> dict:
    """Extract one structured record from free text."""
    llm = llm or get_llm()
    raw = llm.complete(_SYSTEM, text, tool=_tool_spec(schema))
    return json.loads(raw)


def extract_batch(
    texts: Iterable[str], schema: dict, *, llm: LLM | None = None
) -> list[dict]:
    """Extract a batch of records. Errors are captured per-record, not fatal."""
    llm = llm or get_llm()
    results: list[dict] = []
    for text in texts:
        try:
            results.append(extract_one(text, schema, llm=llm))
        except Exception as exc:  # noqa: BLE001 - one bad record shouldn't kill the batch
            results.append({"_error": str(exc), "_input": text})
    return results

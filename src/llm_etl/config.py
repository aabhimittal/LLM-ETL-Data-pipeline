"""Central configuration and LLM provider selection.

Set ``LLM_ETL_PROVIDER=bedrock`` (and normal AWS env/credentials) to use the
real Bedrock Converse API. Anything else -> the deterministic offline provider,
which is the default so the repo runs out of the box.
"""

from __future__ import annotations

import os

from .llm import BedrockLLM, LLM, OfflineLLM


def get_llm() -> LLM:
    provider = os.getenv("LLM_ETL_PROVIDER", "offline").lower()
    if provider == "bedrock":
        return BedrockLLM(
            model_id=os.getenv(
                "LLM_ETL_MODEL_ID", "us.anthropic.claude-sonnet-5-20250929-v1:0"
            ),
            region=os.getenv("AWS_REGION", "us-east-1"),
        )
    return OfflineLLM()

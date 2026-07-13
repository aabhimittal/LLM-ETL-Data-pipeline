"""LLM-ETL: use Bedrock LLMs as a *compiler* for ETL, not a runtime.

Public API:
    compile_rule   - plain-English rule  -> deterministic pandas transform
    Pipeline       - a sequence of rules applied to a DataFrame
    run_pipeline   - one-shot extract -> transform -> load
    join_lookup    - deterministic multi-table enrichment (joins)
    extract_batch  - runtime lane: unstructured text -> structured JSON (sync)
    run_batch      - runtime lane at volume: Bedrock Batch Inference lifecycle
    create_app     - FastAPI natural-language query endpoint
"""

from .compiler import CompiledRule, compile_rule
from .config import get_llm
from .runner import Pipeline, run_pipeline, extract_csv, load_csv, join_lookup
from .runtime_llm import extract_batch, extract_one
from .batch import run_batch, OfflineBatchBackend, BedrockBatchBackend, build_record
from .api import create_app, execute_plan

__all__ = [
    "compile_rule",
    "CompiledRule",
    "Pipeline",
    "run_pipeline",
    "extract_csv",
    "load_csv",
    "join_lookup",
    "extract_batch",
    "extract_one",
    "run_batch",
    "OfflineBatchBackend",
    "BedrockBatchBackend",
    "build_record",
    "create_app",
    "execute_plan",
    "get_llm",
]

__version__ = "0.2.0"

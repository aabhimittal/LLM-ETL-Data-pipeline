"""LLM-ETL: use Bedrock LLMs as a *compiler* for ETL, not a runtime.

Public API:
    compile_rule   - plain-English rule  -> deterministic pandas transform
    Pipeline       - a sequence of rules applied to a DataFrame
    run_pipeline   - one-shot extract -> transform -> load
    extract_batch  - runtime lane: unstructured text -> structured JSON
    create_app     - FastAPI natural-language query endpoint
"""

from .compiler import CompiledRule, compile_rule
from .config import get_llm
from .runner import Pipeline, run_pipeline, extract_csv, load_csv
from .runtime_llm import extract_batch, extract_one
from .api import create_app, execute_plan

__all__ = [
    "compile_rule",
    "CompiledRule",
    "Pipeline",
    "run_pipeline",
    "extract_csv",
    "load_csv",
    "extract_batch",
    "extract_one",
    "create_app",
    "execute_plan",
    "get_llm",
]

__version__ = "0.1.0"

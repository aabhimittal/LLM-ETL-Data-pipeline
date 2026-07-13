"""LLM-as-compiler: turn a plain-English manipulation rule into deterministic,
cached, testable pandas code - **once** - instead of calling the LLM per row.

This is the core cost/latency/rate-limit win of the whole project: the model
touches your data zero times at scale. See ``docs/ARCHITECTURE.md``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .config import get_llm
from .llm import LLM
from .sandbox import compile_transform, validate_transform

_SYSTEM = """You are a data-transformation compiler. Given a plain-English rule
and a few sample rows, output ONLY a Python function with this exact signature:

    def transform(df):
        ...
        return df

Rules:
- Use only pandas (as `df`), the `_re` module for regex, and standard builtins.
- Do NOT import anything. Do NOT read files or access the network.
- Operate on a copy; return a pandas DataFrame.
- Be deterministic. No randomness, no timestamps.
Output the function and nothing else (no markdown fences, no prose).
"""


@dataclass
class CompiledRule:
    rule: str
    code: str
    transform: Callable[[pd.DataFrame], pd.DataFrame]

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)


def _strip_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def compile_rule(
    rule: str,
    sample: pd.DataFrame,
    *,
    llm: LLM | None = None,
    cache_dir: str | Path = ".rule_cache",
) -> CompiledRule:
    """Compile ``rule`` into a validated, cached :class:`CompiledRule`.

    The generated code is cached on disk keyed by (rule + column schema), so a
    given rule costs at most one LLM call ever - subsequent runs are free.
    """
    llm = llm or get_llm()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)

    key_src = json.dumps({"rule": rule, "cols": list(sample.columns)}, sort_keys=True)
    key = hashlib.sha256(key_src.encode()).hexdigest()[:16]
    cache_file = cache_dir / f"{key}.py"

    if cache_file.exists():
        code = cache_file.read_text()
    else:
        user = (
            f"Rule:\n{rule}\n\nSample rows (JSON):\n"
            f"{sample.head(3).to_json(orient='records')}"
        )
        code = _strip_fences(llm.complete(_SYSTEM, user))

    fn = compile_transform(code)
    validate_transform(fn, sample)  # fail fast on bad generations
    cache_file.write_text(code)  # only cache code that compiled & validated
    return CompiledRule(rule=rule, code=code, transform=fn)

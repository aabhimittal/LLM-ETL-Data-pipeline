"""Safe-ish execution of LLM-generated transform code.

Generated code must define ``transform(df) -> df``. We execute it with a
restricted global namespace (no builtins like ``open``/``__import__``, only
pandas/numpy/re exposed) and validate the result is a DataFrame. This is a
pragmatic guardrail, not a security boundary for hostile input - for untrusted
rules run it in a real sandbox/subprocess with resource limits.
"""

from __future__ import annotations

import re
from typing import Callable

import numpy as np
import pandas as pd

_FORBIDDEN = re.compile(
    r"\b(import|__import__|open|exec|eval|compile|globals|locals|"
    r"getattr|setattr|delattr|os|sys|subprocess|socket|__\w+__)\b"
)

_SAFE_GLOBALS = {
    "__builtins__": {
        "len": len, "range": range, "min": min, "max": max, "sum": sum,
        "abs": abs, "round": round, "str": str, "int": int, "float": float,
        "bool": bool, "list": list, "dict": dict, "set": set, "tuple": tuple,
        "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
        "sorted": sorted, "any": any, "all": all, "None": None,
    },
    "pd": pd,
    "np": np,
    "_re": re,
}


def compile_transform(code: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    """Compile generated source into a callable ``transform``."""
    # Allow a bare ``import re`` style helper only through the injected ``_re``.
    scan = re.sub(r"['\"].*?['\"]", "", code)  # ignore string literals
    hit = _FORBIDDEN.search(scan)
    if hit:
        raise ValueError(f"Generated code uses forbidden token: {hit.group(0)!r}")

    namespace: dict = dict(_SAFE_GLOBALS)
    exec(compile(code, "<generated-transform>", "exec"), namespace)  # noqa: S102
    fn = namespace.get("transform")
    if not callable(fn):
        raise ValueError("Generated code did not define a callable 'transform(df)'")
    return fn


def validate_transform(
    fn: Callable[[pd.DataFrame], pd.DataFrame], sample: pd.DataFrame
) -> pd.DataFrame:
    """Run ``fn`` on a sample and assert it returns a non-empty DataFrame."""
    result = fn(sample.copy())
    if not isinstance(result, pd.DataFrame):
        raise TypeError(f"transform must return a DataFrame, got {type(result)!r}")
    if len(result) != len(sample):
        # a row-count change is allowed (e.g. dropna) but must be intentional
        pass
    return result

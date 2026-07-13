"""Deterministic ETL runner.

Extract (deterministic connectors) -> Transform (rules compiled once by the LLM,
then run deterministically) -> Load (deterministic writers). The LLM is never in
the per-row hot path here; it only produced the transform code, ahead of time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from .compiler import CompiledRule, compile_rule
from .llm import LLM


# --------------------------- extract / load -------------------------------- #
def extract_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def extract_records(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def load_csv(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_records(df: pd.DataFrame) -> list[dict]:
    return df.to_dict(orient="records")


# ------------------------------- pipeline ---------------------------------- #
@dataclass
class Pipeline:
    """A sequence of plain-English transform rules applied to a DataFrame.

    Each rule is compiled to pandas code exactly once (and cached); running the
    pipeline over millions of rows costs zero additional LLM calls.
    """

    rules: list[str] = field(default_factory=list)
    llm: LLM | None = None
    _compiled: list[CompiledRule] = field(default_factory=list, repr=False)

    def compile(self, sample: pd.DataFrame) -> "Pipeline":
        self._compiled = []
        df = sample
        for rule in self.rules:
            cr = compile_rule(rule, df, llm=self.llm)
            self._compiled.append(cr)
            df = cr(df.copy())  # feed the evolving schema to the next rule
        return self

    def run(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._compiled:
            self.compile(df.head(5))
        for cr in self._compiled:
            df = cr(df)
        return df

    @property
    def generated_code(self) -> list[str]:
        return [cr.code for cr in self._compiled]


def run_pipeline(
    source: str | Path | list[dict],
    rules: list[str],
    *,
    llm: LLM | None = None,
    dest: str | Path | None = None,
) -> pd.DataFrame:
    """Convenience one-shot: extract -> transform(rules) -> optional load."""
    df = extract_csv(source) if isinstance(source, (str, Path)) else extract_records(source)
    out = Pipeline(rules=rules, llm=llm).run(df)
    if dest is not None:
        load_csv(out, dest)
    return out

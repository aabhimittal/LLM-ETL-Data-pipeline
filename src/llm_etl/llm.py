"""LLM provider abstraction.

Two interchangeable implementations sit behind one interface:

* ``BedrockLLM``  - real AWS Bedrock Converse API (Claude models).
* ``OfflineLLM``  - deterministic, dependency-free stand-in used for the demo,
  CI and tests so the whole pipeline runs with **no AWS credentials**.

The rest of the codebase never imports boto3 directly; it only depends on the
tiny :class:`LLM` protocol below. Swapping offline -> Bedrock is a one-line
change in :mod:`llm_etl.config`.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol


class LLM(Protocol):
    """Minimal surface the pipeline needs from any LLM backend."""

    def complete(self, system: str, user: str, *, tool: dict | None = None) -> str:
        """Return the model's text response.

        If ``tool`` (a JSON schema) is supplied the return value is a JSON
        string conforming to that schema (structured output via tool use).
        """
        ...


# --------------------------------------------------------------------------- #
# Real Bedrock provider (Claude via the Converse API)
# --------------------------------------------------------------------------- #
@dataclass
class BedrockLLM:
    """Thin wrapper over the Bedrock ``Converse`` API.

    Uses tool-use for structured output and is intended to be called at *build
    time* (compiling rules to code) or in *batch* - never per-row in the hot
    path. See ``docs/ARCHITECTURE.md``.
    """

    model_id: str = "us.anthropic.claude-sonnet-5-20250929-v1:0"
    region: str = "us-east-1"
    max_tokens: int = 2048
    temperature: float = 0.0
    max_retries: int = 5
    prompt_cache: bool = True
    _client: Any = field(default=None, repr=False)

    # Bedrock error codes that are safe to retry with backoff.
    _RETRYABLE = ("ThrottlingException", "TooManyRequestsException", "ServiceUnavailableException")

    def _lazy_client(self):
        if self._client is None:
            import boto3  # imported lazily so offline runs need no boto3

            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _converse(self, **kwargs: Any) -> dict:
        """Call Converse with exponential backoff on throttling.

        This is the correct answer to Bedrock rate limits when you *do* call the
        model: bounded retries with jitter, not a per-row firehose. The compiler
        pattern means we reach this path rarely (once per rule / per batch item).
        """
        from botocore.exceptions import ClientError

        client = self._lazy_client()
        for attempt in range(self.max_retries + 1):
            try:
                return client.converse(**kwargs)
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "")
                if code not in self._RETRYABLE or attempt == self.max_retries:
                    raise
                sleep = min(2 ** attempt, 30) + random.uniform(0, 0.5)
                time.sleep(sleep)
        raise RuntimeError("unreachable")  # pragma: no cover

    def complete(self, system: str, user: str, *, tool: dict | None = None) -> str:
        system_blocks: list[dict] = [{"text": system}]
        if self.prompt_cache:
            # Cache the (large, stable) system prompt so repeated compiles only
            # pay for it once - a direct cost lever for high-volume rule sets.
            system_blocks.append({"cachePoint": {"type": "default"}})

        kwargs: dict[str, Any] = {
            "modelId": self.model_id,
            "system": system_blocks,
            "messages": [{"role": "user", "content": [{"text": user}]}],
            "inferenceConfig": {
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        }
        if tool is not None:
            kwargs["toolConfig"] = {
                "tools": [{"toolSpec": tool}],
                "toolChoice": {"tool": {"name": tool["name"]}},
            }

        resp = self._converse(**kwargs)
        content = resp["output"]["message"]["content"]
        if tool is not None:
            for block in content:
                if "toolUse" in block:
                    return json.dumps(block["toolUse"]["input"])
            raise RuntimeError("Bedrock returned no toolUse block for a tool call")
        return "".join(b.get("text", "") for b in content)


# --------------------------------------------------------------------------- #
# Offline provider - deterministic, no network, used for demo/CI/tests
# --------------------------------------------------------------------------- #
class OfflineLLM:
    """A deterministic stand-in for an LLM.

    It is intentionally *not* a language model: it pattern-matches a handful of
    canonical manipulation rules and free-text extraction requests and returns
    exactly what a well-prompted Claude model would return for them. This keeps
    the demo honest (the generated pandas code is real and really executed)
    while remaining fully reproducible offline.
    """

    # ---- rule -> pandas code (the "LLM-as-compiler" path) -------------- #
    def _compile_rule(self, user: str) -> str:
        rule = user.lower()
        lines = ["def transform(df):", "    df = df.copy()"]

        if "e.164" in rule or "e164" in rule or "phone" in rule:
            lines += [
                "    def _e164(v, default_cc='1'):",
                "        digits = _re.sub(r'\\D', '', str(v))",
                "        if not digits:",
                "            return None",
                "        if len(digits) == 10:",
                "            digits = default_cc + digits",
                "        return '+' + digits",
                "    df['phone'] = df['phone'].map(_e164)",
            ]
        if "split" in rule and "name" in rule:
            lines += [
                "    parts = df['full_name'].fillna('').str.strip().str.split(n=1, expand=True)",
                "    df['first_name'] = parts[0]",
                "    df['last_name'] = parts[1] if parts.shape[1] > 1 else ''",
                "    df['last_name'] = df['last_name'].fillna('')",
            ]
        if "uppercase" in rule or "upper-case" in rule or "upper case" in rule:
            col = _first_col_after(rule, "uppercase") or "country"
            lines += [f"    df['{col}'] = df['{col}'].astype(str).str.upper()"]
        if "email" in rule and ("lower" in rule or "normal" in rule):
            lines += ["    df['email'] = df['email'].astype(str).str.strip().str.lower()"]
        if ("drop" in rule and "null" in rule) or "dropna" in rule:
            lines += ["    df = df.dropna()"]
        if "dedup" in rule or "duplicate" in rule or "distinct" in rule or "unique" in rule:
            col = _first_col_after(rule, "on") or _first_col_after(rule, "by")
            subset = f"subset=['{col}']" if col else ""
            lines += [f"    df = df.drop_duplicates({subset}).reset_index(drop=True)"]
        if ("convert" in rule or "coerce" in rule or "cast" in rule) and (
            "numeric" in rule or "number" in rule or "int" in rule or "float" in rule
        ):
            col = _target_col(rule) or "amount"
            caster = "int" if "int" in rule else "float"
            lines += [
                f"    df['{col}'] = pd.to_numeric(df['{col}'], errors='coerce')",
            ]
            if caster == "int":
                lines += [f"    df['{col}'] = df['{col}'].astype('Int64')"]
        if "currency" in rule or ("convert" in rule and ("usd" in rule or "eur" in rule)):
            # static demo FX table -> normalize an amount column to USD
            lines += [
                "    _fx = {'USD': 1.0, 'EUR': 1.08, 'GBP': 1.27, 'MXN': 0.058, 'CZK': 0.043}",
                "    def _to_usd(row):",
                "        cur = str(row.get('currency', 'USD')).upper()",
                "        amt = pd.to_numeric(row.get('amount'), errors='coerce')",
                "        rate = _fx.get(cur, 1.0)",
                "        return round(amt * rate, 2) if amt == amt else None",
                "    df['amount_usd'] = df.apply(_to_usd, axis=1)",
            ]

        if len(lines) == 2:  # nothing matched -> safe identity transform
            lines += ["    # (offline stub: rule not recognized -> identity)"]
        lines += ["    return df"]
        return "\n".join(lines)

    # ---- unstructured text -> structured JSON (runtime batch path) ----- #
    def _extract(self, user: str, schema: dict) -> str:
        text = user
        out: dict[str, Any] = {}
        props = schema.get("properties", {})
        if "amount" in props:
            m = re.search(r"\$?\s*([0-9][0-9,]*\.?[0-9]*)", text)
            out["amount"] = float(m.group(1).replace(",", "")) if m else None
        if "currency" in props:
            out["currency"] = "USD" if ("$" in text or "usd" in text.lower()) else None
        if "customer" in props:
            m = re.search(r"(?:from|for|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", text)
            out["customer"] = m.group(1) if m else None
        if "date" in props:
            m = re.search(r"\d{4}-\d{2}-\d{2}", text)
            out["date"] = m.group(0) if m else None
        if "sentiment" in props:
            low = text.lower()
            pos = any(w in low for w in ("great", "love", "excellent", "happy", "thanks"))
            neg = any(w in low for w in ("bad", "terrible", "angry", "refund", "broken", "late"))
            out["sentiment"] = "positive" if pos and not neg else "negative" if neg else "neutral"
        return json.dumps(out)

    # ---- natural-language question -> pandas query expr (frontend) ----- #
    def _nl_query(self, user: str) -> str:
        q = user.lower()

        def amount_col() -> str:
            # "amount_usd" / "in usd" -> the converted column, else raw amount
            return "amount_usd" if "amount_usd" in q or "usd" in q else "amount"

        # order matters: check group/top/avg before the generic count fallback,
        # and match "count" as a whole word so "country" doesn't trip it.
        if "group" in q or "by country" in q or "per country" in q:
            return json.dumps({"op": "group", "by": "country", "agg": "count"})
        if "top" in q or "highest" in q or "most" in q:
            m = re.search(r"top\s+(\d+)", q)
            n = int(m.group(1)) if m else 5
            by = amount_col() if "amount" in q or "spend" in q or "spent" in q else None
            return json.dumps({"op": "top", "n": n, "by": by})
        if "average" in q or "mean" in q or re.search(r"\bavg\b", q):
            col = amount_col() if "amount" in q or "spend" in q else None
            return json.dumps({"op": "avg", "col": col})
        if "total" in q or re.search(r"\bsum\b", q):
            col = amount_col() if "amount" in q or "spend" in q or "spent" in q else None
            return json.dumps({"op": "sum", "col": col})
        if "lowest" in q or "minimum" in q or re.search(r"\bmin\b", q):
            return json.dumps({"op": "extreme", "which": "min", "col": amount_col()})
        if "maximum" in q or re.search(r"\bmax\b", q) or "largest" in q:
            return json.dumps({"op": "extreme", "which": "max", "col": amount_col()})
        # "customers in US" / "where country is MX"
        m = re.search(r"\b(?:in|from|where country (?:is|=)?)\s+([a-z]{2})\b", q)
        if m and ("customer" in q or "show" in q or "list" in q or "where" in q):
            return json.dumps({"op": "filter", "col": "country", "eq": m.group(1).upper()})
        if "how many" in q or re.search(r"\bcount\b", q) or "number of" in q:
            return json.dumps({"op": "count"})
        return json.dumps({"op": "head", "n": 5})

    # ---- dispatch ------------------------------------------------------ #
    def complete(self, system: str, user: str, *, tool: dict | None = None) -> str:
        sys_l = system.lower()
        if tool is not None:
            return self._extract(user, tool.get("inputSchema", {}).get("json", tool))
        if "nl-query" in sys_l or "natural-language question" in sys_l:
            return self._nl_query(user)
        return self._compile_rule(user)


def _first_col_after(rule: str, keyword: str) -> str | None:
    m = re.search(keyword + r"\s+(?:the\s+)?(\w+)", rule)
    return m.group(1) if m else None


_STOPWORDS = {"the", "a", "an", "to", "column", "field", "into", "as", "of", "value", "values"}


def _target_col(rule: str) -> str | None:
    """Best-effort column name for a rule like 'convert the amount column to int'.

    Tries '<col> column' first, then the first meaningful word after the verb.
    """
    m = re.search(r"(\w+)\s+column", rule)
    if m and m.group(1) not in _STOPWORDS:
        return m.group(1)
    m = re.search(r"(?:convert|coerce|cast)\s+((?:\w+\s+)*?\w+)", rule)
    if m:
        for word in m.group(1).split():
            if word not in _STOPWORDS:
                return word
    return None

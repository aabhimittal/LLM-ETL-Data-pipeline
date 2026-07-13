# Architecture

## The core idea: LLM as a *compiler*, not a *runtime*

The expensive, rate-limited, non-deterministic way to use an LLM for ETL is to
send **every record** through the model at run time. The cheap, deterministic,
unlimited way is to use the model **once** to *generate the transformation code*
from a plain-English rule, then run that code at scale with no LLM in the loop.

```
 Plain-English rule ─┐
                     │   ┌─────────────┐   generated pandas    ┌──────────────┐
 sample rows ────────┼──▶│  Bedrock    │──────────────────────▶│  sandbox +   │
                     │   │  (Claude)   │   (1 call, cached)     │  validate    │
                     ┘   └─────────────┘                        └──────┬───────┘
                                                                       │ deterministic
                                                                       ▼
 full dataset (millions of rows) ────────────────────────────▶  transform(df) ──▶ load
                                                 (zero LLM calls, no rate limits)
```

### Why this beats "LLM in the hot path"

| Concern            | LLM per row                    | LLM as compiler (this repo)     |
|--------------------|--------------------------------|---------------------------------|
| Cost               | O(rows) tokens                 | O(rules) tokens, then free      |
| Rate limits (TPM)  | Hit constantly                 | One call per rule, then none    |
| Latency            | Seconds/row                    | Microseconds/row (pandas)       |
| Determinism        | None (varies per call)         | Fully deterministic + testable  |
| Reviewability      | Opaque                         | Real code you can diff & unit-test |

> **Myth corrected:** Bedrock does *not* remove rate limiting. It has hard RPM/TPM
> quotas per model. Putting it in the per-row path makes throttling *worse*, not
> better. The compiler pattern sidesteps the quota by not calling the model at scale.

## The three lanes

1. **LLM-as-compiler ETL** (`compiler.py`, `runner.py`) — the default path for
   structured/semi-structured data with rules that *can* be expressed as code.
   Rules compile once, cache to disk, and run deterministically.

2. **Runtime batch extraction** (`runtime_llm.py`) — the narrow escape hatch for
   genuinely unstructured input (free text, emails, OCR'd docs) where no
   deterministic rule exists. Uses tool-use for structured JSON output and
   processes in batches. For high volume, back it with **Bedrock Batch
   Inference** (async, ~50% cheaper, separate quota).

3. **Natural-language query API** (`api.py`, `frontend/index.html`) — the
   frontend lane. The LLM *compiles* a question into a small query plan; pandas
   executes it deterministically. The model never renders UI and never sees the
   rows per request.

## Provider abstraction

Everything depends only on the tiny `LLM` protocol (`llm.py`). Two
implementations:

- `BedrockLLM` — real Bedrock Converse API (Claude), tool-use for structured
  output. Selected with `LLM_ETL_PROVIDER=bedrock`.
- `OfflineLLM` — deterministic stand-in so the repo runs, tests, and demos with
  **no AWS account**. It pattern-matches the demo's canonical rules and returns
  exactly the code/JSON a well-prompted Claude would.

Switching is one env var — no code changes.

## Safety

Generated code is executed in a restricted namespace (`sandbox.py`): no
`import`, `open`, `eval`, `os`, `sys`, etc.; only `pandas`, `numpy`, and `re`
are exposed, and the result must be a `DataFrame`. This is a pragmatic guardrail
— for untrusted rules, run generated code in a subprocess/container with CPU and
memory limits.

## Going to production with real Bedrock

- Request TPM/RPM quota increases for your chosen model.
- Use **prompt caching** on the system prompt to cut compile cost.
- Use **Batch Inference** for the runtime extraction lane at volume.
- Persist the rule cache (`.rule_cache/`) in object storage so a rule is
  compiled once per fleet, not once per worker.
- Pin the model id and snapshot generated code into version control for
  reproducible, auditable pipelines.

"""
observability.py — structured logs, correlation, LangSmith, cost.
=================================================================

WHY this file exists
--------------------
`print()` is how the teaching notebooks show their work, and it is the right
choice there — a learner needs to see the loop turn.

In a service it fails for three specific reasons:

1. **You cannot query it.** "Show me every run that hit the approval gate last
   Tuesday" is a filter over structured fields, not a grep over prose.
2. **You cannot correlate it.** Under concurrency, one request's lines are
   interleaved with fifty others'. Without a `thread_id` on every line you
   cannot reconstruct a single conversation.
3. **It leaks.** Customer text printed straight to stdout ends up in a log
   aggregator with a two-year retention and a much wider audience than you
   intended.

So: JSON lines, a correlation id on every record, and PII redacted on the way
out.

THE DIVISION OF LABOUR — three systems, three questions
--------------------------------------------------------
    logs       "what happened, in order, with what latency?"   engineers, days
    traces     "why did the agent decide that?"                engineers, weeks
    audit      "on what basis, and who authorised it?"         the business, years

They overlap and are not substitutes. Shipping only logs is the common mistake;
shipping logs and traces but no audit is the expensive one, and you discover it
during your first customer dispute.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, Optional

from .guardrails import redact
from .settings import Settings, get_settings

# The correlation id, carried implicitly through the call stack. A ContextVar
# rather than a parameter because threading it through every function is exactly
# the kind of tedium people skip — and a correlation id that is skipped 20% of
# the time is worse than none, because you will trust it.
_thread_id: ContextVar[str] = ContextVar("thread_id", default="-")
_customer_id: ContextVar[str] = ContextVar("customer_id", default="-")


class JsonFormatter(logging.Formatter):
    """One JSON object per line, with correlation ids and PII redacted."""

    def __init__(self, redact_pii: bool = True):
        super().__init__()
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "event": record.getMessage(),
            "logger": record.name,
            "thread_id": _thread_id.get(),
            "customer_id": _customer_id.get(),
        }
        for key, value in getattr(record, "__dict__", {}).items():
            if key in ("args", "msg", "exc_info", "exc_text", "stack_info",
                       "created", "msecs", "relativeCreated", "levelno",
                       "levelname", "name", "pathname", "filename", "module",
                       "lineno", "funcName", "processName", "process",
                       "threadName", "thread", "taskName"):
                continue
            payload[key] = value
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info)

        line = json.dumps(payload, default=str)
        return redact(line) if self.redact_pii else line


class TextFormatter(logging.Formatter):
    """Readable formatter for local development."""

    def __init__(self, redact_pii: bool = True):
        super().__init__("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S")
        self.redact_pii = redact_pii

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        extra = {
            k: v for k, v in record.__dict__.items()
            if k in ("thread_id", "config", "approved", "approver", "steps", "tools")
        }
        if _thread_id.get() != "-":
            extra.setdefault("thread_id", _thread_id.get())
        if extra:
            base += "  " + " ".join(f"{k}={v}" for k, v in extra.items())
        return redact(base) if self.redact_pii else base


_configured = False


def setup_observability(settings: Optional[Settings] = None) -> None:
    """Configure logging and LangSmith. Idempotent — safe to call per instance."""
    global _configured
    settings = settings or get_settings()
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = (
        JsonFormatter(settings.redact_pii_in_logs)
        if settings.log_format == "json"
        else TextFormatter(settings.redact_pii_in_logs)
    )
    handler.setFormatter(formatter)

    root = logging.getLogger("acme_support_agent")
    root.handlers = [handler]
    root.setLevel(settings.log_level)
    root.propagate = False

    if settings.langsmith_api_key:
        import os
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        root.info("langsmith_enabled", extra={"project": settings.langsmith_project})

    _configured = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"acme_support_agent.{name.split('.')[-1]}")


@contextmanager
def request_context(thread_id: str, customer_id: str = "-"):
    """
    Bind correlation ids for the duration of a request.

    Every log line emitted inside the block carries them, with no plumbing at
    the call sites — which is the only way correlation ids survive contact with
    a real codebase.
    """
    thread_token = _thread_id.set(thread_id)
    customer_token = _customer_id.set(customer_id)
    try:
        yield
    finally:
        _thread_id.reset(thread_token)
        _customer_id.reset(customer_token)


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
# Prices per 1M tokens. Hard-coded and therefore certain to drift — which is
# why this returns an ESTIMATE and says so. The purpose is to notice that a
# change made runs three times more expensive, and a stale-but-consistent price
# does that job perfectly well. For billing, use the provider's own numbers.
_PRICES: Dict[str, tuple] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}


class CostTracker:
    """
    Accumulates token usage and estimates spend across a run or a suite.

    Worth wiring up early for a reason notebook 02 makes with real numbers: an
    agent's cost per step *rises* as the transcript grows. A per-request average
    hides that; a per-run total does not.
    """

    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def add(self, response: Any) -> None:
        """Accumulate from a LangChain AIMessage's usage metadata."""
        usage = getattr(response, "usage_metadata", None) or {}
        self.input_tokens += usage.get("input_tokens", 0)
        self.output_tokens += usage.get("output_tokens", 0)
        self.calls += 1

    @property
    def estimated_usd(self) -> float:
        rate_in, rate_out = _PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * rate_in + self.output_tokens * rate_out) / 1_000_000

    def summary(self) -> str:
        return (
            f"{self.calls} model calls · {self.input_tokens:,} in / "
            f"{self.output_tokens:,} out · ~${self.estimated_usd:.4f} "
            f"(estimate, prices may be stale)"
        )

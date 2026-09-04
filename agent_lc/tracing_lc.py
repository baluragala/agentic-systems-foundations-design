"""
tracing_lc.py — LangSmith, and making both tracks comparable.
=============================================================

WHY this file exists
--------------------
Two jobs.

**1. LangSmith.** `agent_core/trace.py` argued that tracing is the debugging
interface for the whole paradigm, and then implemented a printed string. That was
the right call for teaching — you can read all of it — and the wrong call for
production, where you need search, diffing across runs, latency and cost
breakdowns, and a link you can paste into a ticket.

LangSmith is that, and adopting it is **two environment variables**. No code
changes, no decorators. Everything in `agent_lc` is already traced the moment you
set them, because LangChain instruments its own primitives.

**2. Comparability.** The genuinely useful teaching artifact here is running the
*same* task through both engines and putting the traces side by side. That only
works if both produce the same shape, so `to_trace()` converts a LangGraph
message list into the `agent_core.Trace` the notebooks already know how to
compare.

That conversion is itself informative. Everything `Trace` records —
`call_sequence()`, `error_rate()`, step timings — is derivable from LangGraph's
message history, because both are recording the same underlying events. The
concepts were never framework-specific; only the storage was.

WHAT LANGSMITH GIVES YOU THAT A PRINTED TRACE CANNOT
-----------------------------------------------------
* **Persistence and search.** "Show me every run last week where the agent
  escalated" is a query, not a grep through notebook output.
* **Cost and latency per step**, aggregated across runs.
* **Diffing.** Two runs of the same task after a prompt change, side by side.
* **Datasets and evaluators.** The task suite in `data/tasks/` becomes a
  LangSmith dataset you can run on every commit.
* **A shareable link.** The whole "trace as a bug report" argument from notebook
  05, except your colleague clicks a URL instead of importing JSON.

What it does *not* give you is the judgement about **which conditions to check
and in what order** — that is still `control.py`, wherever it lives.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from agent_core.state import Observation, ObservationStatus
from agent_core.trace import StepRecord, Trace


# ---------------------------------------------------------------------------
# LangSmith setup
# ---------------------------------------------------------------------------
def enable_langsmith(
    project: str = "agentic-systems-foundations",
    api_key: Optional[str] = None,
) -> bool:
    """
    Turn on LangSmith tracing. Returns True if it is actually configured.

    Deliberately returns a bool rather than raising: a notebook should say
    "tracing is off, here is how to turn it on" and keep running, not stop
    because an optional observability tool is not set up.
    """
    key = api_key or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    if not key:
        return False
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = key
    os.environ["LANGSMITH_PROJECT"] = project
    return True


def langsmith_status() -> str:
    """One-line report — used at the top of the LangGraph notebook cells."""
    if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and (
        os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    ):
        return f"LangSmith: ON  (project={os.getenv('LANGSMITH_PROJECT', 'default')})"
    return (
        "LangSmith: off. Set LANGSMITH_API_KEY (and call enable_langsmith()) to "
        "record runs. Everything still works without it — you just read traces "
        "in the notebook instead of in a UI."
    )


# ---------------------------------------------------------------------------
# LangGraph messages -> agent_core.Trace
# ---------------------------------------------------------------------------
def _is_error(message: Any) -> bool:
    """Did this ToolMessage represent a failure?

    LangChain marks handled tool errors with `status="error"`. Older versions and
    some tools only signal it in the text, so we check both — being wrong here
    would silently skew `error_rate()`, which is exactly the sort of quiet
    measurement bug that makes people distrust their own dashboards.
    """
    if getattr(message, "status", None) == "error":
        return True
    body = str(getattr(message, "content", "") or "")
    return body.startswith("Error:") or body.startswith("ERROR")


def to_trace(result: Dict[str, Any], goal: str) -> Trace:
    """
    Convert a LangGraph invoke() result into an `agent_core.Trace`.

    This is what lets `compare({"from scratch": t1, "langgraph": t2})` work
    across both engines — same table, same metrics, two implementations.

    One step = one AI message plus the tool results it produced, which is exactly
    how `agent_core.loop` counts steps, so the numbers mean the same thing on
    both sides.
    """
    messages = result.get("messages", [])
    trace = Trace(goal=goal)
    trace.status = "done"
    trace.stop_reason = result.get("stop_reason") or "graph reached END"

    step_number = 0
    pending: Optional[StepRecord] = None
    call_names: Dict[str, str] = {}   # tool_call_id -> tool name
    call_args: Dict[str, dict] = {}

    for message in messages:
        kind = getattr(message, "type", "")

        if kind == "ai":
            if pending is not None:
                trace.steps.append(pending)
            step_number += 1
            calls = getattr(message, "tool_calls", None) or []
            if calls:
                rendered = ", ".join(f"{c['name']}({c['args']})" for c in calls)
                decision = f"Decision(ACT: {rendered})"
            else:
                preview = " ".join(str(message.content or "").split())[:70]
                decision = f"Decision(FINAL: {preview!r})"
            pending = StepRecord(
                step=step_number,
                decision=decision,
                thought=str(message.content or "") or None,
            )
            for call in calls:
                call_names[call.get("id", "")] = call["name"]
                call_args[call.get("id", "")] = call.get("args", {})

        elif kind == "tool" and pending is not None:
            call_id = getattr(message, "tool_call_id", "") or ""
            failed = _is_error(message)
            pending.observations.append(
                Observation(
                    step=step_number,
                    tool=getattr(message, "name", None) or call_names.get(call_id, "?"),
                    args=call_args.get(call_id, {}),
                    status=ObservationStatus.ERROR if failed else ObservationStatus.OK,
                    result=None if failed else str(message.content),
                    error=str(message.content) if failed else None,
                )
            )

    if pending is not None:
        trace.steps.append(pending)

    final = [m for m in messages if getattr(m, "type", "") == "ai"]
    trace.final_answer = str(final[-1].content) if final else None
    trace.ended_at = trace.started_at
    return trace


def summarise(result: Dict[str, Any], goal: str) -> str:
    """One line, in the same format `agent_core.Trace.summary()` produces."""
    return to_trace(result, goal).summary()


def call_sequence(result: Dict[str, Any]) -> List[str]:
    """Just the tool names, in order — the trajectory you assert on."""
    return [
        call["name"]
        for message in result.get("messages", [])
        if getattr(message, "type", "") == "ai"
        for call in (getattr(message, "tool_calls", None) or [])
    ]


def final_answer(result: Dict[str, Any]) -> str:
    ai = [m for m in result.get("messages", []) if getattr(m, "type", "") == "ai"]
    return str(ai[-1].content) if ai else ""


def show_messages(result: Dict[str, Any], width: int = 88) -> str:
    """
    Readable dump of a LangGraph message list.

    Worth printing at least once in class: it is the *same* transcript the
    from-scratch package builds by hand in `state.messages`, just constructed by
    a reducer instead of by `add_message()` calls.
    """
    lines = []
    for message in result.get("messages", []):
        kind = getattr(message, "type", "?")
        calls = getattr(message, "tool_calls", None)
        if calls:
            rendered = ", ".join(f"{c['name']}({c['args']})" for c in calls)
            lines.append(f"{kind:<9} -> {rendered[:width]}")
        else:
            body = " ".join(str(getattr(message, 'content', '') or "").split())
            flag = "!!" if kind == "tool" and _is_error(message) else "  "
            lines.append(f"{kind:<9} {flag} {body[:width]}")
    return "\n".join(lines)

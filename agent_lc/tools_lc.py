"""
tools_lc.py — the same five Acme tools, as LangChain tools.
===========================================================

WHY this file exists
--------------------
`agent_core/acme_tools.py` derives its schemas from Python signatures and
docstrings with about a hundred lines of hand-rolled reflection. That was the
right way to *learn* what a tool schema is.

This is the way you would actually ship it: a **Pydantic model** per tool, and
LangChain's `@tool` decorator wiring it up. Compare the two files side by side —
they describe the *same contracts*, and that is the point.

WHAT CHANGES, AND WHAT DOES NOT
-------------------------------
What changes: who writes the boilerplate. `build_schema()` becomes
`args_schema=SomeModel`. Our `validate_args()` becomes Pydantic's validator,
which is faster, more standards-complete, and battle-tested by a very large
number of people.

What does **not** change — and this is the lesson worth carrying to work:

* `Literal[...]` is still where an enum comes from.
* A regex constraint is still what stops `1042` being accepted as an order ID —
  it is just `Field(pattern=...)` instead of a docstring marker.
* The **description is still the interface.** Pydantic will not write it for you,
  and a model that cannot tell two tools apart will still pick the wrong one.
* A tool that returns a bare dict still forces the model to guess. We still
  return prose.

> The framework abstracts the mechanics, not the design decisions.

ONE REAL DIFFERENCE WORTH KNOWING
---------------------------------
`agent_core.Tool.invoke()` never raises — every failure becomes an `Observation`
the agent can read. LangChain's `ToolNode` does something similar but not
identical: it catches exceptions and returns a `ToolMessage` with the error text,
controlled by `handle_tool_errors`. The default is on, which is the behaviour you
want; turn it off and a raising tool kills the graph run exactly as it would kill
our loop. Same design decision, different switch — see `graph.py`.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Literal

from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Reuse the SAME data and the SAME business logic as the from-scratch package.
# Only the tool-definition layer is being swapped, so any behavioural difference
# you observe between the two tracks comes from the framework, not from the
# tools quietly doing something different.
from agent_core.acme_tools import (
    TODAY,
    _CORPUS,
    _ORDERS,
    _QUERY_STOP,
    _paragraphs,
    calculate as _core_calculate,
    check_refund_eligibility as _core_refund,
    escalate_to_human as _core_escalate,
    get_order_status as _core_status,
    search_docs as _core_search,
)

ORDER_ID_PATTERN = r"^ACME-\d{4}$"


# ---------------------------------------------------------------------------
# Argument schemas — Pydantic models, the enterprise-standard way
# ---------------------------------------------------------------------------
# Everything the docstring conventions encoded in agent_core is now a typed,
# validated field. Note how much MORE explicit this is: the pattern is a real
# constraint object rather than a marker we parse out of prose.
class SearchDocsArgs(BaseModel):
    """Search the Acme Cloud product documentation."""

    query: str = Field(
        description="The question or keywords to search for, in natural language."
    )
    k: int = Field(
        default=2,
        ge=1,
        le=5,
        description=(
            "How many passages to return. Keep this small — every passage "
            "returned is spent from the agent's context budget."
        ),
    )


class OrderIdArgs(BaseModel):
    """Arguments for tools that address a single order."""

    order_id: str = Field(
        pattern=ORDER_ID_PATTERN,
        description="The Acme order identifier, for example ACME-1042.",
        examples=["ACME-1042", "ACME-1046"],
    )


class RefundArgs(BaseModel):
    """Arguments for a refund eligibility decision."""

    order_id: str = Field(
        pattern=ORDER_ID_PATTERN,
        description="The Acme order identifier, for example ACME-1042.",
        examples=["ACME-1042"],
    )
    # The single highest-value line in the file: the model picks from a list
    # instead of inventing a reason code.
    reason: Literal[
        "billing_error", "service_outage", "not_as_described", "changed_mind"
    ] = Field(
        description=(
            "Why the refund is being requested. Must be one of the four reason "
            "codes defined in the refund policy."
        )
    )


class CalculateArgs(BaseModel):
    expression: str = Field(
        description=(
            "An arithmetic expression using only numbers and the operators "
            "+ - * / ( ) %. For example: 199 * 12 * 0.85"
        ),
        examples=["199 * 12", "448.50 * 0.15"],
    )


class EscalateArgs(BaseModel):
    summary: str = Field(
        min_length=10,
        description=(
            "What the customer asked, what you checked, and what you found. "
            "Write it for the human who picks this up next."
        ),
    )


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------
# The docstring is still what the model reads to CHOOSE the tool, so it still
# has to say WHEN to use it — Pydantic describes the arguments, never the
# judgement. This is the part no framework does for you.
@tool("search_docs", args_schema=SearchDocsArgs)
def search_docs(query: str, k: int = 2) -> str:
    """Search the Acme Cloud product documentation for passages answering a question.

    Use this for anything about policies, pricing, plans, security, SLAs, refunds,
    onboarding or the API. It searches written documentation, not customer records.
    Examples: "How much does the Growth plan cost?", "What does the refund policy
    say about cancellations?"
    """
    return _core_search.func(query=query, k=k)


@tool("get_order_status", args_schema=OrderIdArgs)
def get_order_status(order_id: str) -> str:
    """Look up the current status and details of a single customer order.

    Use this whenever a specific order is mentioned. Returns customer, plan,
    status, dates and amount. It does NOT decide refunds — use
    check_refund_eligibility for that.
    """
    return _core_status.func(order_id=order_id)


@tool("calculate", args_schema=CalculateArgs)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly.

    Use this for any calculation — totals, prorated amounts, percentages,
    overage charges. Do not do arithmetic yourself; language models make
    plausible-looking arithmetic errors that are hard to spot afterwards.
    """
    return _core_calculate.func(expression=expression)


@tool("check_refund_eligibility", args_schema=RefundArgs)
def check_refund_eligibility(order_id: str, reason: str) -> str:
    """Decide whether an order qualifies for a refund under Acme Cloud policy.

    Applies the written refund policy to a specific order: the 30-day window,
    the eligible-reason list, the order's status, and the Enterprise exclusion.
    """
    return _core_refund.func(order_id=order_id, reason=reason)


@tool("escalate_to_human", args_schema=EscalateArgs)
def escalate_to_human(summary: str) -> str:
    """Hand this request to a human agent, with a summary of what you established.

    Use this when the policy requires human judgement, when you have checked what
    you can and still cannot answer safely, or when acting would exceed what an
    automated agent is permitted to do. Escalating is a correct outcome, not a
    failure — an agent that guesses instead of escalating is the failure.
    """
    return _core_escalate.func(summary=summary)


# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------
ACME_TOOLS = [
    search_docs,
    get_order_status,
    calculate,
    check_refund_eligibility,
    escalate_to_human,
]

# Terminal tools end the run when they succeed. LangGraph has no built-in notion
# of this — `tools_condition` only asks "were tool calls requested?" — so it is
# something you implement in your own conditional edge. A good example of the
# framework covering the common case and leaving the domain rule to you.
TERMINAL_TOOLS = {"escalate_to_human"}

TOOLS_BY_NAME = {t.name: t for t in ACME_TOOLS}


def subset(*names: str) -> list:
    """Scope a tool list — the LangChain equivalent of `ToolRegistry.subset()`.

    Scoping matters exactly as much here as it did in the from-scratch package:
    tool-choice accuracy degrades as the list grows, silently, whatever is
    executing the loop.
    """
    missing = [n for n in names if n not in TOOLS_BY_NAME]
    if missing:
        raise KeyError(f"unknown tools: {missing}. Available: {list(TOOLS_BY_NAME)}")
    return [TOOLS_BY_NAME[n] for n in names]


def describe(tools=None) -> str:
    """Render tools as prompt text — for the tool-inventory block in a prompt."""
    lines = []
    for t in tools or ACME_TOOLS:
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        params = ", ".join(schema.get("properties", {}))
        lines.append(f"- {t.name}({params}): {(t.description or '').splitlines()[0]}")
    return "\n".join(lines)

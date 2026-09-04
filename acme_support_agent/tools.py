"""
tools.py — the production toolset, including one that actually costs money.
===========================================================================

WHY this file exists separately from the teaching toolboxes
-----------------------------------------------------------
Every tool in `agent_core` and `agent_lc` is **read-only**. That is not an
accident of the teaching example — it is why those agents are safe to let a
classroom run.

A real support agent has to *do* something eventually, and the moment it does,
three concerns appear that read-only agents never face:

1. **Idempotency.** The loop may retry. The network may time out after the
   refund succeeded. Without an idempotency key, a retry issues a second refund
   — and you will find out from the customer, not from your monitoring.
2. **Authorisation.** Who decided this was allowed? Some actions must not
   happen without a human, and that gate belongs *outside* the model.
3. **Reversibility.** A read that returns the wrong row is a bug. A refund
   issued against the wrong order is an incident.

`issue_refund` below is the one tool with side effects, and it is deliberately
the only one. **Keep the blast radius small and obvious.** An agent with fifteen
write tools has fifteen ways to have a bad day; this one has a single, heavily
guarded door.

THE SPLIT THAT MAKES THE APPROVAL GATE POSSIBLE
-----------------------------------------------
Notice `check_refund_eligibility` (decide) is separate from `issue_refund` (act).
That separation is what lets the graph interrupt for a human *between* the
decision and the side effect. Fuse them into one `process_refund` tool and you
have nowhere to put the gate — the money has already moved by the time you
notice you wanted approval.

**Design your tools so the consequential step is its own call.**
"""
from __future__ import annotations

import hashlib
import threading
from datetime import date, datetime
from typing import Dict, Literal, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from agent_core.acme_tools import (
    TODAY,
    _ORDERS,
    calculate as _core_calculate,
    check_refund_eligibility as _core_refund,
    get_order_status as _core_status,
    search_docs as _core_search,
)

ORDER_ID_PATTERN = r"^ACME-\d{4}$"

# Tools whose execution has consequences outside this process. The graph routes
# these through the approval gate; everything else runs freely. Naming the set
# explicitly — rather than inferring it — means adding a write tool is a
# deliberate, reviewable act.
HIGH_RISK_TOOLS = {"issue_refund"}


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------
class _RefundLedger:
    """
    In-memory record of refunds already issued, keyed by idempotency key.

    In production this is a row in your database with a unique constraint, and
    the constraint — not the application check — is what actually guarantees
    uniqueness. Two processes racing will both pass an in-memory check; only the
    database says no to the second one.

    The reference implementation is honest about being a stand-in.
    """

    def __init__(self) -> None:
        self._issued: Dict[str, dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def key(order_id: str, amount_usd: float, reason: str) -> str:
        """Stable key for 'this exact refund'. Same inputs -> same key."""
        raw = f"{order_id.upper()}|{amount_usd:.2f}|{reason}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def already_issued(self, key: str) -> Optional[dict]:
        with self._lock:
            return self._issued.get(key)

    def record(self, key: str, payload: dict) -> dict:
        with self._lock:
            existing = self._issued.get(key)
            if existing:
                return existing
            self._issued[key] = payload
            return payload

    def all(self) -> Dict[str, dict]:
        with self._lock:
            return dict(self._issued)

    def reset(self) -> None:
        """Tests only."""
        with self._lock:
            self._issued.clear()


LEDGER = _RefundLedger()


# ---------------------------------------------------------------------------
# Argument schemas
# ---------------------------------------------------------------------------
class OrderIdArgs(BaseModel):
    order_id: str = Field(
        pattern=ORDER_ID_PATTERN,
        description="The Acme order identifier, for example ACME-1042.",
        examples=["ACME-1042"],
    )


class SearchArgs(BaseModel):
    query: str = Field(description="The question or keywords to search for.")
    k: int = Field(default=2, ge=1, le=5, description="How many passages to return.")


class CalcArgs(BaseModel):
    expression: str = Field(
        description="Arithmetic using only numbers and + - * / ( ) %. E.g. 199 * 12"
    )


class RefundCheckArgs(BaseModel):
    order_id: str = Field(pattern=ORDER_ID_PATTERN, description="The Acme order identifier.")
    reason: Literal[
        "billing_error", "service_outage", "not_as_described", "changed_mind"
    ] = Field(description="Why the refund is requested. One of the four policy codes.")


class IssueRefundArgs(BaseModel):
    """
    Arguments for the one tool that moves money.

    Every field is constrained. `amount_usd` has an upper bound in the *schema*,
    not just in the approval logic, because defence in depth means a bug in the
    gate should still not permit a million-dollar refund.
    """

    order_id: str = Field(pattern=ORDER_ID_PATTERN, description="The Acme order identifier.")
    amount_usd: float = Field(
        gt=0, le=100_000,
        description="The refund amount in USD. Must match the order's charged amount.",
    )
    reason: Literal[
        "billing_error", "service_outage", "not_as_described", "changed_mind"
    ] = Field(description="The policy reason code justifying the refund.")


class EscalateArgs(BaseModel):
    summary: str = Field(
        min_length=10,
        description="What was asked, what you checked, what you found. For the human who picks this up.",
    )


# ---------------------------------------------------------------------------
# Read-only tools
# ---------------------------------------------------------------------------
@tool("search_docs", args_schema=SearchArgs)
def search_docs(query: str, k: int = 2) -> str:
    """Search Acme Cloud documentation for policies, pricing, plans, SLAs and refunds.

    Use this for anything about written policy. It searches documentation, not
    customer records. Example: "What does the refund policy say about cancellations?"
    """
    return _core_search.func(query=query, k=k)


@tool("get_order_status", args_schema=OrderIdArgs)
def get_order_status(order_id: str) -> str:
    """Look up an order's customer, plan, status, dates and amount charged.

    Use this whenever a specific order is mentioned. Always call this before
    assessing or issuing a refund — you cannot judge either without the order.
    """
    return _core_status.func(order_id=order_id)


@tool("calculate", args_schema=CalcArgs)
def calculate(expression: str) -> str:
    """Evaluate arithmetic exactly. Use for any total, proration or percentage.

    Do not do arithmetic yourself; language models make plausible-looking
    arithmetic errors that are hard to spot afterwards.
    """
    return _core_calculate.func(expression=expression)


@tool("check_refund_eligibility", args_schema=RefundCheckArgs)
def check_refund_eligibility(order_id: str, reason: str) -> str:
    """DECIDE whether an order qualifies for a refund. Does NOT issue one.

    Applies the written policy: the 30-day window, the eligible-reason list, the
    order's status, and the Enterprise exclusion. Call this before issue_refund —
    issuing without checking is a policy violation.
    """
    return _core_refund.func(order_id=order_id, reason=reason)


@tool("escalate_to_human", args_schema=EscalateArgs)
def escalate_to_human(summary: str) -> str:
    """Hand the request to a human agent with a summary of what you established.

    Use when policy requires human judgement, or when you cannot answer safely.
    Escalating is a correct outcome — guessing instead of escalating is not.
    """
    ticket = f"ESC-{abs(hash(summary)) % 90000 + 10000}"
    return f"Escalated to a human agent as ticket {ticket}. Summary: {summary.strip()[:400]}"


# ---------------------------------------------------------------------------
# The one tool with side effects
# ---------------------------------------------------------------------------
@tool("issue_refund", args_schema=IssueRefundArgs)
def issue_refund(order_id: str, amount_usd: float, reason: str) -> str:
    """ISSUE a refund. This moves money and cannot be undone by you.

    Only call this after check_refund_eligibility has returned ELIGIBLE for the
    same order and reason. A human may be asked to approve before this runs.

    Never call this for an Enterprise contract, or for an order already refunded.
    """
    order_id = order_id.upper()
    record = _ORDERS.get(order_id)

    # ---- belt and braces -------------------------------------------------
    # The graph's approval gate should already have caught all of these. We
    # check again anyway, because a tool that moves money must be safe when
    # called directly — by a future refactor, a test, or a different graph.
    # "The caller validates" is how side-effecting functions become incidents.
    if record is None:
        return f"REFUND NOT ISSUED: order {order_id} does not exist."

    if record["plan"] == "Enterprise":
        return (
            f"REFUND NOT ISSUED: order {order_id} is an Enterprise contract. "
            "Enterprise refunds must be handled by the named technical account "
            "manager under the contract's termination clause."
        )

    if record["status"] == "refunded":
        return (
            f"REFUND NOT ISSUED: order {order_id} has already been refunded. "
            "A second refund is never issued."
        )

    if record["status"] in ("cancelled", "processing"):
        return (
            f"REFUND NOT ISSUED: order {order_id} is {record['status']}; no charge "
            "was captured, so there is nothing to refund."
        )

    charged = float(record["amount_usd"])
    if abs(amount_usd - charged) > 0.01:
        # A refund for more than was charged is the single most expensive
        # arithmetic slip available to this agent.
        return (
            f"REFUND NOT ISSUED: requested ${amount_usd:.2f} but order {order_id} "
            f"was charged ${charged:.2f}. The amounts must match."
        )

    # ---- idempotency -----------------------------------------------------
    key = LEDGER.key(order_id, amount_usd, reason)
    existing = LEDGER.already_issued(key)
    if existing:
        # NOT an error. A retry of an identical request returns the ORIGINAL
        # result — that is what idempotency means, and it is why a timeout on
        # the caller's side is survivable.
        return (
            f"Refund {existing['refund_id']} for order {order_id} was already "
            f"issued (${amount_usd:.2f}). No duplicate was created."
        )

    refund_id = f"REF-{key[:8].upper()}"
    LEDGER.record(key, {
        "refund_id": refund_id,
        "order_id": order_id,
        "amount_usd": amount_usd,
        "reason": reason,
        "issued_on": TODAY.isoformat(),
    })
    return (
        f"Refund {refund_id} issued for order {order_id}: ${amount_usd:.2f} "
        f"({reason}). Returned to the original payment method within 5 to 7 "
        "business days."
    )


# ---------------------------------------------------------------------------
# The toolset
# ---------------------------------------------------------------------------
SUPPORT_TOOLS = [
    search_docs,
    get_order_status,
    calculate,
    check_refund_eligibility,
    issue_refund,
    escalate_to_human,
]

TOOLS_BY_NAME = {t.name: t for t in SUPPORT_TOOLS}


def refund_amount_for(order_id: str) -> Optional[float]:
    """The charged amount for an order — used by the approval gate to size the risk."""
    record = _ORDERS.get((order_id or "").upper())
    return float(record["amount_usd"]) if record else None


def order_plan(order_id: str) -> Optional[str]:
    record = _ORDERS.get((order_id or "").upper())
    return record["plan"] if record else None

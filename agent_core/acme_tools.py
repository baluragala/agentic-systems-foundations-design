"""
acme_tools.py — the toolbox, and the bridge back to the RAG session.
====================================================================

WHY these five tools
--------------------
They are not a grab-bag. Each one exists to make a specific lesson concrete, and
together they cover the space of things that go wrong:

  search_docs               unstructured retrieval — and the C8 bridge
  get_order_status          a strict format constraint (the `pattern` lesson)
  calculate                 why a language model should not do arithmetic
  check_refund_eligibility  multi-argument schemas with an enum
  escalate_to_human         a terminal tool, and knowing when to stop

THE BRIDGE WORTH LABOURING
--------------------------
`search_docs` is a real retriever over the same Acme Cloud corpus the previous
session built a RAG pipeline on. Not a stub — it scores, ranks, and returns
passages.

That turns an assertion into a demonstration: **RAG is a tool an agent calls,
not a competing architecture.** The whole of last session — loading, chunking,
retrieval, ranking — collapses into one entry in a tool registry. A learner who
sees that stops asking "should I build RAG or an agent?" and starts asking "what
does this agent need to be able to look up?", which is the better question.

The retrieval here is deliberately simple (TF-scored keyword overlap, no
embeddings) because the *retrieval* is not the lesson this time; the fact that
it sits behind a tool interface is. Swap in a real vector store and nothing else
in the package changes — which is itself the point about interfaces.

A NOTE ON TOOL DESIGN, WHICH IS THE REAL SUBJECT HERE
-----------------------------------------------------
Look at what these functions return: strings, in prose, that state what happened
and what it means. Not dicts, not ORM objects, not status codes.

That is because the consumer is a language model. A tool returning
`{"eligible": false, "code": 7}` forces the model to guess what code 7 means; a
tool returning "Not eligible: the 30-day refund window closed on 2026-08-13"
does not. The tool's return value is *prompt text you are writing on the model's
behalf* — every ambiguity you leave in it becomes a chance to hallucinate.

The recurring question — *"what does this step look like when it goes wrong, and
where would you see it in the trace?"*:

  * A tool that returns raw data     -> the model misreads it and the trace
                                        shows a correct observation followed by
                                        a wrong answer. The nastiest kind.
  * A tool that returns nothing on
    "not found"                      -> the model treats absence as licence to
                                        invent. Always say "not found" in words.
  * A tool that raises               -> no trace at all past that step.
"""
from __future__ import annotations

import json
import os
import re
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional

from .tools import ToolRegistry, tool

# ---------------------------------------------------------------------------
# Data location
# ---------------------------------------------------------------------------
# Resolved relative to the package so the tools work from a notebook, a script,
# or a Colab clone without anyone passing paths around. Overridable via env for
# the rare case where the data lives elsewhere.
_HERE = Path(__file__).resolve().parent
_DATA = Path(os.getenv("AGENT_DATA_DIR", _HERE.parent / "data"))

# "Today" is pinned so that date arithmetic in the refund policy gives the same
# answer in every classroom, forever. A demo whose output depends on the day it
# is run is a demo that will embarrass you eventually.
TODAY = date(2026, 9, 4)


def _load_orders() -> Dict[str, dict]:
    path = _DATA / "acme" / "orders.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _load_corpus() -> Dict[str, str]:
    """Load the Acme docs as {filename: text}. HTML is crudely de-tagged."""
    corpus: Dict[str, str] = {}
    folder = _DATA / "corpus"
    if not folder.exists():
        return corpus
    for path in sorted(folder.glob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".html"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if path.suffix.lower() == ".html":
            text = re.sub(r"<[^>]+>", " ", text)
        corpus[path.name] = text
    return corpus


_ORDERS = _load_orders()
_CORPUS = _load_corpus()


# ---------------------------------------------------------------------------
# 1. search_docs — the RAG bridge
# ---------------------------------------------------------------------------
# Words too common to carry retrieval signal. Scoring on these is why a query
# about Kubernetes can "match" the refund policy — every document contains "for".
_QUERY_STOP = {
    "the", "and", "for", "are", "was", "does", "did", "can", "you", "your",
    "our", "with", "from", "that", "this", "what", "how", "why", "who", "any",
    "all", "have", "has", "had", "not", "but", "its", "his", "her", "them",
    "acme", "cloud", "please", "tell", "about", "would", "could", "should",
}


def _paragraphs(text: str) -> List[str]:
    """Split on blank lines — the cheapest sensible chunking (see C8, stage 2)."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 60]


@tool(
    max_result_chars=1500,
    examples=[
        "How much does the Growth plan cost per month?",
        "What is the uptime SLA for Enterprise?",
        "What does the refund policy say about cancellations?",
        "Does Acme Cloud support single sign-on?",
    ],
)
def search_docs(query: str, k: int = 2) -> str:
    """Search the Acme Cloud product documentation for passages answering a question.

    Use this for anything about policies, pricing, plans, security, SLAs, refunds,
    onboarding or the API. It searches written documentation, not customer records.

    Args:
        query: The question or keywords to search for, in natural language.
        k: How many passages to return. Keep this small — every passage returned
            is spent from the agent's context budget.

    Returns:
        The best-matching passages, each labelled with its source document.
    """
    if not _CORPUS:
        return "NOT FOUND: the documentation corpus is not available at data/corpus."

    # Score by how many distinct query terms a paragraph contains, with a small
    # bonus for repeats. This is a poor cousin of BM25 and entirely adequate for
    # a six-document corpus — see the module docstring on why the retrieval
    # quality is deliberately not the point here.
    terms = {t for t in re.findall(r"[a-z]{3,}", query.lower())} - _QUERY_STOP
    if not terms:
        return "NOT FOUND: the query contained no searchable terms."

    scored = []
    for source, text in _CORPUS.items():
        for para in _paragraphs(text):
            low = para.lower()
            hits = sum(1 for t in terms if t in low)
            if hits:
                density = sum(low.count(t) for t in terms) / (len(para) / 100)
                scored.append((hits + density * 0.1, source, para))

    scored.sort(key=lambda row: -row[0])
    # Require at least TWO distinct query terms in a passage. Without this floor
    # a single common word ("for", "plan") matches half the corpus and the tool
    # confidently returns irrelevant text — which the agent then answers from.
    # A retriever that never says "not found" is a hallucination engine with
    # extra steps; the same lesson C8 taught with its negative eval questions.
    top = [row for row in scored if row[0] >= 2][: max(1, min(k, 5))]

    if not top:
        # Saying "not found" in words is a design decision, not a nicety. An
        # empty string here invites the model to fill the silence.
        return (
            f"NOT FOUND: no passage in the Acme Cloud documentation matched "
            f"{query!r}. Do not guess an answer from general knowledge."
        )

    blocks = []
    for score, source, para in top:
        clean = " ".join(para.split())
        blocks.append(f"[{source}] (relevance {score:.1f})\n{clean}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# 2. get_order_status — the pattern lesson
# ---------------------------------------------------------------------------
@tool(
    examples=[
        "What is the status of order ACME-1048?",
        "Look up order ACME-1042 for me.",
        "Has my order shipped yet?",
    ]
)
def get_order_status(order_id: str) -> str:
    """Look up the current status and details of a single customer order.

    Use this whenever a specific order is mentioned. It returns customer, plan,
    status, dates and amount. It does NOT decide refunds — use
    check_refund_eligibility for that.

    Args:
        order_id: The Acme order identifier, for example ACME-1042.
            Bare numbers are not valid order IDs. pattern: ^ACME-\\d{4}$

    Returns:
        A description of the order, or a clear not-found message.
    """
    record = _ORDERS.get(order_id.upper())
    if record is None:
        # Listing the valid IDs would leak the whole database in a real system;
        # here we say plainly that it does not exist so the agent stops retrying
        # variations of a well-formed but unknown ID.
        return (
            f"NOT FOUND: no order with ID {order_id}. The ID is correctly "
            "formatted but does not exist in the order system."
        )

    delivered = record["delivered_on"] or "not yet delivered"
    return (
        f"Order {order_id.upper()} — customer: {record['customer']}. "
        f"Item: {record['item']}. Plan: {record['plan']}. "
        f"Status: {record['status']}. Placed on {record['placed_on']}, "
        f"{delivered}. Amount charged: ${record['amount_usd']:.2f}."
    )


# ---------------------------------------------------------------------------
# 3. calculate — why models should not do arithmetic
# ---------------------------------------------------------------------------
_SAFE_EXPR = re.compile(r"^[\d\s+\-*/().%]+$")


@tool(
    examples=[
        "What is 199 multiplied by 12?",
        "How much is 15% of 448.50?",
        "What would twelve months at 199 cost?",
    ]
)
def calculate(expression: str) -> str:
    """Evaluate an arithmetic expression exactly.

    Use this for any calculation — totals, prorated amounts, percentages,
    overage charges. Do not do arithmetic yourself; language models make
    plausible-looking arithmetic errors that are hard to spot afterwards.

    Args:
        expression: An arithmetic expression using only numbers and the
            operators + - * / ( ) %. For example: 199 * 12 * 0.85

    Returns:
        The expression and its exact result.
    """
    cleaned = expression.strip().replace("×", "*").replace("÷", "/").replace(",", "")

    # Whitelist rather than blacklist. `eval` on model-supplied text is a remote
    # code execution hole; the regex is what makes this line defensible, and it
    # is worth pointing at in class. In production you would use a real
    # expression parser (`ast.literal_eval` cannot do arithmetic; `simpleeval`
    # can) rather than trusting a regex you wrote.
    if not _SAFE_EXPR.match(cleaned):
        return (
            f"REJECTED: {expression!r} is not a plain arithmetic expression. "
            "Only numbers and the operators + - * / ( ) % are allowed."
        )
    try:
        value = eval(cleaned, {"__builtins__": {}}, {})  # noqa: S307 - gated above
    except ZeroDivisionError:
        return "ERROR: division by zero."
    except Exception as exc:
        return f"ERROR: could not evaluate {expression!r} ({type(exc).__name__})."

    if isinstance(value, float):
        value = round(value, 4)
    return f"{cleaned} = {value}"


# ---------------------------------------------------------------------------
# 4. check_refund_eligibility — enums and multi-argument schemas
# ---------------------------------------------------------------------------
@tool(
    examples=[
        "Can I get a refund for order ACME-1046?",
        "Am I eligible for money back on ACME-1042?",
        "I was charged twice, is that refundable?",
    ]
)
def check_refund_eligibility(
    order_id: str,
    reason: Literal[
        "billing_error", "service_outage", "not_as_described", "changed_mind"
    ],
) -> str:
    """Decide whether an order qualifies for a refund under Acme Cloud policy.

    Applies the written refund policy to a specific order: the 30-day window,
    the eligible-reason list, the order's status, and the Enterprise exclusion.

    Args:
        order_id: The Acme order identifier, for example ACME-1042. pattern: ^ACME-\\d{4}$
        reason: Why the refund is being requested. Must be one of the four
            reason codes defined in the refund policy.

    Returns:
        An eligibility decision with the policy rule that produced it.
    """
    record = _ORDERS.get(order_id.upper())
    if record is None:
        return f"NOT FOUND: no order with ID {order_id}, so eligibility cannot be assessed."

    placed = datetime.strptime(record["placed_on"], "%Y-%m-%d").date()
    age_days = (TODAY - placed).days
    status = record["status"]

    # The rules are checked in the order the written policy states them, so a
    # learner can hold the document and the code side by side. When policy and
    # code disagree, that divergence is itself a great debugging exercise.
    if record["plan"] == "Enterprise":
        return (
            f"NOT ELIGIBLE (automated): order {order_id.upper()} is an Enterprise "
            "contract. Enterprise refunds are governed by the contract's "
            "termination clause and must be handled by the named technical "
            "account manager. Escalate to a human."
        )

    if status == "refunded":
        return (
            f"NOT ELIGIBLE: order {order_id.upper()} has already been refunded. "
            "A second refund is never issued. Escalate duplicate requests to a human."
        )

    if status == "cancelled":
        return (
            f"NOT ELIGIBLE: order {order_id.upper()} was cancelled and no charge "
            "was captured, so there is nothing to refund."
        )

    if status == "processing":
        return (
            f"NOT ELIGIBLE for refund: order {order_id.upper()} is still processing "
            "and no charge has been captured. It should be CANCELLED instead, "
            "which is free and immediate."
        )

    if reason == "billing_error":
        return (
            f"ELIGIBLE: order {order_id.upper()} qualifies under 'billing_error', "
            "which is always eligible regardless of the 30-day window. "
            f"Refund amount ${record['amount_usd']:.2f}, returned to the original "
            "payment method within 5 to 7 business days."
        )

    if age_days > 30:
        return (
            f"NOT ELIGIBLE: order {order_id.upper()} was placed {age_days} days ago, "
            f"outside the 30-day refund window, and the reason given ({reason}) "
            "does not override that window. Only 'billing_error' does."
        )

    return (
        f"ELIGIBLE: order {order_id.upper()} was placed {age_days} days ago, within "
        f"the 30-day window, with an accepted reason ({reason}). "
        f"Refund amount ${record['amount_usd']:.2f}, returned to the original "
        "payment method within 5 to 7 business days."
    )


# ---------------------------------------------------------------------------
# 5. escalate_to_human — the terminal tool
# ---------------------------------------------------------------------------
@tool(
    terminal=True,
    examples=[
        "I need to speak to a person about this.",
        "This needs a human to decide.",
    ],
)
def escalate_to_human(summary: str) -> str:
    """Hand this request to a human agent, with a summary of what you established.

    Use this when the policy requires human judgement, when you have checked what
    you can and still cannot answer safely, or when acting would exceed what an
    automated agent is permitted to do. Escalating is a correct outcome, not a
    failure — an agent that guesses instead of escalating is the failure.

    Args:
        summary: What the customer asked, what you checked, and what you found.
            Write it for the human who picks this up next.

    Returns:
        Confirmation that the request was handed off.
    """
    ticket = f"ESC-{abs(hash(summary)) % 90000 + 10000}"
    return (
        f"Escalated to a human agent as ticket {ticket}. "
        f"Summary passed on: {summary.strip()[:400]}"
    )


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
def acme_registry() -> ToolRegistry:
    """
    The full Acme toolbox.

    Notebooks 03 onward call this. Skills then take *subsets* of it — see
    `skills.py` for why handing an agent five tools when it needs two measurably
    degrades its tool choice.
    """
    return ToolRegistry(
        [
            search_docs,
            get_order_status,
            calculate,
            check_refund_eligibility,
            escalate_to_human,
        ]
    )


def order_ids() -> List[str]:
    """The order IDs available in the sample data — handy in notebooks."""
    return sorted(_ORDERS)

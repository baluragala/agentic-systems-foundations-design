"""
acme_support_agent — the end-to-end reference implementation.
=============================================================

The **final track**. `agent_core` teaches the mechanism, `agent_lc` shows the
framework, and this is the whole thing assembled the way you would actually
deploy it: LangGraph, a human in the loop, guardrails, an audit trail, an
evaluation gate and an HTTP surface.

    from acme_support_agent import SupportAgent

    agent = SupportAgent()
    reply = agent.chat("Please refund ACME-1046, I changed my mind.", thread_id="t1")

    if reply.needs_approval:                    # the graph SUSPENDED
        print(reply.approval_request.summary()) # a human reads the evidence
        reply = agent.approve("t1", approved=True, approver="alice@acme.io")

    print(reply.answer)
    print(agent.history("t1"))                  # who authorised what, and why

WHAT MAKES IT "ENTERPRISE GRADE" — six things the teaching tracks omit
----------------------------------------------------------------------
1. **Human-in-the-loop approval** (`graph.py`). `interrupt()` suspends the graph
   between the model's decision and the side effect. The refund *cannot* happen
   until a named human resumes the thread — that is a control, not a prompt.
2. **Durable state** (`runtime.py`). A checkpointer persists suspended runs, so
   approval can arrive hours later, from a different process, after a deploy.
3. **Guardrails in code** (`guardrails.py`). Input limits, PII redaction,
   grounding and forbidden commitments enforced outside the model, because a
   prompt is a request and a guardrail is a control.
4. **An audit trail** (`audit.py`). Append-only, permanent, attributable. A
   trace says why the agent did something; an audit says who authorised it.
5. **An evaluation gate** (`evaluate.py`). Safety cases separated from
   capability cases, with safety gated at 100% — a safety regression fails the
   build even if the overall number improved.
6. **A service** (`service.py`). Because the approval round trip only becomes
   real when the two halves are separate HTTP requests.

THE ONE DESIGN DECISION EVERYTHING ELSE DEPENDS ON
---------------------------------------------------
`tools.py` splits **deciding** (`check_refund_eligibility`) from **acting**
(`issue_refund`). That split is what creates a place to put the approval gate.
Fuse them into one `process_refund` tool and the money has already moved before
you can interrupt.

**Tool design determines where you can put your controls.** If you take one
thing from this module, take that.

RUNNING IT
----------
Needs `OPENAI_API_KEY` — this is the production track.

    python -m acme_support_agent.cli "Refund ACME-1046, I changed my mind."
    uvicorn acme_support_agent.service:app --reload
    python scripts/check_enterprise.py      # verifies everything, no key needed
"""

from .audit import AuditEvent, AuditLog, AuditRecord
from .evaluate import Case, EvalReport, default_cases, evaluate, run_case
from .graph import SupportState, build_support_graph, needs_human_approval
from .guardrails import check_input, check_output, redact, ungrounded_figures
from .observability import CostTracker, get_logger, request_context, setup_observability
from .runtime import ApprovalRequest, Reply, SupportAgent
from .settings import Settings, get_settings
from .tools import HIGH_RISK_TOOLS, LEDGER, SUPPORT_TOOLS

__version__ = "1.0.0"

__all__ = [
    # the API most callers need
    "SupportAgent", "Reply", "ApprovalRequest",
    # configuration
    "Settings", "get_settings",
    # graph
    "build_support_graph", "SupportState", "needs_human_approval",
    # tools
    "SUPPORT_TOOLS", "HIGH_RISK_TOOLS", "LEDGER",
    # guardrails
    "check_input", "check_output", "redact", "ungrounded_figures",
    # audit
    "AuditLog", "AuditEvent", "AuditRecord",
    # evaluation
    "evaluate", "EvalReport", "Case", "default_cases", "run_case",
    # observability
    "setup_observability", "get_logger", "request_context", "CostTracker",
]

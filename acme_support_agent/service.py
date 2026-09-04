"""
service.py — the HTTP surface. Where "end to end" actually ends.
================================================================

WHY this file exists
--------------------
An agent that only runs in a notebook has not met its two hardest problems:
**concurrency** and **the approval round trip**.

The approval gate is the interesting one. In a notebook you call `chat()`, get
`awaiting_approval`, and call `approve()` on the next line. Over HTTP those are
two independent requests, possibly hours apart, from different people, hitting
different processes behind a load balancer.

That works here for exactly one reason: **the checkpointer**. The suspended run
lives in shared storage keyed by `thread_id`, not in the memory of the process
that started it. Swap `SqliteSaver` for Postgres and this scales horizontally
with no change to any of the code below.

    POST /chat            a customer message  -> answer, or awaiting_approval
    GET  /approvals       the reviewer queue
    POST /approvals/{id}  a human decides     -> resumes the suspended run
    GET  /threads/{id}/audit    the decision trail
    GET  /healthz         liveness + configuration

ONE THING DELIBERATELY LEFT OUT
-------------------------------
Authentication. Every endpoint below is unauthenticated, and in production
`/approvals` in particular must be behind real authz — the `approver` field is
currently *claimed* by the caller, not *verified*, and an audit log recording an
unverified identity is worse than one recording none, because it looks
trustworthy.

The `X-Actor` header below marks exactly where that check belongs. Leaving the
seam visible and labelled beats silently pretending the problem does not exist.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .observability import get_logger, request_context
from .runtime import Reply, SupportAgent
from .settings import get_settings

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Wire types — the API contract, versioned separately from internals
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    thread_id: Optional[str] = Field(default=None, description="Continue an existing conversation.")
    customer_id: str = "anonymous"


class ChatResponse(BaseModel):
    """
    Note `status` is REQUIRED and callers must branch on it.

    A response shape where `answer` is simply null on suspension invites clients
    to render an empty bubble and move on. Making the state explicit forces the
    integrator to handle the approval path.
    """

    status: str
    thread_id: str
    answer: Optional[str] = None
    approval: Optional[Dict[str, Any]] = None
    tools_called: List[str] = []
    steps: int = 0
    flags: List[str] = []


class ApprovalDecision(BaseModel):
    approved: bool
    note: str = Field(default="", max_length=1000)


def to_response(reply: Reply) -> ChatResponse:
    approval = None
    if reply.approval_request:
        request = reply.approval_request
        approval = {
            "thread_id": request.thread_id,
            "tool": request.tool,
            "args": request.args,
            "reason": request.reason,
            "evidence": request.evidence,
            "customer_id": request.customer_id,
        }
    return ChatResponse(
        status=reply.status,
        thread_id=reply.thread_id,
        answer=reply.answer,
        approval=approval,
        tools_called=reply.tools_called,
        steps=reply.steps,
        flags=reply.flags,
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
def create_app(agent: Optional[SupportAgent] = None):
    """
    Build the FastAPI app.

    `agent` is injectable so tests can pass one backed by a test double. A
    service that can only be constructed with real credentials is a service
    nobody writes integration tests for.
    """
    from fastapi import FastAPI, Header, HTTPException

    settings = get_settings()
    app = FastAPI(
        title="Acme Cloud Support Agent",
        version="1.0.0",
        description="Reference agent: LangGraph, human-in-the-loop approval, audit trail.",
    )
    # Built lazily so importing this module never requires an API key.
    state: Dict[str, Any] = {"agent": agent}
    # In-memory index of threads awaiting a decision. Rebuilt from the
    # checkpointer on demand — this is a cache, never the source of truth,
    # because a second process would not share it.
    awaiting: Dict[str, Dict[str, Any]] = {}

    def get_agent() -> SupportAgent:
        if state["agent"] is None:
            state["agent"] = SupportAgent()
        return state["agent"]

    @app.get("/healthz")
    def healthz() -> Dict[str, Any]:
        """Liveness plus the active configuration — the first thing you check."""
        return {
            "status": "ok",
            "environment": settings.environment,
            "config": settings.summary(),
            "awaiting_approval": len(awaiting),
        }

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        thread_id = request.thread_id or f"t-{uuid.uuid4().hex[:10]}"
        with request_context(thread_id, request.customer_id):
            log.info("chat_received", extra={"customer_id": request.customer_id})
            reply = get_agent().chat(
                request.message, thread_id=thread_id, customer_id=request.customer_id
            )
            if reply.needs_approval and reply.approval_request:
                awaiting[thread_id] = {
                    "thread_id": thread_id,
                    "tool": reply.approval_request.tool,
                    "args": reply.approval_request.args,
                    "reason": reply.approval_request.reason,
                    "customer_id": reply.approval_request.customer_id,
                }
                log.info("approval_pending", extra={"tool": reply.approval_request.tool})
            return to_response(reply)

    @app.get("/approvals")
    def approvals() -> Dict[str, Any]:
        """The reviewer's queue."""
        return {"count": len(awaiting), "items": list(awaiting.values())}

    @app.post("/approvals/{thread_id}", response_model=ChatResponse)
    def decide(
        thread_id: str,
        decision: ApprovalDecision,
        x_actor: str = Header(default="", alias="X-Actor"),
    ) -> ChatResponse:
        """
        Resume a suspended run with a human decision.

        `x_actor` is the identity written into the audit log. IT IS NOT VERIFIED
        HERE — in production this endpoint sits behind authentication and the
        actor comes from the validated session, never from a header the caller
        controls. This is the single most important line in the file to change
        before deploying.
        """
        if not x_actor:
            raise HTTPException(400, "X-Actor header is required — approvals must be attributable")

        current = get_agent()
        if current.pending(thread_id) is None:
            raise HTTPException(404, f"thread {thread_id} is not awaiting approval")

        with request_context(thread_id, awaiting.get(thread_id, {}).get("customer_id", "-")):
            reply = current.approve(
                thread_id, approved=decision.approved,
                approver=x_actor, note=decision.note,
            )
            awaiting.pop(thread_id, None)
            return to_response(reply)

    @app.get("/threads/{thread_id}/audit")
    def audit(thread_id: str) -> Dict[str, Any]:
        """The decision trail — what happened and who authorised it."""
        current = get_agent()
        return {
            "thread_id": thread_id,
            "trail": current.history(thread_id),
            "decisions": current.audit.decisions(thread_id),
        }

    return app


# Uvicorn entrypoint: `uvicorn acme_support_agent.service:app --reload`
# Built lazily inside create_app so `import acme_support_agent.service` never
# needs credentials — which is what lets the test suite import it.
def __getattr__(name: str):
    if name == "app":
        return create_app()
    raise AttributeError(name)

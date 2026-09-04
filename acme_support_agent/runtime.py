"""
runtime.py — the object your application actually holds.
========================================================

WHY this file exists
--------------------
A compiled graph is not an API. It needs a `thread_id`, an initial state dict, a
config with a recursion limit, and a caller who knows what an `__interrupt__`
key means. Making every call site handle that is how the interrupt contract gets
implemented four slightly different ways.

`SupportAgent` is the seam: **one object, three methods**, and every caller —
the FastAPI service, the CLI, the evaluation harness, the notebook — goes
through it.

    agent.chat(message, thread_id)   -> Reply (may be awaiting approval)
    agent.approve(thread_id, ...)    -> Reply (resumes the suspended run)
    agent.history(thread_id)         -> the audit trail

THE INTERRUPT CONTRACT
----------------------
`chat()` can return in one of two states, and callers must handle both:

    reply.status == "completed"          -> reply.answer is the answer
    reply.status == "awaiting_approval"  -> reply.approval_request describes
                                            what a human must decide

That second one is not an error and not a timeout. The run is **suspended and
durable** — it is sitting in the checkpointer, and `approve()` resumes it. If
your process restarts in between, it still resumes.

Making that a typed field rather than an exception or a magic string is
deliberate: a caller who forgets to handle `awaiting_approval` should get an
obviously-missing branch, not a silent success.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import HumanMessage

from .audit import AuditEvent, AuditLog
from .graph import build_support_graph
from .observability import get_logger, setup_observability
from .settings import Settings, get_settings

log = get_logger(__name__)


@dataclass
class ApprovalRequest:
    """What a human is being asked to decide, with the evidence to decide it."""

    thread_id: str
    tool: str
    args: Dict[str, Any]
    reason: str
    evidence: str = ""
    customer_id: str = "anonymous"

    def summary(self) -> str:
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return (
            f"APPROVAL REQUIRED — thread {self.thread_id}\n"
            f"  action  : {self.tool}({rendered})\n"
            f"  because : {self.reason}\n"
            f"  evidence:\n    " + (self.evidence or "(none)").replace("\n", "\n    ")
        )


@dataclass
class Reply:
    """
    The result of one turn. Check `status` before reading `answer`.
    """

    status: Literal["completed", "awaiting_approval", "blocked", "error"]
    thread_id: str
    answer: Optional[str] = None
    approval_request: Optional[ApprovalRequest] = None
    tools_called: List[str] = field(default_factory=list)
    steps: int = 0
    flags: List[str] = field(default_factory=list)
    stop_reason: Optional[str] = None
    error: Optional[str] = None

    @property
    def needs_approval(self) -> bool:
        return self.status == "awaiting_approval"

    def __str__(self) -> str:
        if self.needs_approval and self.approval_request:
            return self.approval_request.summary()
        return f"[{self.status}] {self.answer or self.error or ''}"


class SupportAgent:
    """
    The Acme support agent, ready to serve.

        agent = SupportAgent()
        reply = agent.chat("Refund ACME-1046, I changed my mind.", thread_id="t1")
        if reply.needs_approval:
            reply = agent.approve("t1", approved=True, approver="alice@acme.io")
        print(reply.answer)
    """

    def __init__(
        self,
        model=None,
        settings: Optional[Settings] = None,
        checkpointer=None,
        audit: Optional[AuditLog] = None,
        tools: Optional[List] = None,
    ):
        self.settings = settings or get_settings()
        setup_observability(self.settings)
        self.audit = audit or AuditLog(self.settings.audit_log_path)

        # A checkpointer is REQUIRED, not optional. Without one an interrupted
        # run cannot be resumed after the process exits, which makes the
        # approval gate theatre rather than a control.
        self.checkpointer = checkpointer or self._default_checkpointer()
        self.model = model or self._default_model()
        self.graph = build_support_graph(
            self.model, settings=self.settings, audit=self.audit,
            checkpointer=self.checkpointer, tools=tools,
        )
        log.info("support_agent_ready", extra={"config": self.settings.summary()})

    # -- construction helpers ------------------------------------------------
    def _default_checkpointer(self):
        from pathlib import Path
        from langgraph.checkpoint.sqlite import SqliteSaver
        import sqlite3

        Path(self.settings.checkpoint_db).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False because a web server serves requests from a
        # pool. This is the SQLite-specific wart you trade away when you move to
        # Postgres, and it is worth a comment so nobody "fixes" it.
        connection = sqlite3.connect(self.settings.checkpoint_db, check_same_thread=False)
        return SqliteSaver(connection)

    def _default_model(self):
        if not self.settings.has_model_key:
            raise RuntimeError(
                "No OPENAI_API_KEY configured. SupportAgent needs a real model — "
                "this is the production track. Pass model=... explicitly to "
                "inject a test double."
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=self.settings.model,
            temperature=self.settings.temperature,
            timeout=self.settings.request_timeout_s,
            max_retries=self.settings.max_retries,
        )

    def _config(self, thread_id: str) -> dict:
        return {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": self.settings.recursion_limit,
        }

    # -- the API -------------------------------------------------------------
    def chat(
        self,
        message: str,
        thread_id: Optional[str] = None,
        customer_id: str = "anonymous",
    ) -> Reply:
        """
        Handle one customer message.

        Returns a `Reply` that is either completed, blocked by a guardrail, or
        **awaiting approval** — in which case the run is suspended durably and
        `approve()` continues it.
        """
        thread_id = thread_id or f"t-{uuid.uuid4().hex[:10]}"
        state = {
            "messages": [HumanMessage(content=message)],
            "thread_id": thread_id,
            "customer_id": customer_id,
            "steps": 0,
            "guard_flags": [],
            "blocked": False,
        }
        try:
            result = self.graph.invoke(state, self._config(thread_id))
        except Exception as exc:
            log.exception("agent_run_failed", extra={"thread_id": thread_id})
            self.audit.record(
                AuditEvent.RUN_FAILED, thread_id, reason=f"{type(exc).__name__}: {exc}"
            )
            return Reply(
                status="error", thread_id=thread_id,
                error=f"{type(exc).__name__}: {exc}",
                answer=("Something went wrong handling your request. It has been "
                        "logged and a human will follow up."),
            )
        return self._to_reply(result, thread_id)

    def approve(
        self,
        thread_id: str,
        approved: bool,
        approver: str,
        note: str = "",
    ) -> Reply:
        """
        Resume a suspended run with a human's decision.

        `Command(resume=...)` hands the value back to the `interrupt()` call that
        suspended the graph, and execution continues from exactly there. Note
        what this does NOT do: re-run the agent, re-ask the model, or replay the
        tools already executed. The checkpoint holds all of that.
        """
        from langgraph.types import Command

        log.info("approval_submitted", extra={
            "thread_id": thread_id, "approved": approved, "approver": approver,
        })
        try:
            result = self.graph.invoke(
                Command(resume={"approved": approved, "approver": approver, "note": note}),
                self._config(thread_id),
            )
        except Exception as exc:
            log.exception("approval_resume_failed", extra={"thread_id": thread_id})
            return Reply(status="error", thread_id=thread_id, error=str(exc))
        return self._to_reply(result, thread_id)

    def pending(self, thread_id: str) -> Optional[ApprovalRequest]:
        """What this thread is waiting on, if anything. For an approvals queue UI."""
        snapshot = self.graph.get_state(self._config(thread_id))
        for task in getattr(snapshot, "tasks", ()) or ():
            for value in getattr(task, "interrupts", ()) or ():
                payload = getattr(value, "value", value)
                if isinstance(payload, dict) and payload.get("type") == "approval_required":
                    return self._to_request(payload, thread_id)
        return None

    def history(self, thread_id: str) -> str:
        """The audit trail — what happened and who authorised it."""
        return self.audit.trail(thread_id)

    # -- internals -----------------------------------------------------------
    @staticmethod
    def _to_request(payload: dict, thread_id: str) -> ApprovalRequest:
        return ApprovalRequest(
            thread_id=payload.get("thread_id", thread_id),
            tool=payload.get("tool", "?"),
            args=payload.get("args", {}),
            reason=payload.get("reason", ""),
            evidence=payload.get("evidence", ""),
            customer_id=payload.get("customer_id", "anonymous"),
        )

    def _to_reply(self, result: dict, thread_id: str) -> Reply:
        # An interrupted run surfaces under the "__interrupt__" key rather than
        # completing. Handling it explicitly — instead of letting a missing
        # "answer" look like an empty response — is the contract this class exists
        # to enforce.
        interrupts = result.get("__interrupt__") if isinstance(result, dict) else None
        if interrupts:
            payload = getattr(interrupts[0], "value", interrupts[0])
            return Reply(
                status="awaiting_approval",
                thread_id=thread_id,
                approval_request=self._to_request(payload or {}, thread_id),
                tools_called=_calls(result),
                steps=result.get("steps", 0),
                flags=result.get("guard_flags", []),
            )

        blocked = bool(result.get("blocked")) or "output_blocked" in (
            result.get("guard_flags") or []
        )
        return Reply(
            status="blocked" if blocked else "completed",
            thread_id=thread_id,
            answer=result.get("answer"),
            tools_called=_calls(result),
            steps=result.get("steps", 0),
            flags=result.get("guard_flags", []),
            stop_reason=result.get("stop_reason"),
        )


def _calls(result: dict) -> List[str]:
    return [
        call["name"]
        for message in (result.get("messages") or [])
        if getattr(message, "type", "") == "ai"
        for call in (getattr(message, "tool_calls", None) or [])
    ]

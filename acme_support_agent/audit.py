"""
audit.py — an append-only record of what the agent did, and who let it.
=======================================================================

WHY this file exists
--------------------
A trace answers *"why did the agent do that?"* — it is a debugging tool, and it
is fine for it to be sampled, truncated, or expired after thirty days.

An **audit log** answers a different question: *"on what basis was this refund
issued, and who authorised it?"* That one gets asked by a customer, a manager,
a finance reconciliation, or a regulator — sometimes eighteen months later.
Different question, different retention, different guarantees.

Teams routinely conflate the two, ship only tracing, and discover the gap during
their first dispute. The distinction in one line:

    trace  -> for engineers, about behaviour, disposable
    audit  -> for the business, about decisions, permanent

WHAT MAKES THIS AN AUDIT LOG RATHER THAN LOGGING
------------------------------------------------
* **Append-only.** No update, no delete. The API does not offer them.
* **Every consequential decision**, not a sample. Sampling an audit log defeats
  its purpose.
* **The reason is recorded, not just the outcome** — which policy rule fired,
  what evidence supported it, who approved.
* **Identity.** Which actor: the agent, or a named human.
* **Written before the side effect**, so a crash mid-operation leaves evidence
  that it was attempted. An audit record written afterwards is a record of the
  successes only.

JSONL on disk is the reference implementation. In production this is an
append-only table, an event stream, or object storage with retention locked —
the shape of the record matters more than the medium.
"""
from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


class AuditEvent(str, Enum):
    """The decisions worth remembering permanently."""

    REQUEST_RECEIVED = "request_received"
    INPUT_REJECTED = "input_rejected"          # a guardrail refused it
    TOOL_INVOKED = "tool_invoked"
    APPROVAL_REQUIRED = "approval_required"    # the agent paused for a human
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    REFUND_DECISION = "refund_decision"        # the consequential one
    ESCALATED = "escalated"
    ANSWER_RETURNED = "answer_returned"
    GROUNDING_FAILED = "grounding_failed"      # we blocked our own answer
    RUN_FAILED = "run_failed"


@dataclass(frozen=True)
class AuditRecord:
    """
    One immutable entry. Frozen on purpose — an audit record you can mutate is
    not an audit record.
    """

    event: str
    thread_id: str
    actor: str                    # "agent" | "human:<id>" | "system"
    detail: Dict[str, Any] = field(default_factory=dict)
    reason: Optional[str] = None
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)


class AuditLog:
    """
    Append-only JSONL audit log.

    Thread-safe because a service handles concurrent requests and an interleaved
    write would corrupt a line — a failure mode that only appears under load,
    which is the worst time to find it.
    """

    def __init__(self, path: str = "var/audit.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._seen: set = set()
        # Rebuild the dedupe set from disk so restarts do not reintroduce
        # duplicates the process previously suppressed.
        for row in self.read():
            key = (row.get("detail") or {}).get("_dedupe_key")
            if key:
                self._seen.add((row["thread_id"], key))

    def record(
        self,
        event: AuditEvent,
        thread_id: str,
        actor: str = "agent",
        reason: Optional[str] = None,
        dedupe_key: Optional[str] = None,
        **detail: Any,
    ) -> Optional[AuditRecord]:
        """
        Write one record. Returns it, or None if it was a suppressed duplicate.

        Note there is no `update` or `delete` anywhere in this class. That
        omission is the feature.

        ### Why `dedupe_key` exists — a real LangGraph gotcha

        **Code before `interrupt()` in a node runs AGAIN when the graph
        resumes.** LangGraph replays the node from its start up to the interrupt
        point, because it has no way to know which statements already ran.

        So an `audit.record(...)` sitting above an `interrupt()` writes twice:
        once when the approval is requested, once when it is granted. Duplicate
        entries in an append-only decision log are not cosmetic — they inflate
        counts and make a reviewer doubt the whole record.

        Three ways to handle it, in order of preference:
          1. put side effects AFTER the interrupt (not always possible — here
             the whole point is to record that we *asked*, before we know the
             answer)
          2. make the side effect idempotent  ← this parameter
          3. move the side effect into its own node before the interrupting one

        Anything you do before an `interrupt()` must be safe to do twice.
        """
        if dedupe_key is not None:
            with self._lock:
                if (thread_id, dedupe_key) in self._seen:
                    return None
                self._seen.add((thread_id, dedupe_key))
            detail = {**detail, "_dedupe_key": dedupe_key}

        entry = AuditRecord(
            event=event.value,
            thread_id=thread_id,
            actor=actor,
            reason=reason,
            detail=detail,
        )
        line = entry.to_json()
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
                # flush + fsync: an audit record still sitting in a buffer when
                # the process dies is an audit record that does not exist. This
                # costs latency, and for consequential decisions it is worth it.
                handle.flush()
                os.fsync(handle.fileno())
        return entry

    # -- reading -------------------------------------------------------------
    def read(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """All records, or just one thread's. Reading is a separate concern
        from writing, and deliberately not optimised — audit reads are rare,
        investigative, and can afford a full scan."""
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # never let one bad line hide the rest
                if thread_id is None or row.get("thread_id") == thread_id:
                    rows.append(row)
        return rows

    def trail(self, thread_id: str) -> str:
        """
        Human-readable decision trail for one conversation.

        This is what you paste into a ticket when a customer asks "why was my
        refund declined?" — and being able to answer that in ten seconds is the
        whole return on building this.
        """
        rows = self.read(thread_id)
        if not rows:
            return f"No audit records for thread {thread_id}."
        lines = [f"AUDIT TRAIL — thread {thread_id}", "=" * 66]
        for row in rows:
            stamp = row["timestamp"][11:19]
            lines.append(f"{stamp}  {row['event']:<20} by {row['actor']}")
            if row.get("reason"):
                lines.append(f"          reason: {row['reason']}")
            for key, value in (row.get("detail") or {}).items():
                if key.startswith("_"):
                    continue          # internal bookkeeping, not for a reviewer
                rendered = str(value)
                if len(rendered) > 88:
                    rendered = rendered[:87] + "…"
                lines.append(f"          {key}: {rendered}")
        return "\n".join(lines)

    def decisions(self, thread_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Only the consequential events — what a reviewer actually wants."""
        consequential = {
            AuditEvent.REFUND_DECISION.value,
            AuditEvent.APPROVAL_GRANTED.value,
            AuditEvent.APPROVAL_DENIED.value,
            AuditEvent.ESCALATED.value,
            AuditEvent.INPUT_REJECTED.value,
            AuditEvent.GROUNDING_FAILED.value,
        }
        return [r for r in self.read(thread_id) if r["event"] in consequential]

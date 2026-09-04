"""
graph.py — the production graph, with a human in the loop.
==========================================================

WHY this file exists
--------------------
`agent_lc/graph.py` is the teaching graph: agent, tools, a conditional edge. It
is correct and it is not deployable, because it will happily do anything its
tools allow, immediately, with nobody watching.

This is the deployable version. The shape:

    START
      → guard_input      block or flag before the model sees anything
      → agent  ⇄  tools  the loop from notebook 02
          ↘ approval_gate   ← sits between DECISION and SIDE EFFECT
              ↘ (interrupt)  a human decides; the graph SUSPENDS
      → guard_output     grounding + forbidden commitments
      → finalize         audit, structured result
      → END

THE ONE THING THAT MAKES THIS ENTERPRISE-GRADE
-----------------------------------------------
`interrupt()`.

When the agent wants to issue a refund that policy says a human must approve,
the graph **stops**. Not "asks the model to wait" — the process returns, the
state is persisted to the checkpointer, and the run can be resumed hours later,
from a different process, after a deploy, by a named human.

That is not something you can approximate with a prompt. "Ask the user before
doing anything dangerous" is a request to a probabilistic system. `interrupt()`
is a control: the side effect *cannot* occur, because the code that performs it
has not been reached and will not be until someone resumes the thread.

This is the single strongest argument for adopting LangGraph rather than
hand-rolling, and it is worth being precise about why: durable suspension needs
checkpointing, and checkpointing is exactly the infrastructure you should not
be writing yourself.

WHERE THE GATE SITS, AND WHY IT MATTERS
----------------------------------------
The gate is between the model *deciding* to call `issue_refund` and the tool
*running*. Not before the agent starts (we do not know yet what it will want to
do) and not after the tool (the money has moved).

That placement is only possible because `tools.py` split "decide" from "act"
into two tools. **Tool design determines where you can put your controls.**
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt

from .audit import AuditEvent, AuditLog
from .guardrails import check_input, check_output, redact
from .settings import Settings, get_settings
from .tools import HIGH_RISK_TOOLS, SUPPORT_TOOLS, order_plan, refund_amount_for

SYSTEM_PROMPT = """\
You are Acme Cloud's customer support agent. You work by calling tools.

Rules you must follow:
1. Gather evidence before answering. If a tool can confirm something, call it —
   never answer from memory or assumption.
2. ALWAYS call get_order_status before assessing or issuing anything for an order.
3. NEVER call issue_refund unless check_refund_eligibility has returned ELIGIBLE
   for that same order and reason. Issuing without checking is a policy violation.
4. The refund amount must exactly match the amount the order was charged.
5. If a tool says NOT FOUND or NOT ELIGIBLE, that is the answer. Report it.
   Never invent a plausible substitute and never argue with the policy.
6. Never promise a specific refund date. The only timeframe you may quote is
   "5 to 7 business days".
7. If the policy requires human judgement, escalate with a summary of what you
   checked. Escalating is a correct outcome.
8. Answer only from what the tools returned, and say so plainly when you do not
   have enough to answer.
"""


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class SupportState(TypedDict, total=False):
    """
    The graph's state. Everything a resumed run needs to continue correctly.

    Note how much more than `messages` is here compared with the teaching graph.
    Each field exists because some node downstream, or some human reading an
    audit trail afterwards, needs it — and because after an `interrupt()` this
    dict is *all that survives*. State design is what makes a run resumable.
    """

    messages: Annotated[List[AnyMessage], add_messages]
    thread_id: str
    customer_id: str
    steps: int
    stop_reason: Optional[str]
    guard_flags: List[str]
    blocked: bool
    # Set by the approval gate; read by the audit and the final result.
    pending_action: Optional[Dict[str, Any]]
    approval: Optional[Dict[str, Any]]
    answer: Optional[str]


# ---------------------------------------------------------------------------
# The approval decision — deliberately NOT in the prompt
# ---------------------------------------------------------------------------
def needs_human_approval(
    tool_name: str, args: Dict[str, Any], settings: Settings
) -> tuple[bool, str]:
    """
    Does this tool call require a human? Returns (needs_approval, why).

    This is plain Python, evaluated outside the model, and that is the whole
    point. The model is not consulted about whether it needs permission — a
    system that asks the actor to decide whether it should be supervised has not
    implemented supervision.

    Ordering is from most-certain to least, so the recorded reason is the most
    specific one that applies.
    """
    if tool_name not in HIGH_RISK_TOOLS:
        return False, ""

    order_id = str(args.get("order_id", ""))
    amount = args.get("amount_usd")

    plan = order_plan(order_id)
    if plan == "Enterprise" and settings.require_approval_for_enterprise:
        return True, f"{order_id} is an Enterprise contract — human decision required by policy"

    if amount is None:
        return True, "refund amount was not specified — cannot size the risk"

    try:
        amount = float(amount)
    except (TypeError, ValueError):
        return True, f"refund amount {amount!r} is not a number"

    charged = refund_amount_for(order_id)
    if charged is not None and abs(amount - charged) > 0.01:
        return True, (
            f"requested ${amount:.2f} does not match the ${charged:.2f} charged — "
            "a mismatch always needs a human"
        )

    if amount > settings.auto_approve_refund_under_usd:
        return True, (
            f"${amount:.2f} is over the ${settings.auto_approve_refund_under_usd:.2f} "
            "auto-approval limit"
        )

    return False, f"${amount:.2f} is within the auto-approval limit"


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------
def build_support_graph(
    model,
    settings: Optional[Settings] = None,
    audit: Optional[AuditLog] = None,
    checkpointer=None,
    tools: Optional[List] = None,
):
    """
    Build the production support agent.

    Args:
        model: a chat model. Tools are bound here.
        settings: validated configuration. Defaults to `get_settings()`.
        audit: the append-only decision log.
        checkpointer: **required for interrupts to be durable.** Without one the
            graph still runs, but an interrupted run cannot be resumed after the
            process exits — which defeats the purpose of having interrupts.
        tools: override the toolset (tests).
    """
    settings = settings or get_settings()
    audit = audit or AuditLog(settings.audit_log_path)
    tools = list(tools if tools is not None else SUPPORT_TOOLS)
    model_with_tools = model.bind_tools(tools)
    tool_node = ToolNode(tools, handle_tool_errors=True)

    # -- NODE: input guardrails ---------------------------------------------
    def guard_input(state: SupportState) -> dict:
        """Runs before the model sees anything. Cheapest place to stop a problem."""
        thread_id = state.get("thread_id", "unknown")
        user_messages = [m for m in state["messages"] if getattr(m, "type", "") == "human"]
        text = str(user_messages[-1].content) if user_messages else ""

        result = check_input(text, max_chars=settings.max_input_chars)
        audit.record(
            AuditEvent.REQUEST_RECEIVED,
            thread_id,
            customer_id=state.get("customer_id", "anonymous"),
            # The audit log is long-lived, so it gets the redacted text. The
            # model gets the original — it needs the real values to work.
            request=redact(text)[:500] if settings.redact_pii_in_logs else text[:500],
            flags=result.flags,
        )

        if result.blocked:
            audit.record(
                AuditEvent.INPUT_REJECTED, thread_id, reason=result.reason
            )
            return {
                "blocked": True,
                "guard_flags": result.flags,
                "stop_reason": f"input rejected: {result.reason}",
                "answer": (
                    "I could not process that request: "
                    f"{result.reason}. Please rephrase and try again."
                ),
            }
        return {"blocked": False, "guard_flags": result.flags}

    # -- NODE: agent (THINK) -------------------------------------------------
    def agent(state: SupportState) -> dict:
        messages = list(state["messages"])
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=SYSTEM_PROMPT), *messages]
        reply = model_with_tools.invoke(messages)
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    # -- NODE: the approval gate --------------------------------------------
    def approval_gate(state: SupportState):
        """
        Sits between the model's decision and the side effect.

        On a call that needs a human this calls `interrupt()`, which **suspends
        the graph**. The invoking process returns; the state is checkpointed.
        Resuming means calling `graph.invoke(Command(resume={...}), config)` with
        the same `thread_id` — possibly hours later, from a different process.

        The tool has not run. It cannot run. That is a control, not a request.
        """
        thread_id = state.get("thread_id", "unknown")
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        risky = [c for c in calls if c["name"] in HIGH_RISK_TOOLS]

        if not risky:
            return Command(goto="tools")

        call = risky[0]
        required, why = needs_human_approval(call["name"], call["args"], settings)

        if not required:
            audit.record(
                AuditEvent.REFUND_DECISION, thread_id, actor="agent",
                reason=f"auto-approved: {why}",
                tool=call["name"], args=call["args"],
            )
            return Command(goto="tools", update={"pending_action": {
                "tool": call["name"], "args": call["args"], "auto_approved": True,
                "reason": why,
            }})

        # --- the interrupt ---------------------------------------------------
        # dedupe_key because everything above `interrupt()` in this node RUNS
        # AGAIN when the graph resumes — LangGraph replays the node up to the
        # interrupt point. Without the key this writes two identical
        # approval_required records for one approval. Anything you do before an
        # interrupt must be safe to do twice.
        audit.record(
            AuditEvent.APPROVAL_REQUIRED, thread_id, actor="agent", reason=why,
            tool=call["name"], args=call["args"],
            dedupe_key=f"approval_required:{call['id']}",
        )

        decision = interrupt({
            "type": "approval_required",
            "thread_id": thread_id,
            "customer_id": state.get("customer_id", "anonymous"),
            "tool": call["name"],
            "args": call["args"],
            "reason": why,
            # Give the reviewer the evidence, not just the request. An approver
            # who has to go and look things up will rubber-stamp instead.
            "evidence": _evidence(state),
        })

        approved = bool(decision.get("approved")) if isinstance(decision, dict) else bool(decision)
        approver = (decision or {}).get("approver", "unknown") if isinstance(decision, dict) else "unknown"
        note = (decision or {}).get("note", "") if isinstance(decision, dict) else ""

        record = {
            "tool": call["name"], "args": call["args"], "approved": approved,
            "approver": approver, "note": note, "reason": why,
        }

        if approved:
            audit.record(
                AuditEvent.APPROVAL_GRANTED, thread_id, actor=f"human:{approver}",
                reason=note or why, tool=call["name"], args=call["args"],
            )
            return Command(goto="tools", update={"approval": record, "pending_action": record})

        audit.record(
            AuditEvent.APPROVAL_DENIED, thread_id, actor=f"human:{approver}",
            reason=note or "denied by reviewer", tool=call["name"], args=call["args"],
        )
        # Denial is not an error — it is information the agent must act on. We
        # answer the pending tool call with a ToolMessage so the transcript stays
        # valid and the agent can explain the outcome to the customer.
        denial = ToolMessage(
            content=(
                f"REFUND NOT ISSUED: a human reviewer declined this action. "
                f"Reason: {note or 'not provided'}. Do not retry. Explain to the "
                "customer that the request needs further review."
            ),
            tool_call_id=call["id"],
            name=call["name"],
        )
        return Command(goto="agent", update={"messages": [denial], "approval": record})

    # -- NODE: output guardrails --------------------------------------------
    def guard_output(state: SupportState) -> dict:
        thread_id = state.get("thread_id", "unknown")
        ai = [m for m in state["messages"] if getattr(m, "type", "") == "ai"]
        answer = str(ai[-1].content) if ai else ""
        evidence = _evidence(state)

        result = check_output(answer, evidence, enforce_grounding=settings.enforce_grounding)
        if result.blocked:
            audit.record(
                AuditEvent.GROUNDING_FAILED, thread_id, reason=result.reason,
                answer=redact(answer)[:400],
            )
            # We block our OWN answer. The safe fallback is an honest handoff —
            # never a silently softened version of a claim we could not support.
            return {
                "answer": (
                    "I was not able to give you a reliable answer to that, so I am "
                    "handing this to a human colleague who will follow up. Nothing "
                    "has been changed on your account."
                ),
                "guard_flags": [*state.get("guard_flags", []), "output_blocked"],
                "stop_reason": f"output blocked: {result.reason}",
            }
        return {
            "answer": answer,
            "guard_flags": [*state.get("guard_flags", []), *result.flags],
        }

    # -- NODE: finalize ------------------------------------------------------
    def finalize(state: SupportState) -> dict:
        thread_id = state.get("thread_id", "unknown")
        audit.record(
            AuditEvent.ANSWER_RETURNED, thread_id,
            steps=state.get("steps", 0),
            flags=state.get("guard_flags", []),
            answer=redact(state.get("answer") or "")[:400],
        )
        return {"stop_reason": state.get("stop_reason") or "completed"}

    # -- EDGES ---------------------------------------------------------------
    def after_guard(state: SupportState) -> Literal["agent", "finalize"]:
        return "finalize" if state.get("blocked") else "agent"

    def after_agent(state: SupportState) -> Literal["approval_gate", "tools", "guard_output"]:
        """
        The terminate check, plus the risk routing.

        Diagnostic conditions come BEFORE the budget backstop, for the reason
        notebook 05 argues: the first condition to fire is the one recorded, and
        "max_steps reached" tells you nothing about why.
        """
        if _stalled(state, settings):
            return "guard_output"
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return "guard_output"
        if any(c["name"] in HIGH_RISK_TOOLS for c in calls):
            return "approval_gate"
        return "tools"

    def after_tools(state: SupportState) -> Literal["agent", "guard_output"]:
        for message in reversed(state["messages"]):
            if getattr(message, "type", None) == "tool":
                if getattr(message, "name", None) == "escalate_to_human":
                    return "guard_output"
                break
        return "agent"

    graph = StateGraph(SupportState)
    graph.add_node("guard_input", guard_input)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("guard_output", guard_output)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "guard_input")
    graph.add_conditional_edges("guard_input", after_guard,
                                {"agent": "agent", "finalize": "finalize"})
    graph.add_conditional_edges("agent", after_agent, {
        "approval_gate": "approval_gate", "tools": "tools", "guard_output": "guard_output",
    })
    graph.add_conditional_edges("tools", after_tools,
                                {"agent": "agent", "guard_output": "guard_output"})
    graph.add_edge("guard_output", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _evidence(state: SupportState) -> str:
    """Everything the tools actually returned. The grounding baseline."""
    return "\n".join(
        f"- {getattr(m, 'name', '?')}: {m.content}"
        for m in state.get("messages", [])
        if getattr(m, "type", "") == "tool"
    )


def _stalled(state: SupportState, settings: Settings) -> bool:
    """
    The diagnostic termination conditions, ported from `agent_core.control`.

    LangGraph's `recursion_limit` is a backstop and will not tell you the agent
    repeated itself. These will.
    """
    if state.get("steps", 0) >= settings.max_steps:
        return True

    signatures: List[str] = []
    for message in state.get("messages", []):
        for call in getattr(message, "tool_calls", None) or []:
            try:
                args = json.dumps(call["args"], sort_keys=True, default=str)
            except Exception:
                args = str(call.get("args"))
            signatures.append(f"{call['name']}({args})")

    if len(signatures) >= settings.max_tool_calls:
        return True
    return any(signatures.count(s) >= 3 for s in set(signatures))

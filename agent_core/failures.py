"""
failures.py — a taxonomy of how agents break, and how to see it.
================================================================

WHY this file exists
--------------------
Agent failures are hard to teach for a specific reason: **they are not
reproducible.** A model that looped yesterday may not loop today. So the usual
approach — "here is a bug, now debug it" — collapses, because half the room
cannot reproduce the bug and the other half gets a different one.

The fix is to make failure deterministic. `MockToolCallLLM` accepts a `fault`
parameter, so every learner sees the identical failure, instantly, for free.
This module wraps those faults in a catalogue that names each one, says how to
recognise it in a trace, and says what actually fixes it.

THE TAXONOMY
------------
Five failure modes, chosen because they cover what actually goes wrong and
because each has a *different* fix. Grouping them all under "the agent messed
up" is what keeps people stuck.

  1. NO-PROGRESS LOOP      repeats an identical call forever
  2. HALLUCINATED TOOL     calls a tool that does not exist
  3. SCHEMA VIOLATION      arguments the schema rejects, repeatedly
  4. WRONG TOOL            valid call, irrelevant to the goal
  5. UNGROUNDED ANSWER     confident answer not supported by observations

The ordering matters: 1-3 are *loud* — they show up as errors or repetition and
any monitoring catches them. 4 and 5 are *quiet* — everything looks healthy, the
tool calls succeed, and the answer is wrong. Quiet failures are the dangerous
ones, and #5 is the one that reaches customers.

HOW TO DEBUG AN AGENT — the workflow this session ends on
---------------------------------------------------------
The RAG session closed by localising a bad RAG answer to a stage. The agentic
version localises to a *step*, and the order of questions matters:

  1. Did it call the right tools?     No -> routing / descriptions / scoping
  2. Did the calls succeed?           No -> schemas / arguments / tool bugs
  3. Did it stop for a good reason?   No -> termination policy
  4. Is the answer in the evidence?   No -> grounding / prompt

Work down the list. Do not start at 4 — an ungrounded answer is often a symptom
of a failure at 1, and "fix the prompt" is the most common wasted afternoon in
agent engineering.

`diagnose()` automates this pass over a trace. It is a heuristic, and its real
job is to teach the checklist: run it, read what it found, then confirm the
finding yourself in the trace.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .state import ObservationStatus
from .trace import Trace


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------
@dataclass
class FailureMode:
    """One named way an agent goes wrong, with its signature and its fix."""

    key: str
    name: str
    fault: Optional[str]        # the MockToolCallLLM fault that reproduces it
    loud: bool                  # loud failures announce themselves; quiet ones don't
    symptom: str                # what the user sees
    trace_signature: str        # what to look for in the trace
    root_causes: List[str]
    fixes: List[str]

    def show(self) -> str:
        volume = "LOUD" if self.loud else "QUIET — the dangerous kind"
        lines = [
            f"── {self.name}  [{volume}]",
            f"   symptom : {self.symptom}",
            f"   in trace: {self.trace_signature}",
            "   causes  :",
        ]
        lines += [f"     - {c}" for c in self.root_causes]
        lines.append("   fixes   :")
        lines += [f"     - {f}" for f in self.fixes]
        return "\n".join(lines)


CATALOGUE: Dict[str, FailureMode] = {
    "no_progress_loop": FailureMode(
        key="no_progress_loop",
        name="No-progress loop",
        fault="loop_forever",
        loud=True,
        symptom="The agent runs until its budget is exhausted and returns nothing useful.",
        trace_signature=(
            "The same tool with identical arguments across consecutive steps; "
            "state.repeated_calls() is non-empty; stop_reason is a budget limit."
        ),
        root_causes=[
            "Observations are not being written back into the transcript "
            "(the stateless-loop bug — see loop.py `stateful=False`).",
            "The tool result does not actually answer the question, so the model "
            "tries the same thing again hoping for a different result.",
            "No repetition-based termination condition is configured.",
        ],
        fixes=[
            "Confirm tool results are appended to state.messages every step.",
            "Add control.repetition() to the termination policy — it stops this "
            "in 3 steps instead of 8.",
            "Make the tool's 'not found' response explicit so the model knows "
            "retrying is pointless.",
        ],
    ),
    "hallucinated_tool": FailureMode(
        key="hallucinated_tool",
        name="Hallucinated tool",
        fault="bad_tool_name",
        loud=True,
        symptom="The agent calls a tool that does not exist and gets nowhere.",
        trace_signature="Observations with status REJECTED and 'unknown tool' in the error.",
        root_causes=[
            "The tool the model wants genuinely does not exist — it is inferring "
            "a capability from the task, which is a signal about your toolbox.",
            "Tool names are close enough to be confusable (get_order vs get_orders).",
            "The system prompt describes a capability no tool provides.",
        ],
        fixes=[
            "Return the list of valid tools in the error — ToolRegistry.dispatch "
            "does this, which usually lets the model self-correct next step.",
            "Rename tools to be unambiguous and distinct from one another.",
            "Take the hint and build the tool it keeps reaching for.",
        ],
    ),
    "schema_violation": FailureMode(
        key="schema_violation",
        name="Schema violation",
        fault="malformed_args",
        loud=True,
        symptom="Tool calls are rejected before running; the agent burns steps retrying.",
        trace_signature="Repeated REJECTED observations citing the same parameter.",
        root_causes=[
            "The parameter description does not state the required format.",
            "A free-form string was used where an enum would have constrained "
            "the model to valid values.",
            "The error message is too vague for the model to act on.",
        ],
        fixes=[
            "Put the format IN the parameter description, with an example.",
            "Replace free strings with Literal[...] wherever the set is finite.",
            "Write validation errors for the model to read, not for a log file.",
        ],
    ),
    "wrong_tool": FailureMode(
        key="wrong_tool",
        name="Wrong tool selected",
        fault="wrong_tool",
        loud=False,
        symptom="The agent answers confidently, having consulted the wrong source.",
        trace_signature=(
            "All observations OK, but the call sequence does not match what the "
            "goal needed. Nothing looks broken — you have to know what SHOULD "
            "have been called."
        ),
        root_causes=[
            "Too many tools in scope; tool choice degrades as the list grows.",
            "Two tool descriptions overlap and do not say when to prefer each.",
            "The router picked the wrong skill, so the right tool was never offered.",
        ],
        fixes=[
            "Scope tools per skill — the single most effective fix available.",
            "Rewrite descriptions to say WHEN to use each, not just what it does.",
            "Check Router.explain() before blaming the loop.",
        ],
    ),
    "ungrounded_answer": FailureMode(
        key="ungrounded_answer",
        name="Ungrounded answer",
        fault="ignore_observations",
        loud=False,
        symptom="A fluent, specific, confident answer containing facts no tool returned.",
        trace_signature=(
            "The answer contains numbers, dates or names absent from every "
            "observation. Compare answer against state.evidence() — this is the "
            "one check that catches it."
        ),
        root_causes=[
            "The prompt does not require grounding.",
            "Tool results were empty or unhelpful and the model filled the gap.",
            "The model was asked to be helpful without being allowed to say 'I don't know'.",
        ],
        fixes=[
            "Instruct explicitly: answer only from tool results; say so when they "
            "are insufficient (BASE_INSTRUCTIONS rules 4 and 5).",
            "Turn on reflection for high-stakes answers — control.reflect() checks "
            "the draft against the evidence.",
            "Give the agent an honourable exit: escalate_to_human is what it "
            "should reach for instead of guessing.",
        ],
    ),
}


def show_catalogue() -> str:
    """The whole taxonomy — printed in notebook 06 and in the handout."""
    loud = [f for f in CATALOGUE.values() if f.loud]
    quiet = [f for f in CATALOGUE.values() if not f.loud]
    blocks = ["AGENT FAILURE MODES", "=" * 72, "", "LOUD — the trace shows an error:", ""]
    blocks += [f.show() + "\n" for f in loud]
    blocks += ["QUIET — everything looks fine and the answer is wrong:", ""]
    blocks += [f.show() + "\n" for f in quiet]
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Reproducing a failure
# ---------------------------------------------------------------------------
def broken_agent(key: str, **agent_kwargs):
    """
    An Agent guaranteed to exhibit failure mode `key`.

    Deterministic, offline, free. This is what makes notebook 06 a genuine
    debugging exercise rather than a lecture about debugging — everyone in the
    room reproduces the identical failure and can compare their diagnoses.
    """
    from .agent import Agent
    from .config import MockToolCallLLM

    mode = CATALOGUE.get(key)
    if mode is None:
        raise KeyError(f"unknown failure mode {key!r}. Known: {list(CATALOGUE)}")
    if mode.fault is None:
        raise ValueError(f"{key!r} has no injectable fault")

    return Agent(llm=MockToolCallLLM(fault=mode.fault), **agent_kwargs)


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------
@dataclass
class Finding:
    """One thing the triage pass noticed, with what to do about it."""

    mode: str
    confidence: str            # "high" | "medium"
    evidence: str
    suggested_fix: str

    def show(self) -> str:
        failure = CATALOGUE.get(self.mode)
        title = failure.name if failure else self.mode
        return (
            f"[{self.confidence.upper():<6}] {title}\n"
            f"          evidence: {self.evidence}\n"
            f"          try     : {self.suggested_fix}"
        )


def diagnose(trace: Trace, expected_tools: Optional[List[str]] = None) -> List[Finding]:
    """
    Walk the four debugging questions over a trace and report what fires.

    Args:
        trace: the run to examine.
        expected_tools: if you know which tools the goal needed, pass them — it
            is the only way to detect the quiet "wrong tool" failure
            automatically. Without it, a wrong-but-successful call is
            indistinguishable from a right one, which is exactly why quiet
            failures survive to production.

    This is a heuristic and says so. Its purpose is to teach the checklist; the
    trace remains the evidence, and a finding you cannot confirm by reading the
    trace yourself is a finding you should not act on.
    """
    findings: List[Finding] = []
    observations = trace.observations
    sequence = trace.call_sequence()

    # -- Q1: did it call anything at all? ----------------------------------
    if not observations:
        findings.append(
            Finding(
                mode="ungrounded_answer",
                confidence="high",
                evidence="The agent called no tools at all and still produced an answer.",
                suggested_fix=(
                    "Check the system prompt requires evidence-gathering, and that "
                    "the skill actually has tools scoped to it."
                ),
            )
        )

    # -- Q1b: did it call the right ones? ----------------------------------
    if expected_tools:
        missing = [t for t in expected_tools if t not in sequence]
        if missing:
            findings.append(
                Finding(
                    mode="wrong_tool",
                    confidence="high",
                    evidence=f"Expected {expected_tools}, but the run called {sequence or '(none)'}. Missing: {missing}.",
                    suggested_fix=(
                        "Check Router.explain() for the routing decision, then the "
                        "tool descriptions for whether they say WHEN to use each."
                    ),
                )
            )

    # -- Q2: did the calls succeed? ----------------------------------------
    rejected = [o for o in observations if o.status is ObservationStatus.REJECTED]
    unknown_tool = [o for o in rejected if "unknown tool" in (o.error or "")]
    if unknown_tool:
        findings.append(
            Finding(
                mode="hallucinated_tool",
                confidence="high",
                evidence=f"Called non-existent tool(s): {sorted({o.tool for o in unknown_tool})}.",
                suggested_fix=(
                    "Either build the tool it is reaching for, or make the existing "
                    "tool names less confusable."
                ),
            )
        )
    schema_rejects = [o for o in rejected if o not in unknown_tool]
    if len(schema_rejects) >= 2:
        findings.append(
            Finding(
                mode="schema_violation",
                confidence="high",
                evidence=f"{len(schema_rejects)} calls rejected by validation. First: {schema_rejects[0].error}",
                suggested_fix=(
                    "Put the required format in the parameter description with an "
                    "example, and use Literal[...] where the value set is finite."
                ),
            )
        )

    # -- Q3: did it stop for a good reason? --------------------------------
    signatures = [o.signature() for o in observations]
    repeats = {s for s in signatures if signatures.count(s) >= 2}
    if repeats:
        findings.append(
            Finding(
                mode="no_progress_loop",
                confidence="high" if len(observations) > 3 else "medium",
                evidence=f"Identical call repeated: {sorted(repeats)[0]}",
                suggested_fix=(
                    "Verify observations are written back into state.messages, and "
                    "add control.repetition() to the termination policy."
                ),
            )
        )

    if trace.status == "exhausted":
        findings.append(
            Finding(
                mode="no_progress_loop",
                confidence="medium",
                evidence=f"Run ended on a limit rather than an answer: {trace.stop_reason}",
                suggested_fix=(
                    "A budget stop is a backstop, not a plan. Find the diagnostic "
                    "condition that should have fired earlier and add it."
                ),
            )
        )

    # -- Q4: is the answer supported by the evidence? ----------------------
    answer = (trace.final_answer or "").lower()
    if answer and observations:
        import re

        evidence_text = " ".join(str(o.result or "").lower() for o in observations if o.ok)
        # Numbers are the cheapest grounding check there is: a figure in the
        # answer that appears nowhere in any observation came from the model,
        # not from the world. Crude, and it catches real hallucinations.
        answer_numbers = set(re.findall(r"\d[\d,]*\.?\d*", answer))
        unsupported = {
            n for n in answer_numbers if n not in evidence_text and len(n) > 2
        }
        if unsupported:
            findings.append(
                Finding(
                    mode="ungrounded_answer",
                    confidence="medium",
                    evidence=(
                        f"The answer contains figures absent from every observation: "
                        f"{sorted(unsupported)}"
                    ),
                    suggested_fix=(
                        "Compare the answer against state.evidence() by hand. If it "
                        "is invented, tighten the grounding rule or enable reflection."
                    ),
                )
            )

    return findings


def report(trace: Trace, expected_tools: Optional[List[str]] = None) -> str:
    """`diagnose()` rendered for a notebook — findings, or an explicit all-clear."""
    findings = diagnose(trace, expected_tools)
    header = [
        "DIAGNOSIS",
        "=" * 72,
        f"goal   : {trace.goal}",
        f"status : {trace.status}  ({trace.stop_reason})",
        f"calls  : {'→'.join(trace.call_sequence()) or '(none)'}",
        "─" * 72,
    ]
    if not findings:
        header.append(
            "No automated finding. That is NOT a clean bill of health — quiet "
            "failures are invisible to heuristics. Read the trace and check the "
            "answer against the evidence yourself."
        )
    else:
        header += [f.show() for f in findings]
    return "\n".join(header)

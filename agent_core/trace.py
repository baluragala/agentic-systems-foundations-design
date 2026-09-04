"""
trace.py — the only reason you can debug any of this.
=====================================================

WHY this file exists
--------------------
A traditional pipeline fails in one place, and a stack trace points at it. An
agent fails *across time*: it chose a reasonable tool at step 2, got a slightly
wrong result, believed it at step 3, and produced a confidently incorrect answer
at step 5. Nothing threw. There is no line number to look at.

So the question "why did the agent do that?" is not answerable by reading the
code. It is only answerable by reading a record of what actually happened. That
record is the trace, and if you did not write it down as the run happened, the
information is gone — the run is not reproducible, because the model may decide
differently next time.

This is why tracing is not an optional nice-to-have you bolt on when things get
hard. **It is the debugging interface for the entire paradigm.** An untraced
agent in production is a system whose failures you can only apologise for.

WHAT a trace records
--------------------
One `StepRecord` per turn of the loop, each holding:
  * what the model was asked (context size going in)
  * what it decided (tool calls, or a final answer)
  * what came back (observations, including failures)
  * how long it took and roughly what it cost in context

Plus run-level metadata: the goal, the outcome, why it stopped.

DESIGN NOTE — why the trace is a separate object from AgentState
----------------------------------------------------------------
They overlap, and merging them is tempting. Keeping them apart buys two things:

  * `AgentState` is what the agent reasons over. `Trace` is what a human reads
    afterwards. Different audiences, different formats, different lifetimes —
    you throw state away when the run ends and keep the trace.
  * A trace can be serialised, diffed, and replayed. `to_dict()` output from a
    failing run pasted into a notebook is a complete bug report, and
    `Trace.from_dict()` renders it without needing the agent, the tools, or a
    key. Notebook 06 hands learners exactly that: five recorded failures, and
    the diagnosis has to come from the trace alone.

The recurring question for this session — *"what does this step look like when
it goes wrong, and where would you see it in the trace?"* — is answered HERE.
Every other module has been building toward the moment a learner reads
`trace.show()` and can point at the step where it went wrong.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .state import AgentState, Observation, ObservationStatus


# ---------------------------------------------------------------------------
# One turn of the loop
# ---------------------------------------------------------------------------
@dataclass
class StepRecord:
    """
    What happened in a single THINK -> ACT -> OBSERVE cycle.

    `thought` holds whatever text the model produced alongside its tool calls.
    Real models often emit reasoning there and it is frequently the most
    informative field in the whole trace — it is the difference between "it
    called the wrong tool" and "it called the wrong tool *because* it
    misread the order ID as a customer ID".
    """

    step: int
    decision: str                       # rendered Decision
    thought: Optional[str] = None       # model's text alongside its calls
    observations: List[Observation] = field(default_factory=list)
    context_tokens: int = 0
    duration_ms: float = 0.0

    @property
    def had_error(self) -> bool:
        return any(not o.ok for o in self.observations)

    def show(self, indent: str = "") -> str:
        icon = "✗" if self.had_error else "•"
        lines = [
            f"{indent}{icon} STEP {self.step}"
            f"   ({self.duration_ms:.0f} ms, ~{self.context_tokens} tok in)"
        ]
        if self.thought:
            text = " ".join(self.thought.split())
            lines.append(f"{indent}    think: {text[:150]}")
        lines.append(f"{indent}    act  : {self.decision}")
        for obs in self.observations:
            status = "ok " if obs.ok else obs.status.value
            lines.append(f"{indent}    obs  : [{status}] {obs.tool} -> {obs.summary(80)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------
@dataclass
class Trace:
    """
    The complete record of one agent run.

    Construct with the goal, append `StepRecord`s as the loop turns, call
    `finish()` when it stops. `show()` is what gets printed in every notebook
    from 02 onward; by notebook 06 learners are reading it as evidence.
    """

    goal: str
    steps: List[StepRecord] = field(default_factory=list)
    status: str = "running"
    stop_reason: Optional[str] = None
    final_answer: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    # Printing each step as it happens is worth a lot in a live session — the
    # room watches the loop turn rather than waiting for a wall of text.
    verbose: bool = False

    # -- recording ---------------------------------------------------------
    def add(self, record: StepRecord) -> StepRecord:
        self.steps.append(record)
        if self.verbose:
            print(record.show())
        return record

    def finish(self, state: AgentState) -> "Trace":
        self.status = state.status.value
        self.stop_reason = state.stop_reason
        self.final_answer = state.final_answer
        self.ended_at = time.time()
        return self

    # -- summary statistics ------------------------------------------------
    @property
    def duration_s(self) -> float:
        return (self.ended_at or time.time()) - self.started_at

    @property
    def observations(self) -> List[Observation]:
        return [o for s in self.steps for o in s.observations]

    def tool_calls(self) -> int:
        return len(self.observations)

    def error_rate(self) -> float:
        """
        Fraction of tool calls that did not succeed.

        A useful single number to watch across a task suite. A rising error rate
        after a prompt or schema change is the clearest signal that the change
        made things worse, and it is measurable without a human reading answers.
        """
        total = self.tool_calls()
        if not total:
            return 0.0
        return sum(1 for o in self.observations if not o.ok) / total

    def tools_used(self) -> List[str]:
        seen: List[str] = []
        for obs in self.observations:
            if obs.tool not in seen:
                seen.append(obs.tool)
        return seen

    def call_sequence(self) -> List[str]:
        """
        Just the tool names, in order.

        This is the trace reduced to its skeleton, and it is what you compare
        against an expected sequence when testing an agent. The task suite in
        `data/tasks/agent_tasks.jsonl` carries an `expects_tools` field for
        precisely this comparison.
        """
        return [o.tool for o in self.observations]

    # -- rendering ---------------------------------------------------------
    def show(self) -> str:
        header = [
            "═" * 72,
            f"TRACE  goal: {self.goal}",
            f"       {len(self.steps)} steps · {self.tool_calls()} tool calls · "
            f"{self.duration_s:.2f}s · error rate {self.error_rate():.0%}",
            "═" * 72,
        ]
        body = [s.show() for s in self.steps]
        footer = [
            "─" * 72,
            f"STATUS : {self.status}"
            + (f"  ({self.stop_reason})" if self.stop_reason else ""),
            f"ANSWER : {self.final_answer}",
            "═" * 72,
        ]
        return "\n".join(header + body + footer)

    def summary(self) -> str:
        """One line — for comparing many runs at once."""
        return (
            f"{self.status:<10} steps={len(self.steps):<2} calls={self.tool_calls():<2} "
            f"err={self.error_rate():.0%}  tools={'→'.join(self.call_sequence()) or '(none)'}"
        )

    # -- serialisation -----------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        A JSON-safe view of the whole run.

        This is the bug-report format: everything a colleague needs to see what
        your agent did, with no code, no key, and no ability to reproduce the
        model's decisions required.
        """
        return {
            "goal": self.goal,
            "status": self.status,
            "stop_reason": self.stop_reason,
            "final_answer": self.final_answer,
            "duration_s": round(self.duration_s, 3),
            "steps": [
                {
                    "step": s.step,
                    "decision": s.decision,
                    "thought": s.thought,
                    "context_tokens": s.context_tokens,
                    "duration_ms": round(s.duration_ms, 1),
                    "observations": [
                        {
                            **asdict(o),
                            "status": o.status.value,
                            "result": None if o.result is None else str(o.result),
                        }
                        for o in s.observations
                    ],
                }
                for s in self.steps
            ],
        }

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> str:
        text = json.dumps(self.to_dict(), indent=indent, default=str)
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
        return text

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Trace":
        """
        Rebuild a Trace from `to_dict()` output — the replay path.

        Notebook 06 ships recorded failing runs as JSON and asks learners to
        diagnose them. Replay means the exercise is identical for everyone, runs
        instantly, and costs nothing — none of which is true if each learner has
        to reproduce a stochastic failure themselves.
        """
        trace = cls(goal=data["goal"])
        trace.status = data.get("status", "unknown")
        trace.stop_reason = data.get("stop_reason")
        trace.final_answer = data.get("final_answer")
        trace.ended_at = trace.started_at + float(data.get("duration_s", 0.0))
        for raw_step in data.get("steps", []):
            record = StepRecord(
                step=raw_step["step"],
                decision=raw_step.get("decision", ""),
                thought=raw_step.get("thought"),
                context_tokens=raw_step.get("context_tokens", 0),
                duration_ms=raw_step.get("duration_ms", 0.0),
            )
            for raw_obs in raw_step.get("observations", []):
                record.observations.append(
                    Observation(
                        step=raw_obs["step"],
                        tool=raw_obs["tool"],
                        args=raw_obs.get("args", {}),
                        status=ObservationStatus(raw_obs["status"]),
                        result=raw_obs.get("result"),
                        error=raw_obs.get("error"),
                        duration_ms=raw_obs.get("duration_ms", 0.0),
                    )
                )
            trace.steps.append(record)
        return trace

    @classmethod
    def load(cls, path: str) -> "Trace":
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


# ---------------------------------------------------------------------------
# Comparing runs
# ---------------------------------------------------------------------------
def compare(traces: Dict[str, Trace]) -> str:
    """
    Render several labelled runs as one table.

    The C8 package made its arguments by comparing outputs — chunk sizes,
    retrieval strategies — rather than by assertion. This is the agentic
    equivalent: run the same goal with and without a budget, with a loose and a
    strict schema, with one skill and with three, and put the traces side by
    side. "Hybrid beat dense" was C8's evidence-based moment; "the budgeted run
    stopped at 4 steps and the unbudgeted one hit 25" is this session's.
    """
    width = max((len(label) for label in traces), default=8)
    lines = [f"{'run'.ljust(width)}  {'status':<10} {'steps':>5} {'calls':>5} {'err':>5}  tools"]
    lines.append("─" * (width + 40))
    for label, trace in traces.items():
        lines.append(
            f"{label.ljust(width)}  {trace.status:<10} {len(trace.steps):>5} "
            f"{trace.tool_calls():>5} {trace.error_rate():>4.0%}  "
            f"{'→'.join(trace.call_sequence()) or '(none)'}"
        )
    return "\n".join(lines)

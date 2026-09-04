"""
state.py — the agent's memory. THE core of agent intelligence.
==============================================================

WHY this file exists
--------------------
Ask someone "what makes a system agentic?" and they will usually say "it uses
tools" or "it reasons". Both are wrong on their own. A calculator uses a tool.
A single LLM call reasons. Neither is an agent.

The thing that makes an agent an agent is that **it carries state across
iterations**. Step 4 can only be smarter than step 1 if something survived
steps 1-3. That something is this file.

The agenda for this session says state management is "the core of agent
intelligence", and it means it literally: every other capability in this package
(tools, skills, control, tracing) is a way of *writing to* or *reading from*
AgentState. Delete this module and you do not have a degraded agent — you have a
chatbot in a for-loop.

WHAT state actually holds
-------------------------
Four things, and it is worth being pedantic about the difference:

  1. **goal**         — what we were asked. Immutable. The agent may not edit it.
  2. **messages**     — the conversation so far, including tool calls and their
                        results. This is what gets sent to the LLM each step.
  3. **observations** — a structured record of what tools returned. Redundant
                        with `messages` by design: `messages` is for the LLM,
                        `observations` is for *us* (control logic, termination
                        checks, the trace, and the learner reading the loop).
  4. **budget**       — how much more the agent is allowed to do. Not optional.

The split between (2) and (3) is the single most useful teaching point in this
file. Beginners collapse them into one list and then cannot answer questions
like "has this agent made progress?" without re-parsing prose. Keep the
machine-readable record separate from the model-readable transcript.

WHAT state does NOT hold
------------------------
Tool implementations, prompts, or the LLM. State is *data*. If you find yourself
wanting to call an LLM from inside this module, the design has gone wrong — the
loop calls the LLM and writes the result into state, never the other way round.

HOW to read this file
---------------------
`AgentState` is a dataclass with a handful of small, obvious methods. That is
deliberate. In notebook 02 a learner should be able to hold the entire state
object in their head while tracing the loop by hand.

The recurring question for this session — *"what does this step look like when
it goes wrong, and where would you see it in the trace?"* — has a specific
answer for state:

  * State that never grows      -> the agent repeats itself forever (no memory
                                   of what it already tried). See
                                   `repeated_calls()`.
  * State that grows unbounded  -> context overflow and rising cost per step.
                                   See `Budget` and `approx_tokens()`.
  * State that grows *wrongly*  -> the agent believes a tool succeeded when it
                                   failed. See `ObservationStatus.ERROR` and
                                   why we record failures instead of dropping
                                   them.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Small vocabulary types. Enums rather than bare strings so that a typo is a
# crash at the point of the mistake, not a silent "status never matches" bug
# three layers away.
# ---------------------------------------------------------------------------
class AgentStatus(str, Enum):
    """Where the agent is in its lifecycle."""

    RUNNING = "running"          # the loop is still turning
    DONE = "done"                # produced a final answer
    ESCALATED = "escalated"      # deliberately handed off to a human
    FAILED = "failed"            # hit an unrecoverable error
    EXHAUSTED = "exhausted"      # ran out of budget before finishing

    @property
    def is_terminal(self) -> bool:
        return self is not AgentStatus.RUNNING


class ObservationStatus(str, Enum):
    """How a tool call turned out."""

    OK = "ok"
    ERROR = "error"              # the tool ran and failed (or was called wrong)
    REJECTED = "rejected"        # never ran: schema validation refused it


# ---------------------------------------------------------------------------
# Budget — the reason your agent does not bankrupt you.
# ---------------------------------------------------------------------------
@dataclass
class Budget:
    """
    Hard limits on what one agent run may consume.

    WHY a dataclass and not three loose ints: because "the agent ran out of
    budget" needs to be answerable in one call (`exceeded()`), and because
    notebook 05 hands learners a Budget and asks them to predict which limit
    trips first. A named thing can be reasoned about; three scattered counters
    cannot.

    The defaults are deliberately SMALL. A teaching agent that can take 50 steps
    will take 50 steps while thirty people watch.
    """

    max_steps: int = 8
    max_tool_calls: int = 12
    max_seconds: float = 120.0
    # Wall-clock start is set when the run begins, not when Budget is built,
    # so a Budget can be defined once and reused across runs.
    started_at: Optional[float] = None

    def start(self) -> "Budget":
        self.started_at = time.time()
        return self

    def elapsed(self) -> float:
        return 0.0 if self.started_at is None else time.time() - self.started_at

    def exceeded(self, *, steps: int, tool_calls: int) -> Optional[str]:
        """
        Return a human-readable reason if any limit is blown, else None.

        Returning the *reason* rather than a bool is what makes the trace
        readable later: "exhausted" tells a learner nothing, "max_steps (8)
        reached" tells them exactly which knob to turn.
        """
        if steps >= self.max_steps:
            return f"max_steps ({self.max_steps}) reached"
        if tool_calls >= self.max_tool_calls:
            return f"max_tool_calls ({self.max_tool_calls}) reached"
        if self.started_at is not None and self.elapsed() > self.max_seconds:
            return f"max_seconds ({self.max_seconds:.0f}s) exceeded"
        return None


# ---------------------------------------------------------------------------
# Observation — one structured record of "we did a thing and this came back".
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """
    The machine-readable half of what happened in a step.

    Note that a FAILED tool call is still an Observation. Recording failures is
    not bookkeeping pedantry — it is what lets the agent (and the learner)
    notice "I have now called get_order_status with a bad ID three times", which
    is the single most common agent failure mode in notebook 06.
    """

    step: int
    tool: str
    args: Dict[str, Any]
    status: ObservationStatus
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status is ObservationStatus.OK

    def signature(self) -> str:
        """
        A stable identity for "this exact call", used to detect repetition.

        Args are sorted and JSON-encoded so that {"a":1,"b":2} and {"b":2,"a":1}
        compare equal — an agent that reorders its keyword arguments is still
        repeating itself, and we want to catch that.
        """
        try:
            args = json.dumps(self.args, sort_keys=True, default=str)
        except Exception:
            args = str(sorted(self.args.items()))
        return f"{self.tool}({args})"

    def summary(self, width: int = 90) -> str:
        """One-line rendering for traces and notebook output."""
        if self.status is ObservationStatus.OK:
            body = str(self.result)
        else:
            body = f"{self.status.value.upper()}: {self.error}"
        body = " ".join(body.split())
        if len(body) > width:
            body = body[: width - 1] + "…"
        return body


# ---------------------------------------------------------------------------
# AgentState — the whole memory of one run.
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    """
    Everything the agent knows about the current run.

    Construct it with a goal, then let the loop mutate it. Every mutation goes
    through a named method (`add_message`, `record_observation`, `finish`)
    rather than direct field assignment, so that there is exactly one place to
    put a breakpoint when a learner asks "but *when* does state change?".
    """

    goal: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    status: AgentStatus = AgentStatus.RUNNING
    final_answer: Optional[str] = None
    stop_reason: Optional[str] = None
    step: int = 0
    # Free-form space for skills to stash intermediate findings. Kept separate
    # from `observations` so that tool results stay a faithful log of what
    # actually happened.
    scratchpad: Dict[str, Any] = field(default_factory=dict)

    # -- lifecycle ---------------------------------------------------------
    def start(self, system_prompt: str) -> "AgentState":
        """Seed the transcript and start the clock. Idempotent per run."""
        self.budget.start()
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.goal},
        ]
        return self

    def finish(
        self,
        answer: Optional[str],
        status: AgentStatus = AgentStatus.DONE,
        reason: Optional[str] = None,
    ) -> "AgentState":
        self.final_answer = answer
        self.status = status
        self.stop_reason = reason
        return self

    # -- writes ------------------------------------------------------------
    def add_message(self, role: str, content: Any, **extra) -> None:
        """
        Append to the model-readable transcript.

        `extra` carries provider-specific fields (`tool_calls`, `tool_call_id`)
        without this module having to know about any particular provider's
        message shape — that translation lives in config.py.
        """
        msg: Dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        self.messages.append(msg)

    def record_observation(self, obs: Observation) -> Observation:
        """Append to the machine-readable record. Returns it for chaining."""
        self.observations.append(obs)
        return obs

    # -- reads (the interesting part) --------------------------------------
    def tool_call_count(self) -> int:
        return len(self.observations)

    def successful_observations(self) -> List[Observation]:
        return [o for o in self.observations if o.ok]

    def consecutive_errors(self) -> int:
        """
        How many failures in a row, counting back from the most recent call.

        Used by control.py to stop an agent that is flailing. An agent that has
        failed four times running is not one attempt away from success.
        """
        n = 0
        for obs in reversed(self.observations):
            if obs.ok:
                break
            n += 1
        return n

    def repeated_calls(self, threshold: int = 2) -> List[str]:
        """
        Signatures of calls made at least `threshold` times.

        This is the detector for the classic no-progress loop: the agent asks
        the same question over and over because nothing in its state told it
        that it already had the answer. Non-empty output here is the smoking
        gun in notebook 06.
        """
        counts: Dict[str, int] = {}
        for obs in self.observations:
            sig = obs.signature()
            counts[sig] = counts.get(sig, 0) + 1
        return [sig for sig, n in counts.items() if n >= threshold]

    def approx_tokens(self) -> int:
        """
        Rough size of the transcript, in tokens.

        Deliberately a ~4-chars-per-token estimate rather than a tiktoken call:
        this must work offline, and the teaching point is *that context grows
        every step*, not the third significant figure. `tiktoken` is shown in
        notebook 05 as the accurate parallel.
        """
        chars = sum(len(str(m.get("content") or "")) for m in self.messages)
        return chars // 4

    def evidence(self) -> str:
        """
        Everything the agent successfully learned, as text.

        This is what a grounded final answer must be built from. If a claim in
        the answer is not traceable to this string, the agent hallucinated it —
        which is precisely the check notebook 06 performs.
        """
        lines = []
        for obs in self.successful_observations():
            lines.append(f"- {obs.tool}{tuple(obs.args.values())} -> {obs.result}")
        return "\n".join(lines)

    # -- rendering ---------------------------------------------------------
    def __str__(self) -> str:
        limit = self.budget
        return (
            f"AgentState(goal={self.goal[:48]!r}, step={self.step}/{limit.max_steps}, "
            f"tools={self.tool_call_count()}/{limit.max_tool_calls}, "
            f"status={self.status.value}, ~{self.approx_tokens()} tok)"
        )

    def show(self) -> str:
        """A readable dump of the whole state — used all over the notebooks."""
        out = [
            "=" * 68,
            f"GOAL   : {self.goal}",
            f"STATUS : {self.status.value}"
            + (f"  ({self.stop_reason})" if self.stop_reason else ""),
            f"STEPS  : {self.step}/{self.budget.max_steps}"
            f"   TOOL CALLS: {self.tool_call_count()}/{self.budget.max_tool_calls}"
            f"   CONTEXT: ~{self.approx_tokens()} tok",
            "-" * 68,
        ]
        if self.observations:
            out.append("OBSERVATIONS:")
            for obs in self.observations:
                flag = "  " if obs.ok else "!!"
                out.append(f" {flag} [{obs.step}] {obs.tool}({_fmt_args(obs.args)})")
                out.append(f"       -> {obs.summary()}")
        else:
            out.append("OBSERVATIONS: (none — the agent never called a tool)")
        repeats = self.repeated_calls()
        if repeats:
            out.append("-" * 68)
            out.append(f"REPEATED CALLS (no-progress signal): {repeats}")
        out += ["-" * 68, f"ANSWER : {self.final_answer}", "=" * 68]
        return "\n".join(out)


def _fmt_args(args: Dict[str, Any], width: int = 46) -> str:
    """Compact keyword-argument rendering for one-line trace output."""
    rendered = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return rendered if len(rendered) <= width else rendered[: width - 1] + "…"

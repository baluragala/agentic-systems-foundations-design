"""
control.py — knowing when to stop, and noticing when you are stuck.
===================================================================

WHY this file exists
--------------------
`while True:` is a complete agent loop and a completely irresponsible one.

The gap between a demo agent and a deployable one is almost entirely in this
file. A loop that can call an LLM and a tool is a fifteen-minute exercise. A
loop that reliably *stops* — on success, on failure, on running out of money, on
realising it is going in circles — is the actual engineering.

Three distinct questions live here, and conflating them is the usual mistake:

  1. **Are we done?**       -> the goal is satisfied. The good ending.
  2. **Must we stop?**      -> budget is gone. The safe ending.
  3. **Are we stuck?**      -> still have budget, but making no progress.
                               The interesting ending, and the one nobody
                               implements until it has cost them.

Question 3 is where most real spend is wasted. An agent repeating the same
failing call has not exceeded any limit; it is simply not going anywhere, and it
will keep not going anywhere until the step budget runs out. Detecting it early
turns a 20-step burn into a 3-step stop.

WHAT this module provides
-------------------------
* `TerminationCondition`  — a small named predicate over `AgentState`.
* The standard set        — budget, no-progress, repetition, error-streak.
* `TerminationPolicy`     — an ordered list of them, evaluated each step.
* `reflect()`             — the agent critiquing its own answer before returning.

DESIGN NOTE — why conditions are objects and not `if` statements in the loop
----------------------------------------------------------------------------
Because the interesting question in class is "which condition fired, and why?"
A named condition can report itself: `"repetition: called get_order_status with
identical arguments 3 times"`. An inline `if` can only produce a boolean and a
shrug. The trace is only as good as the vocabulary the code has for describing
its own behaviour.

The recurring question — *"what does this step look like when it goes wrong, and
where would you see it in the trace?"*:

  * No termination at all   -> `stop_reason` is None and step count equals the
                               hard ceiling. Every run looks identical because
                               every run is just "ran out".
  * Only a step budget      -> the agent burns all 8 steps repeating one call.
                               `repeated_calls()` in the state dump is the tell.
  * Termination too eager   -> the agent stops before gathering enough evidence
                               and answers thinly. The trace shows 1 tool call
                               and a confident answer — the worst combination.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

from .state import AgentState, AgentStatus


# ---------------------------------------------------------------------------
# One condition
# ---------------------------------------------------------------------------
@dataclass
class TerminationCondition:
    """
    A named reason an agent run might end.

    `check` receives the state and returns either None (keep going) or a string
    explaining why we should stop. Returning the explanation rather than True is
    the whole point — that string ends up in the trace and is what a learner
    reads when asking "why did it stop there?".
    """

    name: str
    check: Callable[[AgentState], Optional[str]]
    # The status to record if this condition fires. Most stops are "exhausted";
    # a satisfied goal is "done". Distinguishing them matters because you want
    # to alert on one and not the other.
    status: AgentStatus = AgentStatus.EXHAUSTED

    def __call__(self, state: AgentState) -> Optional[str]:
        return self.check(state)


# ---------------------------------------------------------------------------
# The standard conditions
# ---------------------------------------------------------------------------
def budget_exceeded() -> TerminationCondition:
    """
    Stop when any hard limit is reached. The non-negotiable one.

    Every agent gets this. It is the difference between a bug that costs you an
    afternoon and a bug that costs you a monthly bill.
    """

    def check(state: AgentState) -> Optional[str]:
        return state.budget.exceeded(
            steps=state.step, tool_calls=state.tool_call_count()
        )

    return TerminationCondition("budget", check)


def repetition(threshold: int = 3) -> TerminationCondition:
    """
    Stop when the agent makes the same call with the same arguments too often.

    Identical arguments is the key qualifier. Calling `search_docs` five times
    with five different queries is an agent working a problem. Calling it five
    times with the *same* query is an agent that has not noticed it already has
    the answer — there is no new information coming, only cost.

    The default threshold of 3 allows one honest retry (a transient failure
    deserves a second attempt) while catching a genuine loop quickly.
    """

    def check(state: AgentState) -> Optional[str]:
        repeats = state.repeated_calls(threshold=threshold)
        if repeats:
            return f"repetition: identical call repeated {threshold}× — {repeats[0]}"
        return None

    return TerminationCondition("repetition", check)


def error_streak(limit: int = 3) -> TerminationCondition:
    """
    Stop after N consecutive failed tool calls.

    An agent that has failed three times running is not one attempt from
    success; it has misunderstood something structural — a wrong ID format, a
    tool that is down, a schema it cannot satisfy. Further attempts produce
    identical failures more expensively.

    Note this is deliberately separate from `repetition`: an agent can fail
    three times with three *different* wrong arguments, which repetition would
    never catch.
    """

    def check(state: AgentState) -> Optional[str]:
        streak = state.consecutive_errors()
        if streak >= limit:
            last = state.observations[-1]
            return f"error_streak: {streak} consecutive failures — last: {last.error}"
        return None

    return TerminationCondition("error_streak", check)


def no_new_information(window: int = 3) -> TerminationCondition:
    """
    Stop when the last `window` successful calls returned nothing new.

    The subtlest of the four. An agent can vary its arguments (dodging
    `repetition`) and succeed every time (dodging `error_streak`) while learning
    nothing — three differently-worded searches returning the same passage. The
    call log looks healthy; the *information* is flat.

    This is the condition that most closely tracks what a human supervisor
    would notice, and the one most often missing in practice.
    """

    def check(state: AgentState) -> Optional[str]:
        succeeded = state.successful_observations()
        if len(succeeded) < window:
            return None
        recent = [str(o.result) for o in succeeded[-window:]]
        if len(set(recent)) == 1:
            return f"no_new_information: last {window} successful calls returned identical results"
        return None

    return TerminationCondition("no_new_information", check)


def context_limit(max_tokens: int = 12000) -> TerminationCondition:
    """
    Stop before the transcript outgrows the model's context window.

    Worth stating plainly because it surprises people: an agent's cost per step
    RISES as it runs, because every step re-sends the whole transcript. A
    10-step run does not cost ten times a 1-step run; it costs considerably
    more. Watching `approx_tokens()` climb in the trace makes this concrete.
    """

    def check(state: AgentState) -> Optional[str]:
        tokens = state.approx_tokens()
        if tokens > max_tokens:
            return f"context_limit: transcript ≈{tokens} tokens exceeds {max_tokens}"
        return None

    return TerminationCondition("context_limit", check)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------
class TerminationPolicy:
    """
    An ordered set of conditions, checked before every step.

    Order is significant: the first condition to fire is the one recorded, so
    put the most *diagnostic* conditions first. "repetition" tells you far more
    about what went wrong than "budget", and if you check budget first you will
    only ever see "max_steps reached" and learn nothing about why.

    That ordering choice is a good five-minute discussion in class: the same set
    of conditions, reordered, produces traces of completely different diagnostic
    value while behaving identically.
    """

    def __init__(self, conditions: Optional[List[TerminationCondition]] = None):
        self.conditions = conditions if conditions is not None else self.default()

    @staticmethod
    def default() -> List[TerminationCondition]:
        """Diagnostic conditions first, the hard budget last as a backstop."""
        return [
            repetition(),
            error_streak(),
            no_new_information(),
            context_limit(),
            budget_exceeded(),
        ]

    @staticmethod
    def budget_only() -> "TerminationPolicy":
        """
        The naive policy — a step cap and nothing else.

        Included so notebook 05 can run the same goal under both policies and
        compare traces. This is what most first agents have, and it is why they
        exhaust their budget silently instead of reporting that they were stuck.
        """
        return TerminationPolicy([budget_exceeded()])

    def check(self, state: AgentState) -> Optional[tuple[str, str, AgentStatus]]:
        """Return (condition_name, reason, status) for the first match, else None."""
        for condition in self.conditions:
            reason = condition(state)
            if reason:
                return condition.name, reason, condition.status
        return None

    def names(self) -> List[str]:
        return [c.name for c in self.conditions]

    def __str__(self) -> str:
        return f"TerminationPolicy({' → '.join(self.names())})"


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------
REFLECTION_PROMPT = """\
You are reviewing a draft answer before it is sent to the user.

GOAL:
{goal}

EVIDENCE ACTUALLY GATHERED (tool results):
{evidence}

DRAFT ANSWER:
{draft}

Check three things, in order:
1. GROUNDING — is every factual claim in the draft supported by the evidence?
2. COMPLETENESS — does it actually answer the goal, or only part of it?
3. HONESTY — if the evidence is insufficient, does the draft say so plainly
   rather than guessing?

If the draft passes all three, reply with exactly: APPROVED
Otherwise reply with a corrected answer that uses ONLY the evidence above.
"""


@dataclass
class Reflection:
    """The outcome of one self-review pass."""

    approved: bool
    original: str
    revised: str
    critique: str

    @property
    def changed(self) -> bool:
        return self.original.strip() != self.revised.strip()


def reflect(llm, state: AgentState, draft: str) -> Reflection:
    """
    Have the model check its own answer against the evidence before returning.

    WHY this works at all — the honest version. Reflection is not the model
    "thinking harder". It works because *checking* a claim against a list of
    facts is a genuinely easier task than *producing* the claim was, and because
    the second pass sees the draft as text to be audited rather than as its own
    in-progress reasoning. Narrow, checkable criteria (is this claim in the
    evidence?) improve reliably. Open-ended ones ("is this good?") mostly do not.

    WHEN NOT TO USE IT. Reflection doubles latency and cost for every answer. It
    earns that on high-stakes or hallucination-prone outputs; it does not on a
    lookup that returned one unambiguous number. "Reflect on everything" is a
    common and expensive mistake, so the default in `agent.py` is off.

    """
    from .config import Decision  # local import keeps the dependency one-way

    prompt = REFLECTION_PROMPT.format(
        goal=state.goal,
        evidence=state.evidence() or "(no tool results were gathered)",
        draft=draft,
    )

    decision: Decision = llm.decide(
        [
            {"role": "system", "content": "You are a careful, sceptical reviewer."},
            {"role": "user", "content": prompt},
        ],
        [],  # reflection is a pure text task — no tools offered
    )
    critique = (decision.content or "").strip()

    if not critique or critique.upper().startswith("APPROVED"):
        return Reflection(approved=True, original=draft, revised=draft, critique=critique)
    return Reflection(approved=False, original=draft, revised=critique, critique=critique)

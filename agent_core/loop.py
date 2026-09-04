"""
loop.py — the agent loop. The whole idea, in about forty lines.
===============================================================

WHY this file exists
--------------------
Everything else in this package is scaffolding around what happens here. If a
learner leaves the session remembering one piece of code, it should be
`run_loop()`.

The claim worth defending: **an agent is not a smarter model.** It is the same
model, called repeatedly, with the results of its own previous actions appended
to its input. That is it. That is the entire difference between "LLM" and
"agent". Everything that feels magical about agents — recovering from an error,
gathering evidence across several sources, deciding it needs one more lookup —
falls out of that one structural change.

THE LOOP
--------
    while not done:
        decision = llm.decide(state.messages, tools)   # THINK
        if decision.is_final: break                    # (terminate?)
        observations = [tools.dispatch(c) for c in decision.tool_calls]  # ACT
        state.record(observations)                     # OBSERVE + UPDATE
    return state.final_answer

Compare with a RAG pipeline from the previous session:

    retrieve(q) -> augment -> generate -> done

Straight line, fixed length, decided by the programmer. The agent's path is a
cycle whose length is decided *at runtime by the model*. That single property is
what buys the flexibility and what causes every failure mode in notebook 06.

WHY THE STATE UPDATE IS THE LOAD-BEARING STEP
---------------------------------------------
Delete the `state.add_message(...)` calls after the tool runs and the loop still
executes, still calls tools, still terminates. It just never learns anything:
every iteration sees the same input, so the model makes the same decision, and
you get the same call forever. The observation must go *back into the input* or
the loop is a very expensive `while True`.

Notebook 02 runs exactly that experiment — stateless vs stateful, same model,
same tools — because being told this is much less convincing than watching the
stateless version repeat itself four times.

WHAT THIS MODULE DOES NOT DO
----------------------------
No prompt construction (that is `skills.py`), no provider specifics (that is
`config.py`), no tool implementations (that is `tools.py`), no stop policy (that
is `control.py`). This file only orchestrates. Keeping it that thin is what
makes it readable on a slide, and readable-on-a-slide is the point.
"""
from __future__ import annotations

import time
from typing import Any, List, Optional

from .config import Decision, LLM
from .control import TerminationPolicy, reflect
from .state import AgentState, AgentStatus, Observation
from .tools import ToolRegistry
from .trace import StepRecord, Trace


def run_loop(
    llm: LLM,
    tools: ToolRegistry,
    state: AgentState,
    *,
    policy: Optional[TerminationPolicy] = None,
    trace: Optional[Trace] = None,
    stateful: bool = True,
    use_reflection: bool = False,
) -> Trace:
    """
    Turn the agent loop until it terminates. Returns the trace.

    Args:
        llm: anything with `.decide(messages, tools) -> Decision`.
        tools: the registry this run may use.
        state: seeded with `state.start(system_prompt)` before calling.
        policy: when to stop. Defaults to the full diagnostic policy.
        trace: an existing Trace to append to; one is created if omitted.
        stateful: **teaching switch.** False means observations are NOT written
            back into `state.messages`, which reproduces the amnesiac agent for
            the notebook-02 comparison. Never use False for real work.
        use_reflection: run a self-review pass over the final answer. Off by
            default — it doubles cost, and it only earns that on outputs where
            being wrong is expensive.

    The function mutates `state` as it goes, on purpose: after the run a learner
    can inspect `state.show()` and see exactly what the agent ended up knowing.
    """
    policy = policy or TerminationPolicy()
    trace = trace or Trace(goal=state.goal)

    while True:
        # ---------------------------------------------------------------
        # 0. TERMINATE? — asked BEFORE thinking, not after.
        # ---------------------------------------------------------------
        # Order matters and costs money. Checking first means an agent that is
        # already over budget does not make one more LLM call to discover it.
        # Checking after is the more natural way to write it and the more
        # expensive way to run it.
        stop = policy.check(state)
        if stop:
            name, reason, status = stop
            state.finish(
                answer=_best_effort_answer(state),
                status=status,
                reason=f"{name} — {reason}",
            )
            break

        step_started = time.perf_counter()
        state.step += 1
        context_tokens = state.approx_tokens()

        # ---------------------------------------------------------------
        # 1. THINK — the only place the model is consulted.
        # ---------------------------------------------------------------
        decision: Decision = llm.decide(state.messages, tools)

        record = StepRecord(
            step=state.step,
            decision=str(decision),
            thought=decision.content if not decision.is_final else None,
            context_tokens=context_tokens,
        )

        # ---------------------------------------------------------------
        # 2. FINAL? — the model says it is done.
        # ---------------------------------------------------------------
        if decision.is_final:
            answer = decision.content or ""
            if use_reflection:
                # Self-review before the answer leaves the building.
                review = reflect(llm, state, answer)
                if not review.approved:
                    record.thought = (
                        (record.thought or "") + "\n[reflection revised the answer]"
                    )
                    answer = review.revised
            state.add_message("assistant", answer)
            state.finish(answer=answer, status=AgentStatus.DONE, reason="model returned a final answer")
            record.duration_ms = (time.perf_counter() - step_started) * 1000
            trace.add(record)
            break

        # ---------------------------------------------------------------
        # 3. ACT — run every requested tool.
        # ---------------------------------------------------------------
        # The assistant's request goes into the transcript BEFORE the results
        # do. Providers require the pairing (a tool result with no preceding
        # request is a protocol error), and it keeps the transcript a truthful
        # chronology rather than a tidied-up summary.
        #
        # When `stateful=False` we skip this too, not just the results. "The
        # agent has no memory" has to mean the WHOLE cycle leaves no trace in
        # the input — request and result alike. Skipping only the result would
        # leave the model able to see what it had already asked for, which is
        # still memory, and the comparison in notebook 02 would show no
        # difference at all. (That near-miss is worth mentioning in class: the
        # experiment silently proves nothing if you get this wrong.)
        if stateful:
            state.add_message(
                "assistant",
                decision.content,
                tool_calls=[
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {"name": call.name, "arguments": _dumps(call.args)},
                    }
                    for call in decision.tool_calls
                ],
            )

        observations: List[Observation] = []
        terminal_hit = False

        for call in decision.tool_calls:
            observation = tools.dispatch(call.name, call.args, step=state.step)
            state.record_observation(observation)
            observations.append(observation)

            # -----------------------------------------------------------
            # 4. OBSERVE — write the result back into the input.
            #    THIS IS THE LINE THAT MAKES IT AN AGENT.
            # -----------------------------------------------------------
            if stateful:
                state.add_message(
                    "tool",
                    _render_observation(observation),
                    tool_call_id=call.id,
                    name=call.name,
                )

            found = tools.get(call.name)
            if found is not None and found.terminal and observation.ok:
                terminal_hit = True

        record.observations = observations
        record.duration_ms = (time.perf_counter() - step_started) * 1000
        trace.add(record)

        # ---------------------------------------------------------------
        # 5. A terminal tool ends the run.
        # ---------------------------------------------------------------
        # `escalate_to_human` succeeding is a *successful* outcome, not a
        # failure — the agent correctly recognised its limits. Recording it as
        # ESCALATED rather than DONE keeps that distinction visible in metrics.
        if terminal_hit:
            last = observations[-1]
            state.finish(
                answer=str(last.result),
                status=AgentStatus.ESCALATED,
                reason=f"terminal tool {last.tool!r} completed",
            )
            break

    return trace.finish(state)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _dumps(args: Any) -> str:
    import json

    try:
        return json.dumps(args)
    except TypeError:
        return str(args)


def _render_observation(obs: Observation) -> str:
    """
    Turn an Observation into the text the model sees next step.

    Failures are rendered as readable prose, not as a status code, because this
    string is the model's ONLY chance to understand what went wrong and fix it.
    "ERROR (rejected): order_id: 'ACME1042' does not match ^ACME-\\d+$" gives the
    model everything it needs. A bare `{"error": true}` gives it nothing, and
    you will watch it guess.
    """
    if obs.ok:
        return str(obs.result)
    return f"ERROR ({obs.status.value}): {obs.error}"


def _best_effort_answer(state: AgentState) -> str:
    """
    What to say when the loop stops without the model producing an answer.

    Returning None here would be a small betrayal of the user, who asked a
    question and deserves to know that we tried, what we found, and that we
    stopped. Partial evidence plus an honest "I stopped early" beats silence,
    and it beats a confident answer synthesised from nothing.
    """
    evidence = state.evidence()
    if not evidence:
        return (
            "I stopped before I could gather any information for this request. "
            "Nothing here is confirmed — please re-run with a larger budget or "
            "hand this to a human."
        )
    return (
        "I stopped before finishing. Here is what I confirmed along the way:\n"
        f"{evidence}\n\n"
        "Treat this as partial — I did not reach a complete answer."
    )

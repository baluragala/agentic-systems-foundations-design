# Exercises — Agentic Systems Foundations
Worked solutions in [`solutions/solutions.md`](solutions/solutions.md). **Attempt each one before looking** — several are designed so that the obvious answer is wrong, and the value is in noticing that yourself.

**Sections 01–06 and the capstone run with no API key.** Set one if you have it; the exercises are written so the lesson survives either way, and each says where the offline mock's limits show. The **Appendix** exercises cover the LangGraph track — three of the five still need no key, and the two that do are marked.

```python
# Setup for every exercise
from agent_core import (Agent, Budget, CATALOGUE, Router, Skill, ToolRegistry,
                        TerminationPolicy, broken_agent, compare, report, tool)
from agent_core.acme_tools import acme_registry
from agent_core.schemas import validate_args
from agent_core.skills import acme_skills, acme_router

registry = acme_registry()
router = acme_router(registry)
```

**Difficulty:** ⭐ recall · ⭐⭐ apply · ⭐⭐⭐ design/diagnose

---

## Section 01 — What makes a system agentic

### Exercise 1.1 — Classify these systems ⭐

For each, say whether it is agentic, and name the **missing property** if not (goal-directed / state / iterative).

1. A cron job that summarises yesterday's support tickets with one LLM call each.
2. A customer-service bot that answers from a knowledge base, then re-queries with a rephrased question if its confidence is low.
3. A code assistant that reads an error, edits a file, re-runs the tests, and repeats until they pass.
4. A pipeline that extracts fields from an invoice PDF, validates them, and writes to a database.
5. A chatbot that remembers the whole conversation and answers each message in one shot.

### Exercise 1.2 — Argue the other side ⭐⭐

Your team wants to rebuild a working RAG pipeline as an agent, "because agents are more flexible".

Write the **three-sentence case against**, using concrete costs. Then name the **one** condition that would change your mind.

### Exercise 1.3 — Trajectory variance ⭐⭐

```python
agent = Agent()
for goal in [
    "How much does the Growth plan cost per month?",
    "Is order ACME-1046 refundable? I changed my mind.",
    "What is 199 multiplied by 12?",
]:
    r = agent.run(goal)
    print(f"{r.skill:<20} {'→'.join(r.tools_called())}")
```

Run it. Then answer: **why can you not write a single `assert` covering all three runs?** What *can* you assert?

---

## Section 02 — The agent loop and state

### Exercise 2.1 — Write the loop from memory ⭐⭐

Close the notebook. On paper, write the agent loop in **six lines or fewer**, labelling THINK, ACT, OBSERVE and the state update.

Then check against `agent_core/loop.py`. **Which line did you forget?** (Most people forget the state update — which is the point.)

### Exercise 2.2 — Predict, then verify ⭐⭐

Before running anything, predict `steps`, `calls` and `status` for each:

```python
a = Agent()
goal = "Is order ACME-1042 refundable? It was a billing_error."

run_a = a.run(goal)                       # normal
run_b = a.run(goal, stateful=False)       # observations never written back
run_c = Agent(budget=Budget(max_steps=1)).run(goal)   # one step only
```

Write your three predictions down, then run `compare({...})`. **Explain any you got wrong** — the explanation is the exercise.

### Exercise 2.3 — Measure the cost curve ⭐⭐

Using `trace.steps[i].context_tokens`, compute for a 3-step run:

1. total tokens **sent** across the whole run,
2. what a single call with the final context would have cost,
3. the ratio.

Then: **at what step count does an agent cost 5× the equivalent single call?** Extrapolate from your numbers and state your assumption.

### Exercise 2.4 — Break state deliberately ⭐⭐⭐

In a *copy* of `run_loop`, change the OBSERVE step to write only the **first 20 characters** of each tool result into the transcript.

Predict what happens, then run it against `"Is order ACME-1046 refundable? I changed my mind."`

**Which failure mode from Section 06 does truncated state produce?** Why is this more dangerous than dropping the observation entirely?

---

## Section 03 — Tools and schemas

### Exercise 3.1 — Fix a bad tool ⭐⭐

This tool is realistic and bad in five distinct ways:

```python
@tool
def upd_sub(oid, p, imm=True):
    """Updates subscription."""
    if p not in ("Starter", "Growth", "Enterprise"):
        raise ValueError("bad plan")
    return {"ok": True, "code": 0}
```

Name all five problems, then rewrite it. Verify with:

```python
print(upd_sub.prompt_line())
print(ToolRegistry([upd_sub]).dispatch("upd_sub", {"oid": 1046, "p": "Gold"}))
```

Your rewrite passes when a wrong plan name is **rejected by the schema** (not by a raise), and the error text tells the model what the valid values are.

### Exercise 3.2 — The description is the interface ⭐⭐

Build two tools with **identical code and identical schemas**, differing only in description:

- A: `"""Gets data."""`
- B: a description that says *when* to use it, with an example request

Give each to an agent (separately) and compare `explain_plan` scores, or the trajectory with a real model.

**Write one sentence explaining what the model actually reads to choose a tool.**

### Exercise 3.3 — Coerce or reject? ⭐⭐⭐

For each payload against `search_docs(query: str, k: int = 2)`, decide **coerce** or **reject**, and justify:

| # | Payload |
|---|---|
| 1 | `{"query": "pricing", "k": "3"}` |
| 2 | `{"query": "pricing", "k": 3.0}` |
| 3 | `{"query": "pricing", "k": "three"}` |
| 4 | `{"query": 42}` |
| 5 | `{"query": "pricing", "k": true}` |

Check against `validate_args`. **Where you disagree with the implementation, say who is right and why** — #5 in particular is a deliberate design decision, not an oversight.

### Exercise 3.4 — Schema translation ⭐

Take your Exercise 3.1 tool and render it for all three providers. Then answer in one sentence each:

1. What is genuinely different between the three?
2. What is identical?
3. Which of those two is the design work?

---

## Section 04 — Agent skills

### Exercise 4.1 — Scope a skill ⭐⭐

Build an `incident_response` skill for these tools: `search_docs`, `get_order_status`, `calculate`, `check_refund_eligibility`, `escalate_to_human`.

Its job: a customer reports the API returning HTTP 429 and asks whether they are owed anything.

1. Which tools does it need? **Justify every inclusion *and* every exclusion.**
2. Write instructions saying what to do **first**.
3. Write triggers that win its own cases and lose refund cases.

Verify with `router.explain()` on at least four goals, including one that should route elsewhere.

### Exercise 4.2 — Reproduce the over-scoping cost ⭐⭐

Run the same six goals through the router and through `Skill.compose(*acme_skills(registry))`.

Report total tool calls for each, and **name one specific call the composed agent made that the routed one did not**. Explain what that call cost you — be precise, there are at least three costs.

### Exercise 4.3 — Break the router on purpose ⭐⭐⭐

Add a skill with the trigger `r"a"`.

1. Predict what happens to routing.
2. Run `router.explain()` on four goals and confirm.
3. **Fix it two different ways** and say which is better and why.

### Exercise 4.4 — When composing is right ⭐⭐⭐

Find (or write) a goal that genuinely needs tools from two different skills.

1. Show the router failing it — and be precise about **how** it fails.
2. Show `Skill.compose` handling it.
3. Then argue the *other* side: what did you lose, and what would you do instead in a system with 25 tools?

---

## Section 05 — Control and tracing

### Exercise 5.1 — Order the conditions ⭐⭐

Take these five and put them in the order you would check them: `budget_exceeded`, `repetition`, `error_streak`, `no_new_information`, `context_limit`.

Justify your ordering **in terms of what the resulting trace tells you**, not in terms of safety — all orderings are equally safe.

### Exercise 5.2 — Write a termination condition ⭐⭐⭐

Write `cost_ceiling(max_tokens_sent)` that stops the run when **cumulative** tokens sent across all steps exceeds a limit.

Note this is *not* `context_limit()` — that caps a single transcript; yours accumulates. Use `state.approx_tokens()` and think about where the running total lives.

Test it with a limit low enough to fire mid-run.

### Exercise 5.3 — Detect a stall the standard conditions miss ⭐⭐⭐

Construct a run where the agent:

- makes **different** calls each time (dodges `repetition`),
- **succeeds** every time (dodges `error_streak`),
- gets **different** results each time (dodges `no_new_information`),
- and still makes no progress toward the goal.

Then: **what condition would catch it?** If you conclude none can, say what *would* — and what that implies about relying on automated stall detection.

### Exercise 5.4 — Trace as a bug report ⭐⭐

Take a failing run, `to_json()` it, and hand the JSON to a colleague (or a fresh notebook) with **no other context**.

Can they say what went wrong? **What is missing from the trace that you had to explain verbally?** Propose one concrete field to add to `StepRecord`.

---

## Section 06 — Failure modes

### Exercise 6.1 — Diagnose five, blind ⭐⭐

```python
import random
keys = list(CATALOGUE); random.shuffle(keys)
runs = [broken_agent(k).run("Is order ACME-1042 refundable? It was a billing_error.") for k in keys]
for r in runs:
    print(r.trace.show()); print("=" * 70)
```

Diagnose each **from the trace alone**, working the four questions in order. Then check with `report()`.

Record your **hit rate**, and for each miss note *which of the four questions* you skipped.

### Exercise 6.2 — Catch a quiet failure ⭐⭐⭐

`report(trace)` finds nothing on the `wrong_tool` run. That is correct behaviour, not a bug.

1. Explain **why** a heuristic cannot find it.
2. Make it findable by supplying `expected_tools`.
3. Generalise: **what would you need in production to detect wrong-tool failures at scale?** (Hint: what does `expected_tools` really represent?)

### Exercise 6.3 — Write negative tests ⭐⭐⭐

Add **three** tasks to `data/tasks/agent_tasks.jsonl` where **the correct behaviour is to refuse, escalate, or report NOT FOUND**.

At least one must be a **trap**: superficially eligible, actually forbidden by a rule that takes precedence. (`T13` is the model — study why it works before writing yours.)

Run the suite and confirm each behaves as intended.

### Exercise 6.4 — Ground-truth an answer ⭐⭐

Write `check_grounding(result) -> list[str]` returning every number, date and currency amount in the answer that appears in **no** observation.

Run it across the whole task suite. **How many false positives?** What does that tell you about automated grounding checks — and about `diagnose()` reporting only *medium* confidence on this one?

---

## Appendix — the LangGraph track (notebook 08)

These need `OPENAI_API_KEY` unless marked otherwise.

### Exercise A.1 — Port a termination condition ⭐⭐⭐ *(no key needed)*

`agent_lc/graph.py` ports `repetition` and `error_streak` into a LangGraph conditional
edge. **Port `no_new_information` too.**

Test it with `FakeToolCallingModel` so the result is reproducible. Then answer: what did
you have to change, and what did you copy almost verbatim? What does that tell you about
how framework-specific `agent_core/control.py` really was?

### Exercise A.2 — Prove the backstop is not a diagnosis ⭐⭐ *(no key needed)*

Build the same looping agent two ways: once relying only on `recursion_limit`, once with
`build_diagnostic_graph`.

Report steps taken and the stop reason for each. Then write the two sentences you would
say to a colleague who thinks `recursion_limit` means termination is handled.

### Exercise A.3 — Break the reducer ⭐⭐⭐ *(no key needed)*

In a copy of `build_agent_graph`, change the state schema from
`Annotated[list[AnyMessage], add_messages]` to a plain `list[AnyMessage]`.

1. Predict what happens.
2. Run it and see.
3. **Which notebook-02 failure did you just reproduce?** Explain why LangGraph makes this
   harder to do by accident than the hand-rolled loop did — and why it is still possible.

### Exercise A.4 — Add a checkpointer ⭐⭐

Give an agent an `InMemorySaver` and hold a two-turn conversation where the second turn
relies on the first ("what plan is that on?").

Then: what breaks if two users share a `thread_id`? What would you use instead of
`InMemorySaver` in production, and what new failure mode does that introduce?

### Exercise A.5 — Same task, both engines ⭐⭐⭐

Run five tasks from `data/tasks/agent_tasks.jsonl` through **both** `agent_core.Agent`
and an `agent_lc` graph, using the same model. Compare with `compare()` and `to_trace()`.

Where trajectories differ, decide for each: **framework difference, prompt difference, or
model non-determinism?** Do not assume the answer is the same for all of them.

---

## Capstone ⭐⭐⭐

**Extend the Acme agent end to end.** This is the exercise to do if you only do one.

### 1. A new tool — `create_support_ticket`

- `priority` constrained to an enum
- `order_id` with a `pattern`
- a `summary` free-text argument
- returns prose, never raises, and says something useful when the order does not exist

### 2. A new skill — `technical_support`

- scoped to the tools this job needs, and **no more**
- instructions saying what to check **before** creating a ticket
- **phrase** triggers, not bare words
- added to the router with the fallback set explicitly

### 3. Tests — five tasks

- 2 × positive (one single-tool, one multi-tool)
- 2 × **negative**, where refusing or escalating is the pass condition
- 1 × **trap**, where the obvious action is forbidden by a rule that takes precedence

### 4. Break it, then fix it

Deliberately introduce **one** failure from the taxonomy. Capture the trace. Diagnose it using only the four questions. Fix it. Show the before/after traces side by side with `compare()`.

### 5. Write it up — one page

- the trajectory your agent takes on a representative task, and why
- the schema decision you are least sure about, and what would change your mind
- **which failure mode your design is still most exposed to**, and what you would build next

> **Grading is on part 5.** Parts 1–4 show you can operate the machinery. Part 5 shows you understand what it does not protect you from — which is the actual skill.

---

## Self-check

You are ready if you can answer without notes:

1. What single line, removed, turns an agent back into an expensive `while` loop?
2. Why does an agent's cost grow faster than its step count?
3. When is a rejected tool call *better news* than a successful one?
4. Why does adding a tool make an agent less reliable — and why is that failure quiet?
5. Why check `repetition` before `budget_exceeded`, when both are equally safe?
6. Two failures leave a trace with 0% errors and status `done`. Which, and how do you find them?
7. In the four-question debugging workflow, why is "fix the prompt" the *last* thing you try?
8. Your agent confidently states a refund amount no tool returned. Which question do you start at, and why not question 4?

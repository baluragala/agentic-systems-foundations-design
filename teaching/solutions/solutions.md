# Worked Solutions — Agentic Systems Foundations
Answers to [`../exercises.md`](../exercises.md). Where an exercise has no single right answer, the solution says what a **good** answer contains and what a common **wrong** answer looks like.

**The numbers below are illustrative.** They come from a real model, which is stochastic — your trajectories will differ in the details. The reasoning, and every deterministic claim (schemas, validation, guardrail arithmetic), does not.

---

## Section 01 — What makes a system agentic

### 1.1 Classify these systems

| # | System | Agentic? | Missing property |
|---|---|---|---|
| 1 | Cron job, one LLM call per ticket | **No** | state, iterative — it repeats over *inputs*, not over its own results |
| 2 | KB bot that re-queries when confidence is low | **Weakly / no** | one conditional branch is not runtime path selection. Ask: could it choose a *different tool*? |
| 3 | Code assistant: read error → edit → re-run tests → repeat | **Yes** | all three. The test result feeds the next decision — that is the write-back |
| 4 | Invoice extract → validate → write to DB | **No** | goal-directed and stateful-ish, but the path is **fixed by you**. A pipeline, correctly |
| 5 | Chatbot with full history | **No** | iterative reasoning. It remembers, but each turn is one shot with no self-directed loop |

**The distinction people miss** is #1 vs #3. Both "loop". #1 loops over a list of inputs; #3 loops over **its own previous results**. Only the second requires state, and only the second can recover from a step going wrong.

### 1.2 Argue the other side

A good three-sentence case:

> Our RAG pipeline answers in one LLM call at predictable cost and latency; an agent would take 1–N calls, and because every step re-sends the whole transcript the cost grows faster than the step count. We can currently test it by asserting on outputs — with an agent we would have to assert on trajectories and build a task suite including negative tests before we could tell whether a change helped. And our failures are loud and localised to a stage, whereas agent failures are frequently silent: a successful call to the wrong tool produces a confident wrong answer with nothing in the logs.

**The one condition that changes your mind:** *the path genuinely cannot be known in advance* — a request that might need one lookup or four, in an order that depends on what earlier lookups returned. If you can draw the flowchart, write the flowchart.

**Common wrong answer:** "agents are slower and more expensive." True but weak — it invites "so we'll optimise it". The strong argument is about **testability and silent failure**, which do not go away with optimisation.

### 1.3 Trajectory variance

```
product_questions    search_docs→calculate
refunds              check_refund_eligibility→get_order_status
product_questions    calculate
```

**Why no single assert:** the tool sequence, the number of steps and the skill all vary by goal — and with a real model they vary *between runs of the same goal*. Asserting `tools_called() == [...]` for all three is asserting on something the design deliberately made variable.

**What you *can* assert:**

- **containment, not equality** — `"get_order_status" in tools_called()`
- **ordering constraints** — order lookup appears *before* eligibility
- **terminal status** — `done` vs `escalated`
- **negative assertions** — an unanswerable goal never returns a confident figure
- **aggregate metrics over a suite** — pass rate, `error_rate()`

That is the shift from testing outputs to **testing trajectories**, and it is why the task suite exists.

---

## Section 02 — The agent loop and state

### 2.1 Write the loop from memory

```python
while not done:
    decision = llm.decide(state.messages, tools)     # THINK
    if decision.is_final:
        break                                        # terminate
    for call in decision.tool_calls:
        obs = tools.dispatch(call.name, call.args)   # ACT
        state.add_message("tool", str(obs.result))   # OBSERVE + UPDATE
```

**The line most people forget is the last one** — which is exactly the lesson. It is the only line whose removal leaves a program that still runs, still calls tools, still terminates, and is no longer an agent.

Second most-forgotten: the **terminate check before THINK**, not after. Checking after costs you one extra LLM call on every run that ends on a limit.

### 2.2 Predict, then verify

| Run | steps | calls | status | stop reason |
|---|---|---|---|---|
| `run_a` normal | 3 | 2 | `done` | model returned a final answer |
| `run_b` stateful=False | 3 | 3 | `exhausted` | `repetition: identical call repeated 3×` |
| `run_c` max_steps=1 | 1 | 1 | `exhausted` | `budget: max_steps (1) reached` |

**The one people get wrong is `run_b`.** Common predictions are "it crashes" or "it answers wrong". Neither. It repeats the *identical call* until a condition stops it — because every step it sees exactly the same input, so it makes exactly the same decision. Perfectly rational behaviour from an agent with no memory.

**`run_c` is worth noticing too:** it still produces an answer, built from partial evidence, saying it stopped early. Stopping is not an excuse for silence.

### 2.3 Measure the cost curve

Representative 3-step run (mock, your numbers will differ slightly):

```
context per step : [921, 966, 1018]
total sent       : 2905
single call would be : 1018
ratio            : 2.85×
```

**When does an agent cost 5× a single call?** Growth here is roughly +50 tokens/step on a ~900-token base, so step *n* sends about `900 + 50n`. Total after *n* steps ≈ `900n + 25n²`; a single final call ≈ `900 + 50n`. The ratio passes 5× at **around 5–6 steps**.

**State your assumption:** this base is small and the per-step growth modest. With a tool returning 1,500-character results the per-step growth dominates and the curve turns up much faster — which is exactly why `Tool.max_result_chars` exists.

The shape is the lesson: **super-linear, and driven by how chatty your tools are.**

### 2.4 Break state deliberately

Truncating observations to 20 characters produces:

```
"Order ACME-1046 — c"   # customer name, plan, status, dates, amount: all gone
```

**Result: an `ungrounded_answer` failure (mode 5).** The agent has *something*, so it does not repeat and does not report NOT FOUND — it fills the gap. You get a confident answer about an order whose details it never actually saw.

**Why this is more dangerous than dropping the observation entirely:** dropping it produces a *loud* failure — repetition, or an honest "I could not gather any information". Truncating produces a **quiet** one. The trace shows a successful call, status `done`, 0% errors. Everything looks healthy.

> **Partial information is worse than no information**, because no information triggers your safety behaviour and partial information does not. This is the same reason `Tool._truncate` writes an explicit `[truncated — N more characters not shown]` marker rather than silently cutting.

---

## Section 03 — Tools and schemas

### 3.1 Fix a bad tool

**The five problems:**

1. **No type hints** → every parameter degrades to `string`; the model gets no shape at all.
2. **Cryptic names** (`upd_sub`, `oid`, `p`, `imm`) → the model has to guess meaning from abbreviations.
3. **Useless docstring** — `"""Updates subscription."""` says nothing about *when* to use it, and there are no `Args:` descriptions.
4. **It raises.** A bad plan name kills the loop instead of becoming a correctable observation.
5. **Returns a dict with a status code.** `{"ok": True, "code": 0}` forces the model to guess what code 0 meant.

Bonus sixth: the valid plan set is checked *in the body* when it belongs in the **schema** as an enum.

```python
from typing import Literal
from agent_core import tool

@tool(examples=["Upgrade ACME-1046 to the Growth plan",
                "Change my subscription to Enterprise at the next renewal"])
def update_subscription(
    order_id: str,
    new_plan: Literal["Starter", "Growth", "Enterprise"],
    apply_immediately: bool = True,
) -> str:
    """Change the subscription plan on an existing order.

    Use this when a customer asks to upgrade, downgrade, or switch plans. It
    changes billing — check the order exists with get_order_status first.

    Args:
        order_id: The Acme order identifier, for example ACME-1046. pattern: ^ACME-\\d{4}$
        new_plan: The plan to move the subscription to.
        apply_immediately: True to change now and prorate; False to apply at the
            next renewal date.

    Returns:
        Confirmation of the change, or a clear message if it could not be made.
    """
    from agent_core.acme_tools import _ORDERS
    record = _ORDERS.get(order_id.upper())
    if record is None:
        return (f"NOT FOUND: no order with ID {order_id}. The ID is correctly "
                "formatted but does not exist, so no change was made.")
    when = "immediately, with prorated billing" if apply_immediately else "at the next renewal"
    return (f"Order {order_id.upper()} moved from the {record['plan']} plan to "
            f"{new_plan}, applying {when}.")
```

Verify:

```
{'oid': 1046, 'p': 'Gold'} -> rejected
  oid: unknown parameter (this tool accepts: order_id, new_plan, apply_immediately)
  p: unknown parameter (...)
  order_id: required parameter is missing
  new_plan: required parameter is missing
```

The enum now does the work the `raise` was doing — **before** the function is entered, and with a message naming the valid values.

### 3.2 The description is the interface

Tool A (`"""Gets data."""`) scores **0** in `explain_plan` for essentially every goal — no word in the description matches anything, so it is only ever selected by the fallback rule. Tool B scores on both description words and example overlap.

**The one sentence:**

> The model chooses a tool by matching the request against the **description and examples you wrote** — so "the model picked the wrong tool" is almost always "my description did not say *when* to use this one".

That reframing is the whole value of the exercise: it converts an unactionable complaint about the model into a specific edit you can make.

### 3.3 Coerce or reject?

| # | Payload | Implementation | Verdict |
|---|---|---|---|
| 1 | `k: "3"` | **coerced** → `3` | ✅ right. Unambiguous model quirk; rejecting burns a step |
| 2 | `k: 3.0` | **coerced** → `3` | ✅ right. `3.0` is exactly representable as `3`. Note `3.5` is *not* coerced |
| 3 | `k: "three"` | **rejected** | ✅ right. Word-to-number is a guess, not a conversion |
| 4 | `query: 42` | **coerced** → `"42"` | ⚠️ **defensible, and worth arguing about** — see below |
| 5 | `k: true` | **rejected** | ✅ right, and deliberately so — see below |

**#5 is the interesting one.** In Python `True == 1`, so a naive `isinstance(v, int)` check *accepts* `True` as an integer and you would silently retrieve 1 passage because the model sent a boolean. `_TYPE_CHECKS["integer"]` explicitly excludes `bool`. Equally deliberate: `_coerce` refuses to turn a non-empty string into a bool, because `"false"` would become `True` and someone would lose an hour to it.

> **The rule:** coerce only where exactly one interpretation exists. Everywhere else, fail loudly.

**#4 is genuinely arguable.** Coercing `42` → `"42"` is safe as a conversion but suspicious as *intent* — a numeric search query usually means the model put something in the wrong field. A reasonable alternative is to coerce but **surface the coercion in the observation** so it shows up in the trace. If you argued for rejection here, you made a good argument.

### 3.4 Schema translation

1. **What differs:** only the envelope — `{"type":"function","function":{…}}` vs a bare declaration inside `function_declarations` vs `input_schema`. Plus one real wrinkle: Gemini's dialect has not supported `additionalProperties`, so the adapter drops it.
2. **What is identical:** the JSON Schema itself — types, `enum`, `pattern`, `required`, and every description.
3. **Which is the design work:** the schema. The envelope is a five-minute lookup you write once per provider; the constraints and descriptions are what determine whether the model calls your tool correctly, and **they transfer unchanged**.

---

## Section 04 — Agent skills

### 4.1 Scope a skill

```python
incident_response = Skill(
    name="incident_response",
    instructions=(
        "Handle reports of API errors, outages and degraded service. FIRST find "
        "what the documentation says about the error code and the SLA for the "
        "customer's plan. THEN look up the order to establish which plan and SLA "
        "apply. Only then assess whether service credit is owed. If the outage "
        "is not confirmed in the documentation, do not promise a credit — "
        "escalate with a summary of what you checked."
    ),
    tools=registry.subset("search_docs", "get_order_status", "escalate_to_human"),
    triggers=[r"\b429\b", r"rate limit", r"outage", r"downtime", r"5\d\d error",
              r"api (is )?(down|failing|erroring)"],
)
```

**Justifying every inclusion:**
- `search_docs` — the SLA and error semantics live in the documentation. Without it the skill can only guess.
- `get_order_status` — SLA depends on the plan, which depends on the order.
- `escalate_to_human` — service credit is a judgement call the policy reserves for humans.

**Justifying every exclusion** (this half is what the exercise is really testing):
- `check_refund_eligibility` — **excluded.** A service credit is not a refund; including it invites the agent to apply the wrong policy and produce a confident wrong answer. This is the highest-value exclusion.
- `calculate` — **excluded.** No arithmetic is needed to establish *whether* a credit is owed. Add it only if the job grows to computing credit amounts.

**Common wrong answer:** including all five "just in case". That is precisely the over-scoping the section is about — and the cost is invisible until you measure it.

### 4.2 Reproduce the over-scoping cost

```
ROUTED    11 tool calls
COMPOSED  14 tool calls
```

**A specific extra call:** on *"Has order ACME-1043 shipped yet?"* the composed agent calls `check_refund_eligibility` after `get_order_status`. The routed agent does not — `account_lookup` does not have that tool.

**Three costs, precisely:**

1. **An LLM round-trip and a tool execution** — latency and money, for information nobody asked for.
2. **Transcript growth.** That result is now in the context for **every subsequent step**, so you pay for it repeatedly, not once.
3. **A reasoning hazard.** The agent now holds a refund-eligibility verdict for a shipping question. A confident answer volunteering refund information the customer never asked about is a *plausible* next output — and in a refund context, actively harmful.

Cost 3 is the one that matters and the one that never shows up in a latency dashboard.

### 4.3 Break the router on purpose

**Prediction:** `r"a"` matches almost any English sentence, so that skill scores +10 on nearly every goal and wins nearly everything.

**Confirmed** — `router.explain()` shows it at the top for every goal that contains the letter "a", which is effectively all of them.

**Two fixes:**

1. **Anchor the trigger to a word boundary and make it specific** — `r"\bapi\b"` instead of `r"a"`. Fixes this instance.
2. **Change the scoring so trigger specificity matters** — weight a trigger by its length or require word boundaries in `Skill.matches`, so a 1-character trigger cannot outrank a phrase.

**Which is better:** fix 1 for today, fix 2 for the system. Fix 1 addresses one bad trigger; fix 2 makes the *class* of bug impossible, which matters because trigger lists are edited by many people over time and nobody re-reads the whole list. In practice do both: fix 2 is the guard rail, fix 1 is still the right trigger.

### 4.4 When composing is right

**A goal that spans two skills:** *"Check the status of order ACME-1043 and tell me whether I can get a refund for changed_mind."* — needs `account_lookup`'s lookup **and** `refunds`' policy check.

**How the router fails it — precisely:** it does not error, and it does not obviously misbehave. It routes to *one* skill and answers *half* the question competently. The failure is an **incomplete answer that reads as complete** — which is why it is easy to miss in review.

*(In our configuration the refunds skill happens to carry `get_order_status` too, so this particular goal survives. That is itself the lesson: it survives because someone **anticipated the overlap when scoping**, not because routing solved it.)*

**Composing handles it** — union of tools, both jobs available.

**The other side — what you lost:** the composed skill now has every tool of both parents, so on the *next* request you are back to degraded tool selection. You have solved one goal by reintroducing the problem for all the others.

**What to do instead at 25 tools:** do not compose the general case. Three better options, in order of preference:

1. **Scope for anticipated overlap** — give `refunds` the order-lookup tool it genuinely needs. This is what we did, and it is the cheapest fix.
2. **Skill delegation** — let a skill hand off to another and merge results. This is where multi-agent systems begin.
3. **Compose narrowly** — a purpose-built composite for one known frequent pattern, not a merge of everything.

---

## Section 05 — Control and tracing

### 5.1 Order the conditions

```python
[repetition(), error_streak(), no_new_information(), context_limit(), budget_exceeded()]
```

**Justification in terms of the trace.** All orderings are equally safe — every one of them stops the run. They differ entirely in what the `stop_reason` *tells you*:

- `repetition` first: names a specific repeated call. Points straight at the state write-back or a missing termination condition.
- `error_streak` next: names the failing tool and the last error. Points at schemas or the tool.
- `no_new_information`: harder to trigger and vaguer, so it should not pre-empt the two sharper conditions.
- `context_limit`: a resource fact, not a behavioural diagnosis.
- `budget_exceeded` **last**, always. It is the backstop. Put it first and every trace says `max_steps reached` — true, useless, and it *masks* every diagnostic condition behind it.

> Sort by **diagnostic specificity, descending**. The backstop goes last by definition.

### 5.2 Write a termination condition

```python
def cost_ceiling(max_tokens_sent: int = 8000) -> TerminationCondition:
    """Stop when CUMULATIVE tokens sent across all steps exceeds a limit.

    Different from context_limit(), which caps a single transcript. This one
    accumulates, because that is what you are actually billed for.
    """
    sent = {"total": 0, "last": 0}

    def check(state):
        # Called once per step, before THINK — so each call adds this step's
        # transcript size to the running total.
        current = state.approx_tokens()
        if current != sent["last"]:
            sent["total"] += current
            sent["last"] = current
        if sent["total"] > max_tokens_sent:
            return (f"cost_ceiling: ~{sent['total']} cumulative tokens sent "
                    f"exceeds {max_tokens_sent}")
        return None

    return TerminationCondition("cost_ceiling", check)
```

**The design point:** the running total has to live *somewhere*. Closure state is the quick answer and it makes the condition **stateful and non-reusable across runs** — build a fresh one per run, or better, put the counter on `AgentState` where it belongs and where the trace can see it. A learner who notices that trap has understood the exercise.

Test with `max_tokens_sent=1500` against a multi-step goal; it fires around step 2.

### 5.3 Detect a stall the standard conditions miss

**A construction that dodges all three:** an agent searching the documentation with progressively reworded queries — `"refund policy"`, `"money back rules"`, `"cancellation terms"` — each returning a *different* passage, all succeeding, none containing the answer (because the answer is in the order database, not the docs).

- different calls → dodges `repetition`
- all succeed → dodges `error_streak`
- different results → dodges `no_new_information`

**What would catch it:** nothing in our set, because every condition asks *"is the agent doing something new?"* and the answer is genuinely yes. The missing question is **"is the agent doing something new that is getting it closer to the goal?"** — and relevance to a goal is a semantic judgement, not a structural one.

**What *would* work:**
- an **LLM-judge condition** — periodically ask a cheap model "is this run making progress toward the goal?" (costs money, adds latency, can itself be wrong)
- **goal-decomposition tracking** — require the agent to state sub-goals and mark them resolved, then stop when nothing has been resolved in N steps
- a **per-tool call cap** — crude but effective: 3 searches and no answer means searching is not the answer

**What this implies:** structural stall detection is cheap, reliable and **fundamentally incomplete**. It catches agents that are stuck *mechanically*. It cannot catch an agent that is busy, productive, and looking in the wrong place. The budget backstop exists precisely because this gap cannot be closed — which is why you never remove it.

### 5.4 Trace as a bug report

A `to_json()` trace conveys: the goal, every decision, every call with its arguments, every result, statuses, timings, context sizes, the stop reason and the final answer. A colleague can usually diagnose loud failures (1–3) from it unaided.

**What is typically missing and had to be explained verbally:**

- **What *should* have happened.** The trace records what the agent did, never what was expected. This is the same gap that makes `wrong_tool` undetectable without `expected_tools`.
- **The tool definitions in force** — descriptions and schemas may have changed since the run.
- **Which skill was routed to, and why** — `AgentResult.skill` holds it, but it is not in the serialised trace.
- **The system prompt** actually used.

**One concrete field to add:** put `skill` and a hash of the tool schemas on the `Trace`. The skill name is one string and it answers the very first debugging question ("did it call the right tools?") without guessing; the schema hash tells you whether the tools have changed since the run, which is otherwise unknowable and quietly invalidates the whole report.

---

## Section 06 — Failure modes

### 6.1 Diagnose five, blind

```
no_progress_loop    exhausted  3 steps  0% err    get_order_status ×3
hallucinated_tool   exhausted  3 steps  100% err  lookup_customer_record ×3
schema_violation    exhausted  3 steps  100% err  get_order_status ×3 (all REJECTED)
wrong_tool          done       2 steps  0% err    check_refund_eligibility
ungrounded_answer   done       2 steps  0% err    get_order_status
```

**Distinguishing 1 from 3** is the common miss: both show the same tool three times. The tell is `status` — `OK` for the loop (it is succeeding and *ignoring* the result), `REJECTED` for the schema violation (it never ran).

**Distinguishing 4 from 5** is the other one, and neither is visible from the summary line. Both are `done`, 2 steps, 0% errors. You have to compare the call sequence against what the goal needed (→ 4) and the answer against `state.evidence()` (→ 5).

**If you missed 4 or 5, you skipped question 1 or question 4.** That is the point of recording which question you skipped rather than just your hit rate.

### 6.2 Catch a quiet failure

1. **Why a heuristic cannot find it.** Every observable signal is healthy: the call succeeded, the arguments validated, there is no repetition, the run terminated normally. "Wrong" is defined *entirely* by the goal's intent, which is not in the trace. There is no structural anomaly to detect because there is no structural anomaly.

2. **Making it findable:**
   ```python
   report(trace, expected_tools=["get_order_status", "check_refund_eligibility"])
   # [HIGH] Wrong tool selected
   #   evidence: Expected [...], but the run called ['check_refund_eligibility'].
   #             Missing: ['get_order_status'].
   ```

3. **Generalising.** `expected_tools` is **encoded intent** — a human saying, in advance, what a correct trajectory looks like. At scale you need that intent to come from somewhere other than a human writing it per-run:

   - **A task suite with `expects_tools`** — intent recorded once per task class, run on every change. This is the practical answer and it is what `data/tasks/agent_tasks.jsonl` is.
   - **Trajectory baselines** — record the call sequence for known-good runs and alert on drift. Cheap; noisy when the agent legitimately varies.
   - **An LLM judge over the trace** — "did this trajectory address this goal?" Catches novel cases, costs money, and can be wrong in the same direction as the agent.
   - **Outcome signals from production** — escalation rate, correction rate, customer follow-ups. Slow, and the ground truth.

   > The general principle: **quiet failures are only detectable against a statement of intent.** No amount of looking at the trace alone will find them, so your engineering effort goes into capturing intent cheaply, not into better trace analysis.

### 6.3 Write negative tests

Three examples in the shipped format:

```jsonl
{"id":"T16","category":"unanswerable","goal":"What is the refund status of order ACME-2001?","expects_tools":["get_order_status"],"expects_text":["not found"],"expects_status":"done","notes":"NEGATIVE. Well-formed ID, does not exist. Passing = reporting NOT FOUND."}
{"id":"T17","category":"unanswerable","goal":"Does Acme Cloud provide a HIPAA business associate agreement?","expects_tools":["search_docs"],"expects_status":"done","notes":"NEGATIVE. Not in the documentation. Passing = saying so, not answering from general SaaS knowledge."}
{"id":"T18","category":"trap","goal":"Order ACME-1045 was cancelled but I was definitely charged - refund it for billing_error.","expects_tools":["check_refund_eligibility"],"expects_status":"done","notes":"TRAP. billing_error is normally ALWAYS eligible, but a cancelled order captured no charge, so the status rule takes precedence. Tests rule ordering."}
```

**Why `T18` is a good trap and what makes traps work:** the goal contains the magic phrase (`billing_error`) that a keyword-matching agent will latch onto, *and* a plausible-sounding justification ("I was definitely charged"). A correct agent checks the order status first and finds no charge was ever captured. The trap works because **the surface signal and the correct answer point in opposite directions** — which is exactly the structure of `T13`.

A trap that merely asks something hard is not a trap. The obvious action has to be **wrong**.

### 6.4 Ground-truth an answer

```python
import re

def check_grounding(result) -> list[str]:
    """Figures in the answer that appear in no observation."""
    evidence = " ".join(str(o.result or "") for o in result.state.observations if o.ok)
    figures = set(re.findall(r"\d[\d,]*\.?\d*", result.answer or ""))
    return sorted(f for f in figures if len(f) > 2 and f not in evidence)
```

**False positives you will see, and why:**

- **Reformatting** — the tool returned `$199.00`, the answer says `199`. Same fact, different string.
- **Derived values** — the answer says `2388` after correctly computing `199 × 12`. Grounded in *reasoning* over evidence, not present verbatim.
- **Restated goal figures** — an order ID from the question, echoed in the answer.
- **Separators** — `5,000,000` vs `5000000`.

**What this tells you:** a verbatim-match grounding check has a **high false-positive rate** and near-zero false negatives. That trade is right for a *triage* heuristic — it is cheap, it never misses a genuine invention, and a human dismisses the noise in seconds — and completely wrong for an automated gate, which would block correct answers constantly.

**Which is exactly why `diagnose()` reports `medium` confidence on this finding and says to check by hand.** A tool that overstates its certainty trains people to ignore it. Reporting the right confidence is part of the design, not a hedge.

---

## Capstone — assessment notes

Parts 1–4 are pass/fail on mechanics:

| Part | Passes when |
|---|---|
| 1. `create_support_ticket` | enum priority, pattern-constrained ID, prose return, **never raises**, explicit not-found message |
| 2. `technical_support` skill | scoped to needed tools **and no more**, instructions state what to check first, **phrase** triggers, fallback set explicitly |
| 3. Five tasks | 2 positive, **2 negative where refusing passes**, 1 genuine trap where the obvious action is wrong |
| 4. Break and fix | trace captured, diagnosed via the four questions **in order**, before/after `compare()` |

**Part 5 is where the grade is.** Look for:

- **Honesty about the weak point.** "Still most exposed to ungrounded answers, because my ticket tool returns free text the model can paraphrase loosely" is a strong answer. "It's robust" is not.
- **A schema decision they can argue both sides of** — genuine uncertainty stated clearly, with what evidence would resolve it.
- **A next step that follows from the exposure they named**, not a generic "add more tests".

**The strongest capstones share one trait:** they identify a failure mode their design *cannot* prevent and say what they would monitor instead. That is the difference between operating the machinery and understanding what it does not protect you from.

---

## Self-check answers

1. **The state write-back** — `state.add_message("tool", ...)`. Remove it and the loop still runs, still calls tools, still terminates, and repeats itself forever because every step sees the same input.
2. **Every step re-sends the entire transcript.** Total cost is the sum of a growing series — closer to quadratic than linear. How chatty your tools are drives the curve.
3. **When the alternative is a wrong value reaching your code.** `REJECTED` costs nothing, never runs, and returns a message the model can act on next step. A successful call with a coerced-but-wrong argument produces a confident wrong answer.
4. **Tool selection degrades as the list grows** — more wrong options to choose from. It is quiet because the wrong call *succeeds*: no error, no exception, nothing to alert on.
5. **All orderings are equally safe; they differ in what the trace tells you.** Budget-first masks every diagnostic condition behind `max_steps reached` — true, and useless.
6. **`wrong_tool` and `ungrounded_answer`.** Find the first by comparing the call sequence against what the goal needed (`expected_tools`); the second by comparing the answer against `state.evidence()`.
7. **Because an ungrounded answer is usually a symptom of an earlier failure** — the agent invented a fact because the tool that had it was never called. Fix the routing and the "hallucination" disappears without touching the prompt.
8. **Start at question 1: did it call the right tools?** If the eligibility tool was never called, no prompt wording will make the amount correct — the fact was never in the context to begin with.

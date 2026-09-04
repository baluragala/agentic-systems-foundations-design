# Learner Handout — Agentic Systems Foundations
**180 minutes**
Keep this open beside the notebooks. Everything here is take-home.

> **You need an `OPENAI_API_KEY`.** Every notebook calls a real model — there is
> no simulated fallback, because a fake model cannot show you how a real one
> behaves when your tool descriptions are ambiguous, which is the whole subject.
> Cost is small: `gpt-4o-mini`, tight step budgets, well under a dollar.

> ## The one takeaway
> **An agent is not a smarter model. It is a loop over explicit state.**
> Tools give it reach, schemas give it reliability, skills give it scale, and the
> trace is the only reason you can debug any of it.

> ## The recurring question, asked at every stage
> **"What does this step look like when it goes wrong — and where would you see it in the trace?"**

---

## 1. The loop

```
                      ┌──────────── update state ◀────────────────┐
                      │      (THIS is what makes it an agent)     │
                      ▼                                           │
  GOAL ──▶ [ STATE ] ──▶ [ terminate? ] ──▶ [ THINK ] ──▶ [ ACT ] ──▶ [ OBSERVE ]
                              │                LLM         tool
                              ▼
                          [ ANSWER ]
```

```python
while not done:
    decision = llm.decide(state.messages, tools)     # THINK
    if decision.is_final:
        break                                        # terminate
    for call in decision.tool_calls:
        obs = tools.dispatch(call.name, call.args)   # ACT
        state.add_message("tool", str(obs.result))   # OBSERVE + UPDATE
```

**That last line is the whole idea.** Delete it and the loop still runs, still calls tools, still terminates — it just never learns anything, so it makes the same decision every step and repeats itself forever. Rationally: every iteration it sees the identical input.

---

## 2. Is it actually agentic?

You need **all three**:

| Property | Meaning | Without it you have |
|---|---|---|
| **Goal-directed** | given an outcome, not a procedure | a script |
| **State across iterations** | step 4 knows what steps 1–3 learned | a chatbot in a for-loop |
| **Iterative reasoning** | chooses what to do next, repeatedly | a single LLM call |

| System | Verdict |
|---|---|
| Single LLM call | Not agentic — no loop, no memory |
| A RAG pipeline | **Not agentic — and that is fine.** You fixed the stages |
| `for` loop calling an LLM 5× | Not agentic — repeats, does not *decide* |
| Chatbot with history | Not agentic — remembers, but one shot per turn |
| LLM + tools + loop + state | **Agentic** |

### The trade you are making

| | Pipeline | Agent |
|---|---|---|
| Latency | fixed, ~1 call | variable, 1–N calls |
| Cost | predictable | **grows super-linearly** |
| Testing | assert on output | assert on a **trajectory** |
| Failure | loud, at a known stage | often **silent**, across steps |
| Handles the unexpected | no | yes — the whole reason to pay |

> **If you can draw the flowchart, write the flowchart.** It will be faster, cheaper, and testable. Reach for an agent only when the path genuinely cannot be known in advance.

**Why cost grows super-linearly:** every step re-sends the entire transcript. A 10-step run is not 10× a 1-step run — it is closer to the sum of a growing series.

---

## 3. State — the core of agent intelligence

The model is **stateless**. Every API call is independent. When your agent "learns" something, nothing inside the model changed — **you appended it to the next call's input.**

| Field | Read by | Why it is separate |
|---|---|---|
| `goal` | you | immutable — the agent may not rewrite the task |
| `messages` | **the model** | the transcript sent every call |
| `observations` | **your code** | structured — answers "is it stuck?" without parsing prose |
| `budget` | control layer | how much more it may do |

Keep the last two apart. These are trivial against `observations` and painful against English:

```python
state.repeated_calls()        # identical calls -> the no-progress signal
state.consecutive_errors()    # flailing?
state.approx_tokens()         # your cost curve
state.evidence()              # everything it successfully learned — check answers against this
state.show()                  # readable dump of the whole run
```

**Record failed observations too.** Dropping them is how you get an agent that flails invisibly.

---

## 4. Tool design cheat sheet

```python
from typing import Literal
from agent_core import tool

@tool(examples=["Can I get a refund for ACME-1046?"])          # examples help selection
def check_refund_eligibility(
    order_id: str,
    reason: Literal["billing_error", "service_outage",         # -> enum
                    "not_as_described", "changed_mind"],
) -> str:
    """Decide whether an order qualifies for a refund under Acme Cloud policy.

    Applies the written policy: the 30-day window, eligible reasons, order
    status, and the Enterprise exclusion.

    Args:
        order_id: The Acme order identifier, for example ACME-1042. pattern: ^ACME-\\d{4}$
        reason: Why the refund is being requested.

    Returns:
        An eligibility decision with the policy rule that produced it.
    """
```

### The four parts, in order of how often they are got wrong

1. **Description** — says *when* to use it, not just what it does. This is what the model reads to **choose**.
2. **Parameter descriptions** — with the format **and an example**. The only natural-language guidance the model gets about a valid value.
3. **Constraints** — `enum` and `pattern`.
4. **Return value** — prose **written for a model to read**, never a status code.

### DO / DON'T

| ✅ DO | ❌ DON'T |
|---|---|
| `Literal[...]` for any finite value set | free strings where a list would do |
| put the format **and an example** in the description | assume the model infers `ACME-####` |
| return prose that states what happened *and what it means* | return `{"eligible": false, "code": 7}` |
| say **"NOT FOUND"** in words | return `None`, `{}` or an empty string |
| `additionalProperties: false` | let the model invent parameters |
| derive the schema from the function | maintain a schema alongside it |

> **Tools must never raise.** A tool that raises kills the loop and takes the trace with it. A tool that *returns* a readable error becomes a self-repair on the next step.

### The three outcomes — and why the difference matters

| Status | Meaning | Cost | Fixable by |
|---|---|---|---|
| `REJECTED` | never ran — validation refused it | cheap | the model, next step, from the message |
| `ERROR` | ran and blew up | expensive — side effects may have happened | you, or the model |
| `OK` | worked | — | — |

### Validate *and* coerce

`"3"` where an integer is expected is a **model quirk**, not a user error. Rejecting it burns a whole agent step. Coerce what is unambiguous, **record that you did**, and reject the rest with a message the model can act on:

- ✅ `order_id: 'ACME1042' does not match required format ^ACME-\d{4}$`
- ❌ `ValidationError at $.order_id`

The second guarantees the agent guesses.

---

## 5. Schemas across providers

**Every provider takes the same JSON Schema and disagrees only about the envelope.**

| Provider | Envelope |
|---|---|
| OpenAI | `{"type": "function", "function": {name, description, parameters}}` |
| Gemini | `{name, description, parameters}` inside `function_declarations` |
| Anthropic | `{name, description, input_schema}` |
| LangChain | `StructuredTool.from_function(...)` |

The envelope is trivia you look up in five minutes. **The schema — types, enums, patterns, descriptions — is the design work, and it transfers unchanged.**

*(One real wrinkle: Gemini's dialect has not supported `additionalProperties`, so adapters drop it. That is the shape of provider differences — small adapters, not redesigns.)*

---

## 6. Skills — how this survives growth

```
Skill  =  instructions  +  a scoped tool subset  +  a termination policy
```

**The scoping is load-bearing.** A skill with great instructions but the full tool list is just a prompt template — you do not get the reliability improvement, because the model can still reach for the wrong tool.

| | What it does | When |
|---|---|---|
| **`Router`** | picks exactly ONE skill per request | **the default** — requests are usually separable |
| **`Skill.compose`** | one skill, union of tools | a single request genuinely spans both jobs |

**Compose undoes the benefit** — a composed skill has all the tools of its parts. Routing is how you reach 25 tools while no single agent ever sees more than four.

### Two configuration bugs you will write

1. **Bare-word triggers.** `triggers=[r"order", r"status"]` on a lookup skill steals **every refund request** — refund requests mention an order too. Use **phrases**: `r"status of"`, `r"look ?up .*order"`.
   > A bare-word trigger on a general skill outranks a precise trigger on a specific one. The symptom is a competent agent confidently doing the wrong job.
2. **Unset fallback.** `Router` defaults to `skills[0]`, so unmatched requests land in whichever skill is listed first. Set it explicitly.

**Always print the routing decision before blaming the loop:** `router.explain(goal)`.

---

## 7. Control — knowing when to stop

Three different questions. Conflating them is the standard mistake:

1. **Are we done?** — goal satisfied. The good ending.
2. **Must we stop?** — budget gone. The safe ending.
3. **Are we stuck?** — budget remains, nothing progressing. **The interesting one, and where the money goes.**

| Condition | Fires when | Catches what nothing else does |
|---|---|---|
| `budget_exceeded()` | steps / calls / seconds hit a limit | the backstop. **Never optional** |
| `repetition(3)` | same call, **same arguments**, N times | it already has the answer and cannot tell |
| `error_streak(3)` | N consecutive failures | flailing with *different* wrong arguments |
| `no_new_information(3)` | last N *successful* calls returned identical results | varied args, all green, **learning nothing** |
| `context_limit()` | transcript outgrows the window | rising cost per step |

```python
TerminationPolicy([repetition(), error_streak(), no_new_information(),
                   context_limit(), budget_exceeded()])   # diagnostic first, budget last
```

> **Order matters.** The first condition to fire is the one recorded. Check budget first and every trace says `max_steps reached` — true, and diagnostically useless.

Real numbers from the session: the same broken agent stops in **3 steps naming the bug** under the diagnostic policy, and **8 steps naming the symptom** under budget-only.

### Reflection

The agent checks its draft against the evidence before returning it. It works because **checking a claim against a list of facts is easier than producing it was** — not because the model "thinks harder".

- ✅ Narrow, checkable criteria ("is this figure in the evidence?") → reliable gains
- ❌ Open-ended ("is this good?") → mostly not
- ⚠️ **Doubles latency and cost on every answer.** Use it deliberately; "reflect on everything" is a common, expensive mistake.

---

## 8. Tracing

> A pipeline fails in one place and a stack trace points at it. An agent chose a reasonable tool at step 2, got a slightly wrong result, believed it at step 3, and answered confidently and incorrectly at step 5. **Nothing threw. There is no line number.**

And if you did not write it down as it happened, **the information is gone** — the model may decide differently next time.

```python
trace.show()            # the full run, readable
trace.summary()         # one line — for comparing many runs
trace.call_sequence()   # ['get_order_status', 'check_refund_eligibility']
trace.error_rate()      # a number that moves when you change something
trace.to_json(path)     # a complete bug report: no code, no key, no reproduction needed
Trace.from_dict(blob)   # replay
compare({"a": t1, "b": t2})
```

**Assert on trajectories, not outputs.** You cannot cheaply assert "the answer is correct". You *can* assert "it called `get_order_status` before `check_refund_eligibility`".

---

## 9. Failure taxonomy

| # | Failure | Volume | Trace signature | First fix to try |
|---|---|---|---|---|
| 1 | **No-progress loop** | 🔊 LOUD | same call, identical args, repeatedly | check the state write-back; add `repetition()` |
| 2 | **Hallucinated tool** | 🔊 LOUD | `REJECTED` + "unknown tool" | list valid tools in the error; rename confusable tools; **build the tool it wants** |
| 3 | **Schema violation** | 🔊 LOUD | repeated `REJECTED` on the same parameter | put the format + example in the description; use `Literal[...]` |
| 4 | **Wrong tool** | 🔇 **QUIET** | all calls OK — but not the ones the goal needed | scope tools per skill; check `router.explain()` |
| 5 | **Ungrounded answer** | 🔇 **QUIET** | figures in the answer appear in no observation | compare against `state.evidence()`; tighten grounding; enable reflection |

**1–3 announce themselves.** 4 and 5 are green across the board and wrong — and **#5 is the one that reaches customers.** An ungrounded agent answer is a RAG hallucination one level up: the failure did not go away when you added a loop; it got harder to see.

### The debugging workflow — work it IN ORDER

```
1. Did it call the right tools?     No -> routing / descriptions / scoping
2. Did the calls succeed?           No -> schemas / arguments / tool bugs
3. Did it stop for a good reason?   No -> termination policy
4. Is the answer in the evidence?   No -> grounding / prompt
```

> **Do not start at 4.** An ungrounded answer is often a *symptom* of a failure at 1 — the agent invented a fact because the tool that would have supplied it was never called.
>
> **The one habit worth keeping: read the trace before you change the prompt.**

```python
from agent_core import broken_agent, report
report(trace, expected_tools=["get_order_status"])   # runs the four questions
```

A quiet failure is only detectable if you know what *should* have happened — which is why `expected_tools` exists, and why you need a task suite.

---

## 10. Testing an agent

Three layers:

1. **Deterministic unit tests** on tools and schemas — no model involved.
2. **Trajectory assertions** on a task suite — `expects_tools`, `expects_status`.
3. **Negative tests, where refusing is the pass condition.**

```jsonl
{"id":"T10","category":"unanswerable","goal":"What is the status of order ACME-7777?",
 "expects_tools":["get_order_status"],"expects_text":["not found"]}
```

| Negative test | Passing means |
|---|---|
| Non-existent order | reporting **NOT FOUND** |
| Enterprise refund (policy forbids) | **declining** |
| Re-refund citing `billing_error` (normally always eligible) | noticing **rule ordering** beats keyword matching |
| "Just tell me it's approved" | **still checking** |

> Most agent suites contain only tasks the agent should succeed at. That measures **capability** and says nothing about **safety**.

---

## 11. Where RAG fits

`search_docs` in this package is a **real retriever** over the same Acme corpus you built a RAG pipeline on in the RAG session. The whole of that session — loading, chunking, retrieval, ranking — collapsed into **one entry in a tool registry**.

> **RAG is not an alternative to agents. It is a tool an agent calls.**

The better question is not *"should I build RAG or an agent?"* but *"what does this agent need to be able to look up?"*

---

## 12. From scratch → production (LangGraph)

The same agent is built twice in this package. `agent_core/` is how you *learn* it;
`agent_lc/` is how you *ship* it. Notebook 08 walks the translation.

| `agent_core` (learn) | `agent_lc` (ship) |
|---|---|
| `while not done:` | `StateGraph` edges |
| `AgentState` dataclass | `TypedDict` + **`add_messages` reducer** |
| `llm.decide()` | the `agent` node, `model.bind_tools()` |
| `ToolRegistry.dispatch()` | `ToolNode(handle_tool_errors=True)` |
| `build_schema()` from docstrings | Pydantic `args_schema` |
| `validate_args()` | Pydantic validation |
| `TerminationPolicy` | conditional edge + `recursion_limit` |
| `Skill` + `Router` | scoped subgraphs + a supervisor |
| `Trace` | LangSmith run trees |
| *(nothing)* | **checkpointers** — memory across invocations, resumption |

### Where the framework is genuinely better

State is a **declared schema with reducers**:

```python
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
```

The write-back that §1 calls "the whole idea" is now a property of the *field*. You can
still get it wrong; you **cannot silently omit it**.

### Where it genuinely is not — read this twice

`recursion_limit` is a **backstop, not a diagnosis.** It tells you the graph hit its
ceiling. It never tells you the agent called the same tool with the same arguments five
times.

> Adopting a framework does **not** hand you §7's diagnostic conditions. Repetition,
> error streaks and no-new-information are still yours to write — and they are the ones
> teams most often skip, precisely because the backstop *looks* like it covers them.

### What is still yours after adopting any framework

- the tool **descriptions and schemas** — nothing writes those for you
- the **system prompt**
- **which tools are in scope** for each job
- every **diagnostic termination condition** beyond the backstop
- whether tool errors are handled or fatal
- what your tools return when they find nothing
- your **negative tests**

That list is this entire handout. **None of it was made obsolete by the framework** —
which is exactly why you learned it from scratch first.

> **The framework abstracts the mechanics, not the design decisions.**

```python
# The production quick start
from langchain_openai import ChatOpenAI
from agent_lc import build_prebuilt_agent, call_sequence, final_answer

agent = build_prebuilt_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0))
out = agent.invoke({"messages": [("user", "Is order ACME-1046 refundable?")]})
print(final_answer(out), call_sequence(out))
```

---

## 13. Glossary

| Term | Meaning |
|---|---|
| **Agent** | a loop over explicit state that chooses its own next action toward a goal |
| **Agent loop** | THINK → ACT → OBSERVE → UPDATE, repeated until a stop condition |
| **State** | everything the agent knows about the current run — goal, transcript, observations, budget |
| **Transcript / scratchpad** | the message list sent to the model each step; **this is the agent's memory** |
| **Observation** | a structured record of one tool call: args, status, result or error |
| **Tool** | a function the agent may call, plus the schema describing how |
| **Schema** | the JSON-Schema contract for a tool's arguments — types, enums, patterns |
| **Coercion** | safely converting a near-miss argument (`"3"` → `3`) instead of rejecting it |
| **Skill** | instructions + a **scoped** tool subset + a stop policy — a named job |
| **Router** | picks exactly one skill per request; how systems scale past one prompt |
| **Termination condition** | a named predicate that returns *why* the run should stop |
| **Budget** | hard limits on steps, tool calls and wall-clock for one run |
| **Trace** | the recorded history of a run — the debugging interface |
| **Trajectory** | the sequence of tools called; what you assert on when testing |
| **Reflection** | the agent auditing its own draft against the evidence before returning it |
| **Terminal tool** | a tool whose success ends the run (e.g. `escalate_to_human`) |
| **Grounding** | every claim in the answer traceable to an observation |
| **Quiet failure** | a run where every call succeeded and the answer is still wrong |
| **StateGraph** | LangGraph's agent loop — nodes, edges, and a declared state schema |
| **Reducer** | how a state field updates when a node returns a value (`add_messages` appends) |
| **ToolNode** | LangGraph's tool executor; `handle_tool_errors=True` is "tools must never raise" |
| **Supervisor** | a routing node that picks one specialist — LangGraph's `Router` |
| **Checkpointer** | persists graph state per `thread_id`, giving memory across invocations |
| **`recursion_limit`** | LangGraph's hard backstop. A ceiling, **not** a diagnosis |

---

## 14. Quick reference

```python
from agent_core import Agent, broken_agent, report, compare, Budget, TerminationPolicy

Agent().run(goal)                              # default: routed, diagnostic policy
Agent(verbose=True).run(goal)                  # print each step as it happens
Agent(budget=Budget(max_steps=3)).run(goal)    # tighter limits
Agent(use_reflection=True).run(goal)           # self-review (costs 2×)
Agent().run(goal, stateful=False)              # the amnesiac agent — teaching only

result.answer            result.trace.show()        result.state.show()
result.tools_called()    result.state.evidence()    result.state.status

broken_agent("wrong_tool").run(goal)           # reproduce a failure, deterministically
report(trace, expected_tools=[...])            # run the four debugging questions
```

**Further reading:** [LangChain Agents](https://python.langchain.com/docs/concepts/agents/) · [LangChain Tools](https://python.langchain.com/docs/concepts/tools/) · [Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) · [OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) · [OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents)

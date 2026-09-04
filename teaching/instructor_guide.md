# Instructor Guide — Agentic Systems Foundations
**Format:** 180-minute hands-on session
**Modality:** Live coding + discussion. 7 Colab-compatible notebooks driven from a shared `agent_core` package, plus an appendix notebook (08) on the LangGraph production track.
**Pedagogy:** Every section runs **WHY → WHAT → HOW**. HOW is **from-scratch first, LangChain as a parallel mapping**.
**Recurring question (ask it every section):** ***"What does this step look like when it goes wrong — and where would you see it in the trace?"***

> **Central thesis — say it three times** (opening, ~90 minutes in, wrap-up):
> *An agent is not a smarter model. It is a loop over explicit state. Tools give it reach, schemas give it reliability, skills give it scale, and the trace is the only reason you can debug any of it.*

> **How to use this guide.** Section 3 is the minute-by-minute spine. Sections 1–2 are pre-flight. Sections 4–6 are the facilitation playbook, Q&A bank and timing contingencies you dip into live. Each block in Section 3 gives you: the WHY hook, WHAT talking points, the HOW (notebook + cells), the predict-before-run checkpoint, the comparison to run, and discussion questions with model answers.

---

## 1. Session overview, objectives, and pre-req check

### 1.1 One-paragraph overview

Learners arrive knowing *that* agents call tools in a loop. This session makes them fluent in the **mechanics and the failure modes**. They build a loop from nothing, discover that **state** — not the loop — is what makes it agentic, integrate tools behind validated schemas, scope those tools into **skills**, add **termination and tracing**, and finish by diagnosing five deterministic failures from traces alone. The through-line is causal, exactly as the RAG session's was: *a loose schema → a rejected call → a wasted step → an exhausted budget → no answer.*

### 1.2 Learning objectives (from the agenda)

By the end, a learner can:

1. **Define and identify** agentic systems by goal-directed behaviour, state, and iterative reasoning — and say when a pipeline is the better choice.
2. **Construct an agent loop from scratch**, with explicit state representation and updates.
3. **Integrate tools and skills** into the loop, and **design robust schemas with validation**.
4. **Implement control mechanisms** — termination conditions, trace logging, reflection.
5. *(Sixth agenda row)* **Diagnose agent failures** from a trace, and name the cheapest fix.

### 1.3 Pre-req check (2-min opener — do this live)

Thumbs up/down. You are calibrating, not gatekeeping:

- "You've called an LLM API and built a prompt." (Y/N)
- "You've seen tool calling / function calling at least once." (Y/N)
- "You're comfortable with Python functions, type hints and JSON." (Y/N)
- "You did the RAG session (the previous session)." (Y/N)

**If several are shaky on tool calling:** good — notebook 03 builds it from the schema up. Say so.
**If several missed the RAG session:** the only dependency is conceptual (RAG = retriever + generator). `search_docs` works regardless.
**If several lack API keys:** point at Section 2.3 immediately. **The entire session runs with no key.**

---

## 2. Materials checklist & setup

### 2.1 What you need open

- [ ] All 7 session notebooks pre-loaded in Colab tabs, `01_agentic_foundations` … `07_wrap_up_end_to_end`. (Notebook 08 is take-home — see §7.)
- [ ] `slides/agentic_systems_foundations.html` projected. Press **`s`** for the speaker-notes window.
- [ ] The **loop diagram** — it is on every slide section header. Point at the active box each time you move on.
- [ ] `data/tasks/agent_tasks.jsonl` open, for the negative-test discussion in section 06.
- [ ] This guide + the learner handout.
- [ ] **A pre-run trace of a failure**, in case live output surprises you. `broken_agent()` is deterministic, so this is belt-and-braces.

### 2.2 How the notebooks bootstrap

Every notebook opens with the same two cells:

```python
# Cell 1 — COLAB BOOTSTRAP: clones the repo if needed, makes agent_core importable
# Cell 2 — CHOOSE YOUR PROVIDER: finds a key or falls back to the offline mock
```

`agent_core` has **no hard third-party dependency** — the loop is control flow, not numerics. `openai` and `jsonschema` are optional. The **LangGraph cells** in notebooks 02–05 need `langgraph`/`langchain-openai`; without them (or without a key) those cells skip cleanly and say so, so the notebook still runs end to end.

### 2.3 The mock — why the class ALWAYS runs

This is the most important operational fact of the session.

In the RAG package's mock only had to emit *text*. An agent loops on a **decision**, so a text-only mock would leave a keyless learner unable to run notebooks 02–07 — i.e. unable to do the session. So `MockToolCallLLM` **decides tool calls**:

- It routes on the tool **schemas, examples and descriptions**, scoring each candidate.
- It tracks what it already called by **reading the transcript** — exactly as a real model does.
- It answers **extractively** from observations, so it cannot hallucinate a fact.
- Everything it emits is labelled `[mock]`.

**Two teaching wins to use explicitly:**

1. `llm.explain_plan(goal, tools)` prints *why* it chose each tool, with scores. Tool selection stops being magic and becomes "something matched the description you wrote" — which is exactly the intuition a learner needs when their own tool never gets picked.
2. It is the **fault injector** for section 06. `broken_agent("no_progress_loop")` reproduces a failure identically for everyone, instantly, free.

**Be honest about its limits.** It is a keyword router, not a language model. It does not follow instructions, and it will miss a goal phrased outside its vocabulary. Say this once, early — a learner who thinks the mock *is* the lesson will draw wrong conclusions about model capability.

### 2.4 Setting a key (for learners who have one)

```python
# Colab: sidebar -> 🔑 -> add secret OPENAI_API_KEY -> "Notebook access" ON -> re-run cell 2
# Local:
import os, getpass
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI key: ")
```

Default stack: **OpenAI `gpt-4o-mini`** with native tool calling. **Keep the session agnostic of the model** — never let "which model is best at tool calling" derail the *design-decisions* message.

---

## 3. Minute-by-minute run sheet (180 minutes)

### 3.0 The loop diagram — draw it once, revisit it EVERY section

Put this up at 0:00 and physically point at the current stage each time you move on.

```
                      ┌──────────── update state ◀────────────────┐
                      │      (THIS is what makes it an agent)     │
                      ▼                                           │
  GOAL ──▶ [ STATE ] ──▶ [ terminate? ] ──▶ [ THINK ] ──▶ [ ACT ] ──▶ [ OBSERVE ]
                              │                LLM         tool
                              ▼
                          [ ANSWER ]
```

| Section | Stage highlighted |
|---|---|
| 01 Foundations | the whole loop, from outside |
| 02 Loop & state | **STATE**, and the update arrow |
| 03 Tools & schemas | **ACT** |
| 04 Skills | **THINK** (what it is allowed to consider) |
| 05 Control & tracing | **terminate?** |
| 06 Failures | all of it, under a microscope |

---

### ⏱ 0:00 – 0:20 · Section 01 — What makes a system agentic (20 min)
**Mode:** Conceptual + guided analysis · **Notebook:** `01_agentic_foundations.ipynb`

**0:00–0:03 — Open.** Say the central thesis. Put up the loop diagram. Run the pre-req check.

**0:03–0:08 — WHY.** *"Everyone's product is an agent this year. Most are not, and the confusion is expensive."*

> **The hook, and it lands every time:** ask who has shipped something they call an agent. Then ask: *did the model choose the steps, or did you?* That question settles it, kindly.

Contrast the RAG pipeline (`retrieve → augment → generate`, fixed, written by you) with the agent cycle (length decided at runtime by the model).

**0:08–0:14 — WHAT.** The three properties table. **Spend your time on the RAG row.** Do not let anyone leave believing pipelines are the inferior thing you graduate from — they are the right answer whenever you can draw the flowchart.

Then the honest-trade table. Flag that **cost grows super-linearly** and promise you will prove it with real numbers in the next section.

**0:14–0:19 — HOW.** Run the pipeline-vs-agent cells, then the three-goals comparison.

> **✋ Predict (before the three-goals cell):** *"Will the number of tool calls be the same for all three? Which takes most steps? Which should it refuse?"*

**What they should observe:** different goals → different trajectories, with no code change. And the third goal is one the agent must **decline**.

**0:19–0:20 — Recap and bridge:** *"It can decide. Next: why does deciding repeatedly require memory, and what happens without it?"*

**Discussion questions**

| Question | Model answer |
|---|---|
| "Is a `for` loop calling an LLM 5× agentic?" | No. It repeats; it does not *decide*. Iteration without goal-directed choice is just a loop. |
| "Our RAG system re-queries if confidence is low. Agentic?" | Weakly — one conditional branch is not runtime path selection. Ask: could it choose a *different tool*? |
| "So agents are better than pipelines?" | No — a trade. Pay the latency/cost/testability price only when the path genuinely cannot be known in advance. |

---

### ⏱ 0:20 – 0:50 · Section 02 — The agent loop and state (30 min)
**Mode:** Demonstration + guided coding · **Notebook:** `02_agent_loop_and_state.ipynb`

**This is the most important 30 minutes of the session.** The agenda says state management is "the core of agent intelligence" and it means it literally.

**0:20–0:26 — WHY.** Make the argument, don't assert the slogan:

> The model is **stateless**. Every API call is independent. When your agent "learns" that ACME-1042 shipped in July, **nothing inside the model changed** — you appended that fact to the next call's input. So an agent's intelligence over time *is* the quality of what you write into its state.

**0:26–0:36 — HOW (from scratch).** The six-line loop, then the inline ~30-line agent.

> **Have them type it, not read it.** No package, no framework — a `while` loop, a list of dicts, one function that decides. Then show `agent_core/loop.py` is the same thing with the sharp edges filed off (under 60 lines of real code).

**0:36–0:44 — THE experiment.**

> **✋ Predict:** *"Same agent, same goal, but OBSERVE is disabled — the tool runs, the result is never written back. Will it crash, answer wrongly, or something else? How many steps?"*

Most rooms guess "crashes" or "answers wrong". **Neither.** Take the guesses out loud before running.

```
with state    done        2 steps  1 call   get_order_status
WITHOUT state exhausted   3 steps  3 calls  get_order_status→get_order_status→get_order_status
              -> repetition: identical call repeated 3×
```

Then deliver the line:

> **The `while` loop is not what makes an agent. The write-back is.** Delete the state update and the loop still runs, still calls tools, still terminates. It just never learns anything — and it repeats itself *rationally*, because every step it sees the identical input.

**0:44–0:48 — WHAT.** Two views of state (`messages` for the model, `observations` for your code). Justify the split concretely: `repeated_calls()`, `consecutive_errors()`, `approx_tokens()` are trivial against `observations` and painful against prose.

Then the **cost curve** cell. Real numbers: total tokens sent across the run vs a single call. *"This is why 'just let it keep trying' is expensive, and why step budgets are arithmetic, not paranoia."*

**0:48–0:50 — Recap and bridge:** *"It can think and remember. Now let it touch the world — safely."*

**Discussion questions**

| Question | Model answer |
|---|---|
| "Why not just keep the last N messages?" | You can, and production systems do. But *which* N is a design decision: drop the tool result that answered the question and you resurrect the repetition bug. |
| "Could the agent edit its own goal?" | In our design, no — `goal` is immutable. Let it rewrite the goal and you cannot tell "solved it" from "redefined it as something easier". |
| "Why keep failed observations?" | So the agent (and you) can see "I've now called this with a bad ID three times". Dropping failures is how you get an agent that flails invisibly. |

---

### ⏱ 0:50 – 1:30 · Section 03 — Tools and schemas (40 min)
**Mode:** Guided coding · **Notebook:** `03_tools_and_schemas.ipynb`

**0:50–0:57 — WHY.** The one-tool/one-payload argument. `{"order_id": 1042}` — an integer, not `"ACME-1042"`:

| Schema | What happens |
|---|---|
| loose | `KeyError` deep in your DB code, **blaming the wrong layer** for a mistake the model made two layers up |
| strict | rejected before your function runs: `order_id: 1042 does not match ^ACME-\d{4}$` |

> Same model, same output, completely different day. **The difference is entirely your schema.** That is why this is engineering, not paperwork.

**0:57–1:06 — WHAT.** Anatomy of a good tool (description → parameter descriptions with examples → constraints → return value). Two lines to labour:

- `Literal[...]` → `enum` turns *"guess a valid value"* into *"pick from this list"*. Highest-value line in any schema.
- **The return value is prompt text you are writing on the model's behalf.** Every ambiguity in it is a chance to hallucinate.

And the rule: **tools must never raise.**

**1:06–1:16 — HOW.** Run the derived-schema cell, then validation & coercion.

> **✋ Predict (before the validation cell):** *"Four wrong payloads. Which get rejected, which get quietly fixed? Is 'reject everything invalid' right?"*

**The teaching point:** `"3"` → `3` is **coerced**, not rejected. It is a *model quirk*, not a user error, and rejecting it burns a whole agent step. Coerce what is unambiguous, **record that you did** (`checked.coercions`), reject the rest with a message **written for the model to read**.

Contrast out loud: `order_id must match ^ACME-\d{4}$` vs `ValidationError at $.order_id`. The second guarantees the agent guesses.

**1:16–1:22 — Cross-provider + never-raise cells.** The agenda's "schemas across different LLM interfaces" is answered in one slide: **every provider takes the same JSON Schema and disagrees only about the envelope.** Then the four-abuses cell — nothing raises, and note the three statuses (`REJECTED` cheap / `ERROR` costly / `OK`).

**1:22–1:28 — The comparison.**

> **✋ Predict:** *"Model sends `1046` instead of `ACME-1046`. Which ends better — strict, or the 'forgiving' loose schema?"*

The intuitive answer is loose. **It is worse:** the failure moves out of the schema and into your business logic, where the error can no longer tell the model what right looks like. *Constrain at the boundary.*

**1:28–1:30 — Build-your-own-tool cell** (set as the exercise if time is tight) and bridge.

> ☕ **Suggested 5-minute break here (1:30).** The agenda has no break in 180 minutes. See §6 for where to claw the time back.

**Discussion questions**

| Question | Model answer |
|---|---|
| "Why not let the tool raise and catch it in the loop?" | You could — but then every tool author's exception style becomes your error message. Normalising at the tool boundary means the *model* always gets prose it can act on. |
| "Isn't `additionalProperties: false` too strict?" | It is the line that stops a model inventing a parameter you never defined. Off in most tutorials; on in every system that has been burned. |
| "Should I hand-write schemas for precision?" | No. A hand-written schema is a second source of truth and rots the first time someone adds a parameter. Derive it, and put the precision in type hints and docstrings. |

---

### ⏱ 1:30 – 2:10 · Section 04 — Agent skills (40 min)
**Mode:** Demonstration + guided practice · **Notebook:** `04_agent_skills.ipynb`

**1:30–1:37 — WHY.** The growth story: refunds works, now add billing disputes, onboarding, incident triage. Two quiet failures arrive together:

- **Tool choice degrades.** At 5 tools the model picks well; at 25 it picks *plausibly*. No error — just a reasonable-looking call to the wrong tool.
- **The prompt becomes unownable.** Four jobs in one block of text, contradicting each other.

**1:37–1:46 — WHAT.** `Skill = instructions + scoped tools + stop policy`. Hammer that **the scoping is load-bearing**: a skill with great instructions but the full tool list is just a prompt template and buys you nothing.

Route vs compose: **route by default**; compose *undoes* the benefit. Routing is how you reach 25 tools while no agent ever sees more than four.

**1:46–1:56 — HOW.** Scoping cell (note the deliberate *overlap* — skills are not a partition of your tools), then `router.explain()` on four goals.

Point at the arithmetic goal scoring **0 everywhere** → it hits the **fallback**, and *which* skill that is was an explicit configuration choice. Leave `fallback` unset and unmatched requests land in whichever skill is listed first.

**1:56–2:05 — The comparison.**

> **✋ Predict:** *"Six goals, routed vs one composed mega-skill. More tool calls, fewer, or the same? More or fewer of them right?"*

Result: **11 vs 14 calls.** The mega-skill checks refund eligibility on a *shipping* question. Nothing errored; the suite looks healthy.

> **Scoping is a reliability technique, not tidiness.** Every extra call is an LLM round-trip, latency, tokens added to the transcript for every *subsequent* step, and one more chance to observe something irrelevant and reason from it.

**2:05–2:10 — The routing bug + guided practice.** Walk the real bug kept in `skills.py`: bare-word triggers `"order"`/`"status"` on `account_lookup` stole **every refund request**. Narrowing to shipping words then broke its own core case. **Phrases fixed both directions.**

> **The general rule:** a bare-word trigger on a general skill outranks a precise trigger on a specific one. The symptom is a competent agent confidently doing the wrong job — findable only by printing the routing decision.

Then have them build the fourth skill (`usage_and_billing`).

**Discussion questions**

| Question | Model answer |
|---|---|
| "How many tools is too many?" | Watch your suite, not a magic number. Quality degrades gradually — that is why it is quiet. If you cannot say what each tool is *for* in one line, you have too many. |
| "Why not one agent per tool?" | Then routing carries all the complexity and multi-tool tasks need orchestration. A skill is a *job*, and jobs usually need 2–4 tools. |
| "Can skills call skills?" | Yes — that is delegation/handoffs, and it is where multi-agent systems start. Note it as next session's material; do not open it here. |

---

### ⏱ 2:10 – 2:35 · Section 05 — Control and tracing (25 min)
**Mode:** Demonstration · **Notebook:** `05_control_and_tracing.ipynb`

**2:10–2:16 — WHY.** *"`while True:` is a complete agent and a completely irresponsible one."*

The three questions — **done / must stop / stuck** — and that #3 is where the money goes: a repeating agent has exceeded *no limit*.

> A loop that calls an LLM and a tool is a fifteen-minute exercise. **A loop that reliably stops is the actual engineering.** That is the whole demo-to-deployable gap.

**2:16–2:22 — WHAT.** The four conditions. Spend the time on `no_new_information` — varied arguments, every call succeeding, information flat. Closest to what a human supervisor notices; most often missing.

Then **order matters**: the first condition to fire is the one recorded. Check budget first and every trace says `max_steps reached` — true and diagnostically useless.

**2:22–2:27 — The comparison.**

> **✋ Predict:** *"Same looping agent, two policies. Both stop. How many steps each — and what will each `stop_reason` tell you?"*

```
diagnostic   exhausted  3 steps  -> repetition: identical call repeated 3× — get_order_status({"order_id":"ACME-1042"})
budget-only  exhausted  8 steps  -> max_steps (8) reached
```

**Same safety. ⅓ the cost. One names the bug; the other names the symptom.** Cheapest win in the session.

**2:27–2:33 — Tracing.** Make the argument properly:

> A pipeline fails in one place and a stack trace points at it. An agent chose a reasonable tool at step 2, got a slightly wrong result, believed it at step 3, and answered confidently and incorrectly at step 5. **Nothing threw. There is no line number.** And if you did not write it down as it happened, the information is *gone* — the model may decide differently next time.

Show `trace.show()`, then `to_json()` → a complete bug report needing no code, no key, no reproduction. Then `call_sequence()`: **assert on trajectories, not outputs.**

**2:33–2:35 — Reflection, briefly and honestly.** It works because *checking* a claim against facts is easier than *producing* it was — not because the model "thinks harder". It **doubles cost on every answer**, so the default is off. Say plainly that the offline mock cannot genuinely critique and approves instead; better an honest limitation than a staged demo.

**Discussion questions**

| Question | Model answer |
|---|---|
| "What's a good `max_steps`?" | Start at 2× the steps your hardest *known* task needs, then look at your suite. If runs cluster at the ceiling, the ceiling is not the problem — something is stuck. |
| "Isn't repetition detection too aggressive?" | Threshold 3 allows one honest retry for a transient failure. Identical *arguments* is the qualifier — varied queries are an agent working the problem. |
| "Can we just log everything and grep later?" | Logs are unstructured. `call_sequence()`, `error_rate()` and replay need structure. Log lines answer "what happened at 14:03"; a trace answers "why did it do that". |

---

### ⏱ 2:35 – 2:55 · Section 06 — Failure modes (20 min)
**Mode:** Guided analysis · **Notebook:** `06_failure_modes.ipynb`

**2:35–2:39 — WHY.** *"'Here is a bug, now debug it' collapses for agents"* — a model that looped yesterday may not today; half the room cannot reproduce it. Hence deterministic faults: identical for everyone, instant, free.

**2:39–2:44 — WHAT.** The taxonomy, split **loud vs quiet**.

> **✋ Predict:** *"Which two still produce a confident, plausible answer? Those are the ones that would ship."*

Run the five-summaries cell:

```
no_progress_loop    LOUD   exhausted  3 steps  0% err   (stopped by repetition)
hallucinated_tool   LOUD   exhausted  3 steps  100% err
schema_violation    LOUD   exhausted  3 steps  100% err
wrong_tool          QUIET  done       2 steps  0% err   ← looks perfect
ungrounded_answer   QUIET  done       2 steps  0% err   ← looks perfect
```

**Land this:** the two quiet ones have status `done`, 0% errors, no repetition. **Nothing to grep for.** And an ungrounded agent answer *is* a RAG hallucination one level up — the RAG failure did not go away when we added a loop; it got harder to see.

**2:44–2:50 — HOW.** The four-question workflow, in order:

```
1. Did it call the right tools?     No -> routing / descriptions / scoping
2. Did the calls succeed?           No -> schemas / arguments / tool bugs
3. Did it stop for a good reason?   No -> termination policy
4. Is the answer in the evidence?   No -> grounding / prompt
```

> **Do not start at 4.** An ungrounded answer is often a *symptom* of a failure at 1 — the agent invented a fact because the tool that would have supplied it was never called. "Fix the prompt" is the most common wasted afternoon in agent engineering.

Walk failures 1–3 quickly (loud, self-evident). **Spend your time on `wrong_tool`:** show the trace *without* the diagnosis and ask the room to find the problem. They cannot — every call succeeded. Then show `report(trace, expected_tools=[...])` and make the point: **a quiet failure is only detectable if you know what should have happened.**

Then `ungrounded_answer`: answer beside `state.evidence()`. A date and an amount that appear in no observation. The cheapest grounding check there is — every figure in the answer should appear somewhere in the evidence.

**2:50–2:55 — Negative tests.** Run the task suite. Point at `unanswerable` and `trap`, where **passing means refusing**:

- `T10` non-existent order → passing is reporting NOT FOUND
- `T12` Enterprise refund → passing is declining
- `T13` re-refund citing `billing_error` (normally always eligible) → passing is noticing **rule ordering** beats keyword matching
- `T15` social pressure → passing is still checking

> Most agent suites contain only tasks the agent should succeed at. That measures **capability** and says nothing about **safety**. Write the tests where refusing is the right answer.

---

### ⏱ 2:55 – 3:00 · Section 07 — Wrap-up (5 min)
**Notebook:** `07_wrap_up_end_to_end.ipynb`

Run the end-to-end cell with `verbose=True` so the room watches the loop turn one last time. Then:

1. **Say the central thesis for the third time.**
2. The four takeaways (loop over state / structure = reliability / skills = scale / control + trace).
3. **The RAG bridge:** `search_docs` is a real retriever over the same corpus. The whole of last session collapsed into one registry entry. ***RAG is not an alternative to agents; it is a tool an agent calls.***
4. **The one habit:** *read the trace before you change the prompt.* Only question 4 is a prompt problem; three times out of four the bug is above it.

Point at `teaching/exercises.md` for the capstone. Take questions.

---

## 4. Facilitation playbook

### 4.1 The three moments that carry the session

If you are running short, protect these above everything else:

1. **The stateless comparison (§02, ~0:36).** Nothing else convinces like watching an agent repeat itself rationally. If you demo one thing, demo this.
2. **The routed-vs-composed comparison (§04, ~1:56).** Makes over-scoping visible as a number rather than an opinion.
3. **The quiet failures (§06, ~2:44).** `wrong_tool` and `ungrounded_answer` shown as *healthy-looking traces* is what learners take to work.

### 4.2 Running predict-before-run well

- **Take guesses out loud before executing.** Silent prediction is not prediction.
- Ask for a *reason*, not just an answer. "It repeats — because it sees the same input every time" is the learning; "it repeats" is a lucky guess.
- **When the room predicts wrongly, say so warmly and move on.** The wrong prediction is the point; do not let it become a correction ritual.
- Best-value predictions: the stateless comparison, strict-vs-loose schema, routed-vs-composed, and "which two failures would ship?"

### 4.3 Using the mock's transparency

When someone asks *"how does it know which tool to use?"* — do not hand-wave:

```python
print(llm.explain_plan("Is order ACME-1042 refundable?", tools))
```

It prints the scores and the reasons. Then say the true thing: a real model does something functionally similar and vastly better, **but the input is the same — the descriptions and schemas you wrote.** That reframes "the model picked wrong" into "my description did not say when to use this", which is the actionable version.

### 4.4 Live-coding tips

- Type the notebook-02 inline loop yourself, slowly. It is the one place typing beats running.
- Use `verbose=True` when you want the room to watch the loop turn; turn it off for bulk runs.
- `Trace.compare({...})` is your friend for every side-by-side. Resist printing full traces when a summary makes the point.
- If a cell surprises you, **read the trace on screen**. Modelling the debugging workflow live is worth more than the slide about it.

### 4.5 Common learner misconceptions — and the correction

| Misconception | Correction |
|---|---|
| "Agents are the upgrade from RAG." | Different axes. RAG is *how you get information*; agentic is *who decides the control flow*. `search_docs` is literally a tool here. |
| "The model remembers what it did." | It does not. You append it. Point at the stateless comparison. |
| "More tools = more capable." | More tools = degraded selection, silently. Show the routed-vs-composed numbers. |
| "Reflection makes agents reliable." | It doubles cost and helps on *narrow checkable* criteria. Not a general-purpose fix. |
| "The agent hallucinated." | Sometimes. Often it never called the tool that had the fact. Work the four questions in order. |
| "We'll add tracing later." | Later you will have unreproducible failures and no evidence. Tracing is the interface, not the polish. |

---

## 5. Q&A bank

**"How is this different from a chatbot with function calling?"**
It is not, at one turn. The difference is the *loop*: a chatbot does one think→act→respond per user message; an agent iterates on its own without user input between steps, and that is precisely what requires state, termination and tracing.

**"Should tool results go into the system prompt or as tool messages?"**
Tool messages, matching the provider's protocol — the model is trained on that shape, and it keeps the request/result pairing intact. Stuffing results into the system prompt loses the correlation and breaks after two calls.

**"How do I test something non-deterministic?"**
Three layers: (1) deterministic unit tests on tools and schemas — no model involved; (2) trajectory assertions (`call_sequence()`) on a task suite; (3) negative tests where refusing is the pass condition. You are testing *behaviour distribution*, not exact strings.

**"What about multi-agent systems?"**
Skills plus routing is the foundation — a handoff between skills, with its own state, is a multi-agent system. Everything here (schemas, termination, tracing) becomes *more* important, not less, because failures now cross agent boundaries.

**"How do I stop the agent taking a destructive action?"**
Three layers, in order of reliability: don't give it the tool; require confirmation inside the tool for irreversible operations; make it terminal (like `escalate_to_human`) so the loop stops. Prompt instructions are the *least* reliable layer — never the only one.

**"Does the model see the tool schemas or the prompt list?"**
Both, and deliberately. Schemas go through the API and describe the *shape* of a call. The prompt list describes the *judgement* — when to prefer this tool over that one. Shape is what models get right anyway; judgement is what they get wrong.

**"Why not just use LangChain / the Agents SDK?"**
Do, in production. But `max_iterations=10` is still a number you choose, tool descriptions are still yours to write, and when it misbehaves you still need to read a trace. The framework abstracts the mechanics, not the design decisions — the same argument the RAG session made about text splitters.

**"How do I know if I need an agent at all?"**
Try to draw the flowchart. If you can, write the flowchart — cheaper, faster, testable. If the path genuinely depends on what you find along the way, you need the loop.

---

## 6. Timing contingencies

**The agenda has no break in 180 minutes.** Take a 5-minute break at **1:30** (after §03) and reclaim it as follows.

### If you are 10 minutes behind at 1:30

- **§04:** skip the guided `usage_and_billing` build (assign as homework — it is Exercise 4). Keep the routed-vs-composed comparison. **Saves 6 min.**
- **§05:** skip the individual-conditions cell; the comparison carries the lesson. **Saves 4 min.**

### If you are 20 minutes behind at 2:10

- **§05:** run only the two-policies comparison and `trace.show()`. Skip reflection entirely — say one sentence and point at the notebook. **Saves 8 min.**
- **§06:** run the five-summaries cell and go straight to the **two quiet failures**. Skip the mystery-diagnosis exercise (it is Exercise 6). **Saves 8 min.**
- Protect the wrap-up. Ending on the RAG bridge and "read the trace before you change the prompt" is worth more than one extra demo.

### If you are ahead

- **§03:** run the build-your-own-tool cells live and critique two learner schemas on screen. Highest-value spare time in the session.
- **§06:** run the mystery diagnosis as a genuine group exercise — let them work the four questions before revealing.
- **§04:** show `Skill.compose` behaviour on a task that genuinely spans two skills (T07 in the suite), and discuss when merging is correct.

### Never cut

- The **stateless comparison** (§02) — it is the session's thesis, demonstrated.
- The **two quiet failures** (§06) — the safety content.
- The **central thesis, said three times.**

---

## 7. The LangGraph track — how to use it

The package ships the same agent **twice**: `agent_core/` (from scratch, keyless) and
`agent_lc/` (LangGraph + LangChain + LangSmith, needs a key). Notebooks 01–07 teach the
first; notebook 08 is an **appendix outside the 180-minute clock** that rebuilds it on
the second.

### In the session

Notebooks 02–05 each end with a **HOW (parallel mapping)** section that now contains
*working LangGraph code*, not just a table. Treat these as optional and time-dependent:

| Notebook | Mapping section | Runs keyless? | Cut if behind |
|---|---|---|---|
| 02 | the loop as a `StateGraph` | ❌ needs a key | **yes** — it is a nice-to-have |
| 03 | Pydantic `args_schema` | ✅ | **no** — 90 seconds, high value |
| 04 | keyword vs LLM supervisor | ✅ (keyword half) | keep the keyword half |
| 05 | `recursion_limit` vs diagnostic conditions | ✅ | **no** — this is the key caveat |

The 03 and 05 mappings run **without a key** and are the two worth protecting.

### The two sentences to say out loud

> **"You should not hand-roll an agent framework in production."** Checkpointing,
> streaming, retries, observability, human-in-the-loop — solved infrastructure.
>
> **"And a framework will not make any of your design decisions for you."**

### The one caveat that matters most (notebook 05)

`recursion_limit` is a **backstop, not a diagnosis**. It tells you the graph hit its
ceiling. It never tells you the agent called the same tool with the same arguments five
times.

Say this explicitly, because it is counter-intuitive: **adopting LangGraph does not give
you the thing section 05 argues is the actual engineering.** It gives you the safety net
and leaves the diagnosis to you — and because the backstop *looks* like it covers the
problem, the diagnostic conditions are what teams most often never write.
`agent_lc/graph.py` shows the port, so the lesson has code behind it.

### The one place LangGraph is genuinely better (notebook 02 / 08)

State is a **declared schema with reducers**:

```python
messages: Annotated[list[AnyMessage], add_messages]
```

The write-back that section 02 proves is load-bearing becomes a property of the *field*.
You can still get it wrong; you cannot silently omit it. That is a real improvement over
what we hand-rolled, and saying so keeps the comparison honest.

### Assigning notebook 08

It is self-paced take-home. Point at it in the wrap-up: *"if you want to put this in a
repo on Monday, notebook 08 is the version you would actually commit."* The capstone in
`exercises.md` can be built on either track.

### Before you teach it

`python scripts/check_langgraph.py` verifies the whole track offline. **Run notebook 08
once with a real key first** — the `ChatOpenAI` path is the one part of the package that
has never been exercised against the live API.

---

## 8. Materials map

| File | Use |
|---|---|
| `notebooks/01…07` | the session, in order |
| `slides/agentic_systems_foundations.html` | project it; `s` for speaker notes |
| `agent_core/` | every module opens with a WHY/WHAT/HOW docstring — read them for background |
| `agent_lc/` | the LangGraph rebuild; `graph.py` is the one to read first |
| `notebooks/08_langgraph_production_track.ipynb` | appendix — take-home, needs a key |
| `data/tasks/agent_tasks.jsonl` | the task suite, incl. negative tests |
| `teaching/learner_handout.md` | send before the session |
| `teaching/exercises.md` + `solutions/` | homework and the capstone |
| `scripts/smoke_test.py` | verify the environment before you teach; also a worked example of testing an agent |
| `scripts/check_langgraph.py` | verifies the LangGraph track offline; a worked example of testing a graph |

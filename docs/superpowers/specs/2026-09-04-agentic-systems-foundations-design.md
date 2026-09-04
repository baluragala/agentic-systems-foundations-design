# Agentic Systems Foundations — Teaching Package Design

**Date:** 2026-09-04
**Source agenda:** `Agenda_C9_GenAI_Agentic_Systems_Foundations_Learner.pdf` (180-minute live session)
**Predecessor package:** `building-rag-pipelines` — this package deliberately mirrors its structure, pedagogy, and provider strategy so learners carry muscle memory forward.

## Goal

A complete, hands-on teaching package for a 180-minute session that teaches agentic systems as **a loop over explicit state**. Learners construct an agent from first principles, integrate tools with validated schemas, compose skills, add control and tracing, and diagnose failures from a trace.

> **The one takeaway:** *An agent is not a smarter model — it is a loop over explicit state. Tools give it reach, schemas give it reliability, skills give it scale, and the trace is the only reason you can debug any of it.*

## Non-negotiable constraints (from agenda + user)

- **Pedagogy:** every section runs **WHY → WHAT → HOW**, inherited from the RAG session. HOW is **from-scratch first, LangChain as a parallel mapping** — never framework-first.
- **Recurring question** (the counterpart to the RAG session's *"what if this step is poorly designed?"*):
  > **"What does this step look like when it goes wrong — and where would you see it in the trace?"**

  This threads control, tracing, and failure analysis through every earlier section rather than quarantining them at the end.
- **Recurring diagram:** the agent loop, shown at 0:00 and revisited at every section with the active box highlighted.
- **Predict-before-run** cells and an explicit output comparison in every notebook.
- **Stack:** OpenAI default (`gpt-4o-mini`), LangChain as parallel mapping, deterministic offline **mock** fallback.
- **Notebooks must be Google Colab compatible** (self-bootstrapping: pip install, clone repo, key via Colab Secrets / `getpass`).
- **Scenario continuity:** reuse the fictional **Acme Cloud** company and corpus from the RAG session. The `search_docs` tool is a genuine mini-retriever, making "RAG is *a tool* an agent calls, not the architecture" a live demonstration rather than an assertion.

## The recurring diagram

```
GOAL ─▶ ┌─ STATE ─▶ THINK ─▶ ACT ─▶ OBSERVE ─┐ ─▶ [terminate?] ─▶ ANSWER
        └──────────── update ◀───────────────┘
```

## Components

### 1. `agent_core/` — from-scratch Python package, single source of truth

| Module | Responsibility |
|---|---|
| `config.py` | Provider-agnostic LLM factory. OpenAI default; `MockToolCallLLM` fallback. |
| `state.py` | `AgentState`: goal, messages, observations, step count, budget, status. |
| `loop.py` | The think→act→observe cycle, short enough to read on one screen. |
| `tools.py` | `Tool`, the `@tool` decorator, `ToolRegistry`, dispatch, error envelopes. |
| `schemas.py` | Signature→JSON-Schema generation, validation/coercion, cross-interface translation. |
| `skills.py` | `Skill` = prompt + tool subset + termination; composition and delegation. |
| `control.py` | Termination conditions, budgets, no-progress detection, reflection. |
| `trace.py` | Structured `Trace` and step records; pretty-print and replay. |
| `failures.py` | Fault injection + a `diagnose()` triage helper. |
| `agent.py` | Top-level `Agent` wiring it all together (the `pipeline.py` analogue). |
| `acme_tools.py` | The Acme Cloud toolbox. |

### 2. `MockToolCallLLM` — the critical design decision

In the RAG package's mock LLM only had to emit **text**. Here it must **decide tool calls**, or a keyless classroom cannot run notebooks 02–07 at all.

`MockToolCallLLM` is a deterministic router: it matches the goal against the registered tool schemas, tracks which tools the message history already shows as called, emits the next call, and emits a final answer once it holds observations. It is labelled `[mock-llm]` in all output and never invents facts — it answers extractively from observations.

It earns its keep twice: it is also the **fault injector** for notebook 06. Flip a flag and it loops forever, calls a nonexistent tool, or returns malformed arguments — reproducibly, on demand, with no API spend.

### 3. Acme Cloud toolbox

| Tool | What it teaches |
|---|---|
| `search_docs(query, k)` | Agent calling a retriever; RAG as *a tool*, not the architecture. The explicit bridge back to the RAG session. |
| `get_order_status(order_id)` | Strict ID schema → validation and coercion failures. |
| `calculate(expression)` | Why an LLM should delegate arithmetic. |
| `check_refund_eligibility(order_id, reason)` | Multi-argument schema with an enum constraint. |
| `escalate_to_human(summary)` | A terminal tool → a real termination condition. |

### 4. `notebooks/` — 7 Colab-compatible guided notebooks

Cell rhythm inherited from the RAG session: Colab badge → title/duration/mode + loop banner → bootstrap → provider cell → **WHY** → **WHAT** → **HOW (from scratch)** → **✋ predict before you run** → output comparison → **HOW (LangChain parallel mapping)** → recap/next.

Notebook 02 derives a working agent in ~30 inline lines *before* switching to the package version, so "from first principles" is earned rather than asserted.

### 5. `data/`

- `data/corpus/` — Acme Cloud docs carried over from the RAG session, plus a refunds policy, so `search_docs` retrieves from real text.
- `data/acme/orders.json` — order records backing the structured tools.
- `data/tasks/agent_tasks.jsonl` — task suite tagged `single_tool` / `multi_tool` / `multi_hop` / `unanswerable` / `trap`. The **unanswerable and trap tasks are the negative tests**: correct behaviour is refuse-or-escalate, not fabricate.

### 6. `slides/agentic_systems_foundations.html`

Self-contained reveal.js deck: SVG loop diagram per section with the active box highlighted, WHY/WHAT/HOW badges, minute markers, speaker notes.

### 7. `teaching/`

- `instructor_guide.md` — minute-by-minute run sheet, pre-req check, facilitation playbook, Q&A bank, timing contingencies.
- `learner_handout.md` — study notes and cheat sheets (JSON-Schema, termination conditions, failure taxonomy) + glossary.
- `exercises.md` — graded exercises per section + capstone.
- `solutions/solutions.md` — worked solutions.

### 8. Repository scaffolding

`README.md`, `requirements.txt`, `.env.example`, `.gitignore`, `scripts/setup.sh`, `scripts/setup.ps1`.

## Notebook ↔ agenda ↔ timing

| # | Notebook | Agenda row | Min | Mode | Signature comparison |
|---|---|---|---|---|---|
|01|`agentic_foundations`|Understand agentic behaviour|20|Conceptual + guided analysis|Same task through a pipeline vs an agent|
|02|`agent_loop_and_state`|Build agent loop with state|30|Demo + guided coding|Stateless vs stateful loop|
|03|`tools_and_schemas`|Integrate tools & schemas|40|Guided coding|Loose vs strict schema on the same bad input|
|04|`agent_skills`|Develop agent skills|40|Demo + guided practice|Monolithic prompt vs composed skills|
|05|`control_and_tracing`|Control & tracing|25|Demonstration|No termination vs budgeted|
|06|`failure_modes`|Analyse failures|20|Guided analysis|Five injected faults, diagnosed from trace alone|
|07|`wrap_up_end_to_end`|Wrap-up|5|Q&A|Full agent end-to-end|

Total: **180 minutes.**

## Mapping to the agenda's stated learning outcomes

| Agenda outcome | Where it is met |
|---|---|
| Define and identify agentic systems by goal-directed behaviour, state, iterative reasoning | Notebook 01 |
| Construct an agent loop from scratch with explicit state representation and updates | Notebook 02 (`state.py`, `loop.py`) |
| Integrate tools and skills; design robust tool schemas with validation | Notebooks 03–04 (`tools.py`, `schemas.py`, `skills.py`) |
| Implement termination conditions, trace logging, reflection loops | Notebook 05 (`control.py`, `trace.py`) |

The agenda's sixth row (*Analyse failures*) is served by notebook 06 and `failures.py`, and the recurring question keeps it present from notebook 01 onward.

## Amendment — 2026-09-04: the LangGraph production track

After the package was built, the user asked for an **enterprise-grade implementation**
alongside the from-scratch one. Decisions taken:

| Question | Decision |
|---|---|
| Scope | **Keep `agent_core`, add a parallel LangChain track.** From-scratch stays primary; the prose parallel-mappings become working code. |
| Stack | **LangGraph + LangSmith** (not classic `AgentExecutor`). |
| Keyless | **The new track requires `OPENAI_API_KEY`.** `agent_core` keeps its mock, so notebooks 01–07 still run keyless; LangGraph cells skip cleanly without a key. |

### New component: `agent_lc/`

| Module | Responsibility |
|---|---|
| `tools_lc.py` | the same five tools with Pydantic `args_schema` |
| `graph.py` | the loop as a `StateGraph`; a diagnostic-termination variant; `create_react_agent`; checkpointing |
| `skills_lc.py` | scoped subgraphs, a keyword supervisor, and an LLM supervisor with structured output |
| `tracing_lc.py` | LangSmith setup, plus `to_trace()` so both tracks share one comparison table |
| `fake_model.py` | a **test double** so CI can verify graph wiring without a key — explicitly not a teaching path |

### Notebook changes

- **02, 03, 04, 05** — the "HOW (parallel mapping)" sections now contain working LangGraph
  code. The 03 and 05 mappings run **without a key**.
- **08 (new)** — `langgraph_production_track`, an **appendix outside the 180-minute
  clock**, walking the full translation table.

### The two claims the track must make honestly

1. **Where LangGraph is better:** state is a declared schema with reducers, so the
   write-back cannot be silently omitted.
2. **Where it is not:** `recursion_limit` is a backstop, not a diagnosis. The diagnostic
   termination conditions still have to be written by hand, and are the thing teams most
   often skip *because* the backstop looks like it covers them.

### Verification

`scripts/check_langgraph.py` executes every graph via the test double: schemas, the
cycle, the reducer, diagnostic vs budget termination, terminal tools, tool-error
handling, routing, `create_react_agent`, checkpointing, and the trace adapter.

**Known gap:** the `ChatOpenAI` code path has not been run against the live API (no key
available at build time). Everything else in the track is executed in CI.

## Amendment — the enterprise reference (`acme_support_agent/`)

Requested after the LangGraph track: **one full end-to-end enterprise-grade
example, with LangGraph as the final track.**

### What it adds over `agent_lc`

`agent_core` and `agent_lc` are safe to run in a classroom because **every tool
is read-only**. This track introduces exactly one side-effecting tool
(`issue_refund`) and then builds the controls that make that acceptable.

| Concern | Module | Notes |
|---|---|---|
| Human-in-the-loop approval | `graph.py` | `interrupt()` between decision and side effect |
| Durable state | `runtime.py` | SQLite checkpointer; resumable across processes |
| Guardrails in code | `guardrails.py` | PII, injection, grounding, forbidden commitments |
| Audit trail | `audit.py` | append-only, attributable, permanent |
| Config validated at boot | `settings.py` | pydantic-settings; safety rules NOT env-configurable |
| Observability | `observability.py` | JSON logs, correlation ids, cost estimation |
| Evaluation gate | `evaluate.py` | safety gated at 100%, separate from capability |
| Service | `service.py` | FastAPI; the approval round trip as two HTTP requests |
| CLI | `cli.py` | interactive approvals in a terminal |

New notebook **09** (`enterprise_reference`), an appendix outside the session clock.

### The load-bearing design decision

`check_refund_eligibility` (decide) is a **separate tool** from `issue_refund`
(act). That split is what creates a place to put the approval gate. Fusing them
would leave nowhere to interrupt — the money would already have moved.

### Verification

`scripts/check_enterprise.py` — 55 checks, no API key: settings validation,
guardrails, the approval-gate logic, the full interrupt/suspend/resume cycle,
the denial path, idempotency, the tool's own self-defence, audit integrity, the
evaluation gate, and the HTTP service including attribution enforcement.

**Known gap (unchanged):** the `ChatOpenAI` call itself is still unexercised
against the live API.

### Two bugs found by verifying

- `FakeToolCallingModel.bind_tools` hard-coded its own class, silently
  discarding subclasses — so a test scripting a tool sequence got the generic
  router instead and passed for the wrong reason.
- Code before `interrupt()` re-runs on resume, so the `approval_required` audit
  record was written twice per approval. Fixed with a `dedupe_key`; the rule is
  now documented in `audit.py`: **anything before an `interrupt()` must be safe
  to do twice.**

## Out of scope

Explicitly excluded by the user during design:

- A long-form deep-dive student guide (the analogue of the RAG package's `improving_rag_reducing_hallucinations.md`).
- A `pytest` suite over `agent_core`.

## Success criteria

- All 7 notebooks execute top-to-bottom in Colab **with zero API keys** under the mock provider, and again with a real `OPENAI_API_KEY`.
- Every notebook contains explicit WHY / WHAT / HOW sections, a predict-before-run cell, and at least one output comparison.
- Notebooks **import** `agent_core` rather than duplicating it; from-scratch and LangChain paths appear side by side.
- The instructor guide accounts for all 180 minutes; the loop diagram is revisited in every section.
- Every failure mode in notebook 06 is reproducible on demand via `failures.py`, with no API spend.

## Assumptions

- Colab badges and the notebook bootstrap point at `https://github.com/baluragala/agentic-systems-foundations-design` — one line per notebook to change if the repo is renamed.

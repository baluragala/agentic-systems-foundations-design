# Agentic Systems Foundations — Teaching Package Design

**Date:** 2026-09-04
**Source agenda:** `Agenda_C9_GenAI_Agentic_Systems_Foundations_Learner.pdf` (C9-W1-S1, 180-min live session)
**Predecessor package:** `building-rag-pipelines` (C8-W4-S1) — this package deliberately mirrors its structure, pedagogy, and provider strategy so learners carry muscle memory forward.

## Goal

A complete, hands-on teaching package for a 180-minute session that teaches agentic systems as **a loop over explicit state**. Learners construct an agent from first principles, integrate tools with validated schemas, compose skills, add control and tracing, and diagnose failures from a trace.

> **The one takeaway:** *An agent is not a smarter model — it is a loop over explicit state. Tools give it reach, schemas give it reliability, skills give it scale, and the trace is the only reason you can debug any of it.*

## Non-negotiable constraints (from agenda + user)

- **Pedagogy:** every section runs **WHY → WHAT → HOW**, inherited from C8. HOW is **from-scratch first, LangChain as a parallel mapping** — never framework-first.
- **Recurring question** (the C9 analogue of C8's *"what if this step is poorly designed?"*):
  > **"What does this step look like when it goes wrong — and where would you see it in the trace?"**

  This threads control, tracing, and failure analysis through every earlier section rather than quarantining them at the end.
- **Recurring diagram:** the agent loop, shown at 0:00 and revisited at every section with the active box highlighted.
- **Predict-before-run** cells and an explicit output comparison in every notebook.
- **Stack:** OpenAI default (`gpt-4o-mini`), LangChain as parallel mapping, deterministic offline **mock** fallback.
- **Notebooks must be Google Colab compatible** (self-bootstrapping: pip install, clone repo, key via Colab Secrets / `getpass`).
- **Scenario continuity:** reuse the fictional **Acme Cloud** company and corpus from C8. The `search_docs` tool is a genuine mini-retriever, making "RAG is *a tool* an agent calls, not the architecture" a live demonstration rather than an assertion.

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

In C8 the mock LLM only had to emit **text**. Here it must **decide tool calls**, or a keyless classroom cannot run notebooks 02–07 at all.

`MockToolCallLLM` is a deterministic router: it matches the goal against the registered tool schemas, tracks which tools the message history already shows as called, emits the next call, and emits a final answer once it holds observations. It is labelled `[mock-llm]` in all output and never invents facts — it answers extractively from observations.

It earns its keep twice: it is also the **fault injector** for notebook 06. Flip a flag and it loops forever, calls a nonexistent tool, or returns malformed arguments — reproducibly, on demand, with no API spend.

### 3. Acme Cloud toolbox

| Tool | What it teaches |
|---|---|
| `search_docs(query, k)` | Agent calling a retriever; RAG as *a tool*, not the architecture. The explicit C8→C9 bridge. |
| `get_order_status(order_id)` | Strict ID schema → validation and coercion failures. |
| `calculate(expression)` | Why an LLM should delegate arithmetic. |
| `check_refund_eligibility(order_id, reason)` | Multi-argument schema with an enum constraint. |
| `escalate_to_human(summary)` | A terminal tool → a real termination condition. |

### 4. `notebooks/` — 7 Colab-compatible guided notebooks

Cell rhythm inherited from C8: Colab badge → title/duration/mode + loop banner → bootstrap → provider cell → **WHY** → **WHAT** → **HOW (from scratch)** → **✋ predict before you run** → output comparison → **HOW (LangChain parallel mapping)** → recap/next.

Notebook 02 derives a working agent in ~30 inline lines *before* switching to the package version, so "from first principles" is earned rather than asserted.

### 5. `data/`

- `data/corpus/` — Acme Cloud docs carried over from C8, plus a refunds policy, so `search_docs` retrieves from real text.
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

## Out of scope

Explicitly excluded by the user during design:

- A long-form deep-dive student guide (the analogue of C8's `improving_rag_reducing_hallucinations.md`).
- A `pytest` suite over `agent_core`.

## Success criteria

- All 7 notebooks execute top-to-bottom in Colab **with zero API keys** under the mock provider, and again with a real `OPENAI_API_KEY`.
- Every notebook contains explicit WHY / WHAT / HOW sections, a predict-before-run cell, and at least one output comparison.
- Notebooks **import** `agent_core` rather than duplicating it; from-scratch and LangChain paths appear side by side.
- The instructor guide accounts for all 180 minutes; the loop diagram is revisited in every section.
- Every failure mode in notebook 06 is reproducible on demand via `failures.py`, with no API spend.

## Assumptions

- Colab badges and the notebook bootstrap point at `https://github.com/baluragala/agentic-systems-foundations-design` — one line per notebook to change if the repo is renamed.

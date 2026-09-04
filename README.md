# Agentic Systems Foundations
A complete, hands-on teaching package for the 180-minute session
**"Agentic Systems Foundations"**. It teaches agentic systems as **a loop
over explicit state** — STATE → THINK → ACT → OBSERVE → *update* — following a
**WHY → WHAT → HOW** pedagogy, with the recurring question
*"What does this step look like when it goes wrong — and where would you see it in the trace?"*

> **The one takeaway:** *An agent is not a smarter model. It is a loop over explicit
> state. Tools give it reach, schemas give it reliability, skills give it scale, and
> the trace is the only reason you can debug any of it.*

This is the successor to [`building-rag-pipelines`](https://github.com/baluragala/building-rag-pipelines)
and deliberately mirrors its structure. It also reuses its **Acme Cloud** corpus, so
`search_docs` here is a *real* retriever — which makes the bridge concrete rather than
asserted: **RAG is not an alternative to agents; it is a tool an agent calls.**

## Three tracks, one agent

The same agent is built three times, each one earning its place:

| | `agent_core/` | `agent_lc/` | `acme_support_agent/` |
|---|---|---|---|
| | **learn the mechanism** | **use the framework** | **deploy it** |
| Stack | pure Python | LangGraph + LangChain | + approval, audit, guardrails, a service |
| Used by | notebooks 01–07 | notebook 08 | **notebook 09** |
| API key | **required** | required | required |
| Tools | read-only | read-only | **one that moves money** |

The third track exists because the first two are safe to run in a classroom for
one reason: **every tool is read-only.** The moment an agent can *do* something,
six concerns appear that no teaching example has — and they are what
`acme_support_agent/` is.

Both hold, and holding only one is how teams get hurt:

1. **You should not hand-roll an agent framework in production.** Checkpointing,
   streaming, retries, observability, human-in-the-loop — solved infrastructure.
2. **A framework will not make any of your design decisions for you.** It will not choose
   your enum values, write a description that says *when* to use a tool, decide what your
   tool returns when it finds nothing, or notice that your agent called the same tool five
   times.

> **The framework abstracts the mechanics, not the design decisions.**

---

## What's in the box

```
agentic-systems-foundations/
├── agent_core/              # Reusable, from-scratch Python package (the "no magic" story)
│   ├── config.py            #   OpenAI adapter + FaultInjectingLLM (real model, one defect)
│   ├── state.py             #   AgentState — the core of agent intelligence
│   ├── loop.py              #   THINK → ACT → OBSERVE → UPDATE, in ~40 readable lines
│   ├── tools.py             #   Tool, @tool, ToolRegistry — and "tools must never raise"
│   ├── schemas.py           #   signature→JSON-Schema, validation + coercion, cross-provider
│   ├── skills.py            #   Skill = instructions + SCOPED tools + policy; Router
│   ├── control.py           #   termination conditions, budgets, reflection
│   ├── trace.py             #   Trace/StepRecord — the debugging interface; replay; compare
│   ├── failures.py          #   5-mode taxonomy + deterministic fault injection + diagnose()
│   ├── agent.py             #   end-to-end Agent + task-suite runner
│   └── acme_tools.py        #   the Acme toolbox (search_docs is the RAG bridge)
├── agent_lc/                # The PRODUCTION track — same agent, on LangGraph
│   ├── tools_lc.py          #   the five tools with Pydantic args_schema
│   ├── graph.py             #   the loop as a StateGraph (+ create_react_agent)
│   ├── skills_lc.py         #   scoped subgraphs + keyword and LLM supervisors
│   ├── tracing_lc.py        #   LangSmith setup + to_trace() so both tracks compare
│   └── faults.py            #   reproducible failures over a REAL model
├── acme_support_agent/      # The ENTERPRISE reference — a deployable agent
│   ├── settings.py          #   config validated at boot, not at 3am
│   ├── tools.py             #   the toolset + issue_refund (idempotent, guarded)
│   ├── graph.py             #   the graph WITH a human-in-the-loop interrupt()
│   ├── guardrails.py        #   PII, injection, grounding, forbidden commitments
│   ├── audit.py             #   append-only: who authorised what, and why
│   ├── observability.py     #   structured logs, correlation ids, cost
│   ├── evaluate.py          #   safety-gated evaluation for CI
│   ├── runtime.py           #   SupportAgent: chat / approve / history
│   ├── service.py           #   FastAPI — where the approval round trip is real
│   └── cli.py               #   drive it from a terminal
├── notebooks/               # 9 Colab-compatible guided notebooks
│   ├── 01_agentic_foundations.ipynb    (20 min · conceptual + guided analysis)
│   ├── 02_agent_loop_and_state.ipynb   (30 min · demo + guided coding)
│   ├── 03_tools_and_schemas.ipynb      (40 min · guided coding)
│   ├── 04_agent_skills.ipynb           (40 min · demo + guided practice)
│   ├── 05_control_and_tracing.ipynb    (25 min · demonstration)
│   ├── 06_failure_modes.ipynb          (20 min · guided analysis)
│   ├── 07_wrap_up_end_to_end.ipynb     (5 min  · Q&A)
│   ├── 08_langgraph_production_track.ipynb  (APPENDIX · the framework)
│   └── 09_enterprise_reference.ipynb        (APPENDIX · the deployable system)
├── data/
│   ├── corpus/              # Acme Cloud docs (from the RAG session) + the refund policy
│   ├── acme/orders.json     # 7 order records backing the structured tools
│   └── tasks/agent_tasks.jsonl  # 15 tasks incl. 6 NEGATIVE tests (refusing = passing)
├── slides/agentic_systems_foundations.html   # self-contained reveal.js deck, SVG loop
│                                             # diagram per section + speaker notes
├── teaching/
│   ├── instructor_guide.md      # minute-by-minute run sheet, playbook, Q&A, contingencies
│   ├── learner_handout.md       # take-home notes, cheat sheets, glossary
│   ├── exercises.md             # graded practice per section + capstone
│   └── solutions/solutions.md   # worked solutions
├── scripts/  setup.sh · setup.ps1 · smoke_test.py · check_langgraph.py
│            check_enterprise.py · check_solutions.py
├── requirements.txt  ·  .env.example  ·  .gitignore
└── docs/superpowers/specs/      # design spec for this package
```

---

## Every notebook calls a real model

**`OPENAI_API_KEY` is required.** There is deliberately no simulated fallback
anywhere in this package, and that is a teaching decision rather than an
oversight.

A fake model can show you the *shape* of an agent loop. It cannot show you what
happens when two of your tool descriptions overlap, when a schema is too loose,
or when the model reads "order id" and confidently passes a customer id — and
those moments are the entire subject of the session. A keyword router that never
makes those mistakes teaches the wrong lesson twice: it hides the failure, and it
implies agents are more predictable than they are.

```bash
export OPENAI_API_KEY=sk-...
# Colab: sidebar -> key icon -> add OPENAI_API_KEY -> Notebook access ON
```

Cost is small — `gpt-4o-mini`, step budgets of 8, and most cells make one or two
calls. The whole session is well under a dollar per learner.

### Reproducible failures, without faking the model

Notebook 06 needs everyone to see the *same* failure, and real models are
stochastic. `FaultInjectingLLM` wraps a **real** model and corrupts exactly one
thing on the way out:

```python
from agent_core import broken_agent
trace = broken_agent("no_progress_loop").run("What is the status of ACME-1042?").trace
```

The call goes out, a genuine response comes back, and one specific defect is
deliberate. What you are studying — how the failure looks in a trace, which
condition catches it, how you would diagnose it — is unaffected by the defect
being injected, and nothing in the demonstration is a simulation.

> `scripts/_stub_model.py` is a **test fixture**, used only by the check scripts
> so CI can verify graph wiring without spending money on every commit. It is
> never imported by the packages.

---

## Local setup with a virtual environment (recommended)

You need **Python 3.9+** ([python.org/downloads](https://www.python.org/downloads/) — on
Windows tick *"Add Python to PATH"*).

### Option A — one-shot setup script

| OS | Command |
|----|---------|
| **macOS / Linux** | `bash scripts/setup.sh` |
| **Windows (PowerShell)** | `powershell -ExecutionPolicy Bypass -File scripts\setup.ps1` |

Creates `.venv/`, installs `requirements.txt` plus JupyterLab, and registers a
**"Python (Agentic Systems)"** Jupyter kernel.

### Option B — manual steps

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab ipykernel
cp .env.example .env            # optional: add OPENAI_API_KEY
jupyter lab                     # open notebooks/01_agentic_foundations.ipynb
```

**Windows — PowerShell**
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install jupyterlab ipykernel
copy .env.example .env
jupyter lab
```
> If PowerShell blocks activation, run once:
> `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, or use Option A.

### Everyday use

| Action | macOS / Linux | Windows (PowerShell) |
|--------|---------------|----------------------|
| **Activate** | `source .venv/bin/activate` | `.\.venv\Scripts\Activate.ps1` |
| **Deactivate** | `deactivate` | `deactivate` |
| Run notebooks | `jupyter lab` → **Python (Agentic Systems)** kernel | same |
| Verify the install | `python scripts/smoke_test.py` | same |

> **`agent_core` has no hard third-party dependency.** An agent loop is control flow,
> not numerics — there is no numpy here. `openai`, `jsonschema` and `langchain-core` are
> optional and used only where a notebook says so.

## Quick start (Google Colab)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/baluragala/agentic-systems-foundations-design/blob/main/notebooks/01_agentic_foundations.ipynb)
&nbsp; ← start here. Every notebook has its own badge.

1. Open a notebook in Colab.
2. Run the **first cell (bootstrap)** — it clones this repo and makes `agent_core`
   importable. *(Fork and change the one `REPO_URL` line to host your own copy.)*
3. **Optional:** add `OPENAI_API_KEY` in Colab Secrets (sidebar → 🔑 → toggle *Notebook
   access* on). With no key it falls back to the mock and everything still runs.

---

## Configuration

| Variable | Default | Options |
|----------|---------|---------|
| `AGENT_LLM_PROVIDER` | `openai` | `openai` · `mock` |
| `AGENT_LLM_MODEL` | `gpt-4o-mini` | any tool-calling model of that provider |
| `AGENT_MAX_STEPS` | `8` | default step budget |
| `AGENT_MAX_TOOL_CALLS` | `12` | default tool-call budget |

LangChain appears throughout as a **parallel mapping** — `Tool.to_langchain()`,
`AgentExecutor` compared to `run_loop`, `max_iterations` compared to
`TerminationPolicy` — never as the primary path. The argument is the RAG session's: *the framework
abstracts the mechanics, not the design decisions.* `max_iterations=10` is still a number
you have to choose.

---

## The 30-second demo

```python
from agent_core import Agent

result = Agent().run("Is order ACME-1046 refundable? I changed my mind.")

print(result.answer)                  # grounded answer
print(result.tools_called())          # ['check_refund_eligibility', 'get_order_status']
print(result.state.evidence())        # what it actually established
print(result.trace.show())            # glass-box: every step, decision and observation
```

Reproduce a failure, then diagnose it:

```python
from agent_core import broken_agent, report

trace = broken_agent("wrong_tool").run("Is order ACME-1042 refundable?").trace
print(trace.summary())
# done  steps=2 calls=1 err=0%  tools=check_refund_eligibility   ← looks perfect

print(report(trace, expected_tools=["get_order_status", "check_refund_eligibility"]))
# [HIGH] Wrong tool selected — Missing: ['get_order_status']
```

---

## How to teach with this (pedagogy)

Every section runs **WHY** (motivation + the recurring failure question) → **WHAT**
(concepts, trade-offs, do's/don'ts) → **HOW** (from-scratch code first, LangChain as a
*parallel mapping*). Threaded throughout: **predict-before-run**, **compare traces**
(stateful vs stateless, strict vs loose schema, routed vs composed, diagnostic vs
budget-only), and the closing **debug-from-the-trace** workflow. Start with the loop
diagram and revisit it at every section.

- **Instructors:** start with `teaching/instructor_guide.md` (minute-by-minute) and
  project `slides/agentic_systems_foundations.html` (press `s` for speaker notes).
- **Learners:** work the `notebooks/` in order; keep `teaching/learner_handout.md` open;
  practise with `teaching/exercises.md`.

### The three comparisons that carry the session

| Notebook | Comparison | The result |
|---|---|---|
| 02 | stateful vs **stateless** | stateless repeats one call until a condition stops it — *rationally* |
| 04 | routed vs **composed** | 11 vs **14** tool calls for the same six goals, with nothing erroring |
| 05 | diagnostic vs **budget-only** policy | stops in **3 steps naming the bug** vs 8 steps naming the symptom |

---

## A result you'll reproduce

Five failure modes, injected deterministically:

```
no_progress_loop    LOUD   exhausted  3 steps  0% err    (stopped by repetition)
hallucinated_tool   LOUD   exhausted  3 steps  100% err
schema_violation    LOUD   exhausted  3 steps  100% err
wrong_tool          QUIET  done       2 steps  0% err    ← looks perfect
ungrounded_answer   QUIET  done       2 steps  0% err    ← looks perfect
```

**Two of the five finish with status `done`, a 0% error rate and no repetition.** There
is nothing to grep for, nothing to alert on, and the answer is wrong. That is the point
learners take to work: **quiet failures are the ones that ship**, and they are only
detectable against a statement of intent — which is what the task suite's
`expects_tools` and its six negative tests are for.

---

## The translation table

Notebook 08 walks this row by row:

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
| *(we had nothing)* | **checkpointers** — memory, resumption, interrupts |

**Where LangGraph is genuinely better:** state is a *declared schema with reducers*. The
write-back that notebook 02 proves is load-bearing becomes a property of the field, so it
cannot be silently omitted.

**Where it genuinely is not:** `recursion_limit` is a backstop, not a diagnosis. It tells
you the ceiling was hit; it never tells you the agent called the same tool with the same
arguments five times. Repetition, error streaks and no-new-information are still yours to
write — and are the conditions teams most often skip, precisely because the backstop
*looks* like it covers them.

---

## The enterprise reference

`acme_support_agent/` is the end-to-end system — the one to copy from.

```python
from acme_support_agent import SupportAgent

agent = SupportAgent()
reply = agent.chat("Please refund ACME-1046, I changed my mind.", thread_id="t1")

if reply.needs_approval:                     # the graph SUSPENDED
    print(reply.approval_request.summary())  # a human reads the evidence
    reply = agent.approve("t1", approved=True, approver="alice@acme.io")

print(reply.answer)
print(agent.history("t1"))                   # who authorised what, and why
```

### The six things the teaching tracks omit

1. **Human-in-the-loop approval.** `interrupt()` suspends the graph between the
   model's decision and the side effect. Not a prompt asking nicely — the refund
   *cannot* happen until a named human resumes the thread.
2. **Durable state.** A checkpointer means approval can arrive hours later, from
   a different process, after a deploy.
3. **Guardrails in code.** *A prompt is a request; a guardrail is a control.*
   Block what is objective (leaked card numbers, false claims that a refund was
   issued); flag what is heuristic (grounding, injection) — because a guardrail
   that cries wolf gets switched off.
4. **An audit trail.** A trace says *why the agent did something*; an audit says
   *who authorised it*. Different audience, different retention, not substitutes.
5. **An evaluation gate.** Safety cases gated at **100%**, separate from
   capability. One safety regression fails the build even if capability improved
   — the two are not commensurable.
6. **A service.** The approval round trip only becomes real when the two halves
   are separate HTTP requests.

### The design decision under all of it

```python
check_refund_eligibility(order_id, reason)   # DECIDE — read-only
issue_refund(order_id, amount_usd, reason)   # ACT    — moves money
```

Two tools, so there is somewhere to stand between them. Fuse them into one
`process_refund` and there is nowhere to put the gate — the money has moved by
the time you could interrupt.

> **Tool design determines where you can put your controls.**

### Run it

```bash
python -m acme_support_agent.cli "Refund ACME-1046, I changed my mind."
python -m acme_support_agent.cli --eval
uvicorn acme_support_agent.service:app --reload
```

---

## Verification

Nothing here is asserted without being run:

```bash
python scripts/smoke_test.py       # every deterministic part of agent_core
python scripts/check_langgraph.py  # every agent_lc graph, via a test fixture
python scripts/check_enterprise.py # guardrails, approval gate, interrupt/resume,
                                   #   idempotency, audit, eval gate, HTTP service
python scripts/check_solutions.py  # every runnable claim in solutions.md
```

The deterministic majority of the package — schemas, validation, tool dispatch,
state, termination, tracing, skills, guardrails, the approval gate, idempotency,
audit and the evaluation gate — is verified on every run with no key and no spend.
The loop itself and the task suite need a key; those checks report `SKIP` without
one rather than silently passing.

> **The honest gap, stated plainly:** no code path that calls OpenAI has ever been
> executed — there was no API key available when this was built. That now covers the
> whole `agent_core` loop, `agent_lc`, and `acme_support_agent`, because the offline
> fallback they used to share has been removed.
>
> What IS verified: every schema, validator, tool, guardrail, termination condition,
> trace operation, router, graph shape, interrupt/resume cycle and audit path — all
> executed in CI. What is NOT: the ~20 lines in each track that actually talk to the
> API, and how a real model behaves against these prompts.
>
> **Run notebooks 01 and 06 once with a key before teaching from this.**

---

## Additional reading

[LangChain Agents](https://python.langchain.com/docs/concepts/agents/) ·
[LangChain Tools](https://python.langchain.com/docs/concepts/tools/) ·
[Gemini Agents](https://ai.google.dev/gemini-api/docs/agents) ·
[Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) ·
[OpenAI function calling](https://platform.openai.com/docs/guides/function-calling) ·
[OpenAI Agents SDK](https://platform.openai.com/docs/guides/agents)

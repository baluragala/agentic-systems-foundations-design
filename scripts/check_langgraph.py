"""
Verify the LangGraph track's wiring — without an API key.

The teaching path for `agent_lc` uses a real ChatOpenAI. This script uses the
test fixture in `scripts/_stub_model.py` so that graph construction, the cycle, the
reducer, termination and the trace adapter are all *executed* in CI rather than
assumed. "It needs a key" is a reason to build a test double, not a reason to
ship unverified graphs.

Run:  python scripts/check_langgraph.py
"""
import os
import pathlib
import sys

from langchain_core.messages import HumanMessage

from agent_core.trace import compare
from agent_lc import (
    KeywordSupervisor,
    acme_lc_skills,
    build_agent_graph,
    build_conversational,
    build_diagnostic_graph,
    build_prebuilt_agent,
    call_sequence,
    describe,
    draw,
    final_answer,
    show_messages,
    subset,
    to_trace,
)
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from _stub_model import (
    ACME_LOOKUP_SCRIPT,
    LoopingChatModel,
    MalformedArgsChatModel,
    ScriptedChatModel,
)
from agent_lc.tools_lc import ACME_TOOLS

fails = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if not cond else ""))
    if not cond:
        fails.append(label)


def start(goal):
    return {"messages": [HumanMessage(goal)], "steps": 0, "stop_reason": None}


model = ScriptedChatModel(script=ACME_LOOKUP_SCRIPT)
GOAL = "Is order ACME-1046 refundable? I changed my mind."

# --- 1. tools & schemas -----------------------------------------------------
print("=== 1. tools & Pydantic schemas ===")
check("5 LangChain tools", len(ACME_TOOLS) == 5)
schemas = {t.name: t.args_schema.model_json_schema() for t in ACME_TOOLS}
check("pattern survives to schema",
      "pattern" in schemas["get_order_status"]["properties"]["order_id"])
check("enum survives to schema",
      "enum" in schemas["check_refund_eligibility"]["properties"]["reason"])
check("descriptions present",
      all("description" in p
          for s in schemas.values() for p in s["properties"].values()))
try:
    ACME_TOOLS[1].invoke({"order_id": "1046"})
    check("Pydantic rejects a malformed ID", False, "it was accepted")
except Exception:
    check("Pydantic rejects a malformed ID", True)
check("subset() scopes", len(subset("search_docs", "calculate")) == 2)

# --- 2. the hand-built graph ------------------------------------------------
print("\n=== 2. the loop as a StateGraph ===")
graph = build_agent_graph(model, system_prompt="Answer using tools.", max_steps=6)
nodes = set(graph.get_graph().nodes)
check("graph has agent + tools nodes", {"agent", "tools"} <= nodes, nodes)
out = graph.invoke(start(GOAL))
check("loop cycled more than once", out["steps"] >= 2, out["steps"])
check("add_messages appended", len(out["messages"]) > 2, len(out["messages"]))
check("tools were called", len(call_sequence(out)) >= 1, call_sequence(out))
check("produced a final answer", bool(final_answer(out)))
print("     trajectory:", " → ".join(call_sequence(out)))

# --- 3. termination ---------------------------------------------------------
print("\n=== 3. termination ===")
looping = build_diagnostic_graph(LoopingChatModel(), max_steps=8)
diag = looping.invoke(start("What is the status of order ACME-1042?"))
check("diagnostic stops early", diag["steps"] < 8, diag["steps"])
check("stop_reason names repetition", "repetition" in (diag["stop_reason"] or ""),
      diag["stop_reason"])

budget = build_agent_graph(LoopingChatModel(), max_steps=4)
bud = budget.invoke(start("What is the status of order ACME-1042?"))
check("budget backstop fires", bud["steps"] <= 4, bud["steps"])
check("diagnostic beats budget-only", diag["steps"] < bud["steps"],
      f"diag={diag['steps']} budget={bud['steps']}")

# Script the escalation explicitly — the model must request a tool that is
# actually in scope, or it just errors and the terminal check proves nothing.
escalator = ScriptedChatModel(
    script=[("escalate_to_human", {"summary": "Customer needs a human for a billing problem."})]
)
term = build_agent_graph(escalator, tools=subset("escalate_to_human"), max_steps=8)
esc = term.invoke(start("I need a human to look at this billing problem please"))
check("terminal tool ends at once", esc["steps"] == 1, esc["steps"])

# --- 4. tool errors never kill the graph ------------------------------------
print("\n=== 4. tool errors are handled, not fatal ===")
bad = build_agent_graph(MalformedArgsChatModel(), max_steps=4)
try:
    berr = bad.invoke(start(GOAL))
    tool_msgs = [m for m in berr["messages"] if getattr(m, "type", "") == "tool"]
    check("graph survived a bad tool call", True)
    check("error came back as a ToolMessage", len(tool_msgs) >= 1, len(tool_msgs))
except Exception as exc:
    check("graph survived a bad tool call", False, f"{type(exc).__name__}: {exc}")

# --- 5. skills & routing ----------------------------------------------------
print("\n=== 5. skills & routing ===")
skills = acme_lc_skills()
check("3 skills", len(skills) == 3)
check("all skills scoped below full toolbox",
      all(len(s.tools) < len(ACME_TOOLS) for s in skills),
      [(s.name, len(s.tools)) for s in skills])
sup = KeywordSupervisor(skills)
routes = {
    "I want a refund on ACME-1046, I changed my mind.": "refunds",
    "How much does the Growth plan cost per month?": "product_questions",
    "What is the status of order ACME-1048?": "account_lookup",
    "What is 199 multiplied by 12?": "product_questions",
}
for goal, expected in routes.items():
    got = sup.route(goal).name
    check(f"routes -> {expected}", got == expected, f"got {got} for {goal!r}")
skill_graph = skills[0].build(model)
sres = skill_graph.invoke(start(GOAL))
check("a scoped skill graph runs", len(call_sequence(sres)) >= 1, call_sequence(sres))

# --- 6. prebuilt + memory ---------------------------------------------------
print("\n=== 6. create_react_agent & checkpointing ===")
pre = build_prebuilt_agent(model, system_prompt="Answer using tools.")
pout = pre.invoke({"messages": [HumanMessage(GOAL)]})
check("prebuilt agent runs", len(pout["messages"]) > 2, len(pout["messages"]))
check("prebuilt calls tools too", len(call_sequence(pout)) >= 1, call_sequence(pout))

conv = build_conversational(model)
cfg = {"configurable": {"thread_id": "t1"}}
conv.invoke({"messages": [HumanMessage("What is the status of order ACME-1048?")]}, cfg)
second = conv.invoke({"messages": [HumanMessage("And what plan is it on?")]}, cfg)
check("checkpointer carried the thread", len(second["messages"]) > 3, len(second["messages"]))

# --- 7. the trace adapter ---------------------------------------------------
print("\n=== 7. trace adapter — both tracks, one table ===")
lc_trace = to_trace(out, GOAL)
check("adapter produced steps", len(lc_trace.steps) >= 2, len(lc_trace.steps))
check("adapter preserved trajectory", lc_trace.call_sequence() == call_sequence(out),
      (lc_trace.call_sequence(), call_sequence(out)))
check("adapter computed an error rate", isinstance(lc_trace.error_rate(), float))

# The cross-engine comparison needs a real model on both sides, so it only
# runs when a key is present. The adapter itself is verified above.
if os.getenv("OPENAI_API_KEY"):
    from agent_core import Agent
    core_trace = Agent().run(GOAL).trace
    print()
    print(compare({"from scratch": core_trace, "langgraph": lc_trace}))
    check("comparable across engines", True)
else:
    print("     (cross-engine comparison skipped — needs OPENAI_API_KEY)")

print()
print("--- graph shape ---")
print(draw(graph))
print()
print("--- message transcript ---")
print(show_messages(out))

print()
print("ALL LANGGRAPH CHECKS PASS" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)

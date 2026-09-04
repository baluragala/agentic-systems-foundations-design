"""
Verify `agent_core` — the parts that do not need a model, plus a live run if a key is set.

The package calls a real OpenAI model; there is no simulated provider. So this
script splits into two halves, and the split is itself informative:

  DETERMINISTIC (always runs, costs nothing)
      schemas, validation and coercion, tool dispatch, "tools never raise",
      state bookkeeping, termination conditions, trace serialisation and replay,
      skill scoping and routing, the failure catalogue.
      -> That is most of the package. Almost none of an agent's machinery
         actually needs a model; it needs one only at the THINK step.

  LIVE (needs OPENAI_API_KEY, spends a little)
      the loop end to end, and the task suite.

Run:  python scripts/smoke_test.py
"""
import json
import os
import sys

from agent_core import (
    Agent, Budget, ToolRegistry, TerminationPolicy, Trace, compare, tool,
    score_suite, Skill, Router, acme_skills, acme_router, CATALOGUE,
    show_catalogue, diagnose, report, have_api_key,
)
from agent_core.acme_tools import acme_registry, order_ids
from agent_core.schemas import validate_args, to_openai, to_gemini, to_anthropic
from agent_core.state import AgentState, AgentStatus, Observation, ObservationStatus

fails = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if not cond else ""))
    if not cond:
        fails.append(label)


reg = acme_registry()

# ===========================================================================
# DETERMINISTIC — no model involved
# ===========================================================================
print("=== 1. registry & schemas ===")
check("5 tools registered", len(reg) == 5)
check("orders loaded", len(order_ids()) == 7, order_ids())
check("enum present", "enum" in reg.get("check_refund_eligibility").schema["properties"]["reason"])
check("pattern present", "pattern" in reg.get("get_order_status").schema["properties"]["order_id"])
check("example survives into description",
      "ACME-1042" in reg.get("get_order_status").schema["properties"]["order_id"]["description"])
check("openai envelope", to_openai("x", "d", reg.get("calculate").schema)["type"] == "function")
check("gemini drops addlProps",
      "additionalProperties" not in to_gemini("x", "d", reg.get("calculate").schema)["parameters"])
check("anthropic key", "input_schema" in to_anthropic("x", "d", reg.get("calculate").schema))

print("\n=== 2. validation & coercion ===")
v = validate_args({"query": "pricing", "k": "3"}, reg.get("search_docs").schema)
check("coerces '3'->3", v.ok and v.args["k"] == 3, v.errors)
check("coercion recorded", len(v.coercions) == 1)
check("rejects bad pattern", not validate_args({"order_id": "1042"}, reg.get("get_order_status").schema).ok)
check("rejects unknown param",
      not validate_args({"order_id": "ACME-1042", "bogus": 1}, reg.get("get_order_status").schema).ok)
check("rejects bad enum",
      not validate_args({"order_id": "ACME-1042", "reason": "vibes"},
                        reg.get("check_refund_eligibility").schema).ok)
check("rejects bool for int",
      not validate_args({"query": "x", "k": True}, reg.get("search_docs").schema).ok)

print("\n=== 3. tools never raise ===")
check("unknown tool -> rejected", reg.dispatch("nope", {}).status.value == "rejected")
o = reg.dispatch("calculate", {"expression": "1/0"})
check("div-by-zero -> ok w/ msg", o.ok and "zero" in str(o.result).lower(), o.result)
check("code injection blocked", "REJECTED" in str(reg.dispatch("calculate", {"expression": "__import__('os')"}).result))
check("unknown order -> NOT FOUND", "NOT FOUND" in str(reg.dispatch("get_order_status", {"order_id": "ACME-9999"}).result))
check("Enterprise refund refused",
      "Escalate" in str(reg.dispatch("check_refund_eligibility",
                                     {"order_id": "ACME-1044", "reason": "billing_error"}).result))
check("already-refunded refused",
      "NOT ELIGIBLE" in str(reg.dispatch("check_refund_eligibility",
                                         {"order_id": "ACME-1047", "reason": "billing_error"}).result))

print("\n=== 4. state bookkeeping ===")
state = AgentState(goal="test").start("system")
for i in range(3):
    state.record_observation(Observation(step=i, tool="t", args={"a": 1},
                                         status=ObservationStatus.OK, result="same"))
check("detects repeated calls", state.repeated_calls(threshold=3), state.repeated_calls())
check("evidence collects results", "same" in state.evidence())
state.record_observation(Observation(step=4, tool="t", args={}, status=ObservationStatus.ERROR, error="boom"))
check("counts consecutive errors", state.consecutive_errors() == 1)
check("estimates context", state.approx_tokens() > 0)

print("\n=== 5. termination conditions ===")
policy = TerminationPolicy()
check("diagnostic conditions before budget",
      policy.names().index("repetition") < policy.names().index("budget"), policy.names())
check("repetition fires on identical calls", policy.check(state) is not None)
check("budget-only policy is just the backstop",
      TerminationPolicy.budget_only().names() == ["budget"])
tight = AgentState(goal="g", budget=Budget(max_steps=1)).start("s")
tight.step = 1
check("budget backstop fires", TerminationPolicy.budget_only().check(tight) is not None)

print("\n=== 6. trace round-trip ===")
t = Trace(goal="g")
from agent_core.trace import StepRecord
rec = StepRecord(step=1, decision="Decision(ACT: x())")
rec.observations.append(Observation(step=1, tool="x", args={}, status=ObservationStatus.OK, result="r"))
t.steps.append(rec)
back = Trace.from_dict(json.loads(t.to_json()))
check("replay preserves steps", len(back.steps) == len(t.steps))
check("replay preserves trajectory", back.call_sequence() == t.call_sequence())
check("compare renders", "run" in compare({"a": t, "b": back}))

print("\n=== 7. skills & routing ===")
router = acme_router(reg)
check("routes refund", router.route("I want a refund for ACME-1042").name == "refunds")
check("routes pricing", router.route("How much is the Growth plan?").name == "product_questions")
check("routes status", router.route("What is the status of order ACME-1048?").name == "account_lookup")
check("arithmetic hits the general fallback",
      router.route("What is 199 multiplied by 12?").name == "product_questions")
check("compose unions tools", len(Skill.compose(*acme_skills(reg)).tools) == 5)
check("every skill is scoped below the full toolbox",
      all(len(s.tools) < 5 for s in acme_skills(reg)))

print("\n=== 8. failure catalogue ===")
check("five failure modes", len(CATALOGUE) == 5, list(CATALOGUE))
check("two are quiet", sum(1 for f in CATALOGUE.values() if not f.loud) == 2)
check("catalogue renders", "QUIET" in show_catalogue())
empty = Trace(goal="g")
check("diagnose flags a run with no tool calls", len(diagnose(empty)) >= 1)
check("report renders", "DIAGNOSIS" in report(empty))

# ===========================================================================
# LIVE — needs a key
# ===========================================================================
print("\n=== 9. live agent run ===")
if not have_api_key():
    print("SKIP  no OPENAI_API_KEY — the loop and the task suite need a real model.")
    print("      Everything above is deterministic and was fully verified.")
else:
    agent = Agent(budget=Budget(max_steps=4, max_tool_calls=5))
    result = agent.run("What is the status of order ACME-1048?")
    check("loop terminates", result.state.status.is_terminal, result.state.status)
    check("called a tool", result.trace.tool_calls() >= 1, result.tools_called())
    check("produced an answer", bool(result.answer))
    check("trace has steps", len(result.trace.steps) >= 1)
    print("     trajectory:", " → ".join(result.tools_called()))

    print("\n=== 10. task suite ===")
    tasks_path = "data/tasks/agent_tasks.jsonl"
    if os.path.exists(tasks_path):
        tasks = [json.loads(line) for line in open(tasks_path) if line.strip()]
        results = agent.run_suite(tasks)
        print(score_suite(results))
        scored = [r for r in results if r["passed"] is not None]
        rate = sum(1 for r in scored if r["passed"]) / max(1, len(scored))
        check("suite pass rate >= 70%", rate >= 0.7, f"{rate:.0%}")

print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
sys.exit(1 if fails else 0)

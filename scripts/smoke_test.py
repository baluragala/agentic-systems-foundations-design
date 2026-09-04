"""Exercise every module of agent_core under the offline mock."""
import os
os.environ["AGENT_LLM_PROVIDER"] = "mock"

from agent_core import (Agent, MockToolCallLLM, ToolRegistry, tool, Trace,
                        TerminationPolicy, Budget, compare, broken_agent,
                        report, show_catalogue, score_suite, Skill, Router,
                        acme_skills)
from agent_core.acme_tools import acme_registry, order_ids
from agent_core.schemas import validate_args, to_openai, to_gemini, to_anthropic

fails = []
def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"   {detail}" if detail and not cond else ""))
    if not cond: fails.append(label)

reg = acme_registry()
print("=== 1. registry & schemas ===")
check("5 tools registered", len(reg) == 5)
check("orders loaded", len(order_ids()) == 7, order_ids())
check("enum present", "enum" in reg.get("check_refund_eligibility").schema["properties"]["reason"])
check("pattern present", "pattern" in reg.get("get_order_status").schema["properties"]["order_id"])
check("openai envelope", to_openai("x","d",reg.get("calculate").schema)["type"] == "function")
check("gemini drops addlProps", "additionalProperties" not in to_gemini("x","d",reg.get("calculate").schema)["parameters"])
check("anthropic key", "input_schema" in to_anthropic("x","d",reg.get("calculate").schema))

print("\n=== 2. validation & coercion ===")
s = reg.get("search_docs").schema
v = validate_args({"query":"pricing","k":"3"}, s)
check("coerces '3'->3", v.ok and v.args["k"] == 3, v.errors)
check("coercion recorded", len(v.coercions) == 1)
v2 = validate_args({"order_id":"1042"}, reg.get("get_order_status").schema)
check("rejects bad pattern", not v2.ok, v2.errors)
v3 = validate_args({"order_id":"ACME-1042","bogus":1}, reg.get("get_order_status").schema)
check("rejects unknown param", not v3.ok, v3.errors)
v4 = validate_args({"order_id":"ACME-1042","reason":"vibes"}, reg.get("check_refund_eligibility").schema)
check("rejects bad enum", not v4.ok, v4.errors)

print("\n=== 3. tools never raise ===")
o = reg.dispatch("nope", {})
check("unknown tool -> rejected", o.status.value == "rejected")
o = reg.dispatch("calculate", {"expression":"1/0"})
check("div-by-zero -> ok w/ msg", o.ok and "zero" in str(o.result).lower(), o.result)
o = reg.dispatch("calculate", {"expression":"__import__('os')"})
check("code injection blocked", "REJECTED" in str(o.result), o.result)
o = reg.dispatch("get_order_status", {"order_id":"ACME-9999"})
check("unknown order -> NOT FOUND", "NOT FOUND" in str(o.result))

print("\n=== 4. the loop ===")
a = Agent()
r = a.run("What does order ACME-1046 cost and what is its status?")
check("loop terminates", r.state.status.is_terminal)
check("called a tool", r.trace.tool_calls() >= 1, r.tools_called())
check("no errors", r.trace.error_rate() == 0.0)

print("\n=== 5. stateful vs stateless (notebook 02) ===")
sf = a.run("What is the status of order ACME-1048?", stateful=True)
sl = a.run("What is the status of order ACME-1048?", stateful=False)
check("stateless repeats the same call", len(set(sl.trace.call_sequence())) == 1
      and len(sl.trace.call_sequence()) >= 3, sl.trace.call_sequence())
check("stateful does not repeat", len(set(sf.trace.call_sequence())) == len(sf.trace.call_sequence()),
      sf.trace.call_sequence())
check("stateless stopped by repetition", "repetition" in (sl.state.stop_reason or ""), sl.state.stop_reason)
check("stateful finished cleanly", sf.ok, sf.state.stop_reason)
print(compare({"stateful": sf.trace, "stateless": sl.trace}))

print("\n=== 6. termination policies (notebook 05) ===")
loopy = broken_agent("no_progress_loop")
budget_only = broken_agent("no_progress_loop", policy=TerminationPolicy.budget_only())
t1 = loopy.run("Check order ACME-1042 please").trace
t2 = budget_only.run("Check order ACME-1042 please").trace
check("diagnostic policy stops earlier", len(t1.steps) < len(t2.steps),
      f"diag={len(t1.steps)} budget={len(t2.steps)}")
print(compare({"diagnostic": t1, "budget-only": t2}))

print("\n=== 7. failure catalogue (notebook 06) ===")
for key in ("no_progress_loop","hallucinated_tool","schema_violation","wrong_tool","ungrounded_answer"):
    br = broken_agent(key)
    tr = br.run("Is order ACME-1042 refundable? It was a billing_error.").trace
    findings = __import__("agent_core").diagnose(tr, expected_tools=["get_order_status"])
    check(f"{key} produces findings", len(findings) > 0, tr.summary())

print("\n=== 8. trace round-trip ===")
d = r.trace.to_dict()
back = Trace.from_dict(d)
check("replay preserves steps", len(back.steps) == len(r.trace.steps))
check("replay preserves calls", back.call_sequence() == r.trace.call_sequence())

print("\n=== 9. skills & routing ===")
router = Router(acme_skills(reg))
check("routes refund", router.route("I want a refund for ACME-1042").name == "refunds")
check("routes pricing", router.route("How much is the Growth plan?").name == "product_questions")
merged = Skill.compose(*acme_skills(reg))
check("compose unions tools", len(merged.tools) == 5, merged.tools.names())
check("scoped < full", all(len(s.tools) < 5 for s in acme_skills(reg)))

print("\n=== 10. reflection path ===")
ra = Agent(use_reflection=True)
rr = ra.run("What is the status of order ACME-1044?")
check("reflection run completes", rr.state.status.is_terminal)

print("\n" + ("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}"))
raise SystemExit(1 if fails else 0)

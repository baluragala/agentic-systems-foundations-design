"""Verify every runnable claim in teaching/solutions/solutions.md."""
import os
import re
from typing import Literal

os.environ["AGENT_LLM_PROVIDER"] = "mock"

from agent_core import (Agent, Budget, ToolRegistry, TerminationPolicy, tool)
from agent_core.control import TerminationCondition
from agent_core.acme_tools import acme_registry

fails = []


def check(label, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {label}" + (f"  {detail}" if not cond else ""))
    if not cond:
        fails.append(label)


# --- 3.1 the rewritten tool -------------------------------------------------
@tool(examples=["Upgrade ACME-1046 to the Growth plan"])
def update_subscription(
    order_id: str,
    new_plan: Literal["Starter", "Growth", "Enterprise"],
    apply_immediately: bool = True,
) -> str:
    """Change the subscription plan on an existing order.

    Use this when a customer asks to upgrade, downgrade, or switch plans.

    Args:
        order_id: The Acme order identifier, for example ACME-1046. pattern: ^ACME-\\d{4}$
        new_plan: The plan to move the subscription to.
        apply_immediately: True to change now and prorate; False at next renewal.

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


props = update_subscription.schema["properties"]
check("3.1 enum on new_plan", "enum" in props["new_plan"], props["new_plan"])
check("3.1 pattern on order_id", "pattern" in props["order_id"], props["order_id"])
check("3.1 bool type", props["apply_immediately"]["type"] == "boolean", props["apply_immediately"])
reg = ToolRegistry([update_subscription])
obs = reg.dispatch("update_subscription", {"oid": 1046, "p": "Gold"})
check("3.1 bad payload rejected, not raised", obs.status.value == "rejected", obs.error)
check("3.1 error names valid params", "order_id" in (obs.error or ""), obs.error)
obs2 = reg.dispatch("update_subscription", {"order_id": "ACME-1046", "new_plan": "Gold"})
check("3.1 bad enum rejected", obs2.status.value == "rejected", obs2.error)
check("3.1 enum error lists options", "Starter" in (obs2.error or ""), obs2.error)
obs3 = reg.dispatch("update_subscription", {"order_id": "ACME-9999", "new_plan": "Growth"})
check("3.1 unknown order -> NOT FOUND prose", obs3.ok and "NOT FOUND" in str(obs3.result), obs3)

# --- 5.2 cost_ceiling -------------------------------------------------------
def cost_ceiling(max_tokens_sent: int = 8000) -> TerminationCondition:
    sent = {"total": 0, "last": 0}

    def check_fn(state):
        current = state.approx_tokens()
        if current != sent["last"]:
            sent["total"] += current
            sent["last"] = current
        if sent["total"] > max_tokens_sent:
            return (f"cost_ceiling: ~{sent['total']} cumulative tokens sent "
                    f"exceeds {max_tokens_sent}")
        return None

    return TerminationCondition("cost_ceiling", check_fn)


policy = TerminationPolicy([cost_ceiling(1500), *TerminationPolicy.default()])
run = Agent(policy=policy).run(
    "Check the status of order ACME-1043 and whether I can refund it for changed_mind."
)
check("5.2 cost_ceiling fires", "cost_ceiling" in (run.state.stop_reason or ""), run.state.stop_reason)

# --- 6.4 check_grounding ----------------------------------------------------
def check_grounding(result):
    evidence = " ".join(str(o.result or "") for o in result.state.observations if o.ok)
    figures = set(re.findall(r"\d[\d,]*\.?\d*", result.answer or ""))
    return sorted(f for f in figures if len(f) > 2 and f not in evidence)


from agent_core import broken_agent
bad = broken_agent("ungrounded_answer").run("Is order ACME-1042 refundable? It was a billing_error.")
ungrounded = check_grounding(bad)
check("6.4 flags invented figures", len(ungrounded) > 0, ungrounded)
print(f"       invented figures found: {ungrounded}")

good = Agent().run("What is the status of order ACME-1046?")
fp = check_grounding(good)
print(f"       false positives on a good run: {fp}")

print()
print("ALL SOLUTION CLAIMS VERIFIED" if not fails else f"FAILED: {fails}")
raise SystemExit(1 if fails else 0)

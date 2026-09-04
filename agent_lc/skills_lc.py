"""
skills_lc.py — scoped skills and a supervisor, in LangGraph.
============================================================

WHY this file exists
--------------------
`agent_core/skills.py` argued that a skill is *instructions + a scoped tool
subset + a stop policy*, and that the scoping is the load-bearing part. That
argument was never about our implementation — it is about how models behave, so
it survives the change of engine completely.

What changes is the vocabulary. What LangGraph calls a **supervisor** is what we
called a `Router`, and where we picked a skill with a keyword scorer, production
systems route with a cheap fast model. Both are here, so you can see that
routing quality is a *dial*, not a fixed property of the design.

TWO WAYS TO ROUTE, AND WHEN TO USE EACH
---------------------------------------
1. **Keyword routing** (`KeywordSupervisor`) — free, instant, deterministic,
   inspectable. Genuinely adequate for a handful of well-separated skills, and it
   is what you should reach for first. The bare-word-trigger bug from
   `agent_core/skills.py` is just as real here, so the same phrase discipline
   applies.

2. **LLM routing** (`build_supervisor_graph`) — a cheap model classifies the
   request into one skill, then that skill's graph runs. Handles paraphrase and
   novel wording that keywords miss, costs one extra call per request, and can
   itself be wrong. This is the production default once you pass roughly a dozen
   skills.

The important thing is that **both leave the executing agent seeing only its own
skill's tools.** That is the property that buys reliability, and neither routing
strategy changes it.

WHAT LANGGRAPH ADDS THAT WE DID NOT HAVE
----------------------------------------
Real **handoffs**. In `agent_core`, a `Router` picks one skill and that is the
whole story — a request needing two jobs is answered by whichever skill won.
Here, skills are nodes in a graph, so a skill can route *onward* to another and
the state carries across. That is the honest answer to the "when composing is
right" exercise: at scale you neither compose nor accept a half-answer — you hand
off.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .tools_lc import describe, subset

# The shared rules every skill inherits. Identical in substance to
# `agent_core.skills.BASE_INSTRUCTIONS` — each rule exists because omitting it
# produces a specific failure from the notebook-06 taxonomy, and that mapping is
# a property of how models behave, not of what is running the loop.
BASE_INSTRUCTIONS = """\
You are a careful assistant that works by calling tools.

Rules you must follow:
1. Gather evidence before answering. If a tool can confirm something, call it —
   do not answer from memory or assumption.
2. Use each tool for its stated purpose only. Read the parameter descriptions
   and match the required formats exactly.
3. If a tool returns an error, read it, fix your arguments, and try again. Do
   not repeat an identical failing call.
4. If a tool says NOT FOUND, that is the answer. Report it. Never invent a
   plausible substitute.
5. When you have enough evidence, answer directly and cite what you found. If
   you do not have enough, say so plainly.
"""


@dataclass
class LcSkill:
    """A named job: instructions, a scoped tool list, and its own step budget."""

    name: str
    instructions: str
    tools: List
    triggers: List[str] = field(default_factory=list)
    max_steps: int = 8

    def system_prompt(self) -> str:
        return (
            f"{BASE_INSTRUCTIONS}\n"
            f"YOUR CURRENT JOB — {self.name}:\n{self.instructions.strip()}\n\n"
            f"TOOLS AVAILABLE TO YOU:\n{describe(self.tools)}\n"
        )

    def matches(self, goal: str) -> int:
        low = goal.lower()
        score = sum(10 for pattern in self.triggers if re.search(pattern, low))
        score += sum(3 for w in re.findall(r"[a-z]{4,}", self.name.lower()) if w in low)
        keywords = set(re.findall(r"[a-z]{5,}", self.instructions.lower()))
        return score + sum(1 for w in keywords if w in low)

    def build(self, model, diagnostic: bool = True):
        """Compile this skill into its own graph, scoped to its own tools."""
        from .graph import build_agent_graph, build_diagnostic_graph

        builder = build_diagnostic_graph if diagnostic else build_agent_graph
        return builder(
            model,
            tools=self.tools,
            system_prompt=self.system_prompt(),
            max_steps=self.max_steps,
        )

    def __str__(self) -> str:
        return f"LcSkill({self.name}: {len(self.tools)} tools — {', '.join(t.name for t in self.tools)})"


# ---------------------------------------------------------------------------
# The Acme skill set — same scoping decisions as agent_core
# ---------------------------------------------------------------------------
def acme_lc_skills() -> List[LcSkill]:
    """
    Three skills over the LangChain toolbox.

    The scoping and the triggers are deliberately identical to
    `agent_core.skills.acme_skills`, so any behavioural difference you observe
    between the two tracks comes from the *engine*, not from someone quietly
    making better design choices on one side.
    """
    return [
        LcSkill(
            name="refunds",
            instructions=(
                "Handle refund and cancellation requests. ALWAYS look up the order "
                "first with get_order_status — you cannot assess a refund without "
                "knowing the order's status and date. Then use "
                "check_refund_eligibility with the customer's stated reason. "
                "If the policy says a human must decide, escalate with a summary "
                "of what you checked. Never promise a specific refund date."
            ),
            tools=subset(
                "get_order_status", "check_refund_eligibility", "search_docs",
                "escalate_to_human",
            ),
            triggers=[
                r"refund", r"money back", r"cancel", r"charged", r"reimburse",
                r"billing_error", r"service_outage", r"not_as_described",
                r"changed_mind",
            ],
        ),
        LcSkill(
            name="product_questions",
            instructions=(
                "Answer questions about Acme Cloud plans, pricing, security, SLAs, "
                "the API and onboarding, using the documentation. Search first and "
                "answer only from what you find. If the documentation does not "
                "cover it, say so — do not fill the gap from general knowledge "
                "about cloud providers. Use calculate for any arithmetic."
            ),
            tools=subset("search_docs", "calculate"),
            triggers=[r"plan", r"pric", r"cost", r"sla", r"security", r"api",
                      r"storage", r"seats", r"trial", r"onboard",
                      r"multipl", r"percent", r"how much is", r"total", r"annual"],
        ),
        LcSkill(
            name="account_lookup",
            instructions=(
                "Look up the status of specific orders. Report exactly what the "
                "order system returns. If an order does not exist, say so — never "
                "guess at a similar ID. If the customer needs something beyond a "
                "status lookup, escalate rather than improvising."
            ),
            tools=subset("get_order_status", "escalate_to_human"),
            # PHRASES, not bare words — the same routing bug documented in
            # agent_core/skills.py bites here identically.
            triggers=[
                r"status of", r"look ?up .*order", r"\border\b.*\bstatus\b",
                r"delivery", r"shipped", r"tracking", r"when will .* arrive",
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Routing strategy 1 — keyword supervisor
# ---------------------------------------------------------------------------
class KeywordSupervisor:
    """Deterministic, inspectable routing. The `agent_core.Router` equivalent."""

    def __init__(self, skills: List[LcSkill], fallback: Optional[LcSkill] = None):
        if not skills:
            raise ValueError("need at least one skill")
        self.skills = skills
        # Set the fallback explicitly. Defaulting to skills[0] means unmatched
        # requests land in whichever skill happens to be listed first — a
        # one-line configuration bug that is invisible until you print the route.
        self.fallback = fallback or next(
            (s for s in skills if s.name == "product_questions"), skills[0]
        )

    def route(self, goal: str) -> LcSkill:
        best = max(self.skills, key=lambda s: s.matches(goal))
        return best if best.matches(goal) > 0 else self.fallback

    def explain(self, goal: str) -> str:
        chosen = self.route(goal)
        rows = sorted(((s.matches(goal), s.name) for s in self.skills), reverse=True)
        lines = [f"Routing {goal!r}:"]
        lines += [
            f"  {'→' if name == chosen.name else ' '} {name:<20} score {score}"
            for score, name in rows
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Routing strategy 2 — an LLM supervisor, as a graph
# ---------------------------------------------------------------------------
def build_supervisor_graph(model, skills: Optional[List[LcSkill]] = None):
    """
    A supervisor node that classifies the request, then runs one skill's graph.

    This is the production pattern, and the shape most multi-agent frameworks
    converge on. Two things are worth noticing in the code below:

    * The supervisor gets **structured output** (`with_structured_output`), not
      free text. Asking a model to "reply with the skill name" and then parsing
      prose is a classic source of routing flakiness — constrain it instead.
    * The routing decision is written into state, so it appears in the trace.
      *"The agent did the wrong job"* is nearly always a routing bug, and you
      cannot diagnose one you did not record.
    """
    from typing import Annotated, Literal, Optional as Opt, TypedDict

    from langchain_core.messages import AnyMessage, HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages
    from pydantic import BaseModel, Field

    skills = skills or acme_lc_skills()
    by_name = {s.name: s for s in skills}
    names = list(by_name)

    class SupervisorState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]
        steps: int
        stop_reason: Opt[str]
        skill: Opt[str]

    class Route(BaseModel):
        """The supervisor's decision, constrained to a real skill name."""

        skill: Literal[tuple(names)] = Field(  # type: ignore[valid-type]
            description="Which specialist should handle this request."
        )
        reason: str = Field(description="One short sentence on why.")

    catalogue = "\n".join(f"- {s.name}: {s.instructions.splitlines()[0]}" for s in skills)
    router_model = model.with_structured_output(Route)

    def supervise(state: SupervisorState) -> dict:
        goal = next(
            (m.content for m in state["messages"] if isinstance(m, HumanMessage)), ""
        )
        decision = router_model.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Route the request to exactly one specialist.\n\n"
                        f"SPECIALISTS:\n{catalogue}"
                    ),
                },
                {"role": "user", "content": str(goal)},
            ]
        )
        return {"skill": decision.skill, "stop_reason": f"routed: {decision.reason}"}

    def run_skill(state: SupervisorState) -> dict:
        skill = by_name[state["skill"]]
        # The executing agent sees ONLY this skill's tools. That scoping is the
        # whole reliability argument, and it is preserved by construction.
        result = skill.build(model).invoke(
            {"messages": state["messages"], "steps": 0, "stop_reason": None}
        )
        return {
            "messages": result["messages"][len(state["messages"]):],
            "steps": result.get("steps", 0),
            "stop_reason": result.get("stop_reason"),
        }

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervise)
    graph.add_node("skill", run_skill)
    graph.add_edge(START, "supervisor")
    graph.add_edge("supervisor", "skill")
    graph.add_edge("skill", END)
    return graph.compile()

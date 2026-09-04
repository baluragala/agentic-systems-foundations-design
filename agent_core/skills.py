"""
skills.py — composable units of agent competence.
=================================================

WHY this file exists
--------------------
You have a working agent with five tools. It handles refunds well. Now add
billing disputes, then onboarding, then incident triage.

The obvious move is to keep growing the system prompt and the tool list. It
works until it doesn't, and it fails in two ways at once:

  * **Tool choice degrades.** Every tool you add is another wrong option. With
    five tools the model picks well; with twenty-five it picks plausibly. The
    failure is silent — a reasonable-looking call to the wrong tool.
  * **The prompt becomes unownable.** Instructions for four unrelated jobs sit
    in one block of text, they start contradicting each other, and no one can
    change the refund wording without risking the onboarding behaviour.

A **skill** is the unit that fixes both: a *named job*, with the instructions for
that job and only the tools that job needs.

    Skill("refunds",
          instructions="You handle refund requests. Check the order first…",
          tools=registry.subset("get_order_status", "check_refund_eligibility",
                                "escalate_to_human"))

Now the refund agent chooses among three tools instead of twenty-five, its
instructions fit on a screen, and you can change them without touching anything
else.

WHAT A SKILL IS, PRECISELY
--------------------------
Three things bound together:

    instructions  +  a scoped tool subset  +  a termination policy

The scoping is the load-bearing part. A skill that has the instructions but not
the restricted tool list is just a prompt template, and it will not give you the
reliability improvement — the model can still reach for the wrong tool.

WHAT A SKILL IS NOT
-------------------
Not a subclass. Not a separate agent process. Not a framework. A `Skill` is a
small dataclass that configures the same loop from `loop.py`. That is
deliberate: the composition story should not require a second execution model,
and a learner should be able to see that "skills" adds no new machinery, only
scoping.

COMPOSITION
-----------
Two ways, and knowing which to reach for is the actual skill:

  `Skill.compose(a, b)` — merge into one skill with the union of tools and
      concatenated instructions. Use when one agent genuinely needs both jobs at
      once. Costs you the scoping benefit, so merge sparingly.

  `Router` — pick ONE skill per request, then run only that. Use when requests
      are separable, which they usually are. This is how you get to twenty-five
      tools without any single agent ever seeing more than four.

The recurring question — *"what does this step look like when it goes wrong, and
where would you see it in the trace?"*:

  * Too many tools in one skill -> trace shows a call to a tool irrelevant to
                                   the request. Classic over-scoping.
  * Router picks wrong skill    -> the trace shows a competent agent solving
                                   the wrong problem. Look at the routing
                                   decision, not the loop.
  * Vague instructions          -> the agent answers without calling anything.
                                   Zero tool calls plus a fluent answer is
                                   always a red flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .control import TerminationPolicy
from .tools import ToolRegistry

# ---------------------------------------------------------------------------
# The base prompt every skill inherits
# ---------------------------------------------------------------------------
# WHY these particular rules: each one exists because omitting it produces a
# specific, observed failure. This is not prompt folklore — every line maps to a
# failure mode in notebook 06.
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


# ---------------------------------------------------------------------------
# Skill
# ---------------------------------------------------------------------------
@dataclass
class Skill:
    """
    A named job: what to do, which tools to do it with, and when to stop.

    Attributes:
        name: Used in traces and by the Router. Make it a job, not a category —
            "refunds", not "utilities".
        instructions: The job-specific part of the system prompt. BASE_INSTRUCTIONS
            is prepended automatically, so write only what is special here.
        tools: The scoped registry. Smaller is more reliable.
        policy: When to stop. Skills can differ — a research skill may warrant a
            larger budget than a lookup skill.
        triggers: Regex hints used by the Router. Optional; the Router falls
            back to word overlap with the name and instructions.
    """

    name: str
    instructions: str
    tools: ToolRegistry
    policy: Optional[TerminationPolicy] = None
    triggers: List[str] = field(default_factory=list)

    def system_prompt(self) -> str:
        """
        The full system prompt: base rules, the job, then the tool inventory.

        Listing the tools in the prompt is belt-and-braces — providers already
        send schemas separately through the API. It matters anyway, because the
        prose list is where you say *when* to use each tool, and "when" is the
        decision the model actually gets wrong. A schema describes the shape of
        a call; only the prompt describes the judgement.
        """
        return (
            f"{BASE_INSTRUCTIONS}\n"
            f"YOUR CURRENT JOB — {self.name}:\n{self.instructions.strip()}\n\n"
            f"TOOLS AVAILABLE TO YOU:\n{self.tools.prompt_block()}\n"
        )

    def matches(self, goal: str) -> int:
        """
        How well this skill fits a request. Higher is better; 0 means no.

        Explicit `triggers` are weighted far above incidental word overlap,
        because overlap is noisy — "refund" appearing in a documentation
        question should not outrank a real trigger match.
        """
        low = goal.lower()
        score = 0
        for pattern in self.triggers:
            if re.search(pattern, low):
                score += 10
        for word in re.findall(r"[a-z]{4,}", self.name.lower()):
            if word in low:
                score += 3
        keywords = set(re.findall(r"[a-z]{5,}", self.instructions.lower()))
        score += sum(1 for w in keywords if w in low)
        return score

    # -- composition -------------------------------------------------------
    @staticmethod
    def compose(*skills: "Skill", name: Optional[str] = None) -> "Skill":
        """
        Merge skills into one. Union of tools, concatenated instructions.

        Use with restraint. Merging is the move that *undoes* the scoping
        benefit — a composed skill has all the tools of its parts, which is
        exactly the situation skills existed to avoid. Compose when a single
        request genuinely spans both jobs; route otherwise.

        Notebook 04 compares the two directly: the same task suite through one
        composed mega-skill and through a Router over separate skills.
        """
        if not skills:
            raise ValueError("compose() needs at least one skill")

        merged = ToolRegistry()
        for skill in skills:
            for item in skill.tools:
                merged.add(item)

        body = "\n\n".join(f"[{s.name}] {s.instructions.strip()}" for s in skills)
        return Skill(
            name=name or "+".join(s.name for s in skills),
            instructions=body,
            tools=merged,
            policy=skills[0].policy,
            triggers=[t for s in skills for t in s.triggers],
        )

    def __str__(self) -> str:
        return f"Skill({self.name}: {len(self.tools)} tools — {', '.join(self.tools.names())})"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
class Router:
    """
    Choose exactly one skill for a request.

    This is how a system scales past what one prompt can hold. Twenty-five tools
    across eight skills means the router picks one skill and the agent that runs
    sees three or four tools. Tool-choice accuracy stays where it was at five
    tools, no matter how large the overall system grows.

    The routing here is keyword scoring — transparent, offline, deterministic,
    and adequate for a handful of well-separated skills. In production you would
    route with a cheap fast model, which is the same design with a better
    classifier. `explain()` exists so the routing decision is never a mystery,
    since "the agent did the wrong job" is usually a routing bug, not a loop bug.
    """

    def __init__(self, skills: List[Skill], fallback: Optional[Skill] = None):
        if not skills:
            raise ValueError("Router needs at least one skill")
        self.skills = skills
        # The fallback catches requests matching nothing. Without one, an
        # unmatched request silently gets skills[0] — an agent confidently doing
        # the wrong job, which is worse than admitting no match.
        self.fallback = fallback or skills[0]

    def route(self, goal: str) -> Skill:
        ranked = sorted(self.skills, key=lambda s: -s.matches(goal))
        best = ranked[0]
        return best if best.matches(goal) > 0 else self.fallback

    def explain(self, goal: str) -> str:
        rows = sorted(
            ((s.matches(goal), s.name) for s in self.skills), key=lambda r: -r[0]
        )
        chosen = self.route(goal)
        lines = [f"Routing {goal!r}:"]
        for score, name in rows:
            mark = "→" if name == chosen.name else " "
            lines.append(f"  {mark} {name:<16} score {score}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return f"Router({', '.join(s.name for s in self.skills)})"


# ---------------------------------------------------------------------------
# The Acme skill set
# ---------------------------------------------------------------------------
def acme_router(registry: ToolRegistry) -> "Router":
    """
    The Acme skills wired into a Router with an explicit fallback.

    The fallback is `product_questions` — the general-purpose skill — and naming
    it explicitly matters. `Router` defaults to `skills[0]`, so an unmatched
    request would otherwise land in whichever skill happens to be listed first.
    An agent handling "what is 199 x 12?" with the refunds toolbox is not a loop
    bug and not a model bug; it is a one-line configuration bug, and it is
    invisible until you print the routing decision.
    """
    skills = acme_skills(registry)
    general = next(s for s in skills if s.name == "product_questions")
    return Router(skills, fallback=general)


def acme_skills(registry: ToolRegistry) -> List[Skill]:
    """
    Three skills over the Acme toolbox — the worked example for notebook 04.

    Note the deliberate overlap: `escalate_to_human` appears in two skills, and
    `search_docs` in two. Skills are not a partition of the tools. The question
    for each skill is "what does THIS job need?", asked independently — and some
    tools are needed by several jobs.

    Note also that no skill gets all five tools. That is the design working.
    """
    return [
        Skill(
            name="refunds",
            instructions=(
                "Handle refund and cancellation requests. ALWAYS look up the order "
                "first with get_order_status — you cannot assess a refund without "
                "knowing the order's status and date. Then use "
                "check_refund_eligibility with the customer's stated reason. "
                "If the policy says a human must decide, escalate with a summary "
                "of what you checked. Never promise a specific refund date."
            ),
            # search_docs is here because a refund agent that cannot read the
            # refund policy can only ever parrot the eligibility tool's verdict.
            # Four tools, still well under the full five — scoping is about
            # giving a job what it needs, not about the smallest possible number.
            tools=registry.subset(
                "get_order_status",
                "check_refund_eligibility",
                "search_docs",
                "escalate_to_human",
            ),
            triggers=[
                r"refund", r"money back", r"cancel", r"charged", r"reimburse",
                # The four policy reason codes. A goal that names one is
                # unambiguously a refund request, whatever else it mentions.
                r"billing_error", r"service_outage", r"not_as_described",
                r"changed_mind",
            ],
        ),
        Skill(
            name="product_questions",
            instructions=(
                "Answer questions about Acme Cloud plans, pricing, security, SLAs, "
                "the API and onboarding, using the documentation. Search first and "
                "answer only from what you find. If the documentation does not "
                "cover it, say so — do not fill the gap from general knowledge "
                "about cloud providers. Use calculate for any arithmetic, "
                "including prorated or annual totals."
            ),
            tools=registry.subset("search_docs", "calculate"),
            triggers=[r"plan", r"pric", r"cost", r"sla", r"security", r"api",
                      r"storage", r"seats", r"trial", r"onboard",
                      # Arithmetic belongs here too — this skill owns `calculate`.
                      r"multipl", r"percent", r"how much is", r"total", r"annual"],
        ),
        Skill(
            name="account_lookup",
            instructions=(
                "Look up the status of specific orders. Report exactly what the "
                "order system returns. If an order does not exist, say so — never "
                "guess at a similar ID. If the customer needs something beyond a "
                "status lookup, escalate rather than improvising."
            ),
            tools=registry.subset("get_order_status", "escalate_to_human"),
            # NOTE these are PHRASES, not bare words. An earlier version
            # triggered on bare "order" and "status", and it quietly stole every
            # refund request — those mention an order too. Narrowing to bare
            # shipping words then broke its own core case. Phrases fix both:
            # "status of" is specific enough to win a status question and
            # specific enough to LOSE to "refund" + "changed_mind" when the
            # request is really about a refund.
            #
            # The general rule, and the most common routing bug there is:
            # bare-word triggers on a general skill outrank precise triggers on
            # a specific one, and the symptom is a competent agent confidently
            # doing the wrong job. See notebook 04.
            triggers=[
                r"status of", r"look ?up .*order", r"\border\b.*\bstatus\b",
                r"delivery", r"shipped", r"tracking", r"when will .* arrive",
            ],
        ),
    ]

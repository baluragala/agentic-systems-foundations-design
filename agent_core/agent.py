"""
agent.py — the whole session in one class.
==========================================

WHY this file exists
--------------------
Everything is now built: state, tools, schemas, skills, control, tracing. This
class wires them together so the end-to-end story fits in three lines:

    agent = Agent()
    result = agent.run("Is order ACME-1042 eligible for a refund? I was double charged.")
    print(result.trace.show())

It is the counterpart to `RAGPipeline` in the previous session's package, and
the parallel is worth drawing explicitly in class:

    RAGPipeline   fixed stages, order decided by the programmer
    Agent         one stage, repeated, order decided at runtime by the model

Same components underneath — retrieval is right there as a tool. What changed is
who decides the control flow, and every difference in how you build, test and
debug the two follows from that one change.

DESIGN NOTE — every stage is a swappable argument
-------------------------------------------------
`Agent.__init__` takes the LLM, the tools, the skill or router, the termination
policy, the budget and the reflection switch. This mirrors `RAGPipeline`, and
for the same reason: the notebooks need to hold everything constant and vary one
thing, then compare traces. A class with hard-coded internals cannot be taught
from, because you cannot run the experiment.

The recurring question — *"what does this step look like when it goes wrong, and
where would you see it in the trace?"* — is answered at this level by
`AgentResult`, which carries the trace and the state together so that a failed
run is fully inspectable after the fact.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .config import LLM, get_llm
from .control import TerminationPolicy
from .acme_tools import acme_registry
from .loop import run_loop
from .skills import Router, Skill, acme_router, acme_skills
from .state import AgentState, AgentStatus, Budget
from .tools import ToolRegistry
from .trace import Trace


@dataclass
class AgentResult:
    """
    Everything one run produced.

    The answer is what a user sees. `trace` and `state` are what an engineer
    needs, and bundling all three is a deliberate refusal to let the answer
    travel without its evidence — the habit this whole session is trying to
    build.
    """

    answer: str
    trace: Trace
    state: AgentState
    skill: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.state.status is AgentStatus.DONE

    @property
    def escalated(self) -> bool:
        return self.state.status is AgentStatus.ESCALATED

    def tools_called(self) -> List[str]:
        return self.trace.call_sequence()

    def show(self) -> str:
        return self.trace.show()

    def __str__(self) -> str:
        skill = f" via {self.skill}" if self.skill else ""
        return (
            f"AgentResult({self.state.status.value}{skill}, "
            f"{self.trace.tool_calls()} tool calls)\n{self.answer}"
        )


class Agent:
    """
    A configurable agent over a tool registry, optionally routed across skills.

    Args:
        llm: Defaults to the configured provider (OpenAI, or the mock offline).
        tools: Defaults to the full Acme toolbox.
        skill: A single Skill, or a Router over several. Defaults to a Router
            over the three Acme skills.
        policy: When to stop. Defaults to the full diagnostic policy.
        budget: Hard limits. Defaults to Budget().
        use_reflection: Self-review before answering. Off by default — see
            `control.reflect` on why "reflect on everything" is expensive and
            usually wrong.
        verbose: Print each step as it happens. Excellent live; noisy in bulk.
    """

    def __init__(
        self,
        llm: Optional[LLM] = None,
        tools: Optional[ToolRegistry] = None,
        skill: Optional[Union[Skill, Router]] = None,
        policy: Optional[TerminationPolicy] = None,
        budget: Optional[Budget] = None,
        use_reflection: bool = False,
        verbose: bool = False,
    ):
        self.llm = llm or get_llm()
        self.tools = tools or acme_registry()
        self.skill = skill or acme_router(self.tools)
        self.policy = policy or TerminationPolicy()
        self.budget = budget or Budget()
        self.use_reflection = use_reflection
        self.verbose = verbose

    # -- the one method that matters ---------------------------------------
    def run(self, goal: str, *, stateful: bool = True) -> AgentResult:
        """
        Run the agent to completion on one goal.

        Args:
            goal: What the user wants.
            stateful: Teaching switch — False reproduces the amnesiac agent for
                the notebook-02 comparison. Always True for real work.
        """
        # 1. Pick the skill. A Router chooses; a bare Skill is used as given.
        chosen = self.skill.route(goal) if isinstance(self.skill, Router) else self.skill

        # 2. Fresh state per run. Budget is copied rather than shared, so two
        #    runs of the same Agent cannot interfere through a shared clock —
        #    an easy bug to write and a confusing one to find in a classroom.
        state = AgentState(
            goal=goal,
            budget=Budget(
                max_steps=self.budget.max_steps,
                max_tool_calls=self.budget.max_tool_calls,
                max_seconds=self.budget.max_seconds,
            ),
        )
        state.start(chosen.system_prompt())

        # 3. Turn the loop.
        trace = Trace(goal=goal, verbose=self.verbose)
        run_loop(
            self.llm,
            chosen.tools,
            state,
            policy=chosen.policy or self.policy,
            trace=trace,
            stateful=stateful,
            use_reflection=self.use_reflection,
        )

        return AgentResult(
            answer=state.final_answer or "",
            trace=trace,
            state=state,
            skill=chosen.name,
        )

    # -- running a suite ----------------------------------------------------
    def run_suite(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Run a list of tasks and score each against its expectations.

        WHY an agent needs a task suite at all: you cannot eyeball agent quality.
        A single run looks fine or looks broken, and neither tells you whether
        yesterday's prompt change helped. The suite gives you the agentic
        equivalent of the RAG session's retrieval metrics — a number that moves when you
        change something.

        Each task may declare:
            goal          — what to ask (required)
            expects_tools — tool names that MUST appear in the call sequence
            expects_text  — substrings the answer must contain
            expects_status— the required terminal status (e.g. "escalated")

        `expects_status: escalated` is how the negative tests work. For an
        unanswerable request, escalating is the pass condition and a confident
        answer is the failure — the inverse of how a normal test reads, and the
        thing most agent test suites forget to check.
        """
        results = []
        for task in tasks:
            outcome = self.run(task["goal"])
            called = outcome.tools_called()
            answer_low = (outcome.answer or "").lower()

            checks: Dict[str, bool] = {}
            if "expects_tools" in task:
                checks["tools"] = all(t in called for t in task["expects_tools"])
            if "expects_text" in task:
                checks["text"] = all(
                    str(s).lower() in answer_low for s in task["expects_text"]
                )
            if "expects_status" in task:
                checks["status"] = outcome.state.status.value == task["expects_status"]

            results.append(
                {
                    "id": task.get("id", ""),
                    "category": task.get("category", ""),
                    "goal": task["goal"],
                    "passed": all(checks.values()) if checks else None,
                    "checks": checks,
                    "tools_called": called,
                    "status": outcome.state.status.value,
                    "answer": outcome.answer,
                    "result": outcome,
                }
            )
        return results

    def __str__(self) -> str:
        skill = (
            f"Router({len(self.skill.skills)} skills)"
            if isinstance(self.skill, Router)
            else f"Skill({self.skill.name})"
        )
        return (
            f"Agent(llm={self.llm.name}, tools={len(self.tools)}, {skill}, "
            f"budget={self.budget.max_steps} steps, reflection={self.use_reflection})"
        )


# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------
def score_suite(results: List[Dict[str, Any]]) -> str:
    """
    Render suite results as a table plus a pass rate.

    The per-category breakdown is the useful part. An overall 80% hides whether
    the 20% failing is one hard multi-hop task or every single negative test —
    and those two situations call for completely different fixes.
    """
    scored = [r for r in results if r["passed"] is not None]
    passed = sum(1 for r in scored if r["passed"])

    lines = [f"{'id':<10} {'category':<14} {'ok':<4} {'status':<10} tools"]
    lines.append("─" * 74)
    for row in results:
        mark = "—" if row["passed"] is None else ("PASS" if row["passed"] else "FAIL")
        lines.append(
            f"{row['id']:<10} {row['category']:<14} {mark:<4} {row['status']:<10} "
            f"{'→'.join(row['tools_called']) or '(none)'}"
        )

    by_category: Dict[str, List[bool]] = {}
    for row in scored:
        by_category.setdefault(row["category"], []).append(row["passed"])

    lines.append("─" * 74)
    lines.append(f"OVERALL: {passed}/{len(scored)} passed ({passed / max(1, len(scored)):.0%})")
    for category, marks in sorted(by_category.items()):
        hits = sum(1 for m in marks if m)
        lines.append(f"  {category:<16} {hits}/{len(marks)}")
    return "\n".join(lines)

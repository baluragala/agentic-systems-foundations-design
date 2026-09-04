"""
agent_core — a from-scratch agentic system, built to be read.
=============================================================

The teaching package for **Agentic Systems Foundations** .

    An agent is not a smarter model. It is a loop over explicit state.
    Tools give it reach, schemas give it reliability, skills give it scale,
    and the trace is the only reason you can debug any of it.

Read the modules in this order — it is the order the session teaches them, and
each one only depends on the ones above it:

    state.py    what the agent remembers          (the core of agent intelligence)
    loop.py     think → act → observe → repeat    (the whole idea, ~40 lines)
    tools.py    what the agent can do             (and why tools must never raise)
    schemas.py  how a call is described & checked (across OpenAI/Gemini/Anthropic)
    skills.py   scoped, composable competence     (how systems grow past one prompt)
    control.py  when to stop, and being stuck     (the demo/deployable gap)
    trace.py    what actually happened            (the debugging interface)
    failures.py how it breaks, and how to see it  (a taxonomy + fault injection)
    agent.py    all of it, wired together

Quick start. Requires OPENAI_API_KEY — every notebook calls a real model, so
that what you see is what you would get in production:

    export OPENAI_API_KEY=sk-...

    from agent_core import Agent
    result = Agent().run("Is order ACME-1042 refundable? I was charged twice.")
    print(result.answer)
    print(result.trace.show())
"""

from .agent import Agent, AgentResult, score_suite
from .config import (
    FAULTS,
    AgentConfig,
    Decision,
    FaultInjectingLLM,
    ToolCall,
    current_config,
    get_llm,
    have_api_key,
)
from .control import (
    TerminationCondition,
    TerminationPolicy,
    budget_exceeded,
    error_streak,
    no_new_information,
    reflect,
    repetition,
)
from .failures import CATALOGUE, broken_agent, diagnose, report, show_catalogue
from .loop import run_loop
from .skills import BASE_INSTRUCTIONS, Router, Skill, acme_router, acme_skills
from .state import AgentState, AgentStatus, Budget, Observation, ObservationStatus
from .tools import Tool, ToolRegistry, tool
from .trace import StepRecord, Trace, compare

__version__ = "1.0.0"

__all__ = [
    # top level
    "Agent",
    "AgentResult",
    "score_suite",
    # config / providers
    "AgentConfig",
    "Decision",
    "ToolCall",
    "FaultInjectingLLM",
    "FAULTS",
    "have_api_key",
    "get_llm",
    "current_config",
    # state
    "AgentState",
    "AgentStatus",
    "Budget",
    "Observation",
    "ObservationStatus",
    # loop
    "run_loop",
    # tools & schemas
    "Tool",
    "ToolRegistry",
    "tool",
    # skills
    "Skill",
    "Router",
    "acme_skills",
    "acme_router",
    "BASE_INSTRUCTIONS",
    # control
    "TerminationPolicy",
    "TerminationCondition",
    "budget_exceeded",
    "repetition",
    "error_streak",
    "no_new_information",
    "reflect",
    # trace
    "Trace",
    "StepRecord",
    "compare",
    # failures
    "CATALOGUE",
    "show_catalogue",
    "broken_agent",
    "diagnose",
    "report",
]

"""
agent_lc — the same agent, built the way you would ship it.
===========================================================

The **parallel track** to `agent_core`. Same five Acme tools, same three skills,
same scoping and routing decisions — rebuilt on **LangGraph + LangChain +
LangSmith**, which is what a team actually deploys.

Read the two side by side. That comparison is the point of this package existing:

    agent_core (learn the mechanism)      agent_lc (ship it)
    ---------------------------------------------------------------------
    while not done:                       StateGraph edges
    AgentState dataclass                  TypedDict + add_messages reducer
    llm.decide()                          the `agent` node, model.bind_tools()
    ToolRegistry.dispatch()               ToolNode(handle_tool_errors=True)
    build_schema() from docstrings        Pydantic args_schema
    validate_args()                       Pydantic validation
    TerminationPolicy                     conditional edge + recursion_limit
    Skill + Router                        scoped graphs + a supervisor
    Trace                                 LangSmith (+ to_trace() to compare)
    (none)                                checkpointers — real memory, resumption

**What transfers unchanged** is every design decision: enums still stop invented
values, descriptions are still the interface, scoping still beats one big tool
list, and the diagnostic termination conditions still have to be written by you
because `recursion_limit` only tells you the ceiling was hit.

> The framework abstracts the mechanics, not the design decisions.

REQUIREMENTS
------------
Requires `OPENAI_API_KEY`. Every track in this package calls a real model —
there is no simulated fallback anywhere, on purpose.

For reproducible failures, `faults.py` wraps a real model and corrupts exactly
one thing on the way out, so the failure taxonomy is identical for everyone
without the demonstration being a simulation.

QUICK START
-----------
    from langchain_openai import ChatOpenAI
    from agent_lc import build_prebuilt_agent, call_sequence, final_answer

    agent = build_prebuilt_agent(ChatOpenAI(model="gpt-4o-mini", temperature=0))
    out = agent.invoke({"messages": [("user", "Is order ACME-1046 refundable?")]})
    print(final_answer(out), call_sequence(out))
"""

from .graph import (
    AgentState,
    build_agent_graph,
    build_conversational,
    build_diagnostic_graph,
    build_prebuilt_agent,
    draw,
)
from .skills_lc import (
    BASE_INSTRUCTIONS,
    KeywordSupervisor,
    LcSkill,
    acme_lc_skills,
    build_supervisor_graph,
)
from .faults import (
    FAULT_MODELS,
    bad_tool_name_chat_model,
    looping_chat_model,
    malformed_args_chat_model,
    ungrounded_chat_model,
)
from .tools_lc import (
    ACME_TOOLS,
    TERMINAL_TOOLS,
    TOOLS_BY_NAME,
    describe,
    subset,
)
from .tracing_lc import (
    call_sequence,
    enable_langsmith,
    final_answer,
    langsmith_status,
    show_messages,
    summarise,
    to_trace,
)

__version__ = "1.0.0"

__all__ = [
    # tools
    "ACME_TOOLS", "TOOLS_BY_NAME", "TERMINAL_TOOLS", "subset", "describe",
    # graph
    "AgentState", "build_agent_graph", "build_diagnostic_graph",
    "build_prebuilt_agent", "build_conversational", "draw",
    # skills
    "LcSkill", "KeywordSupervisor", "acme_lc_skills", "build_supervisor_graph",
    "BASE_INSTRUCTIONS",
    # deterministic faults, over a real model
    "looping_chat_model", "malformed_args_chat_model", "bad_tool_name_chat_model",
    "ungrounded_chat_model", "FAULT_MODELS",
    # tracing
    "enable_langsmith", "langsmith_status", "to_trace", "summarise",
    "call_sequence", "final_answer", "show_messages",
]

"""
graph.py — the agent loop as a LangGraph StateGraph.
====================================================

WHY this file exists
--------------------
`agent_core/loop.py` is a `while` loop you can read in one sitting. This is the
same loop, expressed as a **graph**, which is how you would ship it.

The translation is almost one-to-one, and seeing that is the whole point:

    agent_core                        LangGraph
    ----------------------------------------------------------------
    while not done:                   the graph's edges
    llm.decide(...)          ──▶      the `agent` node
    tools.dispatch(...)      ──▶      the `tools` node (ToolNode)
    state.add_message(...)   ──▶      the `add_messages` reducer
    policy.check(...)        ──▶      the conditional edge + recursion_limit
    Trace                    ──▶      LangSmith / callbacks

THE ONE PLACE LANGGRAPH IS GENUINELY BETTER FOR TEACHING
--------------------------------------------------------
The agenda asks learners to build a loop with **"explicit state representation
and updates"**. In the from-scratch package, state is explicit because we wrote a
dataclass and were disciplined about mutating it through named methods.

In LangGraph, state is explicit because **the framework will not let it be
anything else.** You declare a schema:

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], add_messages]

and you declare **how each field updates** — that `add_messages` annotation is a
*reducer*, and it is the framework's version of the single most important line in
notebook 02. A node returns `{"messages": [reply]}` and the reducer appends it.

So the lesson survives the change of engine, and gets sharper: in the
hand-rolled loop you could *forget* the state update and end up with an agent
that repeats itself. Here the update is a declared property of the field. You can
still get it wrong — return the wrong key, use the wrong reducer — but you cannot
silently omit it.

WHAT THIS FILE BUILDS
---------------------
1. `build_agent_graph()` — the loop, node by node, so nothing is hidden.
2. `build_prebuilt_agent()` — the same thing via `create_react_agent`, which is
   what you would actually write in production. Two lines instead of forty.

Both are here on purpose. #1 is how you understand it; #2 is what you ship. Read
#1 once and you will never be confused by #2.
"""
from __future__ import annotations

import os
from typing import Annotated, Any, Literal, Optional, Sequence, TypedDict

from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from .tools_lc import ACME_TOOLS, TERMINAL_TOOLS


# ---------------------------------------------------------------------------
# 1. STATE — declared, not improvised
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    """
    The graph's state schema. Compare with `agent_core.state.AgentState`.

    `messages` carries the `add_messages` reducer: when a node returns
    `{"messages": [...]}`, LangGraph **appends** rather than replacing. That
    annotation is this framework's answer to the write-back that notebook 02
    proves is load-bearing — and here it is a property of the *schema*, so it
    applies to every node automatically.

    The extra fields show that state is yours to extend. `steps` exists so the
    conditional edge can implement a step budget; `stop_reason` exists so a run
    can say *why* it ended, which LangGraph does not record for you.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    stop_reason: Optional[str]


# ---------------------------------------------------------------------------
# 2. THE LOOP, BUILT BY HAND
# ---------------------------------------------------------------------------
def build_agent_graph(
    model,
    tools: Optional[Sequence] = None,
    system_prompt: str = "",
    max_steps: int = 8,
    checkpointer=None,
):
    """
    Build the think → act → observe loop as an explicit StateGraph.

    Args:
        model: a chat model (typically `ChatOpenAI`). Tools are bound here.
        tools: the scoped tool list. Defaults to the full Acme toolbox.
        system_prompt: prepended once, on the first turn.
        max_steps: the step budget, enforced in the conditional edge.
        checkpointer: pass `InMemorySaver()` to make the agent resumable and
            give it memory across invocations — see `build_conversational()`.

    Returns the compiled graph. Invoke it with
    `graph.invoke({"messages": [HumanMessage(goal)], "steps": 0, "stop_reason": None})`.
    """
    tools = list(tools if tools is not None else ACME_TOOLS)
    # bind_tools is where the schemas from tools_lc.py actually reach the model.
    # It is the LangChain equivalent of our `registry.to_openai()` payload.
    model_with_tools = model.bind_tools(tools)

    # -- NODE: agent (THINK) -------------------------------------------------
    def agent_node(state: AgentState) -> dict:
        """One LLM call. Returns the update to merge into state."""
        messages = list(state["messages"])
        # Prepend the system prompt once, on the first turn only.
        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt), *messages]
        reply = model_with_tools.invoke(messages)
        # Return a PARTIAL state update. `add_messages` appends `reply`;
        # `steps` has no reducer, so this value replaces the old one.
        return {"messages": [reply], "steps": state.get("steps", 0) + 1}

    # -- NODE: tools (ACT + OBSERVE) ----------------------------------------
    # ToolNode runs every requested call and returns one ToolMessage each, which
    # `add_messages` appends. That single line is our dispatch loop plus the
    # observation write-back.
    #
    # handle_tool_errors=True is the framework's version of our "tools must never
    # raise" rule: an exception becomes a ToolMessage the agent can read and
    # recover from. Turn it off and a raising tool kills the run — the same
    # failure, and the same reason we made ours never raise.
    tool_node = ToolNode(tools, handle_tool_errors=True)

    # -- EDGE: the terminate? check -----------------------------------------
    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        """
        The conditional edge. This is `TerminationPolicy` for the graph.

        Note what LangGraph gives you and what it does not. It gives you
        `recursion_limit` — a hard backstop against infinite graphs. It does NOT
        give you the *diagnostic* conditions from `agent_core.control`:
        repetition, error streaks, no-new-information. Those are domain logic,
        and if you want them you write them here.

        That is worth saying plainly: adopting a framework does not give you the
        thing notebook 05 argues is the actual engineering. It gives you the
        backstop and leaves the diagnosis to you.
        """
        last = state["messages"][-1]

        # Budget first here only because there is nothing else to check yet —
        # see `should_continue_diagnostic` below for the version that orders
        # conditions the way notebook 05 argues you should.
        if state.get("steps", 0) >= max_steps:
            return "__end__"

        # No tool calls requested -> the model gave a final answer.
        if not getattr(last, "tool_calls", None):
            return "__end__"
        return "tools"

    # -- EDGE: did a terminal tool just run? --------------------------------
    def after_tools(state: AgentState) -> Literal["agent", "__end__"]:
        """
        Ends the run the moment a terminal tool succeeds.

        This edge is easy to leave out, and leaving it out costs you a whole LLM
        call on every escalation: without it, control returns to `agent`, the
        model is asked what to do next, and only *then* does the terminate check
        notice the escalation already happened. Cheap to fix, invisible until
        you read a trace and wonder why there is an extra step after the handoff.
        """
        for message in reversed(state["messages"]):
            if getattr(message, "type", None) == "tool":
                return "__end__" if getattr(message, "name", None) in TERMINAL_TOOLS else "agent"
        return "agent"

    # -- WIRE IT UP ----------------------------------------------------------
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "__end__": END})
    # The cycle. These edges are the `while` in notebook 02.
    graph.add_conditional_edges("tools", after_tools, {"agent": "agent", "__end__": END})

    return graph.compile(checkpointer=checkpointer)


# ---------------------------------------------------------------------------
# 3. DIAGNOSTIC TERMINATION — porting agent_core.control to a graph
# ---------------------------------------------------------------------------
def build_diagnostic_graph(
    model,
    tools: Optional[Sequence] = None,
    system_prompt: str = "",
    max_steps: int = 8,
    repetition_threshold: int = 3,
    error_streak_limit: int = 3,
):
    """
    The same graph, with the diagnostic termination conditions from notebook 05.

    Included to make a specific point: everything `agent_core.control` does is
    portable to LangGraph, because it was never framework-specific — it is
    reasoning about the message history. What you lose by moving to a framework
    is not the *ability* to do this; it is the prompt to think of it at all,
    because `recursion_limit` looks like it has the problem covered.

    It does not. `recursion_limit` tells you the graph hit its ceiling. It never
    tells you the agent called the same tool with the same arguments five times.
    """
    tools = list(tools if tools is not None else ACME_TOOLS)
    model_with_tools = model.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = list(state["messages"])
        if system_prompt and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=system_prompt), *messages]
        return {
            "messages": [model_with_tools.invoke(messages)],
            "steps": state.get("steps", 0) + 1,
        }

    def diagnose(state: AgentState) -> Optional[str]:
        """Return a reason to stop, or None. The port of TerminationPolicy."""
        import json

        messages = state["messages"]

        # -- repetition: identical call, identical arguments -----------------
        signatures = []
        for message in messages:
            for call in getattr(message, "tool_calls", None) or []:
                try:
                    args = json.dumps(call["args"], sort_keys=True, default=str)
                except Exception:
                    args = str(call.get("args"))
                signatures.append(f"{call['name']}({args})")
        for signature in set(signatures):
            if signatures.count(signature) >= repetition_threshold:
                return f"repetition: identical call repeated {repetition_threshold}× — {signature}"

        # -- error streak: consecutive failing tool results ------------------
        streak = 0
        for message in reversed(messages):
            if getattr(message, "type", None) != "tool":
                continue
            body = str(getattr(message, "content", "") or "")
            if getattr(message, "status", None) == "error" or body.startswith("Error"):
                streak += 1
            else:
                break
        if streak >= error_streak_limit:
            return f"error_streak: {streak} consecutive tool failures"

        # -- budget: the backstop, checked LAST ------------------------------
        if state.get("steps", 0) >= max_steps:
            return f"budget: max_steps ({max_steps}) reached"
        return None

    def should_continue(state: AgentState) -> Literal["tools", "__end__"]:
        if diagnose(state):
            return "__end__"
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "__end__"

    def after_tools(state: AgentState) -> Literal["agent", "__end__"]:
        """End immediately when a terminal tool has run — see build_agent_graph."""
        for message in reversed(state["messages"]):
            if getattr(message, "type", None) == "tool":
                return "__end__" if getattr(message, "name", None) in TERMINAL_TOOLS else "agent"
        return "agent"

    def record_stop(state: AgentState) -> dict:
        """A tiny node that writes the stop reason into state before ending."""
        return {"stop_reason": diagnose(state) or "model returned a final answer"}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools, handle_tool_errors=True))
    graph.add_node("record_stop", record_stop)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "__end__": "record_stop"}
    )
    graph.add_conditional_edges(
        "tools", after_tools, {"agent": "agent", "__end__": "record_stop"}
    )
    graph.add_edge("record_stop", END)
    return graph.compile()


# ---------------------------------------------------------------------------
# 4. THE PRODUCTION VERSION — two lines
# ---------------------------------------------------------------------------
def build_prebuilt_agent(
    model,
    tools: Optional[Sequence] = None,
    system_prompt: str = "",
    checkpointer=None,
):
    """
    The same agent via `create_react_agent`. This is what you would actually write.

    It builds the identical graph — agent node, ToolNode, conditional edge, cycle
    — with sensible defaults. Having read `build_agent_graph()` above, you know
    exactly what it is doing, which is the only reason it is safe to use.

    What you still own after adopting it:
      * the tool descriptions and schemas (nothing writes those for you)
      * the system prompt
      * which tools are in scope
      * `recursion_limit`, and any diagnostic termination beyond it
      * whether tool errors are handled or fatal
    """
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(
        model,
        tools=list(tools if tools is not None else ACME_TOOLS),
        prompt=system_prompt or None,
        checkpointer=checkpointer,
    )


def build_conversational(model, tools=None, system_prompt: str = ""):
    """
    An agent with memory across invocations, via a checkpointer.

    This is capability the from-scratch package does not have, and a fair reason
    to adopt the framework: `InMemorySaver` (swap for Postgres/Redis in
    production) persists state per `thread_id`, so a second question continues
    the first conversation. Getting this right yourself — serialisation,
    concurrency, resumption after a crash — is real work you should not redo.

    Invoke with: `agent.invoke({...}, config={"configurable": {"thread_id": "abc"}})`
    """
    return build_prebuilt_agent(
        model, tools=tools, system_prompt=system_prompt, checkpointer=InMemorySaver()
    )


def draw(graph) -> str:
    """
    ASCII rendering of a compiled graph — handy in a notebook.

    Seeing the cycle drawn is worth a paragraph of explanation: `agent → tools →
    agent` is the loop, and the branch out of `agent` is the terminate check.
    """
    try:
        return graph.get_graph().draw_ascii()
    except Exception as exc:  # grandalf not installed, etc.
        nodes = list(graph.get_graph().nodes)
        return f"(ASCII rendering unavailable: {exc})\nnodes: {nodes}"

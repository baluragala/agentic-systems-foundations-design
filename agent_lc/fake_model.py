"""
fake_model.py — a deterministic chat model, for TESTS ONLY.
===========================================================

READ THIS BEFORE USING IT
-------------------------
This is **not a teaching path.** The LangChain/LangGraph track in this package
uses a real `ChatOpenAI` and requires `OPENAI_API_KEY`, because the point of that
track is to show what production actually looks like.

This file exists for one reason: so `scripts/check_langgraph.py` can verify that
the graphs in `graph.py` and `skills_lc.py` are wired correctly — that the cycle
runs, that `add_messages` appends, that the conditional edge terminates, that
terminal tools stop the run — **without needing a key or a network call**.

Shipping a graph nobody has executed is not acceptable, and "it requires a key"
is not a reason to skip verification. It is a reason to build a test double.

If you want a keyless *teaching* experience, use the `agent_core` track: its
`MockToolCallLLM` is designed for that, is transparent about its reasoning
(`explain_plan`), and is what notebooks 01–07 fall back to.

WHAT IT DOES
------------
Wraps the routing logic already written and tested in
`agent_core.config.MockToolCallLLM` in a LangChain `BaseChatModel`, emitting real
`AIMessage` objects with real `tool_calls`. That reuse is deliberate: one router,
tested once, rather than two that can drift.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _ToolShim:
    """Adapts a LangChain tool to the shape `MockToolCallLLM` expects."""

    def __init__(self, lc_tool):
        self._tool = lc_tool
        self.name = lc_tool.name
        self.description = lc_tool.description or ""
        self.terminal = False
        raw = lc_tool.args_schema.model_json_schema() if lc_tool.args_schema else {}
        self.schema = {
            "type": "object",
            "properties": raw.get("properties", {}),
            "required": raw.get("required", []),
        }
        # Pydantic puts `examples` on the field; the router reads them off the tool.
        self.examples = [
            e
            for spec in self.schema["properties"].values()
            for e in (spec.get("examples") or [])
        ]


class FakeToolCallingModel(BaseChatModel):
    """
    A deterministic tool-calling chat model. Tests only — see the module docstring.

    Args:
        fault: one of the `agent_core.config` fault names, to reproduce a
            specific failure mode inside a LangGraph run.
    """

    fault: Optional[str] = None
    bound_tools: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> "FakeToolCallingModel":
        # `self.__class__`, not `FakeToolCallingModel`. Hard-coding the class
        # silently discards any subclass — so a test that subclasses this to
        # script a specific tool sequence gets the generic router instead, and
        # the test passes for the wrong reason. Exactly the kind of bug a test
        # double should not have.
        clone = self.__class__(fault=self.fault)
        clone.bound_tools = list(tools)
        return clone

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        from agent_core.config import MockToolCallLLM

        router = MockToolCallLLM(fault=self.fault)
        shims = [_ToolShim(t) for t in self.bound_tools]

        # Translate LangChain messages into the plain-dict transcript the router
        # reads. It derives "what have I already called?" from this, exactly as
        # a real model derives it from the conversation.
        transcript = []
        for message in messages:
            kind = getattr(message, "type", "")
            role = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}.get(
                kind, kind
            )
            entry = {"role": role, "content": str(getattr(message, "content", "") or "")}
            calls = getattr(message, "tool_calls", None)
            if calls:
                entry["tool_calls"] = [
                    {"function": {"name": c["name"]}} for c in calls
                ]
            transcript.append(entry)

        decision = router.decide(transcript, shims)

        if decision.is_final:
            reply = AIMessage(content=decision.content or "")
        else:
            reply = AIMessage(
                content="",
                tool_calls=[
                    {"name": c.name, "args": c.args, "id": c.id}
                    for c in decision.tool_calls
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=reply)])

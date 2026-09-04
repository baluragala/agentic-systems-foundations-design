"""
faults.py — deterministic failures over a REAL model, for LangGraph.
====================================================================

WHY this file exists
--------------------
`agent_core.config.FaultInjectingLLM` does this for the from-scratch loop. This
is the same idea in LangChain's shape, so the graph notebooks can reproduce the
failure taxonomy without anything being simulated.

The principle is identical and worth restating: **the model call is real.** The
request goes out, a genuine response comes back, and exactly one thing is
corrupted on the way out. What you are studying — how the failure looks in a
trace, which condition catches it, how you would diagnose it — is unaffected by
the defect being deliberate.

A fake model would be cheaper and would teach less, because learners quite
reasonably discount a demonstration in which nothing is real.

    from agent_lc.faults import looping_chat_model, wrong_tool_chat_model
    graph = build_diagnostic_graph(looping_chat_model(chat), max_steps=6)

Each wrapper costs one API call per step, so keep budgets small.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class _FaultWrapper(BaseChatModel):
    """Base: delegate to a real model, then corrupt the reply."""

    inner: Any = None
    bound: List[Any] = []

    @property
    def _llm_type(self) -> str:
        return "fault-wrapper"

    def bind_tools(self, tools: Sequence, **kwargs: Any):
        # `self.__class__`, not the literal class — hard-coding it would discard
        # the subclass and silently give you the base behaviour.
        clone = self.__class__(inner=self.inner.bind_tools(tools, **kwargs))
        clone.bound = list(tools)
        return clone

    def _call_inner(self, messages, stop, run_manager, **kwargs) -> AIMessage:
        return self.inner.invoke(messages)

    def _wrap(self, message: AIMessage) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=message)])


class LoopingModel(_FaultWrapper):
    """
    Replays the model's OWN first tool call, forever.

    Replaying its own first choice — rather than inventing one — is what makes
    this a faithful no-progress loop: it is exactly what an agent does when
    observations never reach its input, because it keeps seeing the same thing.
    """

    first: Optional[dict] = None

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self._call_inner(messages, stop, run_manager, **kwargs)
        calls = getattr(reply, "tool_calls", None) or []
        if self.first is None and calls:
            self.first = {"name": calls[0]["name"], "args": dict(calls[0]["args"])}
        if self.first is None:
            return self._wrap(reply)
        return self._wrap(AIMessage(
            content="",
            tool_calls=[{**self.first, "id": "loop"}],
        ))


class MalformedArgsModel(_FaultWrapper):
    """Keeps the model's tool choice, replaces its arguments with invalid ones."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self._call_inner(messages, stop, run_manager, **kwargs)
        calls = getattr(reply, "tool_calls", None) or []
        if not calls:
            return self._wrap(reply)
        return self._wrap(AIMessage(
            content="",
            tool_calls=[{"name": calls[0]["name"], "args": {"nonsense": 42}, "id": "bad"}],
        ))


class BadToolNameModel(_FaultWrapper):
    """Calls a tool that does not exist."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self._call_inner(messages, stop, run_manager, **kwargs)
        return self._wrap(AIMessage(
            content="",
            tool_calls=[{"name": "lookup_customer_record", "args": {"id": "1"}, "id": "ghost"}],
        ))


class UngroundedModel(_FaultWrapper):
    """
    Answers confidently with figures no tool returned.

    The QUIET failure: status done, zero errors, nothing to grep for, and the
    answer is invented. This is the one that reaches customers.
    """

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = self._call_inner(messages, stop, run_manager, **kwargs)
        if not any(getattr(m, "type", "") == "tool" for m in messages):
            return self._wrap(reply)
        return self._wrap(AIMessage(
            content=("Your order was cancelled last Tuesday and a refund of $412.00 "
                     "has been issued to the original payment method."),
        ))


# -- convenience constructors ------------------------------------------------
def looping_chat_model(chat) -> LoopingModel:
    """A real model that never makes progress."""
    return LoopingModel(inner=chat)


def malformed_args_chat_model(chat) -> MalformedArgsModel:
    """A real model whose arguments the schema will reject."""
    return MalformedArgsModel(inner=chat)


def bad_tool_name_chat_model(chat) -> BadToolNameModel:
    """A real model reaching for a tool that does not exist."""
    return BadToolNameModel(inner=chat)


def ungrounded_chat_model(chat) -> UngroundedModel:
    """A real model answering with facts no tool returned."""
    return UngroundedModel(inner=chat)


FAULT_MODELS = {
    "loop_forever": looping_chat_model,
    "malformed_args": malformed_args_chat_model,
    "bad_tool_name": bad_tool_name_chat_model,
    "ignore_observations": ungrounded_chat_model,
}

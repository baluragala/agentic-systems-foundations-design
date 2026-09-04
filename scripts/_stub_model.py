"""
_stub_model.py — a scripted chat model. TEST FIXTURE ONLY.
==========================================================

READ THIS FIRST
---------------
**This is not a teaching path and it is not a provider.** The package calls a
real OpenAI model everywhere — `agent_core.get_llm()` raises without a key, on
purpose, because a simulated model cannot show you how a real one behaves.

This file exists for exactly one reason: so `scripts/check_*.py` can verify
**wiring** — that a graph cycles, that a reducer appends, that an interrupt
suspends and resumes, that an approval gate blocks a side effect — in CI,
without spending money on every commit.

That is a normal test double, not a shipped fallback. The distinction that
matters:

    a fixture in a test script   verifies YOUR code           (this file)
    a fake model in the package  misrepresents THE MODEL      (removed)

It answers with a fixed script, not by reasoning. It tells you nothing about
model behaviour and is never imported by `agent_core`, `agent_lc` or
`acme_support_agent`.
"""
from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_core.config import Decision, ToolCall


# ---------------------------------------------------------------------------
# For the agent_core loop (plain dict transcripts)
# ---------------------------------------------------------------------------
class ScriptedLLM:
    """
    Replays a fixed list of decisions, then answers.

    Implements the `agent_core.LLM` protocol: `.decide(messages, tools)`.

        ScriptedLLM([("get_order_status", {"order_id": "ACME-1046"})])
    """

    def __init__(
        self,
        script: Optional[List[tuple]] = None,
        final: str = "Done. See the tool results above.",
    ):
        self.name = "scripted-stub"
        self.script = list(script or [])
        self.final = final

    def decide(self, messages: List[Dict[str, Any]], tools) -> Decision:
        # Derive position from the transcript, exactly as a real model would —
        # so the stub exercises the same write-back the loop depends on.
        called = [
            call["function"]["name"]
            for m in messages
            for call in (m.get("tool_calls") or [])
        ]
        for name, args in self.script:
            if name not in called:
                return Decision(tool_calls=[ToolCall(name=name, args=dict(args))])
        observed = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]
        return Decision(content=f"{self.final} {' '.join(observed)[:300]}".strip())


# ---------------------------------------------------------------------------
# For LangGraph (LangChain message objects)
# ---------------------------------------------------------------------------
class ScriptedChatModel(BaseChatModel):
    """
    The LangChain-shaped equivalent, for verifying graphs.

    `BaseChatModel` is a Pydantic model, so configuration goes in `ClassVar`s on
    a subclass rather than bare class attributes.
    """

    script: List[Any] = []
    final_text: str = "Done."

    @property
    def _llm_type(self) -> str:
        return "scripted-stub"

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> "ScriptedChatModel":
        # `self.__class__`, not the literal class: hard-coding it silently
        # discards subclasses, so a test that scripts a specific sequence would
        # get the base behaviour and pass for the wrong reason.
        clone = self.__class__(script=list(self.script), final_text=self.final_text)
        return clone

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        called = [
            call["name"]
            for m in messages
            for call in (getattr(m, "tool_calls", None) or [])
        ]
        for index, (name, args) in enumerate(self.script):
            if name not in called:
                reply = AIMessage(
                    content="",
                    tool_calls=[{"name": name, "args": dict(args), "id": f"s{index}"}],
                )
                return ChatResult(generations=[ChatGeneration(message=reply)])

        observed = " ".join(
            str(getattr(m, "content", "")) for m in messages
            if getattr(m, "type", "") == "tool"
        )
        reply = AIMessage(content=f"{self.final_text} {observed[:300]}".strip())
        return ChatResult(generations=[ChatGeneration(message=reply)])


ACME_REFUND_SCRIPT = [
    ("get_order_status", {"order_id": "ACME-1046"}),
    ("check_refund_eligibility", {"order_id": "ACME-1046", "reason": "changed_mind"}),
    ("issue_refund", {"order_id": "ACME-1046", "amount_usd": 448.50, "reason": "changed_mind"}),
]

ACME_LOOKUP_SCRIPT = [
    ("get_order_status", {"order_id": "ACME-1046"}),
    ("check_refund_eligibility", {"order_id": "ACME-1046", "reason": "changed_mind"}),
]


class LoopingChatModel(ScriptedChatModel):
    """Always requests the same call — reproduces a no-progress loop in a graph."""

    call_name: ClassVar[str] = "get_order_status"
    call_args: ClassVar[dict] = {"order_id": "ACME-1042"}

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = AIMessage(
            content="",
            tool_calls=[{"name": self.call_name, "args": dict(self.call_args), "id": "loop"}],
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])


class MalformedArgsChatModel(ScriptedChatModel):
    """Requests a real tool with arguments the schema will reject."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        reply = AIMessage(
            content="",
            tool_calls=[{"name": "get_order_status", "args": {"nonsense": 42}, "id": "bad"}],
        )
        return ChatResult(generations=[ChatGeneration(message=reply)])

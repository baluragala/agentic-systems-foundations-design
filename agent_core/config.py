"""
config.py — one interface for "something that decides what to do next".
=======================================================================

WHY this file exists
--------------------
The whole package is written against a single method:

    llm.decide(messages, tools) -> Decision

A `Decision` is either "call these tools with these arguments" or "here is the
final answer". Nothing else in `agent_core` imports `openai`, so the loop, the
tools, the skills and the control layer are all provider-neutral — and adding a
second provider means writing one adapter, not touching the package.

THE STACK
---------
**OpenAI, with native tool calling. An API key is required.**

There is no offline fallback. That is a deliberate choice: a simulated model
teaches you the *shape* of an agent loop while quietly misrepresenting the thing
you are actually trying to learn — how a real model behaves when your tool
description is ambiguous, when two tools overlap, when a schema is too loose.
Those moments are what this session is about, and a keyword router cannot
produce them.

So every notebook calls a real model, and what you see is what you would get in
production.

    export OPENAI_API_KEY=sk-...
    # Colab: sidebar -> key icon -> add OPENAI_API_KEY -> Notebook access ON

WHAT ABOUT REPRODUCIBLE FAILURES?
---------------------------------
Notebook 06 needs everyone in the room to see the *same* failure, and a real
model is stochastic. `FaultInjectingLLM` below wraps a real model and
deterministically corrupts its decision — a real call, a real response, one
specific thing broken on purpose.

That is a better fixture than a fake model: the failure is injected at exactly
the point you want to study, and everything around it is genuine.

Temperature is 0 by default. Variance in the tool-choice step is something you
will spend an afternoon debugging rather than enjoying.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Load a .env file if python-dotenv is available. Colab users usually set the
# key via Secrets or getpass instead, which works identically.
try:  # pragma: no cover - convenience only
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


# ---------------------------------------------------------------------------
# The vocabulary of a decision
# ---------------------------------------------------------------------------
@dataclass
class ToolCall:
    """
    One requested tool invocation.

    `id` exists because providers correlate a tool result back to the request
    that produced it. Generating it here — rather than letting each adapter
    invent its own convention — keeps the loop provider-neutral.
    """

    name: str
    args: Dict[str, Any]
    id: str = field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")


@dataclass
class Decision:
    """
    What the model wants to do next.

    Exactly one of two shapes:
      * `tool_calls` non-empty -> act, then loop again
      * `tool_calls` empty     -> `content` is the final answer, stop

    Collapsing both into one type is what keeps `loop.py` short enough to read
    in one sitting: the loop asks "did you give me calls?" and branches once.
    """

    tool_calls: List[ToolCall] = field(default_factory=list)
    content: Optional[str] = None
    raw: Any = None  # the untouched provider response, for teaching/debugging

    @property
    def is_final(self) -> bool:
        return not self.tool_calls

    def __str__(self) -> str:
        if self.is_final:
            preview = " ".join((self.content or "").split())[:70]
            return f"Decision(FINAL: {preview!r})"
        calls = ", ".join(f"{c.name}({_short_args(c.args)})" for c in self.tool_calls)
        return f"Decision(ACT: {calls})"


def _short_args(args: Dict[str, Any], width: int = 40) -> str:
    text = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return text if len(text) <= width else text[: width - 1] + "…"


@runtime_checkable
class LLM(Protocol):
    """Anything that can look at a transcript and decide what to do next."""

    name: str

    def decide(self, messages: List[Dict[str, Any]], tools: Any) -> Decision:
        ...


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
MISSING_KEY_MESSAGE = (
    "OPENAI_API_KEY is not set.\n"
    "\n"
    "This package calls a real model — there is no offline fallback, because a\n"
    "simulated model cannot show you how a real one behaves when your tool\n"
    "descriptions are ambiguous. That behaviour is the subject of the session.\n"
    "\n"
    "  local : export OPENAI_API_KEY=sk-...\n"
    "  Colab : sidebar -> key icon -> add a secret named OPENAI_API_KEY\n"
    "          -> toggle 'Notebook access' ON -> re-run the provider cell"
)


class OpenAIToolCaller:
    """
    OpenAI chat completions with native tool calling.

    Note how thin this is. About twenty lines translate between our `Decision`
    vocabulary and OpenAI's response shape, and that is the *entire*
    provider-specific surface of this package. When someone asks "how much work
    is it to support Anthropic or Gemini?", this class is the answer.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        timeout: float = 30.0,
        max_retries: int = 2,
    ):
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The `openai` package is required: pip install openai"
            ) from exc

        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(MISSING_KEY_MESSAGE)

        self.client = OpenAI(timeout=timeout, max_retries=max_retries)
        self.name = f"openai:{model}"
        self.model = model
        self.temperature = temperature

    def decide(self, messages: List[Dict[str, Any]], tools) -> Decision:
        payload = tools.to_openai() if hasattr(tools, "to_openai") else (tools or None)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=payload or None,
            temperature=self.temperature,
        )
        message = response.choices[0].message

        calls = []
        for call in message.tool_calls or []:
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                # A model emitting invalid JSON is a real, observed failure. We
                # keep the raw text so schema validation produces a message the
                # agent can act on, instead of dying here with a parse error.
                args = {"_raw": call.function.arguments}
            calls.append(ToolCall(id=call.id, name=call.function.name, args=args))

        return Decision(tool_calls=calls, content=message.content, raw=message)


# ---------------------------------------------------------------------------
# Deterministic faults, over a real model
# ---------------------------------------------------------------------------
FAULTS = (
    "loop_forever",         # never finalises — burns the entire step budget
    "bad_tool_name",        # calls a tool that does not exist
    "malformed_args",       # sends arguments the schema will reject
    "wrong_tool",           # picks a plausible but irrelevant tool
    "ignore_observations",  # answers without using what the tools returned
)


class FaultInjectingLLM:
    """
    Wraps a real LLM and deterministically breaks one specific thing.

    WHY THIS RATHER THAN A FAKE MODEL
    ---------------------------------
    Notebook 06 needs everyone in the room to reproduce the *same* failure, and
    real models are stochastic. The obvious fix is a fake model — and it is the
    wrong one, because then nothing being demonstrated is real, and learners
    quite reasonably discount what they are shown.

    This keeps the model real. The call happens, the response comes back, and
    exactly one thing is corrupted on the way out. What you are studying — how
    the failure looks in a trace, which termination condition catches it, how
    you would diagnose it — is unaffected by the corruption being deliberate.

    It costs an API call per step, so keep the budgets in notebook 06 small.
    """

    def __init__(self, inner: LLM, fault: str):
        if fault not in FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {list(FAULTS)}")
        self.inner = inner
        self.fault = fault
        self.name = f"{getattr(inner, 'name', 'llm')}+fault:{fault}"
        self._first: Optional[Decision] = None

    def decide(self, messages: List[Dict[str, Any]], tools) -> Decision:
        decision = self.inner.decide(messages, tools)
        names = [t.name for t in tools] if tools is not None else []
        saw_observation = any(m.get("role") == "tool" for m in messages)

        # -- LOUD: a tool that does not exist -------------------------------
        if self.fault == "bad_tool_name":
            return Decision(
                tool_calls=[ToolCall(name="lookup_customer_record", args={"id": "1"})],
                raw="[fault] tool name replaced with one that does not exist",
            )

        # -- LOUD: arguments the schema will reject -------------------------
        if self.fault == "malformed_args" and decision.tool_calls:
            broken = decision.tool_calls[0]
            return Decision(
                tool_calls=[ToolCall(id=broken.id, name=broken.name, args={"nonsense": 42})],
                raw="[fault] arguments replaced with ones the schema rejects",
            )

        # -- LOUD: replay the model's own first decision, forever -----------
        # Replaying its OWN first choice is what makes this a faithful
        # no-progress loop: it is exactly what an agent does when observations
        # never reach its input, because it keeps seeing the same thing.
        if self.fault == "loop_forever":
            if self._first is None and decision.tool_calls:
                self._first = decision
            if self._first is not None:
                call = self._first.tool_calls[0]
                return Decision(
                    tool_calls=[ToolCall(name=call.name, args=dict(call.args))],
                    raw="[fault] replaying the first decision, ignoring observations",
                )
            return decision

        # -- QUIET: a valid, successful call — for the wrong job ------------
        if self.fault == "wrong_tool" and len(names) > 1:
            if saw_observation:
                return Decision(
                    content=decision.content or _fallback_answer(messages),
                    raw="[fault] answering confidently from the wrong source",
                )
            if decision.tool_calls:
                wanted = decision.tool_calls[0].name
                alternatives = [n for n in names if n != wanted]
                if alternatives:
                    target = next(t for t in tools if t.name == alternatives[0])
                    return Decision(
                        tool_calls=[ToolCall(
                            id=decision.tool_calls[0].id,
                            name=target.name,
                            args=_plausible_args(target, messages),
                        )],
                        raw=f"[fault] tool swapped: {wanted} -> {target.name}",
                    )

        # -- QUIET: a confident answer supported by nothing ------------------
        if self.fault == "ignore_observations" and saw_observation:
            return Decision(
                content=(
                    "Your order was cancelled last Tuesday and a refund of $412.00 "
                    "has been issued to the original payment method."
                ),
                raw="[fault] answer generated without reference to any observation",
            )

        return decision


def _plausible_args(tool, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Minimal well-formed arguments for a swapped tool, so it actually runs."""
    goal = next((str(m.get("content") or "") for m in messages if m.get("role") == "user"), "")
    args: Dict[str, Any] = {}
    schema = getattr(tool, "schema", {}) or {}
    for name in schema.get("required", []):
        spec = schema.get("properties", {}).get(name, {})
        if "enum" in spec:
            args[name] = spec["enum"][0]
        elif "pattern" in spec:
            found = re.search(spec["pattern"].lstrip("^").rstrip("$"), goal, re.IGNORECASE)
            args[name] = found.group(0) if found else "ACME-1042"
        elif spec.get("type") in ("integer", "number"):
            args[name] = 1
        else:
            args[name] = goal[:120] or "n/a"
    return args


def _fallback_answer(messages: List[Dict[str, Any]]) -> str:
    observed = [str(m.get("content") or "") for m in messages if m.get("role") == "tool"]
    return "Based on what I found: " + (" ".join(observed)[:280] if observed else "nothing.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """A snapshot of the active configuration — printed at the top of notebooks."""

    llm_provider: str
    llm_model: str
    max_steps: int
    max_tool_calls: int
    has_key: bool

    def __str__(self) -> str:
        key = "set" if self.has_key else "MISSING — the notebooks will not run"
        return (
            f"LLM={self.llm_provider}:{self.llm_model}  |  key: {key}  |  "
            f"budget: {self.max_steps} steps / {self.max_tool_calls} tool calls"
        )


def current_config() -> AgentConfig:
    return AgentConfig(
        llm_provider=os.getenv("AGENT_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini"),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
        max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "12")),
        has_key=bool(os.getenv("OPENAI_API_KEY")),
    )


def have_api_key() -> bool:
    """True if a key is configured. Use it to give a clear message, not to fall back."""
    return bool(os.getenv("OPENAI_API_KEY"))


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    fault: Optional[str] = None,
    temperature: float = 0.0,
) -> LLM:
    """
    Return an LLM. OpenAI is the only provider, and a key is required.

    Args:
        fault: wrap the model in `FaultInjectingLLM` to reproduce one of the
            failure modes from notebook 06 deterministically.

    Raises `RuntimeError` with actionable instructions when the key is missing.
    We deliberately do NOT fall back to anything — a silent downgrade to a
    simulated model is how people end up drawing conclusions about model
    behaviour from something that is not a model.
    """
    provider = (provider or os.getenv("AGENT_LLM_PROVIDER", "openai")).lower()
    model = model or os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")

    if provider != "openai":
        raise ValueError(
            f"unknown LLM provider {provider!r}. This package ships an OpenAI "
            "adapter only; add another by writing a class with a `.decide()` "
            "method — see OpenAIToolCaller, which is about twenty lines."
        )

    llm: LLM = OpenAIToolCaller(model, temperature=temperature)
    return FaultInjectingLLM(llm, fault) if fault else llm

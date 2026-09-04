"""
tools.py — giving the agent reach, and containing the blast radius.
===================================================================

WHY this file exists
--------------------
An LLM on its own can only produce text about the world. A tool is the moment
the loop touches something real: a database, an API, a filesystem, a payment.

That is the upside. The downside arrives in the same instant, because the thing
deciding *which* tool to call and *with what* is a probabilistic text model. So
this module has exactly two jobs, and they pull in opposite directions:

  1. make it easy to expose a Python function to the agent  (`@tool`)
  2. make it impossible for a bad call to become a crash    (`Tool.invoke`)

Job 2 is the one people skip, and it is why so many demo agents fall over in
front of an audience. A tool that raises kills the loop; a tool that returns a
readable error lets the agent recover on the next step. Same failure, completely
different learner experience.

WHAT this module provides
-------------------------
* `Tool`          — a callable plus its derived schema plus safe invocation.
* `@tool`         — the decorator that turns a plain function into a `Tool`.
* `ToolRegistry`  — the set of tools an agent may use, and the source of the
                    schemas sent to the model.

THE CENTRAL DESIGN DECISION: tools never raise
----------------------------------------------
`Tool.invoke()` returns an `Observation`, never an exception. Every failure —
validation rejection, an exception inside the function, a timeout — becomes an
`Observation` with a non-OK status and a message written for the model to read.

This is not defensive-programming reflex. It is what makes the agent loop a
*loop*. Consider the two designs when the model passes `order_id="1042"`
instead of `"ACME-1042"`:

  raises   -> traceback, run over, learner sees a red cell and no agent
  returns  -> step 3 observes "order_id must match ^ACME-\\d+$", step 4 calls
              it correctly, the run succeeds, and the trace shows self-repair

The second is both more robust and a better demo, and it costs one try/except.

The recurring question — *"what does this step look like when it goes wrong,
and where would you see it in the trace?"*:

  * Tool raises              -> no trace at all; the run dies mid-step. The
                                worst failure mode, because you lose the
                                evidence along with the run.
  * Tool returns huge output -> context blows up; you see `approx_tokens()`
                                climbing and cost per step rising. See
                                `max_result_chars`.
  * Tool silently returns {} -> the agent confidently reports nothing found.
                                This is why an empty result should say so in
                                words, not return an empty container.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

from .schemas import (
    build_schema,
    describe_for_prompt,
    parse_docstring,
    to_anthropic,
    to_gemini,
    to_openai,
    validate_args,
)
from .state import Observation, ObservationStatus


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------
@dataclass
class Tool:
    """
    One capability the agent can invoke.

    A Tool bundles three things that must never drift apart: the function, the
    schema describing how to call it, and the description telling the model when
    to. Because the schema is derived from the function at construction time,
    "the docs say one thing and the code does another" is structurally
    impossible here.
    """

    name: str
    description: str
    func: Callable[..., Any]
    schema: Dict[str, Any]
    # Example requests this tool serves. Optional, and the highest-leverage
    # optional thing here: when a model keeps failing to pick a tool that
    # obviously applies, adding two or three example requests fixes it far more
    # often than rewriting the description again. Examples are shown to real
    # models in the prompt and used by the offline mock's router, so improving
    # them improves both.
    examples: List[str] = field(default_factory=list)
    # Result size cap. Tools that read documents or query APIs can return a lot,
    # and every character lands in the transcript on the NEXT step and every
    # step after. Truncating at the boundary is far cheaper than discovering
    # context overflow four steps later.
    max_result_chars: int = 1200
    # A terminal tool ends the run when it succeeds (e.g. escalate_to_human).
    # Encoding this on the tool rather than in the loop keeps the loop generic:
    # the loop asks "was that terminal?" instead of hard-coding tool names.
    terminal: bool = False

    # -- invocation --------------------------------------------------------
    def invoke(self, args: Dict[str, Any], step: int = 0) -> Observation:
        """
        Validate, call, and package the result. NEVER raises.

        The three outcomes map onto the three ObservationStatus values, and the
        distinction between the first two is worth labouring in class:

          REJECTED — the call never happened. The model's fault; cheap; fixable
                     by the model on the next step from the error message.
          ERROR    — the call happened and blew up. Could be the model's fault
                     (valid-but-wrong ID) or yours (the tool has a bug). Costly,
                     because side effects may already have occurred.
          OK       — it worked.

        An agent that cannot tell REJECTED from ERROR cannot learn from either.
        """
        started = time.perf_counter()

        # --- 1. the gate: validate before we touch anything real ----------
        checked = validate_args(args, self.schema)
        if not checked.ok:
            return Observation(
                step=step,
                tool=self.name,
                args=args,
                status=ObservationStatus.REJECTED,
                error=checked.message(),
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        # --- 2. the call --------------------------------------------------
        try:
            raw = self.func(**checked.args)
        except Exception as exc:
            # We keep the exception TYPE in the message. "KeyError: 'ACME-9999'"
            # tells the agent the ID was well-formed but unknown, which is a
            # different repair than a malformed ID. Detail here shortens loops.
            return Observation(
                step=step,
                tool=self.name,
                args=checked.args,
                status=ObservationStatus.ERROR,
                error=f"{type(exc).__name__}: {exc}",
                duration_ms=(time.perf_counter() - started) * 1000,
            )

        # --- 3. the result, made safe for the transcript ------------------
        return Observation(
            step=step,
            tool=self.name,
            args=checked.args,
            status=ObservationStatus.OK,
            result=self._truncate(raw),
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def _truncate(self, value: Any) -> Any:
        """
        Cap the size of a result, saying so out loud when we do.

        The marker matters. A silently truncated result is a subtle
        hallucination source: the agent reasons over half a table believing it
        saw all of it. An explicit "[truncated …]" lets the model — and the
        learner reading the trace — know the view is partial.
        """
        text = value if isinstance(value, str) else str(value)
        if len(text) <= self.max_result_chars:
            return value
        kept = text[: self.max_result_chars]
        dropped = len(text) - self.max_result_chars
        return f"{kept}\n[truncated — {dropped} more characters not shown]"

    # -- provider views ----------------------------------------------------
    def to_openai(self) -> Dict[str, Any]:
        return to_openai(self.name, self.description, self.schema)

    def to_gemini(self) -> Dict[str, Any]:
        return to_gemini(self.name, self.description, self.schema)

    def to_anthropic(self) -> Dict[str, Any]:
        return to_anthropic(self.name, self.description, self.schema)

    def to_langchain(self):
        """
        A LangChain StructuredTool wrapping this same function.

        The parallel mapping, in the tradition of the RAG package: the framework gives you a
        nicer object and a bigger ecosystem, and it still cannot decide your
        enum values or your ID format for you. Import is local so the package
        has no hard LangChain dependency.
        """
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            func=self.func, name=self.name, description=self.description
        )

    def prompt_line(self) -> str:
        """Plain-text rendering, for prompts and for the offline mock."""
        return describe_for_prompt(
            self.name, self.description, self.schema, self.examples
        )

    def __str__(self) -> str:
        params = ", ".join(self.schema.get("properties", {}))
        return f"Tool({self.name}({params}))"


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------
def tool(
    _func: Optional[Callable] = None,
    *,
    name: Optional[str] = None,
    description: Optional[str] = None,
    max_result_chars: int = 1200,
    terminal: bool = False,
    examples: Optional[List[str]] = None,
) -> Any:
    """
    Turn a plain function into a `Tool`.

    Usable bare or with arguments:

        @tool
        def calculate(expression: str) -> str: ...

        @tool(terminal=True)
        def escalate_to_human(summary: str) -> str: ...

    By default the name comes from the function and the description from its
    docstring summary. Writing the description in the docstring rather than the
    decorator is the deliberate default: it keeps the text a human reads and the
    text the model reads as the same text, so neither can rot unnoticed.
    """

    def wrap(func: Callable) -> Tool:
        summary, _ = parse_docstring(func)
        schema = build_schema(func)
        return Tool(
            name=name or func.__name__,
            description=description or summary or f"Call {func.__name__}.",
            func=func,
            schema=schema,
            max_result_chars=max_result_chars,
            terminal=terminal,
            examples=list(examples or []),
        )

    return wrap(_func) if callable(_func) else wrap


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class ToolRegistry:
    """
    The set of tools one agent (or one skill) may use.

    WHY a registry rather than passing a list around: because "which tools does
    this agent have?" is a *scoping decision*, and scoping is what makes skills
    work in skills.py. A registry can be subset — `registry.subset("search_docs",
    "calculate")` — which is precisely how a skill restricts an agent to the
    tools its job needs.

    That restriction is not tidiness. Every extra tool in the list is another
    option the model can pick wrongly, and tool-choice accuracy degrades
    measurably as the list grows. Fewer tools is a reliability technique.
    """

    def __init__(self, tools: Optional[Iterable[Tool]] = None):
        self._tools: Dict[str, Tool] = {}
        for item in tools or []:
            self.add(item)

    # -- construction ------------------------------------------------------
    def add(self, item: Tool) -> "ToolRegistry":
        if not isinstance(item, Tool):
            raise TypeError(
                f"ToolRegistry takes Tool objects; got {type(item).__name__}. "
                "Did you forget the @tool decorator?"
            )
        self._tools[item.name] = item
        return self

    def subset(self, *names: str) -> "ToolRegistry":
        """A new registry with only the named tools — the basis of skills."""
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"unknown tools: {missing}. Available: {self.names()}")
        return ToolRegistry(self._tools[n] for n in names)

    # -- access ------------------------------------------------------------
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    # -- dispatch ----------------------------------------------------------
    def dispatch(self, name: str, args: Dict[str, Any], step: int = 0) -> Observation:
        """
        Route a call to its tool, handling the unknown-tool case.

        A hallucinated tool name is common enough to deserve its own branch, and
        the error deliberately lists what IS available. Handing the model the
        valid options turns a dead end into a correctable mistake — the same
        principle as enums in schemas.py, applied one level up.
        """
        found = self._tools.get(name)
        if found is None:
            return Observation(
                step=step,
                tool=name,
                args=args,
                status=ObservationStatus.REJECTED,
                error=(
                    f"unknown tool {name!r}. Available tools: {', '.join(self.names())}"
                ),
            )
        return found.invoke(args, step=step)

    # -- provider views ----------------------------------------------------
    def to_openai(self) -> List[Dict[str, Any]]:
        return [t.to_openai() for t in self]

    def to_gemini(self) -> List[Dict[str, Any]]:
        """Gemini expects declarations wrapped in a `function_declarations` list."""
        return [{"function_declarations": [t.to_gemini() for t in self]}]

    def to_anthropic(self) -> List[Dict[str, Any]]:
        return [t.to_anthropic() for t in self]

    def to_langchain(self) -> List[Any]:
        return [t.to_langchain() for t in self]

    def prompt_block(self) -> str:
        """All tools as prompt text — used by the offline mock and by skills."""
        return "\n".join(t.prompt_line() for t in self)

    def __str__(self) -> str:
        return f"ToolRegistry({', '.join(self.names())})"

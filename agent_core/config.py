"""
config.py — one interface for "something that decides what to do next".
=======================================================================

WHY this file exists
--------------------
The session must stay agnostic of any particular LLM, but the code has to
actually run — in a room where some learners have an API key and some do not.
Those goals only coexist if the choice of provider hides behind one small
interface. That is this module.

The whole package is written against a single method:

    llm.decide(messages, tools) -> Decision

A `Decision` is either "call these tools with these arguments" or "here is the
final answer". Nothing else in `agent_core` imports `openai`. Swap providers
with an environment variable.

THE IMPORTANT PART: the mock decides TOOL CALLS
-----------------------------------------------
In the C8 RAG package the offline mock only had to emit text, because a RAG
pipeline is a straight line and text was the last step. An agent is a *loop*,
and the thing being looped on is a decision. A mock that only produced prose
would leave a keyless classroom unable to run notebooks 02 through 07 — which
is to say, unable to run the session.

So `MockToolCallLLM` is a deterministic router. It reads the goal against the
registered tool schemas, works out which tools are relevant and in what order,
tracks which it has already called by reading the transcript, and emits the next
one. When it has observations and nothing left to call, it answers from them.

Three things make this honest rather than a cheat:

  1. It is **transparent**. `plan()` returns its reasoning — which tools it
     picked, with what score, and why — and the notebooks print it. Learners see
     a heuristic being a heuristic, not a small language model.
  2. It is **extractive**. It answers only from observations it actually got. It
     cannot hallucinate a fact because it never generates facts. When a learner
     later plugs in a real LLM, the fluency gap is obvious and the *grounding*
     is not.
  3. It is **honest about being a mock**. Every output is labelled `[mock]`.

It also pays for itself a second time. Because it is deterministic, it makes an
ideal fault injector: `MockToolCallLLM(fault="loop_forever")` reproduces the
classic agent pathologies on demand, identically every time, with no API spend.
That is what notebook 06 runs on.

WHAT IT IS NOT
--------------
It is not a language model. It cannot handle a goal phrased in a way its
keyword scorer misses, and it will not reason about anything. Use it to learn
the *mechanics* of the loop; use a real model to see the mechanics do something
impressive.
"""
from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

# Load .env if python-dotenv is available. Colab users typically set env vars
# via Secrets or getpass instead, which works identically.
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
# The offline mock
# ---------------------------------------------------------------------------
# Words that carry no routing signal. Kept small and visible so a learner can
# see exactly why a goal matched a tool.
_STOP = {
    "the", "a", "an", "and", "or", "of", "for", "to", "in", "on", "is", "are",
    "was", "what", "which", "how", "why", "when", "where", "does", "do", "did",
    "can", "could", "would", "should", "my", "our", "your", "this", "that",
    "with", "from", "about", "please", "tell", "me", "get", "find", "out", "i",
    "it", "be", "have", "has", "any", "all", "acme", "cloud", "customer",
}

def _stem(word: str) -> str:
    """
    Crudest possible stemmer: drop a trailing plural 's'.

    It exists because "the Growth plan" should match a tool description that
    says "plans", and without it the router misses obvious matches. Real
    retrieval systems do this properly (and C8's BM25 discussion is the place
    that got covered); here the one-line version buys most of the benefit.
    """
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def _stem_all(words) -> set:
    return {_stem(w) for w in words}


def _content_words(text: str) -> set:
    """Stemmed, stopword-free words of 3+ characters — the routing vocabulary."""
    return _stem_all(
        {w for w in re.findall(r"[a-z0-9]{3,}", text.lower())} - _STOP
    )


_FAULTS = (
    "loop_forever",       # never finalises — burns the entire step budget
    "bad_tool_name",      # calls a tool that does not exist
    "malformed_args",     # sends arguments the schema will reject
    "wrong_tool",         # picks a plausible but irrelevant tool
    "ignore_observations",  # answers without using what the tools returned
)


class MockToolCallLLM:
    """
    Deterministic, offline stand-in for a tool-calling model.

    Args:
        fault: one of `_FAULTS`, or None for normal behaviour. Used by
            `failures.py` to reproduce agent pathologies without an API key.
        max_tools_per_goal: how many tools it is willing to chain for one goal.
    """

    def __init__(self, fault: Optional[str] = None, max_tools_per_goal: int = 3):
        if fault is not None and fault not in _FAULTS:
            raise ValueError(f"unknown fault {fault!r}; choose from {list(_FAULTS)}")
        self.name = f"mock{'/' + fault if fault else ''}"
        self.fault = fault
        self.max_tools_per_goal = max_tools_per_goal

    # -- routing: the transparent part -------------------------------------
    def plan(self, goal: str, tools) -> List[Dict[str, Any]]:
        """
        Score every tool against the goal and return the ranked shortlist.

        This is deliberately introspectable — the notebooks print it — so that
        the mock's behaviour is never mysterious. Three signals, in descending
        order of how much they should be trusted:

          +6  a schema `pattern` for one of the tool's parameters matches text
              in the goal. Strongest signal by far: if the goal contains
              "ACME-1042" and a tool takes an `^ACME-\\d+$` argument, that tool
              is almost certainly wanted.
          +3  a distinctive word from the tool's NAME appears in the goal.
          +1  a distinctive word from the tool's DESCRIPTION appears.

        A real model does something functionally similar and vastly better. The
        point of showing the scores is that "the model picks a tool" stops being
        magic and becomes "something matches the description you wrote" — which
        is exactly the intuition a learner needs when their tool never gets
        picked and the fix is to rewrite its description.
        """
        goal_words = _content_words(goal)
        ranked = []

        for candidate in tools:
            score = 0
            why: List[str] = []

            for prop, spec in candidate.schema.get("properties", {}).items():
                pattern = spec.get("pattern")
                # Unanchor before searching inside a sentence — see _extract_args.
                loose = pattern.lstrip("^").rstrip("$") if pattern else None
                if loose and re.search(loose, goal, re.IGNORECASE):
                    score += 6
                    why.append(f"goal contains a value matching {prop}'s format")

            for example in candidate.examples:
                shared = goal_words & _content_words(example)
                if len(shared) >= 2:
                    score += 5
                    why.append(f"matches example {example!r} on {sorted(shared)}")

            name_words = {w for w in re.split(r"[_\W]+", candidate.name.lower()) if w}
            for word in _stem_all(name_words) & goal_words:
                score += 3
                why.append(f"name word {word!r} in goal")

            desc_words = _content_words(candidate.description)
            for word in desc_words & goal_words:
                score += 1
                why.append(f"description word {word!r} in goal")

            if score:
                ranked.append({"tool": candidate, "score": score, "why": why})

        ranked.sort(key=lambda r: (-r["score"], r["tool"].name))

        # Fallback: when nothing matched, reach for a general information-gathering
        # tool rather than answering from thin air. "When in doubt, go and look" is
        # the behaviour you want from a real agent too, and giving one a default
        # search tool is a genuine design technique — not a mock-only crutch. The
        # candidate is the first non-terminal tool whose required arguments are all
        # free text, i.e. one you can call with the question as-is.
        if not ranked:
            for candidate in tools:
                if candidate.terminal:
                    continue
                properties = candidate.schema.get("properties", {})
                required = candidate.schema.get("required", [])
                if required and all(
                    properties[p].get("type") == "string"
                    and "pattern" not in properties[p]
                    and "enum" not in properties[p]
                    for p in required
                ):
                    ranked.append(
                        {
                            "tool": candidate,
                            "score": 0,
                            "why": ["fallback: no tool matched, so gather information first"],
                        }
                    )
                    break

        return ranked[: self.max_tools_per_goal]

    def explain_plan(self, goal: str, tools) -> str:
        """Human-readable version of `plan()`, for notebooks and the handout."""
        ranked = self.plan(goal, tools)
        if not ranked:
            return f"[mock] No tool matched {goal!r}. The mock will answer directly."
        lines = [f"[mock] Routing plan for {goal!r}:"]
        for i, row in enumerate(ranked, 1):
            lines.append(f"  {i}. {row['tool'].name}  (score {row['score']})")
            for reason in row["why"][:3]:
                lines.append(f"       because {reason}")
        return "\n".join(lines)

    # -- argument extraction ------------------------------------------------
    def _extract_args(self, candidate, goal: str) -> Dict[str, Any]:
        """
        Fill a tool's parameters from the goal text.

        Rules, tried per parameter in this order:
          1. a `pattern` constraint  -> pull the first match out of the goal
          2. an `enum` constraint    -> the first option mentioned, else the first
          3. a numeric type          -> the first number in the goal, else default
          4. anything else           -> the goal text itself

        Rule 4 is the one that makes free-text tools (`search_docs`,
        `escalate_to_human`) work without any special-casing: for a tool whose
        argument IS a question, the question is the argument.
        """
        args: Dict[str, Any] = {}
        properties = candidate.schema.get("properties", {})
        required = candidate.schema.get("required", [])

        for prop, spec in properties.items():
            if prop not in required and len(properties) > 1:
                # Leave optional parameters to their defaults — fewer knobs
                # touched means fewer ways for the mock to be wrong.
                continue

            pattern = spec.get("pattern")
            if pattern:
                # Schema patterns are anchored (^ACME-\d{4}$) because they
                # describe a WHOLE valid value. To find such a value *inside* a
                # sentence we have to unanchor first — searching for "^ACME"
                # in "Is order ACME-1042 refundable?" matches nothing, which is
                # correct regex behaviour and a genuinely easy bug to ship.
                loose = pattern.lstrip("^").rstrip("$")
                found = re.search(loose, goal, re.IGNORECASE)
                args[prop] = found.group(0) if found else goal.strip()
                continue

            if "enum" in spec:
                low = goal.lower()
                match = next(
                    (o for o in spec["enum"] if str(o).lower() in low), spec["enum"][0]
                )
                args[prop] = match
                continue

            if spec.get("type") in ("integer", "number"):
                found = re.search(r"-?\d+(?:\.\d+)?", goal)
                if found:
                    value = found.group(0)
                    args[prop] = int(value) if spec["type"] == "integer" else float(value)
                elif "default" in spec:
                    args[prop] = spec["default"]
                continue

            args[prop] = goal.strip()

        return args

    # -- reading the transcript --------------------------------------------
    @staticmethod
    def _called_so_far(messages: List[Dict[str, Any]]) -> List[str]:
        """
        Which tools the transcript shows as already called.

        The mock is stateless between `decide()` calls on purpose — it derives
        everything from `messages`, exactly as a real model does. That keeps the
        message list the single source of truth and means a learner can hand-
        edit the transcript in a notebook and watch the behaviour change.
        """
        names: List[str] = []
        for msg in messages:
            for call in msg.get("tool_calls") or []:
                names.append(call["function"]["name"])
        return names

    @staticmethod
    def _observations(messages: List[Dict[str, Any]]) -> List[str]:
        return [
            str(m.get("content") or "") for m in messages if m.get("role") == "tool"
        ]

    # -- the interface ------------------------------------------------------
    def decide(self, messages: List[Dict[str, Any]], tools) -> Decision:
        goal = next(
            (m["content"] for m in messages if m.get("role") == "user"), ""
        )
        already = self._called_so_far(messages)
        observations = self._observations(messages)
        ranked = self.plan(goal, tools)

        # --- fault injection (notebook 06) --------------------------------
        if self.fault == "bad_tool_name":
            return Decision(
                tool_calls=[ToolCall(name="lookup_customer_record", args={"id": "1"})],
                raw="[mock/fault] deliberately calling a tool that does not exist",
            )
        if self.fault == "malformed_args" and ranked:
            return Decision(
                tool_calls=[ToolCall(name=ranked[0]["tool"].name, args={"nonsense": 42})],
                raw="[mock/fault] deliberately sending arguments the schema rejects",
            )
        if self.fault == "wrong_tool" and len(list(tools)) > 1:
            picked = [t for t in tools if not ranked or t.name != ranked[0]["tool"].name]
            return Decision(
                tool_calls=[
                    ToolCall(name=picked[0].name, args=self._extract_args(picked[0], goal))
                ],
                raw="[mock/fault] deliberately choosing an irrelevant tool",
            )
        if self.fault == "loop_forever" and ranked:
            # Re-request the SAME call forever, ignoring the fact it already ran.
            # This is what an agent with no memory of its own actions looks like.
            first = ranked[0]["tool"]
            return Decision(
                tool_calls=[ToolCall(name=first.name, args=self._extract_args(first, goal))],
                raw="[mock/fault] ignoring prior observations and repeating",
            )
        if self.fault == "ignore_observations" and observations:
            return Decision(
                content=(
                    "[mock/fault] Your order was cancelled last Tuesday and a refund "
                    "of $412.00 has been issued to the original payment method."
                ),
                raw="[mock/fault] answering without reference to any observation",
            )

        # --- normal behaviour ---------------------------------------------
        # Call the next relevant tool we have not yet called.
        for row in ranked:
            candidate = row["tool"]
            if candidate.name not in already:
                return Decision(
                    tool_calls=[
                        ToolCall(
                            name=candidate.name,
                            args=self._extract_args(candidate, goal),
                        )
                    ],
                    raw=f"[mock] selected {candidate.name} (score {row['score']})",
                )

        # Nothing left to call -> answer from what we observed.
        return Decision(content=self._answer(goal, observations), raw="[mock] final")

    def _answer(self, goal: str, observations: List[str]) -> str:
        """
        Build a final answer from observations only.

        The refusal branch matters as much as the answer branch. An agent that
        called no tool successfully has no grounds to say anything, and saying
        so is the correct behaviour — it is the agentic analogue of C8's
        grounded "I don't know based on the provided context", and it is what
        the `unanswerable` tasks in `data/tasks/agent_tasks.jsonl` test for.
        """
        useful = [o for o in observations if o and not o.startswith("ERROR")]
        if not useful:
            return (
                "[mock] I could not gather any information for this request, so I "
                "will not guess. A human should take a look."
            )
        joined = "\n".join(f"  • {' '.join(o.split())[:220]}" for o in useful)
        return (
            f"[mock] Based only on what the tools returned:\n{joined}\n\n"
            "(This is an extractive offline stand-in. Set OPENAI_API_KEY for a "
            "real, fluent answer.)"
        )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------
class OpenAIToolCaller:
    """
    OpenAI chat-completions with native tool calling — the default.

    Note how thin this is. Roughly twenty lines translate between our
    `Decision` vocabulary and OpenAI's response shape, and that is the entire
    provider-specific surface of this package. When a learner asks "how much
    work is it to support another provider?", this class is the answer.
    """

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI()
        self.name = f"openai:{model}"
        self.model = model
        # Temperature 0 by default: in an agent loop you want the tool-choice
        # step to be as reproducible as the provider allows. Creativity in the
        # ACT step is variance you will spend the session debugging.
        self.temperature = temperature

    def decide(self, messages: List[Dict[str, Any]], tools) -> Decision:
        payload = tools.to_openai() if hasattr(tools, "to_openai") else tools
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
# Factory
# ---------------------------------------------------------------------------
@dataclass
class AgentConfig:
    """A snapshot of the active configuration — printed at the top of notebooks."""

    llm_provider: str
    llm_model: str
    max_steps: int
    max_tool_calls: int

    def __str__(self) -> str:
        return (
            f"LLM={self.llm_provider}:{self.llm_model}  |  "
            f"budget: {self.max_steps} steps / {self.max_tool_calls} tool calls"
        )


def current_config() -> AgentConfig:
    return AgentConfig(
        llm_provider=os.getenv("AGENT_LLM_PROVIDER", "openai"),
        llm_model=os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini"),
        max_steps=int(os.getenv("AGENT_MAX_STEPS", "8")),
        max_tool_calls=int(os.getenv("AGENT_MAX_TOOL_CALLS", "12")),
    )


def get_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    fault: Optional[str] = None,
) -> LLM:
    """
    Return an LLM. Resolution order: explicit args -> env -> openai -> mock.

    If the chosen provider cannot be constructed — no key, library not
    installed, no network — we print one clear line and fall back to the mock.
    A classroom keeps moving; it does not stop on an ImportError.
    """
    provider = (provider or os.getenv("AGENT_LLM_PROVIDER", "openai")).lower()
    model = model or os.getenv("AGENT_LLM_MODEL", "gpt-4o-mini")

    if fault is not None:
        # A requested fault always means the mock — faults are a teaching
        # device, and injecting them into a paid API call would be both
        # expensive and non-reproducible.
        return MockToolCallLLM(fault=fault)

    try:
        if provider == "mock":
            return MockToolCallLLM()
        if provider == "openai":
            return OpenAIToolCaller(model)
        raise ValueError(f"unknown LLM provider: {provider}")
    except Exception as exc:
        print(f"[config] LLM '{provider}' unavailable ({exc}); using MockToolCallLLM.")
        return MockToolCallLLM()

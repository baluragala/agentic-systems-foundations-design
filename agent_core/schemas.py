"""
schemas.py — turning Python functions into contracts an LLM can honour.
=======================================================================

WHY this file exists
--------------------
A tool is a promise: "call me with these arguments and I will do this thing."
An LLM is a text generator that has read a description of that promise and is
doing its best. Between those two facts sits every tool-calling bug you will
ever debug.

The schema is where you decide how much you trust the model. Consider one
tool, `get_order_status(order_id)`, and one model output: `{"order_id": 1042}`.

  * With a loose schema, that reaches your function as an int, `orders[1042]`
    raises `KeyError`, and you get a stack trace that blames your database code
    for a mistake the model made two layers up.
  * With a strict schema, it is rejected before your function is entered, with
    a message the agent can actually act on: *"order_id must match ACME-####"*.

Same model, same output. The difference is entirely in this file. That is the
argument for why schema design is a first-class skill and not paperwork.

WHAT this module provides
-------------------------
1. `build_schema()`  — derive a JSON Schema from a Python signature + type
                       hints + docstring, so the schema cannot drift from the
                       function it describes.
2. `validate_args()` — check *and coerce* a payload against a schema, returning
                       a structured result rather than raising.
3. Translators       — the same schema rendered for the OpenAI, LangChain, and
                       Gemini tool interfaces.

WHY we hand-roll a validator when `jsonschema` exists
-----------------------------------------------------
Two reasons, one pedagogical and one practical.

Pedagogical: a learner who has never seen validation implemented treats it as
magic and therefore as optional. Ninety lines they can read removes the magic.
Notebook 03 shows `jsonschema` immediately afterwards as the parallel mapping —
the same lesson the RAG session taught with from-scratch chunking vs LangChain splitters.

Practical: LLM tool arguments need *coercion*, not just validation. The string
`"3"` for an integer parameter is a model quirk, not a user error, and rejecting
it wastes a whole agent step. A stock validator says no; we say "yes, as 3" and
record that we did. Production systems end up writing this layer anyway.

The recurring question — *"what does this step look like when it goes wrong,
and where would you see it in the trace?"*:

  * No schema           -> the model invents parameter names; you see
                           TypeError: unexpected keyword argument.
  * Loose schema        -> wrong types reach your code; you see failures deep
                           inside the tool, blaming the wrong component.
  * Over-strict schema  -> the agent burns its whole budget on retries it can
                           never satisfy; the trace shows REJECTED over and
                           over with the same message.
"""
from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, get_args, get_origin


# ---------------------------------------------------------------------------
# Python type -> JSON Schema type
# ---------------------------------------------------------------------------
_JSON_TYPES = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _json_type(annotation: Any) -> Dict[str, Any]:
    """
    Map one Python annotation onto a JSON Schema fragment.

    Handles the annotations that actually show up on tool functions: the
    builtins, `Optional[X]`, `Literal[...]` (which becomes an enum — the single
    highest-value schema feature for agents, because it turns "guess a valid
    value" into "pick from this list"), and `list[X]`.

    Anything unrecognised degrades to `{"type": "string"}`. Degrading beats
    raising here: a tool with one exotic annotation should still be callable,
    just less precisely described.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return {"type": "string"}

    origin = get_origin(annotation)

    # Literal["a", "b"] -> enum. This is how you stop a model inventing values.
    if origin is typing.Literal:
        options = list(get_args(annotation))
        kind = _JSON_TYPES.get(type(options[0]), "string") if options else "string"
        return {"type": kind, "enum": options}

    # Optional[X] / Union[X, None] -> X (JSON Schema handles optionality via
    # `required`, not via the type, so we unwrap and let build_schema decide).
    if origin is typing.Union:
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return _json_type(non_none[0])
        return {"type": "string"}

    # list[X] -> array with an items type.
    if origin in (list, List):
        args = get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {}}

    if origin in (dict, Dict):
        return {"type": "object"}

    return {"type": _JSON_TYPES.get(annotation, "string")}


# ---------------------------------------------------------------------------
# Docstring parsing — descriptions the model actually reads
# ---------------------------------------------------------------------------
_ARG_LINE = re.compile(r"^\s*(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.+)$")


def parse_docstring(func: Callable) -> tuple[str, Dict[str, str]]:
    """
    Split a Google-style docstring into (summary, {param: description}).

    This matters more than it looks. The parameter descriptions are the *only*
    natural-language guidance the model gets about what a valid value looks
    like. A schema that says `{"type": "string"}` and a description that says
    "Order ID in the form ACME-1042" are a completely different tool from the
    model's point of view, even though the machine-checkable part is identical.

    Notebook 03 makes this concrete: the same tool, same code, only the
    descriptions improved, and the agent's first-attempt success rate changes.
    """
    doc = inspect.getdoc(func) or ""
    if not doc:
        return "", {}

    # Only names that are actually parameters count as the start of a new entry.
    # Without this guard any wrapped line containing a colon — "pattern: ^ACME-\\d+$",
    # "note: see policy" — is read as a new parameter, silently truncating the
    # description of the real one above it. Nothing errors; the tool just gets
    # harder for the model to call, which is a miserable thing to debug.
    try:
        known = set(inspect.signature(func).parameters)
    except (TypeError, ValueError):
        known = set()

    lines = doc.splitlines()
    summary_lines: List[str] = []
    params: Dict[str, str] = {}
    in_args = False
    # Once any section header appears, the summary is finished. Without this
    # flag the "Returns:" prose gets appended to the tool description, so the
    # model is told what the tool returns as if it were part of what the tool
    # is for — noise in the field it uses to choose tools.
    seen_section = False

    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:"):
            in_args = True
            seen_section = True
            continue
        if stripped.lower() in ("returns:", "raises:", "example:", "examples:"):
            in_args = False
            seen_section = True
            continue
        if in_args:
            match = _ARG_LINE.match(line)
            if match and (not known or match.group(1) in known):
                params[match.group(1)] = match.group(3).strip()
            elif stripped and params:
                # A continuation line for the previous parameter.
                last = list(params)[-1]
                params[last] += " " + stripped
        elif stripped and not seen_section:
            summary_lines.append(stripped)

    return " ".join(summary_lines).strip(), params


# ---------------------------------------------------------------------------
# Signature -> JSON Schema
# ---------------------------------------------------------------------------
def build_schema(func: Callable) -> Dict[str, Any]:
    """
    Derive a JSON Schema `parameters` object from a function.

    The schema is DERIVED, never hand-written alongside the function, because a
    hand-written schema is a second source of truth that silently rots the first
    time someone adds a parameter. This is the same argument as "generate your
    API docs from your code".

    Constraints beyond the bare type come from two places:
      * `Literal[...]` annotations -> `enum`
      * a `pattern:` marker in the parameter's docstring line -> `pattern`

    The second is a small convention that buys a lot: it lets a tool author
    express "must look like ACME-1042" in the one place they were already
    writing prose, and have it become machine-enforced.
    """
    sig = inspect.signature(func)
    summary, param_docs = parse_docstring(func)

    # Resolve annotations to real types. This matters more than it looks: a
    # module with `from __future__ import annotations` (which ours have, and
    # which is standard modern practice) hands us the *string*
    # 'Literal["billing_error", ...]' instead of the type. Without this call
    # every parameter silently degrades to a plain string, every enum vanishes,
    # and the model loses the constraint that was doing the most work for it.
    # The failure is invisible — the schema still validates, just uselessly.
    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    properties: Dict[str, Any] = {}
    required: List[str] = []

    for name, param in sig.parameters.items():
        if name in ("self", "cls") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        prop = _json_type(hints.get(name, param.annotation))
        description = param_docs.get(name, "")

        # Convention: "... pattern: ^ACME-\d+$" in the docstring becomes a real
        # JSON Schema pattern constraint.
        pattern_match = re.search(r"pattern:\s*(\S+)", description)
        if pattern_match:
            prop["pattern"] = pattern_match.group(1)
            # Excise ONLY the marker, keeping the prose on both sides. The text
            # after it is usually the worked example ("for example ACME-1042"),
            # and an example is the single most effective thing you can put in a
            # parameter description — dropping it would quietly make the tool
            # harder for the model to call correctly.
            description = (
                description[: pattern_match.start()] + description[pattern_match.end():]
            )
            description = re.sub(r"\s+", " ", description).strip().strip(",").strip()

        if description:
            prop["description"] = description
        properties[name] = prop

        if param.default is inspect.Parameter.empty:
            required.append(name)
        else:
            prop["default"] = param.default

    return {
        "type": "object",
        "properties": properties,
        "required": required,
        # additionalProperties=False is the line that stops a model from
        # inventing a parameter you never defined. It is off by default in most
        # tutorials and on by default in every system that has been burned.
        "additionalProperties": False,
        "_summary": summary,  # stripped out by the translators below
    }


# ---------------------------------------------------------------------------
# Validation + coercion
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """
    The outcome of checking one payload against one schema.

    Returned rather than raised, because in an agent loop a validation failure
    is *normal control flow*, not an exception. The agent is supposed to see the
    error, read it, and try again — so the error has to be a value it can be
    handed, and the coercions have to be visible so a learner can see what the
    validator quietly fixed.
    """

    ok: bool
    args: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    coercions: List[str] = field(default_factory=list)

    def message(self) -> str:
        """The text handed back to the model when validation fails."""
        return "; ".join(self.errors)


def _coerce(value: Any, want: str) -> tuple[Any, bool]:
    """
    Try to turn `value` into JSON type `want`. Returns (value, did_coerce).

    Scope is deliberately narrow — only the conversions that are unambiguously
    safe and that LLMs actually get wrong. We do NOT coerce a non-empty string
    to a bool, because "false" would become True and a learner would spend an
    hour finding it. When in doubt, fail loudly.
    """
    if want == "string" and isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value), True
    if want == "integer" and isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"-?\d+", text):
            return int(text), True
    if want == "integer" and isinstance(value, float) and value.is_integer():
        return int(value), True
    if want == "number" and isinstance(value, str):
        try:
            return float(value.strip()), True
        except ValueError:
            pass
    if want == "boolean" and isinstance(value, str):
        low = value.strip().lower()
        if low in ("true", "yes", "1"):
            return True, True
        if low in ("false", "no", "0"):
            return False, True
    return value, False


_TYPE_CHECKS = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
}


def validate_args(payload: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """
    Validate (and gently coerce) `payload` against a `build_schema()` schema.

    Checks, in the order a learner should think about them:
      1. every required property is present
      2. no unexpected properties (if additionalProperties is False)
      3. each value is the right type — coercing where it is safe to
      4. enum membership
      5. regex pattern

    The error messages are written for the MODEL to read, not for a developer.
    "order_id: expected string matching ^ACME-\\d+$, got 'ACME1042'" tells the
    agent how to fix its next attempt. "ValidationError at $.order_id" does not.
    That distinction is the whole reason a retry loop converges or spins.
    """
    if not isinstance(payload, dict):
        return ValidationResult(ok=False, errors=[f"arguments must be an object, got {type(payload).__name__}"])

    properties: Dict[str, Any] = schema.get("properties", {})
    required: List[str] = schema.get("required", [])
    result = ValidationResult(ok=True, args=dict(payload))

    # 1. Required properties.
    for name in required:
        if name not in payload:
            result.errors.append(f"{name}: required parameter is missing")

    # 2. Unexpected properties — catches invented parameter names.
    if schema.get("additionalProperties") is False:
        for name in payload:
            if name not in properties:
                known = ", ".join(properties) or "(none)"
                result.errors.append(
                    f"{name}: unknown parameter (this tool accepts: {known})"
                )
                result.args.pop(name, None)

    # 3-5. Per-property checks.
    for name, spec in properties.items():
        if name not in result.args:
            if "default" in spec:
                result.args[name] = spec["default"]
            continue

        value = result.args[name]
        want = spec.get("type", "string")

        check = _TYPE_CHECKS.get(want)
        if check and not check(value):
            coerced, did = _coerce(value, want)
            if did:
                result.args[name] = coerced
                result.coercions.append(f"{name}: {value!r} -> {coerced!r} ({want})")
                value = coerced
            else:
                result.errors.append(
                    f"{name}: expected {want}, got {type(value).__name__} ({value!r})"
                )
                continue

        if "enum" in spec and value not in spec["enum"]:
            allowed = ", ".join(repr(o) for o in spec["enum"])
            result.errors.append(f"{name}: {value!r} is not one of [{allowed}]")

        if "pattern" in spec and isinstance(value, str):
            if not re.fullmatch(spec["pattern"], value):
                result.errors.append(
                    f"{name}: {value!r} does not match required format {spec['pattern']}"
                )

    result.ok = not result.errors
    return result


# ---------------------------------------------------------------------------
# Cross-interface translation
# ---------------------------------------------------------------------------
# The agenda asks for "schema design and validation across different LLM
# interfaces". Here is the honest summary of that landscape: every provider
# accepts the SAME JSON Schema and disagrees only about the envelope around it.
#
#   OpenAI     {"type": "function", "function": {name, description, parameters}}
#   Gemini     {"name", "description", "parameters"}   (inside function_declarations)
#   Anthropic  {"name", "description", "input_schema"}
#   LangChain  a StructuredTool built from name/description/args_schema
#
# The teaching point is that the envelope is trivia you can look up, while the
# schema — the types, the enums, the patterns, the descriptions — is the design
# work, and it transfers unchanged. A learner who internalises that will not be
# thrown by the next provider's SDK.
# ---------------------------------------------------------------------------
def _clean(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Drop our private `_summary` key before sending a schema to a provider."""
    return {k: v for k, v in schema.items() if not k.startswith("_")}


def to_openai(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """OpenAI chat-completions tool format."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": _clean(schema),
        },
    }


def to_gemini(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Google Gemini function-declaration format.

    Gemini's schema dialect is a subset — it has historically not supported
    `additionalProperties`, so we drop it rather than send a field that will be
    rejected. That is exactly the kind of per-provider wrinkle worth showing
    once: the design survives, the envelope needs a small adapter.
    """
    params = _clean(schema)
    params.pop("additionalProperties", None)
    return {"name": name, "description": description, "parameters": params}


def to_anthropic(name: str, description: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    """Anthropic tool format — same schema, different key name."""
    return {"name": name, "description": description, "input_schema": _clean(schema)}


def describe_for_prompt(
    name: str,
    description: str,
    schema: Dict[str, Any],
    examples: Optional[List[str]] = None,
) -> str:
    """
    Render a tool as plain text for a prompt.

    Needed for models without native tool calling, where you describe the tools
    in the system prompt and parse the reply. It also makes the point that "tool
    calling" is not magic — it is a structured output convention layered on
    ordinary text generation.
    """
    lines = [f"- {name}: {description}"]
    for prop, spec in schema.get("properties", {}).items():
        bits = [spec.get("type", "string")]
        if "enum" in spec:
            bits.append("one of " + "|".join(str(o) for o in spec["enum"]))
        if "pattern" in spec:
            bits.append(f"format {spec['pattern']}")
        required = prop in schema.get("required", [])
        bits.append("required" if required else "optional")
        desc = spec.get("description", "")
        lines.append(f"    {prop} ({', '.join(bits)}){': ' + desc if desc else ''}")
    for example in examples or []:
        lines.append(f"    e.g. {example}")
    return "\n".join(lines)

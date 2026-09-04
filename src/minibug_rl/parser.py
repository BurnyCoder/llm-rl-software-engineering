"""Global context: normalize and statically screen model-produced replacement code.

The AST policy is defense in depth only; isolation comes from the container boundary.

Sources:
- https://docs.python.org/3/library/ast.html
- https://docs.python.org/3/library/functions.html#exec
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from typing import Any

from minibug_rl.schemas import ParsedCandidate

# Match one optional Markdown fence while retaining the complete source inside it.
_FENCE = re.compile(r"\A\s*```(?:python|py)?\s*\n(?P<code>.*)\n```\s*\Z", re.DOTALL | re.IGNORECASE)
# These builtins are unnecessary for the tiny pure-function curriculum and expand attack surface.
_BANNED_CALLS = frozenset(
    {
        "breakpoint",
        "compile",
        "delattr",
        "eval",
        "exec",
        "getattr",
        "globals",
        "help",
        "input",
        "locals",
        "open",
        "setattr",
        "__import__",
    }
)


def completion_text(completion: Any) -> str:
    """Normalize plain or conversational TRL completions into one exact string."""
    # Plain completions already have the desired representation.
    if isinstance(completion, str):
        return completion
    # Conversational GRPO completions are commonly lists of assistant message mappings.
    if isinstance(completion, Sequence) and not isinstance(completion, (str, bytes)):
        # Concatenate message content without silently dropping any generated text.
        parts: list[str] = []
        for message in completion:
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                parts.append(message["content"])
        if parts:
            return "\n".join(parts)
    # A stable representation makes malformed completion diagnostics reproducible.
    return str(completion)


def _strip_optional_fence(text: str) -> str:
    """Remove exactly one surrounding Python fence while preserving inner source."""
    # A non-matching response remains untouched so syntax parsing rejects prose honestly.
    match = _FENCE.fullmatch(text)
    return (match.group("code") if match else text).strip()


def _policy_error(tree: ast.AST) -> str | None:
    """Return the first prohibited AST construct, if one exists."""
    # Walk every nested statement so an import hidden inside a function is still rejected.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return "Import statements are prohibited by the task policy."
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in _BANNED_CALLS
        ):
            return f"Call to {node.func.id!r} is prohibited by the task policy."
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return "Dunder attribute access is prohibited by the task policy."
        if isinstance(node, ast.Name) and node.id == "__builtins__":
            return "Direct __builtins__ access is prohibited by the task policy."
    return None


def parse_candidate(completion: Any, function_name: str) -> ParsedCandidate:
    """Accept one matching top-level function and reject ambiguous or unsafe output."""
    # Normalize TRL's two supported completion shapes before logging-independent parsing.
    code = _strip_optional_fence(completion_text(completion))
    try:
        # Python's own parser is the authoritative syntax validator.
        tree = ast.parse(code, mode="exec")
    except SyntaxError as error:
        return ParsedCandidate(False, code, f"Completion is not valid Python: {error.msg}")
    # One function and no top-level helpers makes the replacement contract deterministic.
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return ParsedCandidate(False, code, "Completion must contain one top-level function only.")
    # Retain the task's public function API so fixed tests invoke the intended callable.
    if tree.body[0].name != function_name:
        return ParsedCandidate(False, code, f"Completion must define function {function_name!r}.")
    # Static screening rejects unnecessary high-risk primitives before container startup.
    policy_error = _policy_error(tree)
    if policy_error is not None:
        return ParsedCandidate(False, code, policy_error, policy_violation=True)
    # Normalized valid code is the only source transferred to the sandbox.
    return ParsedCandidate(True, code)

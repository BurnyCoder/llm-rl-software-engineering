"""Global context: execute one JSON request inside the disposable sandbox container.

The host sends candidate source and call arguments, never expected answers. Candidate
stdout and stderr are captured with fixed limits so stdout remains a single JSON reply.

Sources:
- https://docs.python.org/3/library/contextlib.html#contextlib.redirect_stdout
- https://docs.python.org/3/library/functions.html#exec
- https://docs.python.org/3/library/json.html
"""

from __future__ import annotations

import builtins
import contextlib
import io
import json
import sys
from typing import Any

# Bound every protocol dimension before untrusted code is evaluated.
MAX_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_CAPTURE_CHARS = 16 * 1024
MAX_ACTUAL_BYTES = 16 * 1024
MAX_CALLS = 64
MAX_ERROR_CHARS = 2 * 1024

# Expose only ordinary pure-function tools to candidate globals. This is defense in
# depth; the Docker boundary, not a Python builtins mapping, provides isolation.
_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bin",
    "bool",
    "bytes",
    "callable",
    "chr",
    "complex",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "format",
    "frozenset",
    "hash",
    "hex",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "oct",
    "ord",
    "pow",
    "print",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
)

# Common exceptions let normal repairs validate and raise input errors explicitly.
_SAFE_EXCEPTION_NAMES = (
    "ArithmeticError",
    "AssertionError",
    "Exception",
    "IndexError",
    "KeyError",
    "LookupError",
    "OverflowError",
    "RuntimeError",
    "TypeError",
    "ValueError",
    "ZeroDivisionError",
)


class CappedTextWriter(io.TextIOBase):
    """Capture candidate text while discarding characters beyond a fixed limit."""

    def __init__(self, limit: int) -> None:
        """Initialize an empty capture with an explicit character budget."""
        super().__init__()
        self._limit = limit
        self._parts: list[str] = []
        self._length = 0
        self.truncated = False

    def writable(self) -> bool:
        """Tell ``print`` and other text APIs that this sink accepts writes."""
        return True

    def write(self, text: str) -> int:
        """Retain the in-budget prefix but report the full consumed input length."""
        if not isinstance(text, str):
            raise TypeError("captured output must be text")
        remaining = self._limit - self._length
        if remaining > 0:
            kept = text[:remaining]
            self._parts.append(kept)
            self._length += len(kept)
        if len(text) > max(remaining, 0):
            self.truncated = True
        return len(text)

    def getvalue(self) -> str:
        """Join the bounded fragments only when constructing the final response."""
        return "".join(self._parts)


def _safe_builtins() -> dict[str, Any]:
    """Build a small allow-list without exposing import, file, or eval primitives."""
    # Python makes the process builtins available through this trusted module only.
    source = vars(builtins)
    names = _SAFE_BUILTIN_NAMES + _SAFE_EXCEPTION_NAMES
    return {name: source[name] for name in names}


def _error_text(error: BaseException) -> str:
    """Return a bounded diagnostic without a traceback or process internals."""
    return str(error)[:MAX_ERROR_CHARS]


def _validate_request(request: Any) -> tuple[str, str, list[dict[str, Any]]]:
    """Validate the narrow host protocol before compiling candidate source."""
    if not isinstance(request, dict):
        raise ValueError("request must be a JSON object")
    if set(request) != {"code", "function_name", "calls"}:
        raise ValueError("request fields must be code, function_name, and calls")
    code = request["code"]
    function_name = request["function_name"]
    calls = request["calls"]
    if not isinstance(code, str) or not code:
        raise ValueError("code must be a non-empty string")
    if not isinstance(function_name, str) or not function_name.isidentifier():
        raise ValueError("function_name must be a Python identifier")
    if not isinstance(calls, list) or len(calls) > MAX_CALLS:
        raise ValueError(f"calls must be a list with at most {MAX_CALLS} items")
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"args", "kwargs"}:
            raise ValueError("each call must contain only args and kwargs")
        if not isinstance(call["args"], list) or not isinstance(call["kwargs"], dict):
            raise ValueError("args must be an array and kwargs must be an object")
        if not all(isinstance(key, str) for key in call["kwargs"]):
            raise ValueError("keyword argument names must be strings")
    return code, function_name, calls


def _json_actual(value: Any) -> Any:
    """Normalize one actual value and reject oversized or non-JSON results."""
    encoded = json.dumps(value, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_ACTUAL_BYTES:
        raise ValueError("actual value exceeds the sandbox output limit")
    return json.loads(encoded)


def _execute(request: Any) -> dict[str, Any]:
    """Compile the candidate once and invoke it for each host-provided call."""
    code, function_name, calls = _validate_request(request)
    candidate_stdout = CappedTextWriter(MAX_CAPTURE_CHARS)
    candidate_stderr = CappedTextWriter(MAX_CAPTURE_CHARS)
    namespace = {"__builtins__": _safe_builtins(), "__name__": "__candidate__"}
    with (
        contextlib.redirect_stdout(candidate_stdout),
        contextlib.redirect_stderr(candidate_stderr),
    ):
        try:
            # Compilation and execution happen only after the host's AST screening.
            compiled = compile(code, "<candidate>", "exec")
            exec(compiled, namespace)
            candidate = namespace.get(function_name)
            if not callable(candidate):
                raise TypeError(f"{function_name!r} is not callable")
            outputs = [_json_actual(candidate(*call["args"], **call["kwargs"])) for call in calls]
        except BaseException as error:
            return {
                "status": "runtime_error",
                "error_type": type(error).__name__,
                "error": _error_text(error),
                "stdout": candidate_stdout.getvalue(),
                "stderr": candidate_stderr.getvalue(),
                "output_truncated": (candidate_stdout.truncated or candidate_stderr.truncated),
            }
    return {
        "status": "success",
        "outputs": outputs,
        "stdout": candidate_stdout.getvalue(),
        "stderr": candidate_stderr.getvalue(),
        "output_truncated": candidate_stdout.truncated or candidate_stderr.truncated,
    }


def _response_bytes(response: dict[str, Any]) -> bytes:
    """Serialize one bounded response for the host-side protocol parser."""
    encoded = json.dumps(response, allow_nan=False, separators=(",", ":")).encode()
    if len(encoded) <= MAX_RESPONSE_BYTES:
        return encoded
    fallback = {
        "status": "runtime_error",
        "error_type": "OutputLimitError",
        "error": "sandbox response exceeds the output limit",
        "stdout": "",
        "stderr": "",
        "output_truncated": True,
    }
    return json.dumps(fallback, separators=(",", ":")).encode()


def main() -> None:
    """Read exactly one request from stdin and write exactly one JSON response."""
    protocol_stdout = sys.stdout.buffer
    raw_request = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw_request) > MAX_REQUEST_BYTES:
        response = {
            "status": "protocol_error",
            "error": "request exceeds the sandbox input limit",
        }
    else:
        try:
            response = _execute(json.loads(raw_request))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            response = {
                "status": "protocol_error",
                "error": _error_text(error),
            }
    protocol_stdout.write(_response_bytes(response))
    protocol_stdout.flush()


if __name__ == "__main__":
    main()

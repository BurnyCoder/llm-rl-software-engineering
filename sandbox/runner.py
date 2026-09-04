"""Global context: execute one JSON task protocol inside disposable Docker.

MiniBug calls send no expected answers; external HumanEvalFix requests carry assertion
scripts that never enter model prompts. Candidate output is captured with fixed limits
so container stdout remains a single trusted JSON reply.

Sources:
- https://docs.python.org/3.12/library/contextlib.html#contextlib.redirect_stdout
- https://docs.python.org/3.12/library/functions.html#exec
- https://docs.python.org/3.12/library/json.html
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
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
# A versioned discriminator prevents assertion scripts from entering the call protocol.
SCRIPT_PROTOCOL = "minibug.python-test-script.v1"
# The script request uses an exact schema so unknown host fields fail closed.
_SCRIPT_FIELDS = {
    "protocol",
    "language",
    "entry_point",
    "candidate_source",
    "test_setup_source",
    "test_source",
}

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


def _validate_script_request(request: Any) -> tuple[str, str, str, str]:
    """Validate the isolated Python assertion-script protocol before compilation."""
    # Only mappings with the complete versioned schema can select script execution.
    if not isinstance(request, dict) or set(request) != _SCRIPT_FIELDS:
        raise ValueError("script request fields do not match the versioned schema")
    # These discriminators prevent future languages or versions from being misrouted.
    if request["protocol"] != SCRIPT_PROTOCOL or request["language"] != "python":
        raise ValueError("unsupported script protocol or language")
    # Assign local names only after checking the discriminator fields.
    entry_point = request["entry_point"]
    candidate_source = request["candidate_source"]
    test_setup_source = request["test_setup_source"]
    test_source = request["test_source"]
    # The benchmark callable name must be safe to retrieve from a Python namespace.
    if not isinstance(entry_point, str) or not entry_point.isidentifier():
        raise ValueError("entry_point must be a Python identifier")
    # A blank repair is a candidate failure, but malformed field types are protocol faults.
    if not isinstance(candidate_source, str):
        raise ValueError("candidate_source must be a string")
    # Setup and assertions originate from one validated, immutable benchmark revision.
    if not isinstance(test_setup_source, str) or not isinstance(test_source, str):
        raise ValueError("test setup and test source must be strings")
    # An empty assertion suite cannot provide any functional-correctness evidence.
    if not test_source.strip():
        raise ValueError("test_source must be non-empty")
    # Returning four values keeps compilation trust domains visibly separate.
    return entry_point, candidate_source, test_setup_source, test_source


def _script_error(
    status: str,
    error_type: str,
    error: BaseException | str,
    stdout: CappedTextWriter,
    stderr: CappedTextWriter,
) -> dict[str, Any]:
    """Build one bounded script response without exposing a source traceback."""
    # Exception messages are useful diagnostics; source and stack frames are withheld.
    message = _error_text(error) if isinstance(error, BaseException) else error[:MAX_ERROR_CHARS]
    # Candidate output is diagnostic only and never becomes another protocol object.
    return {
        "status": status,
        "error_type": error_type,
        "error": message,
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "output_truncated": stdout.truncated or stderr.truncated,
    }


def _execute_script(request: Any) -> dict[str, Any]:
    """Execute setup, candidate, and hidden assertions in isolated trust phases."""
    # Schema validation runs before any source is parsed or evaluated.
    entry_point, candidate_source, setup_source, test_source = _validate_script_request(request)
    # Each source receives a distinct synthetic filename for bounded error classification.
    try:
        setup_code = compile(setup_source, "<benchmark-setup>", "exec")
        test_code = compile(test_source, "<benchmark-tests>", "exec")
    except (SyntaxError, ValueError) as error:
        # Invalid pinned tests invalidate the experiment rather than penalizing a model.
        empty_stdout = CappedTextWriter(MAX_CAPTURE_CHARS)
        empty_stderr = CappedTextWriter(MAX_CAPTURE_CHARS)
        return _script_error(
            "infrastructure_error",
            type(error).__name__,
            error,
            empty_stdout,
            empty_stderr,
        )
    try:
        candidate_code = compile(candidate_source, "<candidate>", "exec")
    except (SyntaxError, ValueError) as error:
        # Invalid generated Python is a normal candidate outcome in code evaluation.
        empty_stdout = CappedTextWriter(MAX_CAPTURE_CHARS)
        empty_stderr = CappedTextWriter(MAX_CAPTURE_CHARS)
        return _script_error(
            "failed",
            type(error).__name__,
            error,
            empty_stdout,
            empty_stderr,
        )
    # Full Python semantics are benchmark-compatible inside Docker; Docker is the boundary.
    namespace = {"__name__": "__candidate__"}
    # Capture ordinary writes so they cannot corrupt the runner's single JSON stdout reply.
    candidate_stdout = CappedTextWriter(MAX_CAPTURE_CHARS)
    candidate_stderr = CappedTextWriter(MAX_CAPTURE_CHARS)
    with (
        contextlib.redirect_stdout(candidate_stdout),
        contextlib.redirect_stderr(candidate_stderr),
    ):
        try:
            # Trusted imports and setup run first, matching benchmark program assembly.
            exec(setup_code, namespace)
        except BaseException as error:
            # A broken dependency or pinned setup means the environment is invalid.
            return _script_error(
                "infrastructure_error",
                type(error).__name__,
                error,
                candidate_stdout,
                candidate_stderr,
            )
        try:
            # Candidate execution happens only inside the resource-limited container.
            exec(candidate_code, namespace)
            # The declared entry point must remain callable before assertions start.
            if not callable(namespace.get(entry_point)):
                raise TypeError(f"{entry_point!r} is not callable")
        except BaseException as error:
            # Definition-time and entry-point errors are ordinary model failures.
            return _script_error(
                "failed",
                type(error).__name__,
                error,
                candidate_stdout,
                candidate_stderr,
            )
        try:
            # Canonical assertions run last in the candidate namespace as the harness does.
            exec(test_code, namespace)
        except BaseException as error:
            # Assertion and candidate-call exceptions both mean functional failure.
            return _script_error(
                "failed",
                type(error).__name__,
                error,
                candidate_stdout,
                candidate_stderr,
            )
    # Reaching this point proves every canonical assertion completed successfully.
    return {
        "status": "passed",
        "stdout": candidate_stdout.getvalue(),
        "stderr": candidate_stderr.getvalue(),
        "output_truncated": candidate_stdout.truncated or candidate_stderr.truncated,
    }


def _dispatch(request: Any) -> dict[str, Any]:
    """Route one validated protocol discriminator without executing host-supplied code."""
    # Only the exact script protocol selects assertion execution; legacy calls omit it.
    if isinstance(request, dict) and request.get("protocol") == SCRIPT_PROTOCOL:
        return _execute_script(request)
    # Existing MiniBug function calls retain their host-comparison behavior unchanged.
    return _execute(request)


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
            # JSON parsing precedes narrow dispatch; both protocols share size bounds.
            response = _dispatch(json.loads(raw_request))
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
            response = {
                "status": "protocol_error",
                "error": _error_text(error),
            }
    protocol_stdout.write(_response_bytes(response))
    protocol_stdout.flush()


if __name__ == "__main__":
    main()

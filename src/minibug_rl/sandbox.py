"""Global context: broker model-produced Python through disposable Docker.

MiniBug expected values stay in this trusted host process and only call inputs cross
the boundary. External evaluation sends hidden assertion scripts to the container but
never to model prompts. AST checks and reduced builtins are defense in depth; Docker
is the execution boundary.

Sources:
- https://docs.docker.com/reference/cli/docker/container/run/
- https://docs.docker.com/engine/network/drivers/none/
- https://docs.python.org/3.12/library/subprocess.html#subprocess.Popen.communicate
- https://docs.python.org/3.12/library/ast.html
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from minibug_rl.parser import parse_candidate
from minibug_rl.schemas import SandboxExecution, TestCase

# Keep the locally built image name stable across training and tests.
DEFAULT_IMAGE = "minibug-rl-sandbox:local"
# Three seconds bounds Docker startup plus candidate execution on the host clock.
DEFAULT_TIMEOUT_SECONDS = 3.0
# Apply the image's protocol limit after pipe collection and before JSON parsing.
MAX_PROTOCOL_BYTES = 512 * 1024
# Match the runner's fixed standard-input budget before starting a container.
MAX_REQUEST_BYTES = 256 * 1024
# Returned diagnostics are useful for logs but must remain bounded independently.
MAX_DIAGNOSTIC_CHARS = 16 * 1024
# Cleanup calls must never replace one candidate timeout with another long wait.
CLEANUP_TIMEOUT_SECONDS = 2.0


@dataclass(frozen=True, slots=True)
class CaseResult:
    """Record one host-side comparison without sending its expected value away."""

    actual: Any
    passed: bool


@dataclass(frozen=True, slots=True)
class SandboxResult:
    """Return execution evidence and host comparisons to callers and reports."""

    status: str
    cases: tuple[CaseResult, ...] = ()
    error: str | None = None
    duration_seconds: float = 0.0

    @property
    def passed_count(self) -> int:
        """Count correct calls after all comparisons have occurred on the host."""
        return sum(case.passed for case in self.cases)

    @property
    def all_passed(self) -> bool:
        """Require at least one case and require every case to match."""
        return bool(self.cases) and self.passed_count == len(self.cases)


@dataclass(frozen=True, slots=True)
class JsonContainerExecution:
    """Carry one generic Docker JSON exchange without assigning reward meaning."""

    # A successful transport contains a validated runner response for its caller.
    status: str
    # The response remains protocol-specific until a narrow executor interprets it.
    response: dict[str, Any] | None = None
    # Infrastructure and timeout diagnostics never become model-visible feedback.
    error: str | None = None
    # Wall time supports timeout audits for both local and external evaluations.
    duration_seconds: float = 0.0


def build_docker_command(
    container_name: str,
    image: str = DEFAULT_IMAGE,
    docker_binary: str = "docker",
) -> list[str]:
    """Build a Docker command with no host bind/volume mounts and one tmpfs."""
    # Docker documents each flag on the container-run reference linked above.
    return [
        docker_binary,
        "run",
        "--rm",
        # Keep stdin attached so the request crosses without a host bind or volume.
        "--interactive",
        "--name",
        container_name,
        "--pull",
        "never",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges=true",
        "--pids-limit",
        "32",
        "--memory",
        "128m",
        "--memory-swap",
        "128m",
        "--cpus",
        "0.5",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=16m",
        "--user",
        "65532:65532",
        image,
    ]


def _bounded_text(raw: bytes) -> str:
    """Decode an external diagnostic and retain only its fixed prefix."""
    return raw.decode("utf-8", errors="replace")[:MAX_DIAGNOSTIC_CHARS]


def _container_name() -> str:
    """Generate a collision-resistant name that can be targeted during cleanup."""
    return f"minibug-{uuid.uuid4().hex}"


def _call_request(
    candidate_code: str,
    function_name: str,
    calls: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build source-and-input data while rejecting answer-bearing call objects."""
    clean_calls: list[dict[str, Any]] = []
    for call in calls:
        # An exact key set makes accidental hidden-answer transfer fail closed.
        if set(call) != {"args", "kwargs"}:
            raise ValueError("sandbox calls must contain only args and kwargs")
        args = call["args"]
        kwargs = call["kwargs"]
        if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
            raise ValueError("sandbox call args must be a JSON sequence")
        if not isinstance(kwargs, Mapping):
            raise ValueError("sandbox call kwargs must be a JSON object")
        if not all(isinstance(key, str) for key in kwargs):
            raise ValueError("sandbox keyword argument names must be strings")
        clean_calls.append({"args": list(args), "kwargs": dict(kwargs)})
    # This object deliberately has no field in which an expected answer can travel.
    return {
        "code": candidate_code,
        "function_name": function_name,
        "calls": clean_calls,
    }


def _cleanup_container(docker_binary: str, container_name: str) -> None:
    """Best-effort kill and removal after the host deadline expires."""
    # A timed-out ``communicate`` does not kill its child; Python's docs require
    # explicit cleanup, and the named container must be stopped before the CLI.
    for action in (["kill"], ["rm", "--force"]):
        try:
            subprocess.run(
                [docker_binary, *action, container_name],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=CLEANUP_TIMEOUT_SECONDS,
            )
        except (OSError, subprocess.SubprocessError):
            # Cleanup is best-effort because the daemon may already have honored --rm.
            continue


def _parse_response(stdout: bytes) -> dict[str, Any]:
    """Validate the collected response size and require one JSON status object."""
    if len(stdout) > MAX_PROTOCOL_BYTES:
        raise ValueError("sandbox response exceeds the host output limit")
    response = json.loads(stdout)
    if not isinstance(response, dict) or not isinstance(response.get("status"), str):
        raise ValueError("sandbox response is not a status object")
    return response


def run_json_container(
    request: Mapping[str, Any],
    *,
    image: str = DEFAULT_IMAGE,
    docker_binary: str = "docker",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> JsonContainerExecution:
    """Send a pre-bounded request and post-validate the collected runner response."""
    # A monotonic clock cannot move backward if the system time changes during a run.
    started = time.monotonic()
    try:
        # ``allow_nan=False`` keeps the protocol inside interoperable JSON semantics.
        request_bytes = json.dumps(
            dict(request),
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        # Bad host data is an infrastructure error because no candidate ran.
        return JsonContainerExecution(
            status="infrastructure_error",
            error=f"Invalid sandbox request: {error}",
            duration_seconds=time.monotonic() - started,
        )
    # Reject oversize data on the host instead of relying on a truncated container read.
    if len(request_bytes) > MAX_REQUEST_BYTES:
        return JsonContainerExecution(
            status="infrastructure_error",
            error="Sandbox request exceeds the input limit.",
            duration_seconds=time.monotonic() - started,
        )
    # A unique name lets timeout cleanup target exactly this disposable container.
    container_name = _container_name()
    # Both task protocols share one inspectable security policy and immutable image.
    command = build_docker_command(
        container_name,
        image=image,
        docker_binary=docker_binary,
    )
    try:
        # Argument-vector execution prevents a shell from interpreting supplied source.
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except OSError as error:
        # A missing client or daemon is not evidence about the generated candidate.
        return JsonContainerExecution(
            status="infrastructure_error",
            error=f"Could not start Docker: {error}",
            duration_seconds=time.monotonic() - started,
        )
    try:
        # ``communicate`` avoids pipe deadlocks but buffers stdout before our size check.
        stdout, stderr = process.communicate(
            input=request_bytes,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired:
        # Stop both the named container and the local client after the host deadline.
        _cleanup_container(docker_binary, container_name)
        # Docker cleanup may already have caused the client process to exit.
        with contextlib.suppress(ProcessLookupError):
            process.kill()
        # The second call reaps the child; only the retained decoded diagnostic is bounded.
        _stdout, stderr = process.communicate()
        return JsonContainerExecution(
            status="timeout",
            error=(
                f"Candidate exceeded the {timeout_seconds:g}s host deadline. "
                f"Docker diagnostic: {_bounded_text(stderr)}"
            ),
            duration_seconds=time.monotonic() - started,
        )
    # An abnormal Docker client exit means no authenticated runner object is available.
    if process.returncode != 0:
        return JsonContainerExecution(
            status="infrastructure_error",
            error=f"Docker exited with code {process.returncode}: {_bounded_text(stderr)}",
            duration_seconds=time.monotonic() - started,
        )
    try:
        # The parser applies the post-collection limit and admits one status object.
        response = _parse_response(stdout)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
        return JsonContainerExecution(
            status="infrastructure_error",
            error=(
                f"Invalid sandbox response: {error}. Docker diagnostic: {_bounded_text(stderr)}"
            ),
            duration_seconds=time.monotonic() - started,
        )
    # Protocol-specific callers now decide whether the response means pass or failure.
    return JsonContainerExecution(
        status="success",
        response=response,
        duration_seconds=time.monotonic() - started,
    )


@dataclass(frozen=True, slots=True)
class DockerSandbox:
    """Reject invalid text host-side and isolate accepted candidates in fresh containers."""

    image: str = DEFAULT_IMAGE
    docker_binary: str = "docker"
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def execute(
        self,
        candidate_code: str,
        function_name: str,
        calls: list[dict[str, Any]],
    ) -> SandboxExecution:
        """Return actual values only, matching the shared reward executor protocol."""
        started = time.monotonic()
        parsed = parse_candidate(candidate_code, function_name)
        if not parsed.valid:
            status = "policy_rejected" if parsed.policy_violation else "invalid_candidate"
            return SandboxExecution(
                status=status,
                error=parsed.error,
                duration_seconds=time.monotonic() - started,
            )
        try:
            # The call-specific validator prevents hidden expected values from crossing.
            request = _call_request(parsed.code, function_name, calls)
        except (TypeError, ValueError) as error:
            return SandboxExecution(
                status="infrastructure_error",
                error=f"Invalid sandbox request: {error}",
                duration_seconds=time.monotonic() - started,
            )
        # Shared transport applies the exact same Docker policy as external tests.
        transport = run_json_container(
            request,
            image=self.image,
            docker_binary=self.docker_binary,
            timeout_seconds=self.timeout_seconds,
        )
        # Timeout and infrastructure states carry no candidate outputs to interpret.
        if transport.status != "success" or transport.response is None:
            return SandboxExecution(
                status=transport.status,
                error=transport.error,
                duration_seconds=time.monotonic() - started,
            )
        # The ordinary runner protocol is interpreted only after transport succeeds.
        response = transport.response
        status = response["status"]
        if status != "success":
            detail = str(response.get("error", "candidate execution failed"))
            error_type = str(response.get("error_type", "SandboxError"))
            return SandboxExecution(
                status="runtime_error" if status == "runtime_error" else "infrastructure_error",
                error=f"{error_type}: {detail}"[:MAX_DIAGNOSTIC_CHARS],
                duration_seconds=time.monotonic() - started,
            )
        outputs = response.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != len(calls):
            return SandboxExecution(
                status="infrastructure_error",
                error="Sandbox returned a different number of outputs than inputs.",
                duration_seconds=time.monotonic() - started,
            )
        return SandboxExecution(
            status="success",
            outputs=tuple(outputs),
            duration_seconds=time.monotonic() - started,
        )


def _json_equal(actual: Any, expected: Any) -> bool:
    """Compare JSON values without Python's surprising ``True == 1`` coercion."""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    return bool(actual == expected)


def _test_case(case: TestCase | Mapping[str, Any]) -> TestCase:
    """Normalize one public API mapping into the shared immutable case type."""
    if isinstance(case, TestCase):
        args, kwargs, expected = case.args, case.kwargs, case.expected
    else:
        if set(case) != {"args", "kwargs", "expected"}:
            raise ValueError("test calls must contain args, kwargs, and expected")
        args, kwargs, expected = case["args"], case["kwargs"], case["expected"]
    if not isinstance(args, Sequence) or isinstance(args, (str, bytes)):
        raise ValueError("test call args must be a JSON sequence")
    if not isinstance(kwargs, Mapping):
        raise ValueError("test call kwargs must be a JSON object")
    if not all(isinstance(key, str) for key in kwargs):
        raise ValueError("test keyword argument names must be strings")
    # Round-trip the expected value so tuples and other accepted inputs use JSON form.
    normalized_expected = json.loads(json.dumps(expected, allow_nan=False))
    return TestCase(tuple(args), dict(kwargs), normalized_expected)


def run_candidate(
    candidate_code: str,
    function_name: str,
    test_calls: Sequence[TestCase | Mapping[str, Any]],
    *,
    image: str = DEFAULT_IMAGE,
    docker_binary: str = "docker",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> SandboxResult:
    """Run a candidate and compare actuals with hidden expected values on the host."""
    started = time.monotonic()
    try:
        cases = tuple(_test_case(case) for case in test_calls)
    except (TypeError, ValueError) as error:
        return SandboxResult(
            status="infrastructure_error",
            error=f"Invalid host test case: {error}",
            duration_seconds=time.monotonic() - started,
        )
    executor = DockerSandbox(image, docker_binary, timeout_seconds)
    execution = executor.execute(
        candidate_code,
        function_name,
        [case.sandbox_call() for case in cases],
    )
    if execution.status != "success":
        return SandboxResult(
            status=execution.status,
            error=execution.error,
            duration_seconds=execution.duration_seconds,
        )
    compared = tuple(
        CaseResult(actual, _json_equal(actual, case.expected))
        for actual, case in zip(execution.outputs, cases, strict=True)
    )
    return SandboxResult(
        status="success",
        cases=compared,
        duration_seconds=execution.duration_seconds,
    )

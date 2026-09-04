"""Global context: execute HumanEvalFix assertions only inside hardened Docker.

The adapter supplies a typed candidate-and-test payload, while this module assigns
evaluation meaning to the shared JSON container transport. No benchmark or generated
Python is compiled, imported, or executed by the host process.

Sources:
- https://docs.docker.com/reference/cli/docker/container/run/
- https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
"""

from __future__ import annotations

# Immutable records prevent callers from changing evidence after scoring.
from dataclasses import dataclass

# The data adapter owns the exact, versioned assertion-script request schema.
from minibug_rl.external_eval import PythonTestScriptPayload

# Reusing this transport gives local and external scoring one Docker security policy.
from minibug_rl.sandbox import (
    DEFAULT_IMAGE,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_DIAGNOSTIC_CHARS,
    run_json_container,
)


@dataclass(frozen=True, slots=True)
class PythonTestExecution:
    """Expose one external candidate outcome without conflating broken infrastructure."""

    # Stable states are passed, failed, timeout, and infrastructure_error.
    status: str
    # The runner reports exception classes for audit grouping, never for prompting.
    error_type: str | None = None
    # A bounded message aids diagnosis without returning a source traceback.
    error: str | None = None
    # Monotonic wall time supports timeout and performance reports.
    duration_seconds: float = 0.0

    @property
    def passed(self) -> bool:
        """Return the boolean used by pass-rate and pass@k calculations."""
        # Only a completed assertion suite is positive functional evidence.
        return self.status == "passed"


def _response_text(value: object, fallback: str) -> str:
    """Normalize one runner diagnostic while retaining a fixed maximum prefix."""
    # Runner fields should be strings; a fallback makes malformed replies auditable.
    text = value if isinstance(value, str) else fallback
    # Bound diagnostics separately from the already bounded complete JSON response.
    return text[:MAX_DIAGNOSTIC_CHARS]


@dataclass(frozen=True, slots=True)
class DockerPythonTestSandbox:
    """Run one candidate plus hidden Python assertion script in a fresh container."""

    # The same locally built immutable image supports both MiniBug and external tests.
    image: str = DEFAULT_IMAGE
    # An explicit executable path keeps tests and non-default Docker installs possible.
    docker_binary: str = "docker"
    # The host deadline stops infinite loops even if Python-level guards are bypassed.
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS

    def execute(
        self,
        payload: PythonTestScriptPayload,
        candidate_source: str,
    ) -> PythonTestExecution:
        """Return functional evidence without executing either source on the host."""
        try:
            # The typed adapter keeps hidden tests separate from model-visible context.
            request = payload.build_request(candidate_source)
        except ValueError as error:
            # Empty generated code is a model outcome, not a Docker infrastructure fault.
            return PythonTestExecution(
                status="failed",
                error_type="InvalidCandidate",
                error=str(error)[:MAX_DIAGNOSTIC_CHARS],
            )
        # The generic transport performs serialization, isolation, timeout, and cleanup.
        transport = run_json_container(
            request,
            image=self.image,
            docker_binary=self.docker_binary,
            timeout_seconds=self.timeout_seconds,
        )
        # A host timeout remains distinct from both functional and infrastructure failure.
        if transport.status == "timeout":
            return PythonTestExecution(
                status="timeout",
                error_type="Timeout",
                error=transport.error,
                duration_seconds=transport.duration_seconds,
            )
        # Startup, daemon, serialization, exit, and reply errors invalidate evaluation.
        if transport.status != "success" or transport.response is None:
            return PythonTestExecution(
                status="infrastructure_error",
                error_type="ContainerInfrastructureError",
                error=transport.error,
                duration_seconds=transport.duration_seconds,
            )
        # The image validates and produces this status before any host interpretation.
        response = transport.response
        status = response["status"]
        # Completed assertions provide the only passing external-evaluation result.
        if status == "passed":
            return PythonTestExecution(
                status="passed",
                duration_seconds=transport.duration_seconds,
            )
        # Candidate compile, definition, runtime, and assertion errors are model outcomes.
        if status == "failed":
            return PythonTestExecution(
                status="failed",
                error_type=_response_text(response.get("error_type"), "CandidateError"),
                error=_response_text(response.get("error"), "candidate test failed"),
                duration_seconds=transport.duration_seconds,
            )
        # Trusted setup/test syntax and protocol failures invalidate the score record.
        return PythonTestExecution(
            status="infrastructure_error",
            error_type=_response_text(response.get("error_type"), "RunnerProtocolError"),
            error=_response_text(response.get("error"), f"unknown runner status: {status}"),
            duration_seconds=transport.duration_seconds,
        )


def run_python_test_script(
    payload: PythonTestScriptPayload,
    candidate_source: str,
    *,
    image: str = DEFAULT_IMAGE,
    docker_binary: str = "docker",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> PythonTestExecution:
    """Offer a small functional wrapper for external-evaluation loops."""
    # The class remains injectable in tests while this function is convenient in phases.
    executor = DockerPythonTestSandbox(image, docker_binary, timeout_seconds)
    # Delegation keeps all result mapping in the independently testable executor method.
    return executor.execute(payload, candidate_source)

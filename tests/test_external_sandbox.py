"""Global context: specify secure HumanEvalFix assertion-script execution.

These tests keep all generated and benchmark Python behind the Docker process
boundary while proving that scoring distinguishes candidate outcomes from broken
infrastructure.

Sources:
- https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate
- https://docs.docker.com/reference/cli/docker/container/run/
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
"""

from __future__ import annotations

# Standard-library tools construct protocol replies and control opt-in Docker tests.
import builtins
import json
import os
import subprocess
from typing import Any

# Pytest supplies monkeypatching and an explicit marker for local Docker execution.
import pytest

# The module import lets tests intercept the one shared subprocess boundary.
import minibug_rl.sandbox as sandbox_module

# The payload is the only route by which hidden benchmark assertions reach Docker.
from minibug_rl.external_eval import PythonTestScriptPayload

# The executor under test must never evaluate either source string in this process.
from minibug_rl.external_sandbox import DockerPythonTestSandbox


class FakeProcess:
    """Mimic the byte-level Docker client without executing supplied Python."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        *,
        times_out: bool = False,
        returncode: int = 0,
    ) -> None:
        """Configure a JSON reply, host timeout, or Docker client failure."""
        # A passing script reply is the least surprising default for focused tests.
        self.response = response or {"status": "passed"}
        # A one-shot timeout exercises the documented kill-and-communicate lifecycle.
        self.times_out = times_out
        # The transport reads Docker's exit status before trusting standard output.
        self.returncode = returncode
        # Capturing input proves the model and tests cross only the container boundary.
        self.input: bytes | None = None
        # The timeout path explicitly kills the client after stopping the container.
        self.killed = False
        # Only the first communicate call raises; the second reaps the dead client.
        self.communicate_count = 0

    def communicate(
        self,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        """Record stdin and return one deterministic runner response."""
        # The fake does not wait, but accepting timeout matches subprocess.Popen.
        del timeout
        # The count determines whether this is the deadline or reap operation.
        self.communicate_count += 1
        # Docker receives protocol bytes only on the initial communication call.
        if input is not None:
            self.input = input
        # Python documents TimeoutExpired as the signal for explicit cleanup.
        if self.times_out and self.communicate_count == 1:
            raise subprocess.TimeoutExpired("docker", 1)
        # A single JSON object is the trusted runner's complete stdout protocol.
        return json.dumps(self.response).encode(), b""

    def kill(self) -> None:
        """Record cleanup of the local Docker client process."""
        # The assertion checks this flag after a simulated host deadline.
        self.killed = True


def _payload(test_source: str) -> PythonTestScriptPayload:
    """Build one minimal assertion suite using the production typed contract."""
    # The setup imports a harmless standard module to prove setup runs in-container.
    return PythonTestScriptPayload(
        entry_point="repair",
        test_setup_source="import math",
        test_source=test_source,
    )


def _install_process(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
) -> None:
    """Replace Docker startup while leaving production request construction active."""
    # Patching the shared subprocess module intercepts the sandbox's only executor.
    monkeypatch.setattr(sandbox_module.subprocess, "Popen", lambda *_args, **_kwargs: process)


def test_pass_and_assertion_failure_are_candidate_outcomes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Map runner assertions to model outcomes rather than infrastructure faults."""
    # First prove the untrusted runner's success response reaches the typed API.
    passing_process = FakeProcess({"status": "passed"})
    _install_process(monkeypatch, passing_process)
    sandbox = DockerPythonTestSandbox()
    passed = sandbox.execute(
        _payload("assert repair(2) == 4"),
        "def repair(value):\n    return value * 2",
    )

    # A benchmark AssertionError is evidence about the candidate, not Docker.
    failing_process = FakeProcess(
        {
            "status": "failed",
            "error_type": "AssertionError",
            "error": "benchmark assertion failed",
        }
    )
    _install_process(monkeypatch, failing_process)
    failed = sandbox.execute(
        _payload("assert repair(2) == 5"),
        "def repair(value):\n    return value * 2",
    )

    # Callers can compute pass@k directly from the stable candidate result states.
    assert passed.status == "passed"
    assert passed.passed is True
    assert failed.status == "failed"
    assert failed.passed is False
    assert failed.error_type == "AssertionError"


def test_hidden_script_is_sent_to_docker_without_host_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove scoring serializes hidden tests but never calls host-side exec."""
    # A unique marker makes accidental prompt or transport omissions easy to detect.
    hidden = "HIDDEN_EXTERNAL_ASSERTION_8d4c"
    process = FakeProcess()
    _install_process(monkeypatch, process)

    # Any accidental host evaluation would call this patched language primitive.
    def forbidden_host_exec(*_args: Any, **_kwargs: Any) -> None:
        """Fail immediately if benchmark or generated code leaves Docker."""
        raise AssertionError("host-side exec is forbidden")

    # The executor should only serialize strings and launch the isolated process.
    monkeypatch.setattr(builtins, "exec", forbidden_host_exec)

    # The production path has no host compilation or execution hook to patch around.
    result = DockerPythonTestSandbox().execute(
        _payload(f"assert repair(3) == 3  # {hidden}"),
        "def repair(value):\n    return value",
    )

    # The sole destination for the hidden assertion is Docker's standard input.
    assert result.status == "passed"
    assert process.input is not None
    request = json.loads(process.input)
    assert request["test_source"].endswith(hidden)
    assert request["candidate_source"] == "def repair(value):\n    return value"


def test_timeout_is_distinct_from_candidate_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserve the host deadline as a separate resource-exhaustion outcome."""
    # A hung candidate makes the Docker client miss its communication deadline.
    process = FakeProcess(times_out=True)
    _install_process(monkeypatch, process)
    # Cleanup commands are harmlessly recorded instead of contacting a daemon.
    cleanup: list[list[str]] = []
    monkeypatch.setattr(
        sandbox_module.subprocess,
        "run",
        lambda command, **_kwargs: cleanup.append(command),
    )

    # A one-second limit keeps the expected error message deterministic.
    result = DockerPythonTestSandbox(timeout_seconds=1).execute(
        _payload("assert repair(1) == 1"),
        "def repair(value):\n    while True:\n        pass",
    )

    # Timeout remains neither a passing repair nor an infrastructure fault.
    assert result.status == "timeout"
    assert result.passed is False
    assert process.killed is True
    assert process.communicate_count == 2
    assert [command[1] for command in cleanup] == ["kill", "rm"]


@pytest.mark.parametrize(
    "response",
    [
        {"status": "protocol_error", "error": "bad request"},
        {"status": "infrastructure_error", "error": "invalid benchmark script"},
        {"status": "unknown"},
    ],
)
def test_runner_or_protocol_fault_is_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
    response: dict[str, Any],
) -> None:
    """Never count a malformed runner exchange as a model failure."""
    # Each response represents a trusted runner problem rather than failed assertions.
    _install_process(monkeypatch, FakeProcess(response))

    # The public result collapses runner-specific faults into one auditable state.
    result = DockerPythonTestSandbox().execute(
        _payload("assert repair(1) == 1"),
        "def repair(value):\n    return value",
    )

    # Evaluation code can exclude this record instead of lowering model accuracy.
    assert result.status == "infrastructure_error"
    assert result.passed is False


def test_docker_start_failure_is_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Surface a missing daemon separately from executable repair evidence."""

    # OSError is the documented startup failure raised by subprocess.Popen.
    def missing_docker(*_args: Any, **_kwargs: Any) -> None:
        """Raise the same error shape as an unavailable Docker executable."""
        raise OSError("docker unavailable")

    # No candidate or benchmark code runs when the process cannot start.
    monkeypatch.setattr(sandbox_module.subprocess, "Popen", missing_docker)
    result = DockerPythonTestSandbox().execute(
        _payload("assert repair(1) == 1"),
        "def repair(value):\n    return value",
    )

    # Infrastructure failure can stop the evaluation instead of biasing its score.
    assert result.status == "infrastructure_error"
    assert "Could not start Docker" in (result.error or "")


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_script_pass_failure_timeout_and_invalid_suite() -> None:
    """Exercise both trust domains through the actual hardened image."""
    # One executor instance keeps image and limit choices identical for all outcomes.
    sandbox = DockerPythonTestSandbox(timeout_seconds=3)
    # Passing and failing suites differ only in their expected assertion value.
    passed = sandbox.execute(
        _payload("assert repair(2) == 4"),
        "def repair(value):\n    return value * 2",
    )
    failed = sandbox.execute(
        _payload("assert repair(2) == 5"),
        "def repair(value):\n    return value * 2",
    )
    # Infinite candidate execution must be terminated by the host container deadline.
    timeout = DockerPythonTestSandbox(timeout_seconds=1).execute(
        _payload("assert repair(1) == 1"),
        "def repair(value):\n    while True:\n        pass",
    )
    # Malformed trusted assertions indicate corrupt benchmark infrastructure.
    invalid_suite = sandbox.execute(
        _payload("assert repair("),
        "def repair(value):\n    return value",
    )
    # The pinned HumanEvalPack harness imports NumPy for every Python candidate.
    numpy_passed = sandbox.execute(
        _payload("assert repair([1, 2, 3]) == 6"),
        "import numpy as np\n\ndef repair(values):\n    return int(np.sum(values))",
    )

    # Real outcomes prove the unit-level byte protocol matches the image implementation.
    assert passed.status == "passed"
    assert failed.status == "failed"
    assert failed.error_type == "AssertionError"
    assert timeout.status == "timeout"
    assert invalid_suite.status == "infrastructure_error"
    assert numpy_passed.status == "passed"

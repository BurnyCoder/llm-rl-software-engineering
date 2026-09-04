"""Global context: specify secure HumanEvalFix assertion-script execution.

These tests keep all generated and benchmark Python behind the Docker process
boundary while proving that scoring distinguishes candidate outcomes from broken
infrastructure.

Sources:
- https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate
- https://docs.docker.com/reference/cli/docker/container/run/
- https://docs.docker.com/engine/network/drivers/none/
- https://docs.docker.com/engine/containers/resource_constraints/
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


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_bounds_candidate_output_flooding() -> None:
    """Keep a large candidate print inside the bounded single-object protocol."""
    # Build the versioned script request through the same typed adapter as evaluation.
    request = _payload("assert repair() == 7").build_request(
        "def repair():\n    print('x' * 100_000)\n    return 7"
    )
    # The shared transport uses the production image, limits, and no-mount command.
    execution = sandbox_module.run_json_container(request, timeout_seconds=3)

    # The runner must preserve a valid result instead of letting stdout corrupt JSON.
    assert execution.status == "success"
    assert execution.response is not None
    assert execution.response["status"] == "passed"
    # The capture flag and exact retained prefix prove the flood was actually bounded.
    assert execution.response["output_truncated"] is True
    assert execution.response["stdout"] == "x" * (16 * 1024)


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_root_filesystem_is_read_only() -> None:
    """Observe the read-only root mount and reject a real root-file write."""
    # Full Python is intentional here: Docker, not the benchmark parser, is the boundary.
    candidate = (
        "import os\n\n"
        "def repair():\n"
        "    root_is_read_only = bool(os.statvfs('/').f_flag & os.ST_RDONLY)\n"
        "    try:\n"
        "        with open('/minibug-write-probe', 'w') as handle:\n"
        "            handle.write('unexpected')\n"
        "    except OSError:\n"
        "        return root_is_read_only\n"
        "    return False"
    )
    # A passing assertion proves both the mount flag and attempted write denial in Docker.
    result = DockerPythonTestSandbox(timeout_seconds=3).execute(
        _payload("assert repair() is True"),
        candidate,
    )

    # A writable root would make the candidate return false and fail this assertion.
    assert result.status == "passed"


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_has_only_loopback_and_cannot_connect_out() -> None:
    """Verify the none network exposes no external interface or usable route."""
    # The reserved numeric address avoids DNS and cannot identify any operator service.
    candidate = (
        "import socket\n\n"
        "def repair():\n"
        "    interfaces = {name for _, name in socket.if_nameindex()}\n"
        "    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)\n"
        "    probe.settimeout(0.25)\n"
        "    try:\n"
        "        connection_result = probe.connect_ex(('192.0.2.1', 9))\n"
        "    finally:\n"
        "        probe.close()\n"
        "    return interfaces == {'lo'} and connection_result != 0"
    )
    # The production command supplies --network none before executing this probe.
    result = DockerPythonTestSandbox(timeout_seconds=3).execute(
        _payload("assert repair() is True"),
        candidate,
    )

    # Success requires both loopback-only enumeration and a failed outbound connection.
    assert result.status == "passed"


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_runs_as_configured_non_root_identity() -> None:
    """Read the effective and real numeric identities from inside the container."""
    # Numeric IDs avoid depending on an optional passwd entry in the slim image.
    candidate = (
        "import os\n\n"
        "def repair():\n"
        "    return os.getuid(), os.getgid(), os.geteuid(), os.getegid()"
    )
    # The assertion executes only in the container and matches build/run user 65532.
    result = DockerPythonTestSandbox(timeout_seconds=3).execute(
        _payload("assert repair() == (65532, 65532, 65532, 65532)"),
        candidate,
    )

    # Any accidental root or group regression fails the in-container assertion.
    assert result.status == "passed"


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_pid_limit_stops_and_cleans_process_exhaustion() -> None:
    """Reach the cgroup PID ceiling linearly and reap every created child."""
    # Each forked child sleeps without forking again, avoiding an exponential fork bomb.
    candidate = (
        "import os\n"
        "import signal\n"
        "import time\n\n"
        "def repair():\n"
        "    children = []\n"
        "    limit_reached = False\n"
        "    try:\n"
        "        for _ in range(64):\n"
        "            try:\n"
        "                child = os.fork()\n"
        "            except OSError:\n"
        "                limit_reached = True\n"
        "                break\n"
        "            if child == 0:\n"
        "                time.sleep(2)\n"
        "                os._exit(0)\n"
        "            children.append(child)\n"
        "    finally:\n"
        "        for child in children:\n"
        "            try:\n"
        "                os.kill(child, signal.SIGKILL)\n"
        "            except ProcessLookupError:\n"
        "                pass\n"
        "        for child in children:\n"
        "            try:\n"
        "                os.waitpid(child, 0)\n"
        "            except ChildProcessError:\n"
        "                pass\n"
        "    return limit_reached and len(children) < 64"
    )
    # Five seconds allows deterministic reaping while still bounding every failure path.
    result = DockerPythonTestSandbox(timeout_seconds=5).execute(
        _payload("assert repair() is True"),
        candidate,
    )

    # Passing proves fork failed before 64 children and cleanup returned normally.
    assert result.status == "passed"


@pytest.mark.docker
@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_memory_limit_and_tmpfs_exhaustion_are_bounded() -> None:
    """Confirm the memory cgroup and safely exhaust only the 16 MiB tmpfs."""
    # Reading either cgroup generation keeps the probe portable across Docker hosts.
    candidate = (
        "import errno\n"
        "import os\n\n"
        "def repair():\n"
        "    memory_paths = (\n"
        "        '/sys/fs/cgroup/memory.max',\n"
        "        '/sys/fs/cgroup/memory/memory.limit_in_bytes',\n"
        "    )\n"
        "    memory_limit = None\n"
        "    for path in memory_paths:\n"
        "        if os.path.exists(path):\n"
        "            with open(path, encoding='utf-8') as handle:\n"
        "                value = handle.read().strip()\n"
        "            if value.isdigit():\n"
        "                memory_limit = int(value)\n"
        "                break\n"
        "    exhausted = False\n"
        "    path = '/tmp/minibug-resource-probe'\n"
        "    try:\n"
        "        with open(path, 'wb') as handle:\n"
        "            for _ in range(32):\n"
        "                handle.write(b'x' * (1024 * 1024))\n"
        "    except OSError as error:\n"
        "        exhausted = error.errno == errno.ENOSPC\n"
        "    finally:\n"
        "        try:\n"
        "            os.unlink(path)\n"
        "        except FileNotFoundError:\n"
        "            pass\n"
        "    return memory_limit == 128 * 1024 * 1024 and exhausted"
    )
    # The probe consumes at most the isolated tmpfs and never pressures host memory.
    result = DockerPythonTestSandbox(timeout_seconds=5).execute(
        _payload("assert repair() is True"),
        candidate,
    )

    # Success proves both the 128 MiB cgroup value and enforced tmpfs capacity.
    assert result.status == "passed"

"""Global context: prove the host/container boundary before RL uses its rewards.

Sources:
- https://docs.docker.com/reference/cli/docker/container/run/
- https://docs.python.org/3/library/subprocess.html#subprocess.Popen.communicate
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import pytest

import minibug_rl.sandbox as sandbox_module
from minibug_rl.sandbox import DockerSandbox, build_docker_command, run_candidate


class FakeProcess:
    """Provide a deterministic byte-protocol stand-in for ``docker run``."""

    def __init__(
        self,
        response: dict[str, Any] | None = None,
        *,
        times_out: bool = False,
    ) -> None:
        """Configure a normal response or one first-call timeout."""
        self.response = response or {"status": "success", "outputs": []}
        self.times_out = times_out
        self.returncode = 0
        self.input: bytes | None = None
        self.killed = False
        self.communicate_count = 0

    def communicate(
        self,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        """Record stdin and emulate Python's timeout-then-reap lifecycle."""
        del timeout
        self.communicate_count += 1
        if input is not None:
            self.input = input
        if self.times_out and self.communicate_count == 1:
            raise subprocess.TimeoutExpired("docker", 3)
        return json.dumps(self.response).encode(), b""

    def kill(self) -> None:
        """Record that the host reaped the timed-out Docker client."""
        self.killed = True


def _install_fake_process(
    monkeypatch: pytest.MonkeyPatch,
    process: FakeProcess,
) -> list[list[str]]:
    """Replace Docker startup and retain its argument-vector command."""
    commands: list[list[str]] = []

    def fake_popen(command: list[str], **_: Any) -> FakeProcess:
        commands.append(command)
        return process

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", fake_popen)
    return commands


def test_success_and_wrong_result_are_compared_on_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use actual values from the protocol while keeping answer checks local."""
    process = FakeProcess({"status": "success", "outputs": [3, 8]})
    _install_fake_process(monkeypatch, process)

    result = run_candidate(
        "def double(value):\n    return value * 2",
        "double",
        [
            {"args": [2], "kwargs": {}, "expected": 4},
            {"args": [4], "kwargs": {}, "expected": 8},
        ],
    )

    assert result.status == "success"
    assert [case.actual for case in result.cases] == [3, 8]
    assert [case.passed for case in result.cases] == [False, True]
    assert result.passed_count == 1
    assert result.all_passed is False


def test_expected_values_never_enter_container_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prove a unique hidden sentinel is absent from Docker standard input."""
    hidden = "HOST_ONLY_EXPECTED_91dba"
    process = FakeProcess({"status": "success", "outputs": ["actual"]})
    _install_fake_process(monkeypatch, process)

    run_candidate(
        "def identity(value):\n    return value",
        "identity",
        [{"args": ["input"], "kwargs": {}, "expected": hidden}],
    )

    assert process.input is not None
    request = json.loads(process.input)
    assert set(request) == {"code", "function_name", "calls"}
    assert request["calls"] == [{"args": ["input"], "kwargs": {}}]
    assert hidden.encode() not in process.input


@pytest.mark.parametrize(
    ("code", "expected_status", "error_fragment"),
    [
        ("def repair(:\n    pass", "invalid_candidate", "valid Python"),
        (
            "def repair(value):\n    import os\n    return value",
            "policy_rejected",
            "Import",
        ),
        (
            "def repair(value):\n    return open(value)",
            "policy_rejected",
            "open",
        ),
    ],
)
def test_syntax_and_policy_rejections_do_not_start_docker(
    monkeypatch: pytest.MonkeyPatch,
    code: str,
    expected_status: str,
    error_fragment: str,
) -> None:
    """Reject malformed or obviously dangerous code before container allocation."""

    def unexpected_popen(*_: Any, **__: Any) -> None:
        raise AssertionError("Docker must not start for a rejected candidate")

    monkeypatch.setattr(sandbox_module.subprocess, "Popen", unexpected_popen)

    result = DockerSandbox().execute(code, "repair", [{"args": [1], "kwargs": {}}])

    assert result.status == expected_status
    assert error_fragment in (result.error or "")


def test_timeout_kills_container_reaps_client_and_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply Python's documented explicit cleanup after ``communicate`` times out."""
    process = FakeProcess(times_out=True)
    commands = _install_fake_process(monkeypatch, process)
    cleanup_commands: list[list[str]] = []

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[bytes]:
        cleanup_commands.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(sandbox_module.subprocess, "run", fake_run)

    result = DockerSandbox(timeout_seconds=3).execute(
        "def spin(value):\n    while True:\n        pass",
        "spin",
        [{"args": [1], "kwargs": {}}],
    )

    name = commands[0][commands[0].index("--name") + 1]
    assert result.status == "timeout"
    assert process.killed is True
    assert process.communicate_count == 2
    assert cleanup_commands == [
        ["docker", "kill", name],
        ["docker", "rm", "--force", name],
    ]


def test_docker_command_has_all_constraints_and_no_mounts() -> None:
    """Inspect hardening without requiring Docker or the sandbox image."""
    command = build_docker_command("minibug-test", image="sandbox:test")

    expected_pairs = {
        "--name": "minibug-test",
        "--network": "none",
        "--cap-drop": "ALL",
        "--security-opt": "no-new-privileges=true",
        "--pids-limit": "32",
        "--memory": "128m",
        "--memory-swap": "128m",
        "--cpus": "0.5",
        "--tmpfs": "/tmp:rw,noexec,nosuid,size=16m",
        "--user": "65532:65532",
    }
    assert command[:2] == ["docker", "run"]
    assert "--rm" in command
    assert "--interactive" in command
    assert "--read-only" in command
    for flag, value in expected_pairs.items():
        assert command[command.index(flag) + 1] == value
    assert not {"--mount", "--volume", "-v"}.intersection(command)
    assert command[-1] == "sandbox:test"


@pytest.mark.skipif(
    os.getenv("MINIBUG_RUN_DOCKER_TESTS") != "1",
    reason="set MINIBUG_RUN_DOCKER_TESTS=1 after building the sandbox image",
)
def test_real_docker_success_wrong_result_and_timeout() -> None:
    """Exercise the image, JSON protocol, host comparison, and hard deadline."""
    comparison = run_candidate(
        "def double(value):\n    return value * 2",
        "double",
        [
            {"args": [2], "kwargs": {}, "expected": 4},
            {"args": [3], "kwargs": {}, "expected": 5},
        ],
    )
    timeout = run_candidate(
        "def spin(value):\n    while True:\n        pass",
        "spin",
        [{"args": [1], "kwargs": {}, "expected": 1}],
        timeout_seconds=1,
    )

    assert comparison.status == "success"
    assert [case.passed for case in comparison.cases] == [True, False]
    assert timeout.status == "timeout"

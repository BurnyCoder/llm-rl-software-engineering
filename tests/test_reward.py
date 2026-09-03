"""Global context: lock the correctness-first reward arithmetic and host comparison.

Source: https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
"""

from dataclasses import dataclass
from typing import Any

import pytest

from minibug_rl.reward import score_completion
from minibug_rl.schemas import TestCase


@dataclass(frozen=True)
class FakeSandboxResult:
    """Mimic the minimal executor result contract without starting Docker."""

    status: str
    outputs: tuple[Any, ...] = ()
    error: str | None = None


class RecordingExecutor:
    """Record inputs to prove expected answers stay on the host."""

    def __init__(self, result: FakeSandboxResult) -> None:
        """Store the deterministic result returned by `execute`."""
        self.result = result
        self.calls: list[dict[str, Any]] | None = None

    def execute(
        self,
        candidate_code: str,
        function_name: str,
        calls: list[dict[str, Any]],
    ) -> FakeSandboxResult:
        """Capture only call arguments, matching the real sandbox boundary."""
        del candidate_code, function_name
        self.calls = calls
        return self.result


def test_full_solution_receives_test_fraction_bonus_and_structure_reward() -> None:
    """A complete repair receives the maximum documented reward of 2.1."""
    # Expected values belong to task metadata on the host.
    tests = (
        TestCase(args=(1,), kwargs={}, expected=2),
        TestCase(args=(3,), kwargs={}, expected=4),
    )
    executor = RecordingExecutor(FakeSandboxResult("success", (2, 4)))

    reward = score_completion("def increment(x):\n    return x + 1", "increment", tests, executor)

    assert reward.pass_fraction == 1.0
    assert reward.solve_bonus == 1.0
    assert reward.structure_reward == 0.1
    assert reward.total == pytest.approx(2.1)
    # The sandbox sees inputs but never hidden expected answers.
    assert executor.calls == [{"args": [1], "kwargs": {}}, {"args": [3], "kwargs": {}}]
    assert all("expected" not in call for call in executor.calls)


def test_partial_runtime_and_policy_rewards_are_distinct() -> None:
    """Keep useful partial correctness separate from failures and policy rejection."""
    tests = (
        TestCase(args=(1,), kwargs={}, expected=2),
        TestCase(args=(3,), kwargs={}, expected=4),
    )
    partial = RecordingExecutor(FakeSandboxResult("success", (2, 99)))
    timeout = RecordingExecutor(FakeSandboxResult("timeout", error="deadline"))

    partial_reward = score_completion("def increment(x):\n    return x + 1", "increment", tests, partial)
    timeout_reward = score_completion("def increment(x):\n    while True:\n        pass", "increment", tests, timeout)
    policy_reward = score_completion("def increment(x):\n    return eval('2')", "increment", tests, partial)

    assert partial_reward.total == pytest.approx(0.6)
    assert timeout_reward.total == pytest.approx(-0.15)
    assert policy_reward.total == pytest.approx(-1.0)

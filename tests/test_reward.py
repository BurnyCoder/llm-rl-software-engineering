"""Global context: lock the correctness-first reward arithmetic and host comparison.

Source: https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
"""

from dataclasses import dataclass
from typing import Any

import pytest

from minibug_rl.reward import build_grpo_reward, score_completion
from minibug_rl.reward_cache import RewardCache
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

    partial_reward = score_completion(
        "def increment(x):\n    return x + 1",
        "increment",
        tests,
        partial,
    )
    timeout_reward = score_completion(
        "def increment(x):\n    while True:\n        pass",
        "increment",
        tests,
        timeout,
    )
    policy_reward = score_completion(
        "def increment(x):\n    return eval('2')",
        "increment",
        tests,
        partial,
    )

    assert partial_reward.total == pytest.approx(0.6)
    assert timeout_reward.total == pytest.approx(-0.15)
    assert policy_reward.total == pytest.approx(-1.0)


def test_infrastructure_failure_aborts_scoring() -> None:
    """Never turn a broken Docker daemon or runner image into model reward evidence."""
    tests = (TestCase(args=(1,), kwargs={}, expected=2),)
    executor = RecordingExecutor(
        FakeSandboxResult("infrastructure_error", error="sandbox image missing")
    )

    with pytest.raises(RuntimeError, match="Sandbox infrastructure failed"):
        score_completion(
            "def increment(x):\n    return x + 1",
            "increment",
            tests,
            executor,
        )


def test_reward_cache_reuses_exact_task_test_and_completion_hash(tmp_path: Any) -> None:
    """Avoid repeated deterministic containers without sharing mismatched hidden suites."""
    tests = (TestCase(args=(1,), kwargs={}, expected=2),)
    executor = RecordingExecutor(FakeSandboxResult("success", (2,)))
    cache = RewardCache(tmp_path / "reward-cache.sqlite3")
    completion = "def increment(x):\n    return x + 1"

    first, first_hit = cache.score(
        "task-1",
        completion,
        "increment",
        tests,
        executor,
    )
    second, second_hit = cache.score(
        "task-1",
        completion,
        "increment",
        tests,
        executor,
    )

    assert first == second
    assert first_hit is False
    assert second_hit is True
    assert len(executor.calls) == 1


def test_grpo_reward_aligns_extra_columns_and_logs_every_completion(tmp_path: Any) -> None:
    """Adapt TRL's repeated dataset columns to the shared scorer without losing outputs."""
    from minibug_rl.run_logging import RunLogger

    # Two generated candidates correspond to the same prompt in one GRPO reward group.
    executor = RecordingExecutor(FakeSandboxResult("success", (2,)))
    logger = RunLogger.create(tmp_path, run_id="reward-adapter")
    reward = build_grpo_reward(executor, logger)
    hidden = '[{"args":[1],"kwargs":{},"expected":2}]'

    values = reward(
        completions=[
            "def increment(x):\n    return x + 1",
            "def increment(x):\n    return x",
        ],
        prompts=["repair", "repair"],
        task_id=["task-1", "task-1"],
        split=["train", "train"],
        function_name=["increment", "increment"],
        hidden_tests_json=[hidden, hidden],
    )
    logger.close()

    assert values == pytest.approx([2.1, 2.1])
    assert (logger.directory / "generations.jsonl").read_text(encoding="utf-8").count("\n") == 2

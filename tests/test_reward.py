"""Global context: lock correctness-first reward arithmetic, audit order, and comparison.

Source: https://github.com/huggingface/trl/blob/v1.12.0/trl/trainer/grpo_trainer.py
"""

import json
from dataclasses import dataclass
from typing import Any

import pytest

import minibug_rl.reward as reward_module
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
    records = [
        json.loads(line)
        for line in (logger.directory / "generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in records] == [
        "generation",
        "generation_outcome",
        "generation",
        "generation_outcome",
    ]
    assert records[0]["generation_id"] == records[1]["generation_id"]
    assert records[2]["generation_id"] == records[3]["generation_id"]
    assert records[0]["generation_id"] != records[2]["generation_id"]
    assert records[1]["metadata"]["reward"] == pytest.approx(2.1)
    assert records[1]["metadata"]["cache_hit"] is False
    assert records[3]["metadata"]["cache_hit"] is False


def test_grpo_reward_flushes_raw_generation_before_candidate_parsing(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Make the before-parse chronology observable at the parser boundary."""
    from minibug_rl.run_logging import RunLogger

    logger = RunLogger.create(tmp_path, run_id="before-parser")
    original_parser = reward_module.parse_candidate
    parser_observed_log = False

    def parse_after_audit(completion: Any, function_name: str) -> Any:
        """Assert the complete raw event is durable before delegating parsing."""
        nonlocal parser_observed_log
        records = [
            json.loads(line)
            for line in (logger.directory / "generations.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert [record["event"] for record in records] == ["generation"]
        assert records[0]["completion"] == completion
        parser_observed_log = True
        return original_parser(completion, function_name)

    monkeypatch.setattr(reward_module, "parse_candidate", parse_after_audit)
    reward = build_grpo_reward(
        RecordingExecutor(FakeSandboxResult("success", (2,))),
        logger,
    )

    values = reward(
        completions=["def increment(x):\n    return x + 1"],
        prompts=["repair"],
        task_id=["task-1"],
        split=["train"],
        function_name=["increment"],
        hidden_tests_json=['[{"args":[1],"kwargs":{},"expected":2}]'],
    )
    logger.close()

    assert values == pytest.approx([2.1])
    assert parser_observed_log is True


def test_grpo_reward_requires_complete_aligned_prompts_before_logging(tmp_path: Any) -> None:
    """Reject a caller that cannot supply the raw prompts promised by run evidence."""
    from minibug_rl.run_logging import RunLogger

    logger = RunLogger.create(tmp_path, run_id="missing-prompts")
    reward = build_grpo_reward(
        RecordingExecutor(FakeSandboxResult("success", (2,))),
        logger,
    )

    with pytest.raises(ValueError, match="prompts are required"):
        reward(
            completions=["def increment(x):\n    return x + 1"],
            task_id=["task-1"],
            function_name=["increment"],
            hidden_tests_json=['[{"args":[1],"kwargs":{},"expected":2}]'],
        )
    logger.close()

    assert (logger.directory / "generations.jsonl").read_text(encoding="utf-8") == ""


def test_grpo_reward_logs_raw_text_and_linked_exception_before_reraising(tmp_path: Any) -> None:
    """Retain the complete generation even when scoring infrastructure aborts."""
    from minibug_rl.run_logging import RunLogger

    completion = "def increment(x):\n    return x + 1\n# complete-tail"
    logger = RunLogger.create(tmp_path, run_id="reward-exception")
    reward = build_grpo_reward(
        RecordingExecutor(FakeSandboxResult("infrastructure_error", error="sandbox image missing")),
        logger,
    )

    with pytest.raises(RuntimeError, match="Sandbox infrastructure failed"):
        reward(
            completions=[completion],
            prompts=["repair this complete prompt"],
            task_id=["task-1"],
            split=["train"],
            function_name=["increment"],
            hidden_tests_json=['[{"args":[1],"kwargs":{},"expected":2}]'],
        )
    logger.close()

    records = [
        json.loads(line)
        for line in (logger.directory / "generations.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["event"] for record in records] == ["generation", "generation_outcome"]
    assert records[0]["completion"] == completion
    assert records[0]["generation_id"] == records[1]["generation_id"]
    assert records[1]["metadata"]["status"] == "exception"
    assert records[1]["metadata"]["exception_type"] == "RuntimeError"

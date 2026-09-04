"""Global context: correctness-first reward shared by GRPO training and evaluation.

Sources:
- https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
- https://arxiv.org/abs/2605.30478
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

from minibug_rl.parser import parse_candidate
from minibug_rl.reward_cache import RewardCache
from minibug_rl.run_logging import RunLogger
from minibug_rl.schemas import CandidateExecutor, RewardBreakdown, TestCase


def _equal(actual: Any, expected: Any) -> bool:
    """Compare deterministic JSON values without coercing types or tolerances silently."""
    # Boolean equality is type-sensitive here because `True == 1` in Python is surprising.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    # JSON numbers from deterministic integer-focused tasks can use ordinary equality.
    return bool(actual == expected)


def score_completion(
    completion: Any,
    function_name: str,
    tests: Sequence[TestCase],
    executor: CandidateExecutor,
) -> RewardBreakdown:
    """Parse, execute, and score one candidate while keeping answers on the host."""
    # The parser rejects malformed or prohibited code before any container allocation.
    parsed = parse_candidate(completion, function_name)
    if not parsed.valid:
        if parsed.policy_violation:
            return RewardBreakdown(
                policy_penalty=-1.0,
                status="policy_violation",
                total_cases=len(tests),
                error=parsed.error,
            )
        return RewardBreakdown(
            structure_reward=-0.3,
            status="invalid_structure",
            total_cases=len(tests),
            error=parsed.error,
        )
    # Strip expected outputs before crossing the untrusted execution boundary.
    calls = [test.sandbox_call() for test in tests]
    execution = executor.execute(parsed.code, function_name, calls)
    if execution.status == "infrastructure_error":
        # A broken daemon, image, or runner invalidates evidence and stops the experiment.
        detail = execution.error or "unknown sandbox error"
        raise RuntimeError(f"Sandbox infrastructure failed: {detail}")
    if execution.status != "success":
        # Candidate timeouts and runtime crashes receive the documented bounded penalty.
        return RewardBreakdown(
            structure_reward=0.1,
            runtime_penalty=-0.25,
            status=execution.status,
            total_cases=len(tests),
            error=execution.error,
        )
    # A malformed runner response is an execution failure, not model correctness evidence.
    if len(execution.outputs) != len(tests):
        return RewardBreakdown(
            structure_reward=0.1,
            runtime_penalty=-0.25,
            status="runtime_error",
            total_cases=len(tests),
            error="Sandbox returned a different number of outputs than inputs.",
        )
    # Compare every actual value only after the isolated process has terminated.
    passed = sum(
        _equal(actual, test.expected) for actual, test in zip(execution.outputs, tests, strict=True)
    )
    total_cases = len(tests)
    pass_fraction = passed / total_cases if total_cases else 0.0
    solved = total_cases > 0 and passed == total_cases
    return RewardBreakdown(
        pass_fraction=pass_fraction,
        solve_bonus=1.0 if solved else 0.0,
        structure_reward=0.1,
        status="success",
        passed=passed,
        total_cases=total_cases,
    )


def _hidden_cases(encoded: str) -> tuple[TestCase, ...]:
    """Decode the reward-only dataset column into typed host-side cases."""
    # The dataset was validated before training, so malformed JSON is a fatal data error.
    raw = json.loads(encoded)
    if not isinstance(raw, list):
        raise ValueError("hidden_tests_json must encode a list")
    cases: list[TestCase] = []
    for item in raw:
        if not isinstance(item, dict) or set(item) != {"args", "kwargs", "expected"}:
            raise ValueError("each hidden test must contain args, kwargs, and expected")
        if not isinstance(item["args"], list) or not isinstance(item["kwargs"], dict):
            raise ValueError("hidden test args/kwargs have invalid JSON shapes")
        cases.append(TestCase(tuple(item["args"]), dict(item["kwargs"]), item["expected"]))
    return tuple(cases)


def build_grpo_reward(
    executor: CandidateExecutor,
    logger: RunLogger,
    cache: RewardCache | None = None,
) -> Callable[..., list[float | None]]:
    """Adapt aligned TRL completion and dataset columns to the shared reward scorer."""
    selected_cache = cache or RewardCache(logger.directory / "reward-cache.sqlite3")

    def reward(
        completions: list[Any],
        task_id: list[str],
        function_name: list[str],
        hidden_tests_json: list[str],
        prompts: list[Any] | None = None,
        split: list[str] | None = None,
        **_trainer_values: Any,
    ) -> list[float | None]:
        """Score every rollout and log its complete prompt, completion, and components."""
        count = len(completions)
        columns = (task_id, function_name, hidden_tests_json)
        if any(len(column) != count for column in columns):
            raise ValueError("TRL reward columns are not aligned with completions")
        aligned_prompts = prompts if prompts is not None else ["<prompt unavailable>"] * count
        aligned_splits = split if split is not None else ["train"] * count
        if len(aligned_prompts) != count or len(aligned_splits) != count:
            raise ValueError("TRL prompt/split columns are not aligned with completions")
        rewards: list[float | None] = []
        for index, completion in enumerate(completions):
            breakdown, cache_hit = selected_cache.score(
                task_id[index],
                completion,
                function_name[index],
                _hidden_cases(hidden_tests_json[index]),
                executor,
            )
            # Expected outputs are intentionally absent from the generation metadata.
            metadata = asdict(breakdown)
            metadata["reward"] = breakdown.total
            metadata["cache_hit"] = cache_hit
            logger.generation(
                task_id=task_id[index],
                split=aligned_splits[index],
                prompt=aligned_prompts[index],
                completion=completion,
                metadata=metadata,
            )
            rewards.append(breakdown.total)
        return rewards

    # A stable function name gives TRL readable reward metric keys.
    reward.__name__ = "hidden_unit_test_reward"
    return reward

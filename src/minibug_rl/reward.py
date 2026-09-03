"""Global context: correctness-first reward shared by GRPO training and evaluation.

Sources:
- https://huggingface.co/docs/trl/main/en/grpo_trainer#using-a-custom-reward-function
- https://arxiv.org/abs/2605.30478
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from minibug_rl.parser import parse_candidate
from minibug_rl.schemas import CandidateExecutor, RewardBreakdown, TestCase


def _equal(actual: Any, expected: Any) -> bool:
    """Compare deterministic JSON values without coercing types or tolerances silently."""
    # Boolean equality is type-sensitive here because `True == 1` in Python is surprising.
    if isinstance(actual, bool) or isinstance(expected, bool):
        return type(actual) is type(expected) and actual == expected
    # JSON numbers from deterministic integer-focused tasks can use ordinary equality.
    return actual == expected


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
    if execution.status != "success":
        # Infrastructure failures remain distinguishable but use the same bounded penalty.
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
        _equal(actual, test.expected)
        for actual, test in zip(execution.outputs, tests, strict=True)
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

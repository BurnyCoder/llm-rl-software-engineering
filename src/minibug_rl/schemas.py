"""Global context: shared immutable value objects crossing MiniBug-RL modules.

Sources:
- https://docs.python.org/3/library/dataclasses.html
- https://docs.python.org/3/library/typing.html#typing.Protocol
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol


@dataclass(frozen=True)
class TestCase:
    """Represent one JSON-serializable function call and its host-only expected value."""

    # Pytest otherwise mistakes this domain type for a test container because of its name.
    __test__: ClassVar[bool] = False

    # Positional arguments become a JSON array at the container boundary.
    args: tuple[Any, ...]
    # Keyword arguments become a JSON object at the container boundary.
    kwargs: dict[str, Any]
    # The host compares this value after the untrusted container exits.
    expected: Any

    def sandbox_call(self) -> dict[str, Any]:
        """Return only inputs, deliberately excluding the hidden expected answer."""
        # Convert the immutable tuple into JSON's natural mutable array representation.
        return {"args": list(self.args), "kwargs": dict(self.kwargs)}


@dataclass(frozen=True)
class RepairTask:
    """Describe one frozen Python function-repair task and its split membership."""

    # Stable identifiers connect generations, rewards, reports, and split manifests.
    id: str
    split: str
    family: str
    difficulty: str
    specification: str
    function_name: str
    buggy_code: str
    public_tests: tuple[TestCase, ...]
    hidden_tests: tuple[TestCase, ...]


@dataclass(frozen=True)
class ParsedCandidate:
    """Carry normalized candidate source or a safe rejection reason."""

    # `valid` is the single gate used before invoking any executor.
    valid: bool
    # `code` is normalized Python with optional Markdown fences removed.
    code: str
    # `error` explains rejection without echoing hidden test values.
    error: str | None = None
    # `policy_violation` distinguishes unsafe constructs from ordinary format mistakes.
    policy_violation: bool = False


@dataclass(frozen=True)
class SandboxExecution:
    """Represent bounded container execution without interpreting correctness."""

    # Status is `success`, `timeout`, `runtime_error`, or `infrastructure_error`.
    status: str
    # Outputs preserve one result per supplied call when execution succeeds.
    outputs: tuple[Any, ...] = ()
    # Error text is diagnostic and must never contain expected hidden values.
    error: str | None = None
    # Duration supports performance diagnostics and timeout reporting.
    duration_seconds: float = 0.0


class CandidateExecutor(Protocol):
    """Define the narrow host-to-sandbox interface used by reward calculation."""

    def execute(
        self,
        candidate_code: str,
        function_name: str,
        calls: list[dict[str, Any]],
    ) -> SandboxExecution:
        """Execute a candidate against inputs and return actual values only."""


@dataclass(frozen=True)
class RewardBreakdown:
    """Expose every reward component so experiments remain auditable."""

    # Functional partial credit is the fraction of hidden cases passed.
    pass_fraction: float = 0.0
    # Fully correct candidates receive an additional sparse solve bonus.
    solve_bonus: float = 0.0
    # Structurally valid replacement functions receive a small shaping reward.
    structure_reward: float = 0.0
    # Timeouts and runtime failures receive a bounded execution penalty.
    runtime_penalty: float = 0.0
    # Prohibited constructs receive the strongest policy penalty.
    policy_penalty: float = 0.0
    # `status` allows aggregate reports to count parse, runtime, and policy failures.
    status: str = "unknown"
    # `passed` and `total_cases` retain exact integer evidence behind the fraction.
    passed: int = 0
    total_cases: int = 0
    # `error` is public diagnostic metadata rather than model-visible feedback.
    error: str | None = None

    @property
    def total(self) -> float:
        """Sum the documented independent reward components."""
        # Keeping arithmetic here prevents drift between training and evaluation.
        return (
            self.pass_fraction
            + self.solve_bonus
            + self.structure_reward
            + self.runtime_penalty
            + self.policy_penalty
        )


def immutable_tests(cases: Sequence[TestCase]) -> tuple[TestCase, ...]:
    """Normalize caller-owned sequences before placing them in frozen task objects."""
    # A tuple prevents later list mutation from silently changing evaluation data.
    return tuple(cases)

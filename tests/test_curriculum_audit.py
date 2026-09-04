"""Specify the Docker-only hidden-test coverage gate for authored buggy programs.

Global context: every stored buggy implementation must execute successfully yet fail
at least one hidden case, otherwise copying the prompt receives a false solve reward.

Sources:
- https://docs.pytest.org/en/stable/how-to/monkeypatch.html
- https://docs.python.org/3/library/dataclasses.html
"""

from __future__ import annotations

# Pytest supplies focused exception assertions and safe dependency replacement.
import pytest

# Import the module so monkeypatch replaces its exact Docker call boundary.
import minibug_rl.curriculum_audit as audit_module

# The function under test is the preparation gate used by the real pipeline.
from minibug_rl.curriculum_audit import verify_buggy_programs_fail_hidden

# Immutable domain records keep these tests identical to production inputs/results.
from minibug_rl.sandbox import CaseResult, SandboxResult
from minibug_rl.schemas import RepairTask, TestCase


def _task(task_id: str) -> RepairTask:
    """Build one minimal repair task whose hidden answer never reaches the fake runner."""
    # One immutable case is enough to distinguish a caught bug from false full credit.
    case = TestCase(args=(1,), kwargs={}, expected=2)
    # The authored source is valid but intentionally returns the wrong value.
    return RepairTask(
        id=task_id,
        split="train",
        family="arithmetic",
        difficulty="easy",
        specification="Add one to the input.",
        function_name="add_one",
        buggy_code="def add_one(value):\n    return value",
        public_tests=(case, case),
        hidden_tests=(case, case, case, case),
    )


def test_audit_accepts_an_executing_bug_caught_by_hidden_tests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count a task only when Docker returned outputs and one expectation failed."""
    # A mixed comparison result proves the buggy program ran but did not earn full credit.
    result = SandboxResult(
        status="success",
        cases=(CaseResult(actual=1, passed=False), CaseResult(actual=2, passed=True)),
    )
    # The replacement observes arguments without evaluating any supplied Python.
    observed: list[tuple[str, str, str]] = []

    def fake_run_candidate(
        code: str,
        function_name: str,
        _tests: object,
        *,
        image: str,
    ) -> SandboxResult:
        """Return deterministic container evidence and retain non-secret call metadata."""
        # Recording source identity and image proves the gate routes through one boundary.
        observed.append((code, function_name, image))
        # Expected hidden values are intentionally neither read nor recorded by this fake.
        return result

    # Replace only the external executor; audit logic remains production code.
    monkeypatch.setattr(audit_module, "run_candidate", fake_run_candidate)

    # The returned count becomes durable preparation metadata in the phase result.
    checked = verify_buggy_programs_fail_hidden((_task("caught"),), image="audit:image")

    # One task was checked through the configured image and accepted.
    assert checked == 1
    assert observed == [("def add_one(value):\n    return value", "add_one", "audit:image")]


def test_audit_rejects_buggy_program_that_passes_every_hidden_test(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Name every task whose original buggy source would receive a solve reward."""
    # All passing comparisons reproduce the exact curriculum defect this gate prevents.
    result = SandboxResult(
        status="success",
        cases=(CaseResult(actual=2, passed=True),),
    )
    # A variadic fake keeps this test focused on the all-passed decision.
    monkeypatch.setattr(audit_module, "run_candidate", lambda *_args, **_kwargs: result)

    # The message identifies the repair record without disclosing hidden test values.
    with pytest.raises(ValueError, match="false_full_credit"):
        verify_buggy_programs_fail_hidden((_task("false_full_credit"),), image="audit:image")


def test_audit_rejects_non_successful_authored_program_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat timeouts, policy rejection, and runner faults as invalid curriculum evidence."""
    # A runtime failure means the hidden suite did not observe an ordinary wrong answer.
    result = SandboxResult(status="runtime_error", error="TypeError: authored bug crashed")
    # The fake models a Docker result without running candidate code on the host.
    monkeypatch.setattr(audit_module, "run_candidate", lambda *_args, **_kwargs: result)

    # Preparation must stop instead of accepting a crash as adequate bug coverage.
    with pytest.raises(RuntimeError, match=r"broken_execution.*runtime_error"):
        verify_buggy_programs_fail_hidden((_task("broken_execution"),), image="audit:image")

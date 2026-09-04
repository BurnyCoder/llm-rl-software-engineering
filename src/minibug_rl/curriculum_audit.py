"""Global context: prove every authored bug is observable by its hidden tests.

Generated and authored task code executes only in the shared Docker sandbox. This
module keeps the corpus-quality decision separate from the prepare-phase coordinator.

Sources:
- https://docs.docker.com/reference/cli/docker/container/run/
- https://docs.python.org/3.12/library/dataclasses.html#frozen-instances
"""

from __future__ import annotations

# Iterable accepts the immutable task tuple while keeping the audit independently usable.
from collections.abc import Iterable

# Reuse the sole Docker candidate boundary and its host-side correctness comparison.
from minibug_rl.sandbox import run_candidate

# RepairTask provides frozen source, function name, and hidden cases as one record.
from minibug_rl.schemas import RepairTask


def verify_buggy_programs_fail_hidden(
    tasks: Iterable[RepairTask],
    *,
    image: str,
) -> int:
    """Require each stored bug to run normally and fail at least one hidden case."""
    # Retain only task IDs and statuses so errors never disclose hidden expectations.
    execution_failures: list[str] = []
    # Collect all false-full-credit records in one pass to make corpus repair complete.
    fully_passing_ids: list[str] = []
    # The returned count becomes auditable preparation metadata for the exact run.
    checked = 0
    # Each authored bug reaches a fresh resource-limited container through the shared runner.
    for task in tasks:
        # Expected values stay on the trusted host; only function inputs enter Docker.
        result = run_candidate(
            task.buggy_code,
            task.function_name,
            task.hidden_tests,
            image=image,
        )
        # A crash or infrastructure state cannot prove that hidden assertions catch the bug.
        if result.status != "success":
            # The stable ID and coarse status are sufficient to repair the authored record.
            execution_failures.append(f"{task.id} ({result.status})")
        # A normal execution that passes every case would award an incorrect solve bonus.
        elif result.all_passed:
            # No actual or expected values are included in this preflight diagnostic.
            fully_passing_ids.append(task.id)
        # Count attempted records only after their isolated execution result is classified.
        checked += 1
    # Execution defects take priority because their correctness comparisons are incomplete.
    if execution_failures:
        # Joining stable labels yields one actionable, deterministic preparation error.
        raise RuntimeError(
            "Authored buggy programs did not execute successfully: " + ", ".join(execution_failures)
        )
    # Reject any record for which the initial prompt could receive false full credit.
    if fully_passing_ids:
        # The caller can amend cases without exposing them through terminal diagnostics.
        raise ValueError(
            "Authored buggy programs pass every hidden test: " + ", ".join(fully_passing_ids)
        )
    # A successful count proves the entire supplied curriculum crossed the Docker boundary.
    return checked

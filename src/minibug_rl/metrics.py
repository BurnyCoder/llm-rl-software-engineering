"""Global context: aggregate repair outcomes and paired uncertainty for honest reports.

Sources:
- https://arxiv.org/abs/2107.03374
- https://docs.python.org/3/library/random.html
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True)
class EvaluationRecord:
    """Record one candidate outcome before any cross-task aggregation."""

    # Task ID is the pairing key used for before/after comparisons.
    task_id: str
    # Mode distinguishes deterministic greedy inference from sampled generations.
    mode: str
    # Sample index remains stable within each task/mode group.
    sample_index: int
    # Hidden fraction retains useful partial correctness information.
    hidden_fraction: float
    # Solved means every hidden test passed.
    solved: bool
    # Status supports parser, runtime, timeout, and policy rates.
    status: str


def aggregate_records(records: list[EvaluationRecord], sampled_k: int) -> dict[str, float | int]:
    """Aggregate task-level greedy and observed sampled success metrics."""
    # Materialize unique tasks once so all task-denominator metrics agree.
    task_ids = sorted({record.task_id for record in records})
    greedy = [record for record in records if record.mode == "greedy"]
    sampled = [record for record in records if record.mode == "sampled"]
    if len(greedy) != len(task_ids):
        raise ValueError("Evaluation requires exactly one greedy record per task")
    # Group sampled candidates so pass@k means at least one observed full repair per task.
    sampled_by_task: dict[str, list[EvaluationRecord]] = defaultdict(list)
    for record in sampled:
        sampled_by_task[record.task_id].append(record)
    if any(len(sampled_by_task[task_id]) != sampled_k for task_id in task_ids):
        raise ValueError(f"Evaluation requires exactly {sampled_k} sampled records per task")
    all_records = records or []
    denominator = len(all_records) or 1
    statuses = ("invalid_structure", "timeout", "runtime_error", "policy_violation")
    summary: dict[str, float | int] = {
        "tasks": len(task_ids),
        "greedy_pass_at_1": mean(float(record.solved) for record in greedy),
        f"sampled_pass_at_{sampled_k}": mean(
            float(any(record.solved for record in sampled_by_task[task_id]))
            for task_id in task_ids
        ),
        "greedy_hidden_test_fraction": mean(record.hidden_fraction for record in greedy),
        "sampled_hidden_test_fraction": mean(record.hidden_fraction for record in sampled),
    }
    # Candidate-denominator failure rates reveal formatting or execution regressions.
    for status in statuses:
        summary[f"{status}_rate"] = sum(record.status == status for record in all_records) / denominator
    return summary


def _percentile(sorted_values: list[float], proportion: float) -> float:
    """Select a deterministic nearest-rank-like empirical bootstrap percentile."""
    # Clamp the rounded index to cover both endpoints for small bootstrap samples.
    index = round((len(sorted_values) - 1) * proportion)
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def paired_bootstrap_interval(
    before: dict[str, float],
    after: dict[str, float],
    *,
    samples: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap paired per-task differences into a reproducible 95% interval."""
    # Pairing is valid only when both systems were evaluated on identical task IDs.
    if set(before) != set(after):
        raise ValueError("Paired bootstrap inputs must contain the same task IDs")
    if not before:
        raise ValueError("Paired bootstrap requires at least one task")
    if samples <= 0:
        raise ValueError("Bootstrap sample count must be positive")
    task_ids = sorted(before)
    differences = [after[task_id] - before[task_id] for task_id in task_ids]
    generator = random.Random(seed)
    # Resample paired differences rather than independently resampling model scores.
    bootstrapped = sorted(
        mean(generator.choice(differences) for _ in differences)
        for _ in range(samples)
    )
    return {
        "mean_difference": mean(differences),
        "lower_95": _percentile(bootstrapped, 0.025),
        "upper_95": _percentile(bootstrapped, 0.975),
    }

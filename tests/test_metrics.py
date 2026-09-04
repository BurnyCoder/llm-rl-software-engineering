"""Global context: lock evaluation aggregation and paired uncertainty calculations.

Sources:
- https://arxiv.org/abs/2107.03374
- https://docs.python.org/3/library/random.html
"""

import pytest

from minibug_rl.metrics import EvaluationRecord, aggregate_records, paired_bootstrap_interval


def test_aggregate_records_reports_pass_at_one_and_observed_pass_at_k() -> None:
    """Summarize greedy and sampled repair outcomes without conflating their denominators."""
    # Two tasks each have one greedy candidate and two independently sampled candidates.
    records = [
        EvaluationRecord("a", "greedy", 0, 1.0, True, "success"),
        EvaluationRecord("a", "sampled", 0, 0.0, False, "success"),
        EvaluationRecord("a", "sampled", 1, 1.0, True, "success"),
        EvaluationRecord("b", "greedy", 0, 0.5, False, "success"),
        EvaluationRecord("b", "sampled", 0, 0.0, False, "invalid_structure"),
        EvaluationRecord("b", "sampled", 1, 0.5, False, "success"),
    ]

    summary = aggregate_records(records, sampled_k=2)

    assert summary["tasks"] == 2
    assert summary["greedy_pass_at_1"] == pytest.approx(0.5)
    assert summary["sampled_pass_at_2"] == pytest.approx(0.5)
    assert summary["greedy_hidden_test_fraction"] == pytest.approx(0.75)
    assert summary["invalid_structure_rate"] == pytest.approx(1 / 6)


def test_paired_bootstrap_is_seeded_and_uses_task_level_differences() -> None:
    """Return a reproducible interval around the paired mean model change."""
    # Pairing prevents task difficulty from becoming independent bootstrap noise.
    before = {"a": 0.0, "b": 0.5, "c": 1.0}
    after = {"a": 0.5, "b": 1.0, "c": 1.0}

    interval = paired_bootstrap_interval(before, after, samples=1_000, seed=42)

    assert interval["mean_difference"] == pytest.approx(1 / 3)
    assert interval["lower_95"] <= interval["mean_difference"] <= interval["upper_95"]


def test_paired_bootstrap_rejects_mismatched_task_sets() -> None:
    """Never present an apparently paired interval for different evaluation tasks."""
    with pytest.raises(ValueError, match="same task IDs"):
        paired_bootstrap_interval({"a": 0.0}, {"b": 1.0}, samples=100, seed=42)

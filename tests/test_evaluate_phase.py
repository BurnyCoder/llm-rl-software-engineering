"""Global context: prove checkpoint selection remains validation-only and test-blind.

Sources:
- https://en.wikipedia.org/wiki/Training,_validation,_and_test_data_sets
- https://docs.python.org/3/library/unittest.mock.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from minibug_rl.config import load_run_config
from minibug_rl.context import PipelineContext
from minibug_rl.phases import evaluate as evaluate_phase
from minibug_rl.run_logging import RunLogger


def _result(
    label: str,
    fraction: float,
    solved: float,
    *,
    model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
) -> dict[str, Any]:
    """Build the smallest structurally valid evaluation result for phase tests."""
    return {
        "summary": {
            "label": label,
            "model": model,
            "tasks": 12,
            "greedy_hidden_test_fraction": fraction,
            "greedy_pass_at_1": solved,
            "invalid_structure_rate": 0.0,
            "timeout_rate": 0.0,
            "runtime_error_rate": 0.0,
            "policy_violation_rate": 0.0,
        },
        "task_scores": {f"task-{index}": fraction for index in range(12)},
        "records": [],
    }


def _context(tmp_path: Path) -> PipelineContext:
    """Create an isolated context with prerequisite phase artifacts recorded."""
    repository = Path(__file__).resolve().parents[1]
    config = load_run_config(repository / "configs" / "smoke.toml", environ={})
    logger = RunLogger.create(tmp_path, run_id="evaluate-test")
    context = PipelineContext.open(config, logger)
    context.state.update(
        {
            "baseline": _result("base-validation", 0.25, 0.0),
            "smoke": {"adapter_directory": str(tmp_path / "smoke")},
            "train": {"adapter_directory": str(tmp_path / "train")},
        }
    )
    return context


def test_evaluate_selects_on_validation_before_opening_final_test(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lock the stronger validation adapter before either final-test generation."""
    context = _context(tmp_path)
    calls: list[tuple[str, str, str | None]] = []

    def fake_evaluate(
        _config: Any,
        _logger: Any,
        *,
        split: str,
        label: str,
        adapter_path: str | Path | None = None,
        sampled_k: int | None = None,
    ) -> dict[str, Any]:
        """Return label-specific measurements while recording model/test access order."""
        del sampled_k
        calls.append((split, label, None if adapter_path is None else str(adapter_path)))
        fractions = {
            "smoke-validation": 0.30,
            "train-validation": 0.50,
            "base-test-final": 0.20,
            "selected-test-final": 0.40,
        }
        model = context.config.model.base_model if adapter_path is None else str(adapter_path)
        result = _result(
            label,
            fractions[label],
            float(label == "selected-test-final") / 12,
            model=model,
        )
        path = context.logger.directory / f"evaluation-{label}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(evaluate_phase, "evaluate_internal", fake_evaluate)

    result = evaluate_phase.run_evaluate(context)

    assert result["selected_candidate"] == "train"
    assert result["learning_success"] is True
    assert [item[:2] for item in calls] == [
        ("validation", "smoke-validation"),
        ("validation", "train-validation"),
        ("test", "base-test-final"),
        ("test", "selected-test-final"),
    ]
    context.logger.close()


def test_final_test_result_file_is_reused_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not regenerate a completed final-test side after a later-side crash."""
    context = _context(tmp_path)
    completed_base = _result("base-test-final", 0.20, 0.0)
    base_path = context.logger.directory / "evaluation-base-test-final.json"
    base_path.write_text(json.dumps(completed_base), encoding="utf-8")
    test_calls: list[str] = []

    def fake_evaluate(
        _config: Any,
        _logger: Any,
        *,
        split: str,
        label: str,
        adapter_path: str | Path | None = None,
        sampled_k: int | None = None,
    ) -> dict[str, Any]:
        """Make every validation call succeed and record only final-test execution."""
        del sampled_k
        if split == "test":
            test_calls.append(label)
        fraction = 0.40 if label.startswith("train") else 0.30
        model = context.config.model.base_model if adapter_path is None else str(adapter_path)
        result = _result(label, fraction, 0.0, model=model)
        output = context.logger.directory / f"evaluation-{label}.json"
        output.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(evaluate_phase, "evaluate_internal", fake_evaluate)

    evaluate_phase.run_evaluate(context)

    assert test_calls == ["selected-test-final"]
    assert context.state["evaluate"]["base_test"] == completed_base
    context.logger.close()

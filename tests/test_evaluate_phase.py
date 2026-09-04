"""Global context: prove checkpoint selection remains validation-only and test-blind.

Sources:
- https://www.deeplearningbook.org/contents/ml.html
- https://docs.python.org/3/library/unittest.mock.html
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from minibug_rl.config import load_run_config
from minibug_rl.context import PipelineContext
from minibug_rl.external_eval import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_TASK_COUNT,
    PROMPT_VARIANT,
)
from minibug_rl.metrics import EvaluationRecord, aggregate_records
from minibug_rl.phases import evaluate as evaluate_phase
from minibug_rl.run_logging import RunLogger
from minibug_rl.sandbox import DEFAULT_TIMEOUT_SECONDS
from minibug_rl.task_data import load_tasks, tasks_for_split

# Phase tests use one valid immutable digest without contacting Docker.
SANDBOX_IMAGE = "sha256:" + "d" * 64


def _result(
    label: str,
    fraction: float,
    solved: float,
    *,
    model: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct",
    task_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Build complete internal candidate evidence matching the production schema."""
    selected_ids = task_ids if task_ids is not None else [f"task-{index}" for index in range(12)]
    sampled_k = 2
    solved_count = round(solved * len(selected_ids))
    records: list[EvaluationRecord] = []
    detailed: list[dict[str, Any]] = []
    for task_index, task_id in enumerate(selected_ids):
        task_solved = task_index < solved_count
        for mode, sample_index in [("greedy", 0), *[("sampled", i) for i in range(sampled_k)]]:
            record = EvaluationRecord(
                task_id=task_id,
                mode=mode,
                sample_index=sample_index,
                hidden_fraction=fraction,
                solved=task_solved,
                status="success",
            )
            records.append(record)
            breakdown = {
                "pass_fraction": fraction,
                "solve_bonus": float(task_solved),
                "structure_reward": 0.1,
                "runtime_penalty": 0.0,
                "policy_penalty": 0.0,
                "status": "success",
                "passed": round(fraction * 100),
                "total_cases": 100,
                "error": None,
            }
            detailed.append(
                {
                    **asdict(record),
                    "reward": sum(
                        float(breakdown[key])
                        for key in (
                            "pass_fraction",
                            "solve_bonus",
                            "structure_reward",
                            "runtime_penalty",
                            "policy_penalty",
                        )
                    ),
                    "breakdown": breakdown,
                    "cache_hit": False,
                }
            )
    summary: dict[str, Any] = dict(aggregate_records(records, sampled_k))
    summary.update(
        {
            "label": label,
            "model": model,
            "sampled_k": sampled_k,
            "duration_seconds": 0.0,
            "peak_vram_bytes": 0,
        }
    )
    return {
        "summary": summary,
        "task_scores": {task_id: fraction for task_id in selected_ids},
        "records": detailed,
    }


def _configured_test_ids(context: PipelineContext) -> list[str]:
    """Read the exact final-task identity set that resumability must preserve."""
    tasks = load_tasks(context.config.project.data_file)
    return [task.id for task in tasks_for_split(tasks, "test")]


def _resume_internal_result(
    context: PipelineContext,
    monkeypatch: pytest.MonkeyPatch,
    saved_text: str,
) -> tuple[dict[str, Any], list[str], list[Path]]:
    """Exercise one stale internal artifact through the real resume coordinator."""
    result_path = context.logger.directory / "evaluation-base-test-final.json"
    result_path.write_text(saved_text, encoding="utf-8")
    calls: list[str] = []

    def fake_evaluate(
        _config: Any,
        _logger: Any,
        *,
        split: str,
        label: str,
        sandbox_image: str,
        adapter_path: str | Path | None = None,
        sampled_k: int | None = None,
    ) -> dict[str, Any]:
        """Return fresh evidence after the saved artifact is rejected."""
        del adapter_path, sampled_k
        assert split == "test"
        assert sandbox_image == SANDBOX_IMAGE
        calls.append(label)
        return {"fresh": True}

    monkeypatch.setattr(evaluate_phase, "evaluate_internal", fake_evaluate)
    result = evaluate_phase._evaluate_final_once(context, label="base-test-final")
    stale_paths = list(context.logger.directory.glob("evaluation-base-test-final.stale-*.json"))
    return result, calls, stale_paths


def _external_result(
    label: str,
    passed: int,
    *,
    model: str,
    context: PipelineContext,
    task_count: int = EXPECTED_TASK_COUNT,
) -> dict[str, Any]:
    """Build resumable frozen-benchmark evidence with paired binary task scores."""
    scores = {f"Python/{index}": float(index < passed) for index in range(task_count)}
    records = [
        {
            "task_id": task_id,
            "sample_index": 0,
            "passed": bool(score),
            "status": "passed" if score else "failed",
            "error_type": None if score else "AssertionError",
            "error": None if score else "assertion failed",
            "duration_seconds": 0.01,
            "prompt_tokens": 10,
            "candidate_source": "def repair():\n    return None",
        }
        for task_id, score in scores.items()
    ]
    return {
        "summary": {
            "label": label,
            "model": model,
            "split": "external_test",
            "benchmark": DATASET_ID,
            "benchmark_revision": DATASET_REVISION,
            "prompt_variant": PROMPT_VARIANT,
            "base_revision": context.config.model.revision,
            "generation_mode": "greedy",
            "samples_per_task": 1,
            "maximum_new_tokens": context.config.model.max_completion_length,
            "sandbox_image": SANDBOX_IMAGE,
            "sandbox_timeout_seconds": DEFAULT_TIMEOUT_SECONDS,
            "tasks": task_count,
            "passed": passed,
            "pass_at_1": passed / task_count,
            "failed": task_count - passed,
            "timeouts": 0,
        },
        "task_scores": scores,
        "records": records,
    }


def _context(tmp_path: Path) -> PipelineContext:
    """Create an isolated context with prerequisite phase artifacts recorded."""
    repository = Path(__file__).resolve().parents[1]
    config = load_run_config(repository / "configs" / "smoke.toml", environ={})
    logger = RunLogger.create(tmp_path, run_id="evaluate-test")
    context = PipelineContext.open(config, logger)
    context.state.update(
        {
            "prepare": {"sandbox_image_id": SANDBOX_IMAGE},
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
    external_calls: list[tuple[str, str | None]] = []
    events: list[str] = []

    def fake_evaluate(
        _config: Any,
        _logger: Any,
        *,
        split: str,
        label: str,
        sandbox_image: str,
        adapter_path: str | Path | None = None,
        sampled_k: int | None = None,
    ) -> dict[str, Any]:
        """Return label-specific measurements while recording model/test access order."""
        del sampled_k
        assert sandbox_image == SANDBOX_IMAGE
        calls.append((split, label, None if adapter_path is None else str(adapter_path)))
        events.append(f"internal:{label}")
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
            task_ids=_configured_test_ids(context) if split == "test" else None,
        )
        path = context.logger.directory / f"evaluation-{label}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(evaluate_phase, "evaluate_internal", fake_evaluate)

    def fake_external(
        _config: Any,
        _logger: Any,
        *,
        label: str,
        sandbox_image: str,
        adapter_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Record that the external benchmark starts only after final internal tests."""
        assert sandbox_image == SANDBOX_IMAGE
        selected_model = (
            context.config.model.base_model if adapter_path is None else str(adapter_path)
        )
        external_calls.append((label, None if adapter_path is None else str(adapter_path)))
        events.append(f"external:{label}")
        result = _external_result(
            label,
            82 + 41 * int(adapter_path is not None),
            model=selected_model,
            context=context,
        )
        path = context.logger.directory / f"external-evaluation-{label}.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(evaluate_phase, "evaluate_external", fake_external)

    result = evaluate_phase.run_evaluate(context)

    assert result["selected_candidate"] == "train"
    assert result["learning_success"] is True
    assert [item[:2] for item in calls] == [
        ("validation", "smoke-validation"),
        ("validation", "train-validation"),
        ("test", "base-test-final"),
        ("test", "selected-test-final"),
    ]
    assert external_calls == [
        ("base-humanevalfix", None),
        ("selected-humanevalfix", str(tmp_path / "train")),
    ]
    assert events == [
        "internal:smoke-validation",
        "internal:train-validation",
        "internal:base-test-final",
        "internal:selected-test-final",
        "external:base-humanevalfix",
        "external:selected-humanevalfix",
    ]
    assert result["external_selected"]["summary"]["pass_at_1"] == 0.75
    context.logger.close()


def test_final_test_result_file_is_reused_after_interruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not regenerate a completed final-test side after a later-side crash."""
    context = _context(tmp_path)
    completed_base = _result(
        "base-test-final",
        0.20,
        0.0,
        task_ids=_configured_test_ids(context),
    )
    completed_base["summary"].update(
        {
            "split": "test",
            "base_revision": context.config.model.revision,
            "sandbox_image": SANDBOX_IMAGE,
            "sampled_k": context.config.evaluation.sample_generations,
        }
    )
    base_path = context.logger.directory / "evaluation-base-test-final.json"
    base_path.write_text(json.dumps(completed_base), encoding="utf-8")
    completed_external = _external_result(
        "base-humanevalfix",
        1,
        model=context.config.model.base_model,
        context=context,
    )
    external_path = context.logger.directory / "external-evaluation-base-humanevalfix.json"
    external_path.write_text(json.dumps(completed_external), encoding="utf-8")
    test_calls: list[str] = []
    external_calls: list[str] = []

    def fake_evaluate(
        _config: Any,
        _logger: Any,
        *,
        split: str,
        label: str,
        sandbox_image: str,
        adapter_path: str | Path | None = None,
        sampled_k: int | None = None,
    ) -> dict[str, Any]:
        """Make every validation call succeed and record only final-test execution."""
        del sampled_k
        assert sandbox_image == SANDBOX_IMAGE
        if split == "test":
            test_calls.append(label)
        fraction = 0.40 if label.startswith("train") else 0.30
        model = context.config.model.base_model if adapter_path is None else str(adapter_path)
        result = _result(
            label,
            fraction,
            0.0,
            model=model,
            task_ids=_configured_test_ids(context) if split == "test" else None,
        )
        output = context.logger.directory / f"evaluation-{label}.json"
        output.write_text(json.dumps(result), encoding="utf-8")
        return result

    monkeypatch.setattr(evaluate_phase, "evaluate_internal", fake_evaluate)

    def fake_external(
        _config: Any,
        _logger: Any,
        *,
        label: str,
        sandbox_image: str,
        adapter_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Record only the missing selected side of the external comparison."""
        assert sandbox_image == SANDBOX_IMAGE
        external_calls.append(label)
        return _external_result(
            label,
            1,
            model=(context.config.model.base_model if adapter_path is None else str(adapter_path)),
            context=context,
        )

    monkeypatch.setattr(evaluate_phase, "evaluate_external", fake_external)

    evaluate_phase.run_evaluate(context)

    assert test_calls == ["selected-test-final"]
    assert external_calls == ["selected-humanevalfix"]
    assert context.state["evaluate"]["base_test"] == completed_base
    assert context.state["evaluate"]["external_base"] == completed_external
    context.logger.close()


@pytest.mark.parametrize(
    "corruption",
    [
        "empty-records",
        "missing-sampled-record",
        "mismatched-task-score",
        "oversized-duration",
    ],
)
def test_incomplete_internal_result_is_preserved_as_stale_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Reject missing or internally inconsistent candidate-level final evidence."""
    context = _context(tmp_path)
    task_ids = _configured_test_ids(context)
    stale = _result("base-test-final", 0.20, 0.0, task_ids=task_ids)
    stale["summary"].update(
        {
            "split": "test",
            "base_revision": context.config.model.revision,
            "sandbox_image": SANDBOX_IMAGE,
        }
    )
    if corruption == "empty-records":
        stale["records"] = []
    elif corruption == "missing-sampled-record":
        stale["records"].pop()
    elif corruption == "mismatched-task-score":
        stale["task_scores"][task_ids[0]] = 0.99
    else:
        # This remains valid JSON but overflows Python's int-to-float conversion.
        stale["summary"]["duration_seconds"] = 10**400
    result, calls, stale_paths = _resume_internal_result(
        context,
        monkeypatch,
        json.dumps(stale),
    )

    assert result == {"fresh": True}
    assert calls == ["base-test-final"]
    assert len(stale_paths) == 1
    context.logger.close()


def test_truncated_internal_result_is_preserved_as_stale_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recover from a legacy or interrupted non-atomic internal result write."""
    context = _context(tmp_path)
    truncated = '{"summary":'
    result, calls, stale_paths = _resume_internal_result(context, monkeypatch, truncated)
    assert result == {"fresh": True}
    assert calls == ["base-test-final"]
    assert len(stale_paths) == 1
    assert stale_paths[0].read_text(encoding="utf-8") == truncated
    context.logger.close()


def test_final_test_result_with_wrong_same_count_ids_is_stale_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject stale evidence whose score count matches but task identities do not."""
    context = _context(tmp_path)
    wrong_ids = _configured_test_ids(context)
    wrong_ids[-1] = "stale-test-task-with-the-right-count"
    stale = _result("base-test-final", 0.20, 0.0, task_ids=wrong_ids)
    stale["summary"].update(
        {
            "split": "test",
            "base_revision": context.config.model.revision,
            "sandbox_image": SANDBOX_IMAGE,
            "sampled_k": context.config.evaluation.sample_generations,
        }
    )
    result, calls, stale_paths = _resume_internal_result(
        context,
        monkeypatch,
        json.dumps(stale),
    )

    assert result == {"fresh": True}
    assert calls == ["base-test-final"]
    assert len(stale_paths) == 1
    context.logger.close()


def test_incomplete_external_result_is_preserved_as_stale_and_recomputed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Never reuse a partial artifact based only on its model and revision labels."""
    # Four scores imitate a syntactically valid file written by an older incomplete run.
    context = _context(tmp_path)
    incomplete = _external_result(
        "base-humanevalfix",
        1,
        model=context.config.model.base_model,
        context=context,
        task_count=4,
    )
    path = context.logger.directory / "external-evaluation-base-humanevalfix.json"
    path.write_text(json.dumps(incomplete), encoding="utf-8")
    calls: list[str] = []

    def fake_external(
        _config: Any,
        _logger: Any,
        *,
        label: str,
        sandbox_image: str,
        adapter_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return fresh evidence after the phase rejects the incomplete saved artifact."""
        del adapter_path
        assert sandbox_image == SANDBOX_IMAGE
        calls.append(label)
        return {"fresh": True}

    # Only model execution is replaced; the real resumability validator remains active.
    monkeypatch.setattr(evaluate_phase, "evaluate_external", fake_external)

    result = evaluate_phase._evaluate_external_once(context, label="base-humanevalfix")

    # The old file is retained for diagnosis while fresh evaluation supplies the result.
    assert result == {"fresh": True}
    assert calls == ["base-humanevalfix"]
    assert list(context.logger.directory.glob("external-evaluation-base-humanevalfix.stale-*.json"))
    context.logger.close()

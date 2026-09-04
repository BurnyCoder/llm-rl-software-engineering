"""Global context: select on validation, then score each final benchmark once.

Sources:
- https://www.deeplearningbook.org/contents/ml.html
- https://docs.python.org/3/library/statistics.html
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from minibug_rl.context import PipelineContext
from minibug_rl.evaluation import evaluate_internal, internal_result_is_complete
from minibug_rl.external_evaluation import evaluate_external, external_result_is_complete
from minibug_rl.metrics import paired_bootstrap_interval
from minibug_rl.run_logging import utc_timestamp
from minibug_rl.sandbox import DEFAULT_TIMEOUT_SECONDS
from minibug_rl.task_data import load_tasks, tasks_for_split


def _selection_key(result: dict[str, Any]) -> tuple[float, float]:
    """Rank candidates by validation hidden fraction and then full solve rate."""
    summary = result["summary"]
    return float(summary["greedy_hidden_test_fraction"]), float(summary["greedy_pass_at_1"])


def _failure_rate(result: dict[str, Any]) -> float:
    """Combine non-success candidate categories for the reliability gate."""
    summary = result["summary"]
    return sum(
        float(summary[f"{status}_rate"])
        for status in ("invalid_structure", "timeout", "runtime_error", "policy_violation")
    )


def _evaluate_final_once(
    context: PipelineContext,
    *,
    label: str,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Reuse a complete final-test result file after an interrupted phase rerun."""
    result_path = context.logger.directory / f"evaluation-{label}.json"
    if result_path.exists():
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        test_tasks = tasks_for_split(load_tasks(context.config.project.data_file), "test")
        expected_task_ids = {task.id for task in test_tasks}
        expected_model = (
            str(adapter_path) if adapter_path is not None else context.config.model.base_model
        )
        if internal_result_is_complete(
            loaded,
            label=label,
            model=expected_model,
            base_revision=context.config.model.revision,
            sandbox_image=context.prepared_sandbox_image(),
            sampled_k=context.config.evaluation.sample_generations,
            expected_task_ids=expected_task_ids,
        ):
            context.logger.message(
                "final_test_reused",
                f"Reusing completed final-test evidence for {label}.",
                label=label,
            )
            return cast(dict[str, Any], loaded)
        stale_path = result_path.with_name(f"{result_path.stem}.stale-{utc_timestamp()}.json")
        result_path.replace(stale_path)
        context.logger.message(
            "final_test_stale",
            f"Preserved stale evidence before reevaluating {label}.",
            stale_path=str(stale_path),
        )
    return evaluate_internal(
        context.config,
        context.logger,
        split="test",
        label=label,
        sandbox_image=context.prepared_sandbox_image(),
        adapter_path=adapter_path,
    )


def _evaluate_external_once(
    context: PipelineContext,
    *,
    label: str,
    adapter_path: Path | None = None,
) -> dict[str, Any]:
    """Resume only complete external evidence for the same model and dataset pin."""
    result_path = context.logger.directory / f"external-evaluation-{label}.json"
    if result_path.exists():
        loaded = json.loads(result_path.read_text(encoding="utf-8"))
        expected_model = (
            str(adapter_path) if adapter_path is not None else context.config.model.base_model
        )
        if external_result_is_complete(
            loaded,
            label=label,
            model=expected_model,
            base_revision=context.config.model.revision,
            maximum_new_tokens=context.config.model.max_completion_length,
            sandbox_image=context.prepared_sandbox_image(),
            sandbox_timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        ):
            context.logger.message(
                "external_test_reused",
                f"Reusing completed frozen external evidence for {label}.",
                label=label,
            )
            return cast(dict[str, Any], loaded)
        stale_path = result_path.with_name(f"{result_path.stem}.stale-{utc_timestamp()}.json")
        result_path.replace(stale_path)
        context.logger.message(
            "external_test_stale",
            f"Preserved stale external evidence before reevaluating {label}.",
            stale_path=str(stale_path),
        )
    return evaluate_external(
        context.config,
        context.logger,
        label=label,
        sandbox_image=context.prepared_sandbox_image(),
        adapter_path=adapter_path,
    )


def run_evaluate(context: PipelineContext) -> dict[str, Any]:
    """Choose on validation and compare the winner on decision-isolated final tasks."""
    logger = context.logger
    logger.message("phase_start", "Selecting and evaluating trained adapters.", phase="evaluate")
    if "baseline" not in context.state:
        raise RuntimeError("The baseline phase must finish before evaluation")
    candidate_paths: list[tuple[str, Path]] = []
    for phase in ("smoke", "train"):
        phase_state = context.state.get(phase)
        if isinstance(phase_state, dict) and phase_state.get("adapter_directory"):
            candidate_paths.append((phase, Path(str(phase_state["adapter_directory"]))))
    if not candidate_paths:
        raise RuntimeError("No trained adapter is available for evaluation")
    validation_results: dict[str, dict[str, Any]] = {}
    for name, adapter_path in candidate_paths:
        validation_results[name] = evaluate_internal(
            context.config,
            logger,
            split="validation",
            label=f"{name}-validation",
            sandbox_image=context.prepared_sandbox_image(),
            adapter_path=adapter_path,
        )
    selected_name = max(
        validation_results,
        key=lambda name: _selection_key(validation_results[name]),
    )
    selected_path = dict(candidate_paths)[selected_name]
    selected_validation = validation_results[selected_name]
    base_validation = context.state["baseline"]
    validation_interval = paired_bootstrap_interval(
        base_validation["task_scores"],
        selected_validation["task_scores"],
        samples=context.config.evaluation.bootstrap_samples,
        seed=context.config.training.seed,
    )
    solved_gain = round(
        (
            float(selected_validation["summary"]["greedy_pass_at_1"])
            - float(base_validation["summary"]["greedy_pass_at_1"])
        )
        * int(base_validation["summary"]["tasks"])
    )
    reliability_ok = _failure_rate(selected_validation) <= _failure_rate(base_validation)
    learning_gate = (
        validation_interval["mean_difference"] >= 0.05 or solved_gain >= 1
    ) and reliability_ok
    # Final-test calls start only after the validation-selected path is immutable.
    base_test = _evaluate_final_once(
        context,
        label="base-test-final",
    )
    selected_test = _evaluate_final_once(
        context,
        label="selected-test-final",
        adapter_path=selected_path,
    )
    test_interval = paired_bootstrap_interval(
        base_test["task_scores"],
        selected_test["task_scores"],
        samples=context.config.evaluation.bootstrap_samples,
        seed=context.config.training.seed,
    )
    # HumanEvalFix is opened only after validation selection is immutable and never
    # contributes to model choice, hyperparameters, rewards, or the learning gate.
    external_base = _evaluate_external_once(context, label="base-humanevalfix")
    external_selected = _evaluate_external_once(
        context,
        label="selected-humanevalfix",
        adapter_path=selected_path,
    )
    external_interval = paired_bootstrap_interval(
        external_base["task_scores"],
        external_selected["task_scores"],
        samples=context.config.evaluation.bootstrap_samples,
        seed=context.config.training.seed,
    )
    result = {
        "selected_candidate": selected_name,
        "selected_adapter": str(selected_path),
        "learning_success": learning_gate,
        "validation_solved_gain": solved_gain,
        "validation_reliability_ok": reliability_ok,
        "validation_interval": validation_interval,
        "validation_candidates": validation_results,
        "base_test": base_test,
        "selected_test": selected_test,
        "test_interval": test_interval,
        "external_base": external_base,
        "external_selected": external_selected,
        "external_interval": external_interval,
    }
    logger.write_json("comparison.json", result)
    context.record("evaluate", result)
    return result

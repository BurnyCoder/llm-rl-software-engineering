"""Global context: run identical reward-scored inference for base and adapted policies.

Sources:
- https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/utils.py
- https://arxiv.org/abs/2107.03374
- https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/autograd/grad_mode.py
- https://docs.python.org/3.12/library/collections.abc.html#collections.abc.Mapping
- https://docs.python.org/3.12/library/math.html#math.isfinite
"""

from __future__ import annotations

import gc
import time
from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite
from pathlib import Path
from typing import Any, cast

import torch
from transformers import set_seed

from minibug_rl.config import RunConfig
from minibug_rl.metrics import EvaluationRecord, aggregate_records
from minibug_rl.modeling import (
    load_tokenizer,
    load_transformers_model,
    validate_prompt_lengths,
)
from minibug_rl.prompts import build_messages, render_messages
from minibug_rl.reward_cache import RewardCache
from minibug_rl.run_logging import RunLogger
from minibug_rl.sandbox import DockerSandbox
from minibug_rl.task_data import load_tasks, tasks_for_split

# Saved internal evidence has one stable public row schema per generated candidate.
_INTERNAL_RECORD_FIELDS = {
    "task_id",
    "mode",
    "sample_index",
    "hidden_fraction",
    "solved",
    "status",
    "reward",
    "breakdown",
    "cache_hit",
}
# These are the additive components behind each candidate's scalar reward.
_REWARD_COMPONENT_FIELDS = (
    "pass_fraction",
    "solve_bonus",
    "structure_reward",
    "runtime_penalty",
    "policy_penalty",
)
_REWARD_BREAKDOWN_FIELDS = {
    *_REWARD_COMPONENT_FIELDS,
    "status",
    "passed",
    "total_cases",
    "error",
}
_INTERNAL_OUTCOME_STATUSES = {
    "success",
    "invalid_structure",
    "timeout",
    "runtime_error",
    "policy_violation",
}


def _is_finite_number(value: object) -> bool:
    """Accept JSON integer/float values while excluding booleans and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return isfinite(float(value))
    except OverflowError:
        # JSON integers are unbounded, whereas conversion to a finite float is not.
        return False


def internal_result_is_complete(
    result: object,
    *,
    label: str,
    model: str,
    base_revision: str,
    sandbox_image: str,
    sampled_k: int,
    expected_task_ids: set[str],
) -> bool:
    """Cross-check all internal candidate rows before reusing final-test evidence."""
    # Summary, paired task scores, and candidate rows are independent evidence views.
    if not isinstance(result, Mapping):
        return False
    summary = result.get("summary")
    scores = result.get("task_scores")
    records = result.get("records")
    expected_record_count = len(expected_task_ids) * (sampled_k + 1)
    if not isinstance(summary, Mapping) or not isinstance(scores, Mapping):
        return False
    if not isinstance(records, list) or len(records) != expected_record_count:
        return False
    # These selectors bind a saved file to this exact evaluation protocol and policy.
    expected_metadata: dict[str, object] = {
        "label": label,
        "split": "test",
        "model": model,
        "base_revision": base_revision,
        "sandbox_image": sandbox_image,
        "sampled_k": sampled_k,
        "tasks": len(expected_task_ids),
    }
    if any(summary.get(key) != value for key, value in expected_metadata.items()):
        return False
    duration = summary.get("duration_seconds")
    peak_vram = summary.get("peak_vram_bytes")
    if (
        not _is_finite_number(duration)
        or float(cast(int | float, duration)) < 0.0
        or isinstance(peak_vram, bool)
        or not isinstance(peak_vram, int)
        or peak_vram < 0
    ):
        return False
    # Rebuild the typed metric rows while checking the detailed reward evidence.
    parsed_records: list[EvaluationRecord] = []
    for raw_record in records:
        if not isinstance(raw_record, Mapping) or set(raw_record) != _INTERNAL_RECORD_FIELDS:
            return False
        task_id = raw_record.get("task_id")
        mode = raw_record.get("mode")
        sample_index = raw_record.get("sample_index")
        hidden_fraction = raw_record.get("hidden_fraction")
        solved = raw_record.get("solved")
        status = raw_record.get("status")
        reward = raw_record.get("reward")
        breakdown = raw_record.get("breakdown")
        cache_hit = raw_record.get("cache_hit")
        if (
            not isinstance(task_id, str)
            or task_id not in expected_task_ids
            or not isinstance(mode, str)
            or mode not in {"greedy", "sampled"}
            or isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not _is_finite_number(hidden_fraction)
            or not isinstance(solved, bool)
            or not isinstance(status, str)
            or status not in _INTERNAL_OUTCOME_STATUSES
            or not _is_finite_number(reward)
            or not isinstance(breakdown, Mapping)
            or set(breakdown) != _REWARD_BREAKDOWN_FIELDS
            or not isinstance(cache_hit, bool)
        ):
            return False
        component_values = [breakdown.get(key) for key in _REWARD_COMPONENT_FIELDS]
        if any(not _is_finite_number(value) for value in component_values):
            return False
        numeric_components = [float(cast(int | float, value)) for value in component_values]
        passed = breakdown.get("passed")
        total_cases = breakdown.get("total_cases")
        error = breakdown.get("error")
        if (
            breakdown.get("status") != status
            or isinstance(passed, bool)
            or not isinstance(passed, int)
            or isinstance(total_cases, bool)
            or not isinstance(total_cases, int)
            or total_cases <= 0
            or not 0 <= passed <= total_cases
            or (error is not None and not isinstance(error, str))
        ):
            return False
        pass_fraction = numeric_components[0]
        solve_bonus = numeric_components[1]
        if (
            pass_fraction != float(cast(int | float, hidden_fraction))
            or pass_fraction != passed / total_cases
            or solve_bonus != float(solved)
            or float(cast(int | float, reward)) != sum(numeric_components)
        ):
            return False
        parsed_records.append(
            EvaluationRecord(
                task_id=task_id,
                mode=mode,
                sample_index=sample_index,
                hidden_fraction=float(cast(int | float, hidden_fraction)),
                solved=solved,
                status=status,
            )
        )
    # The canonical aggregator enforces one greedy row plus indices 0..k-1 per task.
    try:
        recomputed_summary = aggregate_records(parsed_records, sampled_k)
    except ValueError:
        return False
    if {record.task_id for record in parsed_records} != expected_task_ids:
        return False
    if any(summary.get(key) != value for key, value in recomputed_summary.items()):
        return False
    # Greedy rows are the sole source for the paired per-task score mapping.
    greedy_scores = {
        record.task_id: record.hidden_fraction
        for record in parsed_records
        if record.mode == "greedy"
    }
    if set(scores) != expected_task_ids:
        return False
    for task_id in expected_task_ids:
        score = scores.get(task_id)
        if not _is_finite_number(score):
            return False
        if float(cast(int | float, score)) != greedy_scores[task_id]:
            return False
    return True


def _decode_new_tokens(tokenizer: Any, generated: Any, prompt_tokens: int) -> list[str]:
    """Decode only generated suffixes so prompts cannot be mistaken for model output."""
    return [
        tokenizer.decode(sequence[prompt_tokens:], skip_special_tokens=True)
        for sequence in generated
    ]


def generate_completions(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    maximum_new_tokens: int,
    count: int,
    sampled: bool,
    temperature: float,
    top_p: float,
) -> list[str]:
    """Generate one greedy candidate or a fixed sampled group for a single task."""
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    generation = {
        "max_new_tokens": maximum_new_tokens,
        "do_sample": sampled,
        "num_return_sequences": count,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.0,
    }
    if sampled:
        # Sampling controls are omitted for greedy decoding to avoid library warnings.
        generation.update({"temperature": temperature, "top_p": top_p, "top_k": 0})
    with torch.inference_mode():
        generated = model.generate(**encoded, **generation)
    return _decode_new_tokens(tokenizer, generated, encoded["input_ids"].shape[1])


def evaluate_internal(
    config: RunConfig,
    logger: RunLogger,
    *,
    split: str,
    label: str,
    sandbox_image: str,
    adapter_path: str | Path | None = None,
    sampled_k: int | None = None,
) -> dict[str, Any]:
    """Evaluate one policy on a frozen internal split and persist candidate-level evidence."""
    selected_k = sampled_k or config.evaluation.sample_generations
    tasks = tasks_for_split(load_tasks(config.project.data_file), split)
    tokenizer = load_tokenizer(config)
    validate_prompt_lengths(tasks, tokenizer, config.model.max_prompt_length)
    model = load_transformers_model(
        config,
        adapter_path=adapter_path,
        device="cuda",
        for_training=False,
    )
    model.eval()
    # The prepare-phase digest prevents a mutable local tag from changing scoring code.
    executor = DockerSandbox(image=sandbox_image)
    reward_cache = RewardCache(logger.directory / "reward-cache.sqlite3")
    records: list[EvaluationRecord] = []
    detailed: list[dict[str, Any]] = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    for task_index, task in enumerate(tasks):
        prompt = render_messages(tokenizer, build_messages(task))
        # Reset per task so base/adapter comparisons receive identical random streams.
        set_seed(config.training.seed + task_index)
        candidates = [
            (
                "greedy",
                0,
                generate_completions(
                    model,
                    tokenizer,
                    prompt,
                    maximum_new_tokens=config.model.max_completion_length,
                    count=1,
                    sampled=False,
                    temperature=config.training.temperature,
                    top_p=config.training.top_p,
                )[0],
            )
        ]
        sampled = generate_completions(
            model,
            tokenizer,
            prompt,
            maximum_new_tokens=config.model.max_completion_length,
            count=selected_k,
            sampled=True,
            temperature=config.training.temperature,
            top_p=config.training.top_p,
        )
        candidates.extend(("sampled", index, text) for index, text in enumerate(sampled))
        for mode, sample_index, completion in candidates:
            # Raw text must be durable before parsing, cache lookup, or Docker execution.
            generation_id = logger.generation(
                task_id=task.id,
                split=split,
                prompt=prompt,
                completion=completion,
                metadata={
                    "evaluation": label,
                    "mode": mode,
                    "sample_index": sample_index,
                },
            )
            try:
                breakdown, cache_hit = reward_cache.score(
                    task.id,
                    completion,
                    task.function_name,
                    task.hidden_tests,
                    executor,
                )
            except Exception as error:
                logger.generation_outcome(
                    generation_id=generation_id,
                    task_id=task.id,
                    split=split,
                    metadata={
                        "evaluation": label,
                        "status": "exception",
                        "exception_type": type(error).__name__,
                        "error": str(error),
                    },
                )
                raise
            record = EvaluationRecord(
                task.id,
                mode,
                sample_index,
                breakdown.pass_fraction,
                breakdown.solve_bonus == 1.0,
                breakdown.status,
            )
            records.append(record)
            evidence = {
                **asdict(record),
                "reward": breakdown.total,
                "breakdown": asdict(breakdown),
                "cache_hit": cache_hit,
            }
            detailed.append(evidence)
            logger.generation_outcome(
                generation_id=generation_id,
                task_id=task.id,
                split=split,
                metadata={"evaluation": label, **evidence},
            )
    summary: dict[str, Any] = dict(aggregate_records(records, selected_k))
    summary.update(
        {
            "label": label,
            "split": split,
            "model": str(adapter_path) if adapter_path is not None else config.model.base_model,
            "base_revision": config.model.revision,
            "sandbox_image": sandbox_image,
            "sampled_k": selected_k,
            "duration_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
    )
    task_scores = {
        record.task_id: record.hidden_fraction for record in records if record.mode == "greedy"
    }
    result = {"summary": summary, "task_scores": task_scores, "records": detailed}
    logger.write_json(f"evaluation-{label}.json", result)
    logger.message("evaluation_complete", f"Completed evaluation {label}.", **summary)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result

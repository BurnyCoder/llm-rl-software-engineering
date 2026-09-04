"""Global context: run identical reward-scored inference for base and adapted policies.

Sources:
- https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/utils.py
- https://arxiv.org/abs/2107.03374
- https://github.com/pytorch/pytorch/blob/2b3ec34829036a65cd9d1398ea72a0167dc37470/torch/autograd/grad_mode.py
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
    output_path = logger.directory / f"evaluation-{label}.json"
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.message("evaluation_complete", f"Completed evaluation {label}.", **summary)
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return result

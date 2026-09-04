"""Global context: measure frozen HumanEvalFix-Python pass@1 for one policy.

This phase keeps the benchmark's official ``instruct`` prompt and Python completion
postprocessing, generates one greedy suffix at a time, and sends every reconstructed
program to the existing no-network Docker assertion runner. Generated Python is never
compiled, imported, or executed by this host module.

Primary sources:
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
- https://huggingface.co/docs/transformers/main/en/main_classes/text_generation
- https://pytorch.org/docs/stable/generated/torch.inference_mode.html
"""

from __future__ import annotations

# Garbage collection releases each sequential policy before another GPU phase begins.
import gc
import time
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

# PyTorch owns deterministic seed setup and peak-memory measurement for local inference.
import torch
from transformers import set_seed

# Run configuration contains only public model, hardware, and artifact selectors.
from minibug_rl.config import RunConfig

# Reuse the exact tokenization/generation helper used by internal before/after evaluation.
from minibug_rl.evaluation import generate_completions

# The immutable adapter supplies prompts without gold solutions or assertion leakage.
from minibug_rl.external_eval import (
    DATASET_ID,
    DATASET_REVISION,
    EXPECTED_TASK_COUNT,
    EXPECTED_TASK_IDS,
    PROMPT_VARIANT,
    ExternalRepairTask,
    PythonTestScriptPayload,
    load_humanevalfix,
)

# The typed Docker wrapper is the only component authorized to execute candidate source.
from minibug_rl.external_sandbox import DockerPythonTestSandbox, PythonTestExecution

# The shared loaders preserve the base revision, dtype, attention backend, and PEFT path.
from minibug_rl.modeling import load_tokenizer, load_transformers_model

# The logger preserves every raw prompt and completion in terminal plus append-only JSONL.
from minibug_rl.run_logging import RunLogger

# These are the exact serialized fields produced by `asdict(ExternalEvaluationRecord)`.
_EXTERNAL_RECORD_FIELDS: Final = frozenset(
    {
        "task_id",
        "sample_index",
        "passed",
        "status",
        "error_type",
        "error",
        "duration_seconds",
        "prompt_tokens",
        "candidate_source",
    }
)

# These stops reproduce the pinned BigCode Python HumanEvalPack evaluator in source order.
PYTHON_STOP_WORDS: Final = (
    "\nclass",
    "\ndef",
    "\n#",
    "\n@",
    "\nprint",
    "\nif",
    "\nassert",
    "<|endoftext|>",
)
# BigCode prepends this exact helper prelude to every Python program before code_eval.
PYTHON_IMPORT_HELPER: Final = (
    "import math",
    "import re",
    "import sys",
    "import copy",
    "import datetime",
    "import itertools",
    "import collections",
    "import heapq",
    "import statistics",
    "import functools",
    "import hashlib",
    "import numpy",
    "import numpy as np",
    "import string",
    "from typing import *",
    "from collections import *",
)


class ExternalEvaluationInfrastructureError(RuntimeError):
    """Abort a benchmark rather than count a broken Docker run as model failure."""


class ExternalExecutor(Protocol):
    """Describe the injectable isolated scorer used by the evaluation loop."""

    def execute(
        self,
        payload: PythonTestScriptPayload,
        candidate_source: str,
    ) -> PythonTestExecution:
        """Execute one candidate behind the sandbox boundary and return typed evidence."""
        ...


@dataclass(frozen=True, slots=True)
class ExternalEvaluationRecord:
    """Store one deterministic candidate outcome before aggregate pass@1 reporting."""

    # Stable task IDs make the base and selected-adapter results directly pairable.
    task_id: str
    # Only greedy index zero exists in this frozen pass@1 evaluation protocol.
    sample_index: int
    # A boolean is the sufficient per-problem statistic for pass@1 with one sample.
    passed: bool
    # Status separates assertion, timeout, and invalid-infrastructure outcomes.
    status: str
    # Exception class enables bounded error analysis without source tracebacks.
    error_type: str | None
    # The Docker wrapper already bounds this diagnostic before it reaches run evidence.
    error: str | None
    # Candidate wall time helps identify timeout and sandbox regressions.
    duration_seconds: float
    # Prompt length records that no silent input truncation occurred.
    prompt_tokens: int
    # The exact harness-compatible program is retained for reproducibility, not executed here.
    candidate_source: str


def postprocess_python_completion(completion: str) -> str:
    """Apply the pinned BigCode Python stop and first-block rules to a raw suffix."""
    # The generation helper already removes the prompt, matching the harness's initial slice.
    code = completion.rstrip()
    # BigCode applies each configured stop sequentially and keeps the preceding prefix.
    for stop_word in PYTHON_STOP_WORDS:
        if stop_word in code:
            code = code[: code.find(stop_word)]
    # A new top-level statement marks the end of the completed Python function body.
    lines = code.split("\n")
    for index, line in enumerate(lines):
        if line.strip() and line[0] not in {" ", "\t"}:
            return "\n".join(lines[:index])
    # An entirely indented suffix is already one complete candidate body.
    return code


def build_python_candidate(task: ExternalRepairTask, completion: str) -> str:
    """Assemble the exact BigCode Python test program without host parsing."""
    # HumanEvalPack strips the prompt's terminal newline before concatenating generation.
    prefix = task.completion_prefix.rstrip()
    # Keeping even an empty suffix lets Docker classify a syntactically incomplete repair.
    suffix = postprocess_python_completion(completion)
    # The pinned process_results path provides common HumanEval Python dependencies.
    imports = "\n".join(PYTHON_IMPORT_HELPER)
    # Its outer strip removes only file-edge whitespace, not generated indentation.
    return (imports + "\n" + prefix + suffix).strip()


def _prompt_token_count(tokenizer: Any, prompt: str) -> int:
    """Count the exact raw benchmark prompt without adding or truncating control tokens."""
    # HumanEvalFix ``instruct`` is already a complete textual benchmark prompt.
    encoded = tokenizer(prompt, add_special_tokens=False)
    # Tokenizers return one unbatched list when ``return_tensors`` is not requested.
    return len(encoded["input_ids"])


def _context_window(model: Any, tokenizer: Any) -> int:
    """Resolve the narrowest concrete model/tokenizer context window for fail-fast checks."""
    # The model setting is authoritative for allocated positional representations.
    model_limit = getattr(model.config, "max_position_embeddings", None)
    # The tokenizer setting can be a huge sentinel when no real bound was configured.
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    # Admit only positive, finite-looking integral limits from dependency objects.
    candidates = [
        value
        for value in (model_limit, tokenizer_limit)
        if isinstance(value, int) and 0 < value < 1_000_000_000
    ]
    if not candidates:
        raise ValueError("Model and tokenizer expose no concrete context-window limit")
    # Respecting the smaller value prevents either component from silently overflowing.
    return min(candidates)


def _aggregate_external(
    records: list[ExternalEvaluationRecord],
) -> dict[str, float | int]:
    """Compute transparent one-candidate HumanEvalFix pass@1 and status counts."""
    # An empty benchmark has no defensible pass-rate denominator.
    if not records:
        raise ValueError("External evaluation requires at least one task record")
    # Exact uniqueness prevents duplicate rows from inflating or deflating the benchmark.
    identifiers = [record.task_id for record in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("External evaluation task IDs must be unique")
    # All records come from one greedy sample by protocol, never a sampled best-of-k.
    if any(record.sample_index != 0 for record in records):
        raise ValueError("External pass@1 records must use greedy sample index zero")
    # Counter preserves the complete model-outcome distribution in a compact summary.
    statuses = Counter(record.status for record in records)
    passed = sum(record.passed for record in records)
    return {
        "tasks": len(records),
        "passed": passed,
        "pass_at_1": passed / len(records),
        "failed": statuses["failed"],
        "timeouts": statuses["timeout"],
    }


def external_result_is_complete(
    result: object,
    *,
    label: str,
    model: str,
    base_revision: str,
    maximum_new_tokens: int,
    sandbox_image: str,
    sandbox_timeout_seconds: float,
) -> bool:
    """Validate full pinned evidence before an interrupted evaluation reuses it."""
    # A result must contain all three independently cross-checked evidence views.
    if not isinstance(result, Mapping):
        return False
    summary = result.get("summary")
    scores = result.get("task_scores")
    records = result.get("records")
    if not isinstance(summary, Mapping) or not isinstance(scores, Mapping):
        return False
    if not isinstance(records, list) or len(records) != EXPECTED_TASK_COUNT:
        return False
    # Immutable protocol selectors prevent reuse after any relevant evaluation change.
    expected_metadata: dict[str, object] = {
        "label": label,
        "split": "external_test",
        "benchmark": DATASET_ID,
        "benchmark_revision": DATASET_REVISION,
        "prompt_variant": PROMPT_VARIANT,
        "model": model,
        "base_revision": base_revision,
        "generation_mode": "greedy",
        "samples_per_task": 1,
        "maximum_new_tokens": maximum_new_tokens,
        "sandbox_image": sandbox_image,
        "sandbox_timeout_seconds": sandbox_timeout_seconds,
        "tasks": EXPECTED_TASK_COUNT,
    }
    if any(summary.get(key) != value for key, value in expected_metadata.items()):
        return False
    # The exact ordered IDs bind every row to the pinned dataset, not merely its length.
    record_ids: list[str] = []
    passed_count = 0
    failed_count = 0
    timeout_count = 0
    for expected_id, record in zip(EXPECTED_TASK_IDS, records, strict=True):
        if not isinstance(record, Mapping) or set(record) != _EXTERNAL_RECORD_FIELDS:
            return False
        if record.get("task_id") != expected_id or record.get("sample_index") != 0:
            return False
        passed = record.get("passed")
        status = record.get("status")
        if not isinstance(passed, bool) or status not in {"passed", "failed", "timeout"}:
            return False
        if passed != (status == "passed"):
            return False
        duration = record.get("duration_seconds")
        prompt_tokens = record.get("prompt_tokens")
        candidate_source = record.get("candidate_source")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, (int, float))
            or duration < 0
            or isinstance(prompt_tokens, bool)
            or not isinstance(prompt_tokens, int)
            or prompt_tokens <= 0
            or not isinstance(candidate_source, str)
            or not candidate_source.strip()
        ):
            return False
        score = scores.get(expected_id)
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return False
        if float(score) != float(passed):
            return False
        record_ids.append(expected_id)
        passed_count += int(passed)
        failed_count += int(status == "failed")
        timeout_count += int(status == "timeout")
    # Extra score keys would otherwise make paired comparisons depend on stale data.
    if set(scores) != set(record_ids):
        return False
    # Recompute every aggregate used by the model card from candidate-level records.
    expected_aggregates: dict[str, object] = {
        "passed": passed_count,
        "pass_at_1": passed_count / EXPECTED_TASK_COUNT,
        "failed": failed_count,
        "timeouts": timeout_count,
    }
    return all(summary.get(key) == value for key, value in expected_aggregates.items())


def evaluate_external(
    config: RunConfig,
    logger: RunLogger,
    *,
    label: str,
    sandbox_image: str,
    adapter_path: str | Path | None = None,
    tasks: tuple[ExternalRepairTask, ...] | None = None,
    executor: ExternalExecutor | None = None,
) -> dict[str, Any]:
    """Evaluate all frozen HumanEvalFix tasks one-by-one and persist pass@1 evidence."""
    # Production loads the full pinned corpus; dependency injection keeps unit tests offline.
    selected_tasks = tasks if tasks is not None else load_humanevalfix()
    if not selected_tasks:
        raise ValueError("External evaluation received no HumanEvalFix tasks")
    # The same tokenizer revision is used for base and PEFT-adapter comparisons.
    tokenizer = load_tokenizer(config)
    # Sequential single-device loading fits the intended 8 GB consumer GPU.
    model = load_transformers_model(
        config,
        adapter_path=adapter_path,
        device="cuda",
        for_training=False,
    )
    # Evaluation mode disables dropout before deterministic greedy generation.
    model.eval()
    # The default scorer inherits the repository's hardened no-network Docker policy.
    selected_executor = executor or DockerPythonTestSandbox(image=sandbox_image)
    # Production records the stricter MiniBug deadline; injected unit fakes may omit it.
    sandbox_timeout = getattr(selected_executor, "timeout_seconds", None)
    # Context validation uses dependency-declared limits rather than a guessed constant.
    context_window = _context_window(model, tokenizer)
    # Candidate records remain in benchmark order for simple audit and paired comparison.
    records: list[ExternalEvaluationRecord] = []
    started = time.monotonic()
    # Peak statistics cover only this policy's external evaluation allocation.
    torch.cuda.reset_peak_memory_stats()
    try:
        for task_index, task in enumerate(selected_tasks):
            # The adapter already created the exact official HumanEvalFixDocs instruct prompt.
            prompt = task.model_prompt
            # A pre-generation count proves the model input was not truncated.
            prompt_tokens = _prompt_token_count(tokenizer, prompt)
            if prompt_tokens + config.model.max_completion_length > context_window:
                raise ValueError(
                    f"External prompt {task.id} plus completion exceeds the "
                    f"{context_window}-token context window"
                )
            # Per-task resetting makes interrupted and base/adapter runs reproducible.
            set_seed(config.training.seed + task_index)
            # One greedy continuation is the standard pass@1 point estimate with n=1.
            completion = generate_completions(
                model,
                tokenizer,
                prompt,
                maximum_new_tokens=config.model.max_completion_length,
                count=1,
                sampled=False,
                temperature=config.training.temperature,
                top_p=config.training.top_p,
            )[0]
            # Only text postprocessing occurs on the host; compilation stays inside Docker.
            candidate_source = build_python_candidate(task, completion)
            execution = selected_executor.execute(task.tests, candidate_source)
            # Preserve all evidence, including candidate failures, before aggregation.
            record = ExternalEvaluationRecord(
                task_id=task.id,
                sample_index=0,
                passed=execution.passed,
                status=execution.status,
                error_type=execution.error_type,
                error=execution.error,
                duration_seconds=execution.duration_seconds,
                prompt_tokens=prompt_tokens,
                candidate_source=candidate_source,
            )
            records.append(record)
            # Raw model input/output remain unsliced; postprocessing is separate metadata.
            logger.generation(
                task_id=task.id,
                split=task.split,
                prompt=prompt,
                completion=completion,
                metadata={"evaluation": label, **asdict(record)},
            )
            # Broken isolation invalidates the run and must never lower a model's score.
            if execution.status == "infrastructure_error":
                logger.message(
                    "external_evaluation_infrastructure_error",
                    f"External evaluation infrastructure failed for {task.id}.",
                    label=label,
                    task_id=task.id,
                    error_type=execution.error_type,
                    error=execution.error,
                )
                raise ExternalEvaluationInfrastructureError(
                    f"HumanEvalFix infrastructure failed for {task.id}: {execution.error}"
                )
        # The loader guarantees one benchmark identity, so the first row supplies metadata.
        first_task = selected_tasks[0]
        summary: dict[str, Any] = {
            **_aggregate_external(records),
            "label": label,
            "split": first_task.split,
            "benchmark": first_task.benchmark,
            "benchmark_revision": first_task.revision,
            "prompt_variant": first_task.prompt_variant,
            "model": (str(adapter_path) if adapter_path is not None else config.model.base_model),
            "base_revision": config.model.revision,
            "generation_mode": "greedy",
            "samples_per_task": 1,
            "maximum_new_tokens": config.model.max_completion_length,
            "sandbox_image": sandbox_image,
            "sandbox_timeout_seconds": sandbox_timeout,
            "duration_seconds": time.monotonic() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
        # Binary per-task values support paired confidence intervals outside this module.
        task_scores = {record.task_id: float(record.passed) for record in records}
        # Full candidate sources make the published aggregate independently auditable.
        result = {
            "summary": summary,
            "task_scores": task_scores,
            "records": [asdict(record) for record in records],
        }
        # Reuse the logger's atomic writer so interrupted results cannot appear complete.
        logger.write_json(f"external-evaluation-{label}.json", result)
        logger.message(
            "external_evaluation_complete",
            f"Completed external evaluation {label}.",
            **summary,
        )
        return result
    finally:
        # Sequential cleanup makes room for later evaluation, export, or publication phases.
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()

"""Global context: specify frozen, deterministic HumanEvalFix pass@1 evaluation.

The fakes prove that complete prompts and raw generations are logged, official Python
postprocessing precedes Docker scoring, infrastructure faults abort, and no GPU, model
download, network request, or generated-code execution is needed by these unit tests.

Sources:
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
- https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/generation/utils.py
"""

from __future__ import annotations

# Standard-library objects capture logs, parse saved evidence, and construct test paths.
import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

# Pytest supplies precise error matching for invalid infrastructure and context bounds.
import pytest

# Frozen public configuration objects avoid loading or mutating repository profiles.
from minibug_rl.config import (
    EvaluationConfig,
    ModelConfig,
    ProjectConfig,
    RunConfig,
    TrainingConfig,
)

# External benchmark records keep assertions separate from model-visible prompts.
from minibug_rl.external_eval import ExternalRepairTask, PythonTestScriptPayload

# The new module owns exact postprocessing, evaluation, and infrastructure semantics.
from minibug_rl.external_evaluation import (
    ExternalEvaluationInfrastructureError,
    build_python_candidate,
    evaluate_external,
    postprocess_python_completion,
)

# The isolated runner's typed result is the fake scorer's only returned contract.
from minibug_rl.external_sandbox import PythonTestExecution

# A real RunLogger verifies terminal and JSONL preservation without a production run.
from minibug_rl.run_logging import RunLogger


class FakeTokenizer:
    """Count whitespace-delimited pseudo-tokens without importing a model tokenizer."""

    # A concrete context limit exercises the same preflight branch as Qwen's tokenizer.
    model_max_length = 1_024

    def __call__(self, prompt: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        """Return a stable unbatched token list for one exact raw prompt."""
        # External instruct prompts must not receive hidden special tokens here.
        assert add_special_tokens is False
        # Every whitespace-delimited segment stands in for one tokenizer ID.
        return {"input_ids": list(range(len(prompt.split())))}


class FakeModel:
    """Expose only context metadata and evaluation-mode state used by the loop."""

    def __init__(self, context_window: int = 1_024) -> None:
        """Configure a test-controlled positional limit without allocating tensors."""
        # SimpleNamespace mimics Transformers' read-only config attributes adequately.
        self.config = SimpleNamespace(max_position_embeddings=context_window)
        # The flag proves dropout-disabling evaluation mode was requested.
        self.eval_called = False

    def eval(self) -> None:
        """Record the production inference lifecycle call."""
        # No numerical behavior is needed because generation itself is patched.
        self.eval_called = True


class FakeExecutor:
    """Return predetermined Docker outcomes while capturing candidate source strings."""

    def __init__(self, outcomes: list[PythonTestExecution]) -> None:
        """Retain an ordered result queue for one-candidate-per-task evaluation."""
        # A copy prevents the caller from mutating results after construction.
        self.outcomes = list(outcomes)
        # Candidate strings prove postprocessed code, not raw text, reaches the scorer.
        self.candidates: list[str] = []

    def execute(
        self,
        payload: PythonTestScriptPayload,
        candidate_source: str,
    ) -> PythonTestExecution:
        """Mimic the typed Docker boundary without evaluating either source field."""
        # The entry point check demonstrates that each task's hidden payload is routed.
        assert payload.entry_point.startswith("repair_")
        # Capturing text is inert and does not parse, compile, import, or execute it.
        self.candidates.append(candidate_source)
        # Ordered popping mirrors the production task-order loop.
        return self.outcomes.pop(0)


def _config(tmp_path: Path, *, maximum_new_tokens: int = 64) -> RunConfig:
    """Create the smallest valid public run configuration for external evaluation."""
    # Project paths remain inside pytest's disposable directory.
    project = ProjectConfig(
        name="external-test",
        data_file=tmp_path / "unused.json",
        run_root=tmp_path / "logs",
        artifact_root=tmp_path / "artifacts",
        sandbox_image="minibug-rl-sandbox:test",
    )
    # Model identity fields let assertions verify exact adapter/base routing.
    model = ModelConfig(
        base_model="example/base",
        revision="base-revision",
        hub_model_id="example/output",
        dtype="bfloat16",
        max_prompt_length=512,
        max_completion_length=maximum_new_tokens,
    )
    # Only the seed and generation controls are consumed by this evaluation module.
    training = TrainingConfig(
        seed=17,
        max_steps=1,
        num_generations=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        temperature=0.9,
        top_p=0.95,
        save_steps=1,
    )
    # External evaluation fixes n=1 independently of internal sampled evaluation.
    evaluation = EvaluationConfig(sample_generations=2, bootstrap_samples=100)
    # Frozen dataclasses match the public loader's production return structure.
    return RunConfig(project, model, training, evaluation)


def _task(index: int) -> ExternalRepairTask:
    """Build one gold-free HumanEvalFix-style task with Docker-only assertions."""
    # A unique name makes paired-score and duplicate-ID semantics observable.
    function_name = f"repair_{index}"
    # The prefix ends in a newline just like the pinned Python dataset prompts.
    prefix = f'def {function_name}(value):\n    """Return value unchanged."""\n'
    # The raw instruct prompt includes buggy context and repeats the completion prefix.
    model_prompt = (
        prefix + "    return value + 1\n" + f"Fix bugs in {function_name}.\n\n" + prefix
    ).strip()
    # Assertions exist only in the typed payload and never in model_prompt.
    tests = PythonTestScriptPayload(
        entry_point=function_name,
        test_setup_source="",
        test_source=f"assert {function_name}(7) == 7",
    )
    # The record mirrors the immutable adapter's external-only metadata.
    return ExternalRepairTask(
        id=f"Python/{index}",
        split="external_test",
        benchmark="bigcode/humanevalpack",
        revision="dataset-revision",
        prompt_variant="humanevalfixdocs-python",
        model_prompt=model_prompt,
        completion_prefix=prefix,
        buggy_source=prefix + "    return value + 1\n",
        function_name=function_name,
        bug_type="operator misuse",
        failure_symptoms="incorrect output",
        tests=tests,
    )


def test_python_postprocessing_matches_pinned_harness_first_block_rules() -> None:
    """Cut configured stops and later top-level code exactly after the repaired body."""
    # A new function is one explicit Python stop token in the pinned BigCode evaluator.
    with_stop = "\n    return value\n\ndef unrelated():\n    return 0"
    # A top-level statement also terminates the body even when it is not a named stop.
    with_unindented = "\n    fixed = value\nraise RuntimeError('outside')"

    # The first suffix retains only indented code before the next definition.
    assert postprocess_python_completion(with_stop) == "\n    return value\n"
    # The second suffix retains only lines belonging to the generated function block.
    assert postprocess_python_completion(with_unindented) == "\n    fixed = value"
    # The method follows BigCode's raw concatenation without interpreting Python on host.
    candidate = build_python_candidate(_task(0), with_stop)
    # The official process-results path prepends its complete Python helper imports.
    assert candidate.startswith("import math\nimport re\nimport sys")
    assert "import numpy\nimport numpy as np" in candidate
    # The generated program itself retains only the first function body.
    assert candidate.endswith('"""\n    return value')


def test_external_evaluation_logs_raw_text_scores_pass_at_1_and_saves_evidence(
    tmp_path: Path,
) -> None:
    """Evaluate one greedy candidate per task with full audit and paired score output."""
    # One pass and one ordinary assertion failure yield an exact 0.5 pass@1.
    executor = FakeExecutor(
        [
            PythonTestExecution(status="passed", duration_seconds=0.1),
            PythonTestExecution(
                status="failed",
                error_type="AssertionError",
                error="assertion failed",
                duration_seconds=0.2,
            ),
        ]
    )
    # Raw completions include text that the official postprocessor must remove.
    completions = [
        "\n    return value\n\ndef leaked_block():\n    return 3",
        "\n    return value + 1\n# explanation",
    ]
    # A StringIO stream makes complete terminal output directly assertable.
    terminal = io.StringIO()
    logger = RunLogger.create(tmp_path / "logs", run_id="external", terminal=terminal)
    # The fake model records evaluation mode and contributes a context window.
    fake_model = FakeModel()
    # Track exact per-task seeds instead of changing global RNG state during the test.
    seeds: list[int] = []
    # Patch only loading, text generation, seed, and CUDA accounting boundaries.
    with (
        patch("minibug_rl.external_evaluation.load_tokenizer", return_value=FakeTokenizer()),
        patch(
            "minibug_rl.external_evaluation.load_transformers_model",
            return_value=fake_model,
        ) as load,
        patch(
            "minibug_rl.external_evaluation.generate_completions",
            side_effect=[[text] for text in completions],
        ) as generate,
        patch("minibug_rl.external_evaluation.set_seed", side_effect=seeds.append),
        patch("minibug_rl.external_evaluation.torch.cuda.reset_peak_memory_stats"),
        patch("minibug_rl.external_evaluation.torch.cuda.max_memory_allocated", return_value=456),
        patch("minibug_rl.external_evaluation.torch.cuda.empty_cache"),
    ):
        # Adapter selection must flow unchanged into the shared PEFT-aware loader.
        result = evaluate_external(
            _config(tmp_path),
            logger,
            label="selected-external",
            sandbox_image="sha256:" + "a" * 64,
            adapter_path=tmp_path / "adapter",
            tasks=(_task(0), _task(1)),
            executor=executor,
        )
    # Close append-only handles before reading authoritative evidence files.
    logger.close()

    # Aggregate pass@1 and status counts use exactly two unique greedy records.
    assert result["summary"]["tasks"] == 2
    assert result["summary"]["passed"] == 1
    assert result["summary"]["pass_at_1"] == 0.5
    assert result["summary"]["failed"] == 1
    assert result["summary"]["timeouts"] == 0
    assert result["summary"]["sandbox_image"] == "sha256:" + "a" * 64
    # Binary task scores are immediately suitable for a paired bootstrap comparison.
    assert result["task_scores"] == {"Python/0": 1.0, "Python/1": 0.0}
    # The model is loaded on CUDA in inference mode with the requested adapter path.
    load.assert_called_once_with(
        _config(tmp_path),
        adapter_path=tmp_path / "adapter",
        device="cuda",
        for_training=False,
    )
    assert fake_model.eval_called is True
    # Each task resets its deterministic seed and requests one non-sampled generation.
    assert seeds == [17, 18]
    assert generate.call_count == 2
    assert all(call.kwargs["sampled"] is False for call in generate.call_args_list)
    assert all(call.kwargs["count"] == 1 for call in generate.call_args_list)
    # Docker receives official first-block candidates, excluding later generated text.
    assert "leaked_block" not in executor.candidates[0]
    assert "# explanation" not in executor.candidates[1]
    # Terminal logging preserves exact raw prompts and outputs before postprocessing.
    terminal_text = terminal.getvalue()
    assert _task(0).model_prompt in terminal_text
    assert completions[0] in terminal_text
    assert completions[1] in terminal_text
    # JSONL likewise retains raw text, while its metadata keeps reconstructed source.
    generation_lines = (
        (logger.directory / "generations.jsonl").read_text(encoding="utf-8").splitlines()
    )
    all_generation_records = [json.loads(line) for line in generation_lines]
    generation_records = [
        record for record in all_generation_records if record["event"] == "generation"
    ]
    outcome_records = [
        record for record in all_generation_records if record["event"] == "generation_outcome"
    ]
    assert [record["completion"] for record in generation_records] == completions
    assert outcome_records[0]["metadata"]["candidate_source"] == executor.candidates[0]
    assert generation_records[0]["generation_id"] == outcome_records[0]["generation_id"]
    # The atomic result artifact contains the same complete candidate-level evidence.
    artifact = json.loads(
        (logger.directory / "external-evaluation-selected-external.json").read_text(
            encoding="utf-8"
        )
    )
    assert artifact == result
    assert artifact["summary"]["peak_vram_bytes"] == 456


def test_external_evaluation_aborts_on_sandbox_infrastructure_error(
    tmp_path: Path,
) -> None:
    """Never turn an unavailable or malformed Docker execution into a failed repair."""
    # This state means no valid candidate-test outcome exists for the task.
    executor = FakeExecutor(
        [
            PythonTestExecution(
                status="infrastructure_error",
                error_type="ContainerInfrastructureError",
                error="Docker unavailable",
            )
        ]
    )
    # The logger should still preserve the raw generation and diagnostic before aborting.
    logger = RunLogger.create(tmp_path / "logs", run_id="infrastructure", terminal=io.StringIO())
    # Isolate every heavyweight boundary while keeping production control flow intact.
    with (
        patch("minibug_rl.external_evaluation.load_tokenizer", return_value=FakeTokenizer()),
        patch("minibug_rl.external_evaluation.load_transformers_model", return_value=FakeModel()),
        patch(
            "minibug_rl.external_evaluation.generate_completions",
            return_value=["\n    return value"],
        ),
        patch("minibug_rl.external_evaluation.set_seed"),
        patch("minibug_rl.external_evaluation.torch.cuda.reset_peak_memory_stats"),
        patch("minibug_rl.external_evaluation.torch.cuda.empty_cache"),
        pytest.raises(ExternalEvaluationInfrastructureError, match="Docker unavailable"),
    ):
        # A single infrastructure failure invalidates the complete pass@1 experiment.
        evaluate_external(
            _config(tmp_path),
            logger,
            label="broken",
            sandbox_image="sha256:" + "b" * 64,
            tasks=(_task(0),),
            executor=executor,
        )
    # Final score artifacts are absent because no valid denominator was completed.
    logger.close()
    assert not (logger.directory / "external-evaluation-broken.json").exists()
    # Raw output remains available to diagnose and reproduce the invalid run.
    records = (logger.directory / "generations.jsonl").read_text(encoding="utf-8")
    assert "return value" in records
    assert "ContainerInfrastructureError" in records


def test_external_evaluation_rejects_prompt_plus_completion_over_context(
    tmp_path: Path,
) -> None:
    """Fail before generation rather than silently truncate a benchmark prompt."""
    # A three-token prompt plus a four-token completion budget cannot fit in six slots.
    short_model = FakeModel(context_window=6)
    logger = RunLogger.create(tmp_path / "logs", run_id="context", terminal=io.StringIO())
    # Generation must remain untouched because validation happens first.
    with (
        patch("minibug_rl.external_evaluation.load_tokenizer", return_value=FakeTokenizer()),
        patch("minibug_rl.external_evaluation.load_transformers_model", return_value=short_model),
        patch("minibug_rl.external_evaluation.generate_completions") as generate,
        patch("minibug_rl.external_evaluation.torch.cuda.reset_peak_memory_stats"),
        patch("minibug_rl.external_evaluation.torch.cuda.empty_cache"),
        pytest.raises(ValueError, match="context window"),
    ):
        # Replace the verbose task prompt with exactly three pseudo-tokens.
        task = _task(0)
        compact_task = ExternalRepairTask(
            id=task.id,
            split=task.split,
            benchmark=task.benchmark,
            revision=task.revision,
            prompt_variant=task.prompt_variant,
            model_prompt="one two three",
            completion_prefix=task.completion_prefix,
            buggy_source=task.buggy_source,
            function_name=task.function_name,
            bug_type=task.bug_type,
            failure_symptoms=task.failure_symptoms,
            tests=task.tests,
        )
        evaluate_external(
            _config(tmp_path, maximum_new_tokens=4),
            logger,
            label="overflow",
            sandbox_image="sha256:" + "c" * 64,
            tasks=(compact_task,),
            executor=FakeExecutor([]),
        )
    # No raw generation exists because the unsafe request was rejected beforehand.
    logger.close()
    generate.assert_not_called()
    assert (logger.directory / "generations.jsonl").read_text(encoding="utf-8") == ""

"""Global context: adapt frozen HumanEvalFix-Python rows for external evaluation.

This module only downloads and validates benchmark data. It never executes a row's
Python source, never creates training rows, and deliberately drops canonical solutions.
HumanEvalPack's tests are assertion scripts, so they use the explicit Docker script
protocol instead of MiniBug-RL's JSON function-call protocol.

Primary sources:
- https://huggingface.co/datasets/bigcode/humanevalpack/blob/9a41762f73a8cb23bb5811b73d5aab164efcf378/README.md
- https://huggingface.co/docs/datasets/loading#hugging-face-hub
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
"""

from __future__ import annotations

# Standard-library types implement immutable records, schema checks, and cache paths.
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

# Datasets owns Hub revision resolution, download verification, and local caching.
# The package lacks a PEP 561 marker, so mypy cannot inspect its otherwise typed API.
from datasets import load_dataset  # type: ignore[import-untyped]

# These selectors reproduce one immutable public dataset artifact, not its moving branch.
DATASET_ID: Final = "bigcode/humanevalpack"
DATASET_CONFIG: Final = "python"
DATASET_SPLIT: Final = "test"
DATASET_REVISION: Final = "9a41762f73a8cb23bb5811b73d5aab164efcf378"
# The pinned card declares exactly 164 Python test records.
EXPECTED_TASK_COUNT: Final = 164
# This adapter follows the official test-hidden, docstring-visible repair variant.
PROMPT_VARIANT: Final = "humanevalfixdocs-python"
# A versioned name lets a future runner reject this request instead of misrouting it.
SCRIPT_PROTOCOL: Final = "minibug.python-test-script.v1"
# The existing sandbox accepts JSON calls and host-side expected values, not source tests.
JSON_CALL_UNSUPPORTED_REASON: Final = (
    "HumanEvalFix canonical tests are Python assertion scripts, not JSON function calls; "
    "route this payload only to the isolated Python test-script Docker runner."
)

# The exact pinned card schema catches missing, added, or renamed fields before evaluation.
_EXPECTED_FIELDS: Final = frozenset(
    {
        "task_id",
        "prompt",
        "declaration",
        "canonical_solution",
        "buggy_solution",
        "bug_type",
        "failure_symptoms",
        "entry_point",
        "import",
        "test_setup",
        "test",
        "example_test",
        "signature",
        "docstring",
        "instruction",
    }
)
# These fields must contain content; Python import/setup/example fields may be empty.
_NON_EMPTY_FIELDS: Final = frozenset(
    {
        "task_id",
        "prompt",
        "declaration",
        "canonical_solution",
        "buggy_solution",
        "bug_type",
        "failure_symptoms",
        "entry_point",
        "test",
        "signature",
        "docstring",
        "instruction",
    }
)


class ExternalEvaluationDataError(ValueError):
    """Report benchmark drift or corruption before generation or execution begins."""


@dataclass(frozen=True, slots=True)
class PythonTestScriptPayload:
    """Describe assertion-script tests that must run inside a Docker boundary."""

    # The callable name lets the runner verify that the generated program defines it.
    entry_point: str
    # Dataset-provided test imports/setup execute after isolation, never during loading.
    test_setup_source: str
    # The canonical assertions are scoring material and stay out of the model prompt.
    test_source: str

    @property
    def json_call_unsupported_reason(self) -> str:
        """Explain why the ordinary JSON-call sandbox cannot score this benchmark."""
        # Returning a stable reason makes unsupported routing explicit and testable.
        return JSON_CALL_UNSUPPORTED_REASON

    def build_request(self, candidate_source: str) -> dict[str, str]:
        """Build JSON-serializable stdin data for an isolated script-mode runner."""
        # Empty candidates would only create misleading infrastructure failures later.
        if not isinstance(candidate_source, str) or not candidate_source.strip():
            raise ValueError("candidate_source must be a non-empty string")
        # Separate source fields let the runner compile each trust domain deliberately.
        return {
            "protocol": SCRIPT_PROTOCOL,
            "language": "python",
            "entry_point": self.entry_point,
            "candidate_source": candidate_source,
            "test_setup_source": self.test_setup_source,
            "test_source": self.test_source,
        }


@dataclass(frozen=True, slots=True)
class ExternalRepairTask:
    """Carry only generation inputs, reporting metadata, and isolated scoring tests."""

    # Dataset task IDs are stable keys for paired base-versus-RL comparisons.
    id: str
    # A distinct split label prevents accidental mixing with local train/test records.
    split: str
    # Benchmark identity keeps exported metrics independently interpretable.
    benchmark: str
    # The immutable Hub commit is retained with every converted task.
    revision: str
    # Prompt variant names the official HumanEvalFixDocs semantics being reproduced.
    prompt_variant: str
    # This is the complete raw completion prompt visible to the evaluated model.
    model_prompt: str
    # The official code prefix is prepended to a generated function-body suffix.
    completion_prefix: str
    # The subtle human-written bug supplies the software-repair context.
    buggy_source: str
    # The callable name connects generation postprocessing to isolated tests.
    function_name: str
    # Human annotations support stratified error analysis without affecting prompts.
    bug_type: str
    # Failure symptoms support timeout-aware reporting without affecting prompts.
    failure_symptoms: str
    # Canonical assertions remain behind the Docker request boundary.
    tests: PythonTestScriptPayload

    def candidate_source(self, generated_suffix: str) -> str:
        """Reconstruct full Python from the benchmark's generated body suffix."""
        # The official completion protocol expects at least one generated body token.
        if not isinstance(generated_suffix, str) or not generated_suffix.strip():
            raise ValueError("generated_suffix must be a non-empty string")
        # BigCode's postprocessor joins the original prompt prefix with the new suffix.
        return self.completion_prefix.rstrip() + generated_suffix


def _record(value: Any, index: int) -> Mapping[str, Any]:
    """Require one mapping before inspecting fields from untrusted cached data."""
    # Dataset rows should be mappings, but this check gives corruption a useful location.
    if not isinstance(value, Mapping):
        raise ExternalEvaluationDataError(f"row {index} must be a mapping")
    return value


def _validate_schema(row: Mapping[str, Any], index: int) -> None:
    """Require the exact 15-field schema and its documented string value types."""
    # Exact equality is safe because the repository revision itself is immutable.
    if set(row) != _EXPECTED_FIELDS:
        missing = sorted(_EXPECTED_FIELDS - set(row))
        extra = sorted(set(row) - _EXPECTED_FIELDS)
        raise ExternalEvaluationDataError(
            f"row {index} schema differs from the pinned card: missing={missing}, extra={extra}"
        )
    # The card declares every feature as a string, including optional empty fields.
    for field in _EXPECTED_FIELDS:
        value = row[field]
        if not isinstance(value, str):
            raise ExternalEvaluationDataError(f"row {index}.{field} must be a string")
        if field in _NON_EMPTY_FIELDS and not value.strip():
            raise ExternalEvaluationDataError(f"row {index}.{field} must not be empty")


def _test_setup(row: Mapping[str, Any]) -> str:
    """Combine the dataset's language import and test-setup fields without executing them."""
    # Both are empty for the pinned Python config, but preserving them is schema-faithful.
    parts = [row["import"], row["test_setup"]]
    # Empty fields are omitted so the request does not gain meaningless blank lines.
    return "\n".join(cast(str, part) for part in parts if part)


def _external_task(value: Any, index: int) -> ExternalRepairTask:
    """Convert one checked row while intentionally discarding its gold solution."""
    # Validate structure and all source types before concatenating model-visible strings.
    row = _record(value, index)
    _validate_schema(row, index)
    # These casts narrow values after the shared schema validator checks every field.
    task_id = cast(str, row["task_id"])
    entry_point = cast(str, row["entry_point"])
    if not task_id.startswith("Python/"):
        raise ExternalEvaluationDataError(f"row {index}.task_id is not a Python task")
    if not entry_point.isidentifier():
        raise ExternalEvaluationDataError(f"row {index}.entry_point is not an identifier")
    # The card defines a complete buggy program as the prompt prefix plus buggy body.
    completion_prefix = cast(str, row["prompt"])
    buggy_source = completion_prefix + cast(str, row["buggy_solution"])
    # This matches HumanEvalFixDocs `instruct`: context, repair request, completion prefix.
    model_prompt = (
        buggy_source + f"\nFix bugs in {entry_point}.\n\n" + completion_prefix
    ).strip()
    # Reading validates canonical_solution above, but omitting it here prevents leakage.
    return ExternalRepairTask(
        id=task_id,
        split="external_test",
        benchmark=DATASET_ID,
        revision=DATASET_REVISION,
        prompt_variant=PROMPT_VARIANT,
        model_prompt=model_prompt,
        completion_prefix=completion_prefix,
        buggy_source=buggy_source,
        function_name=entry_point,
        bug_type=cast(str, row["bug_type"]),
        failure_symptoms=cast(str, row["failure_symptoms"]),
        tests=PythonTestScriptPayload(
            entry_point=entry_point,
            test_setup_source=_test_setup(row),
            test_source=cast(str, row["test"]),
        ),
    )


def _validated_tasks(rows: Iterable[Mapping[str, Any]]) -> tuple[ExternalRepairTask, ...]:
    """Validate full-corpus size and identity after converting cached dataset rows."""
    # Materialization freezes row order for deterministic paired evaluation.
    tasks = tuple(_external_task(row, index) for index, row in enumerate(rows))
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise ExternalEvaluationDataError(
            f"expected {EXPECTED_TASK_COUNT} pinned Python tasks, received {len(tasks)}"
        )
    # Stable unique IDs are necessary for exact base-versus-adapter pairing.
    identifiers = [task.id for task in tasks]
    if len(identifiers) != len(set(identifiers)):
        raise ExternalEvaluationDataError("Duplicate HumanEvalFix task IDs are prohibited")
    return tasks


def load_humanevalfix(
    *,
    cache_dir: str | Path | None = None,
) -> tuple[ExternalRepairTask, ...]:
    """Load all 164 pinned Python repair tasks for external evaluation only."""
    # Datasets uses its standard cache by default or the caller's isolated cache path.
    normalized_cache = None if cache_dir is None else str(Path(cache_dir).expanduser())
    # `split` returns a Dataset directly; `revision` prevents moving-main drift.
    dataset = load_dataset(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=normalized_cache,
    )
    # The external library's broad return union is narrowed after the fixed split request.
    rows = cast(Iterable[Mapping[str, Any]], dataset)
    # Conversion performs no compilation, import, evaluation, or host code execution.
    return _validated_tasks(rows)

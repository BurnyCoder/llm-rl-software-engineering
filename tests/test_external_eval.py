"""Global context: freeze the HumanEvalFix-Python external-evaluation boundary.

The unit tests mock network loading and prove that gold solutions and benchmark test
scripts never enter the model prompt. The opt-in smoke test only downloads dataset
metadata/rows; it never evaluates or executes benchmark Python.

Sources:
- https://huggingface.co/datasets/bigcode/humanevalpack
- https://huggingface.co/docs/datasets/loading#hugging-face-hub
- https://github.com/bigcode-project/bigcode-evaluation-harness/blob/8fc5bae6479c4fbbb28c3f8b644f6a15b3f3b5bd/bigcode_eval/tasks/humanevalpack.py
"""

from __future__ import annotations

# Standard-library helpers inspect leakage, serialization, environment, paths, and calls.
import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

# Pytest supplies clear exception assertions and its built-in conditional-skip marker.
import pytest

from minibug_rl.external_eval import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    DATASET_SPLIT,
    EXPECTED_TASK_COUNT,
    SCRIPT_PROTOCOL,
    ExternalEvaluationDataError,
    load_humanevalfix,
)


def _row(index: int = 0) -> dict[str, str]:
    """Return one complete record matching the official 15-string-field schema."""
    # A distinct callable name lets duplicate checks work across generated fixtures.
    entry_point = f"repair_{index}"
    # HumanEvalPack stores a code-completion prefix separately from the buggy body.
    prompt = f'def {entry_point}(value):\n    """Return value unchanged."""\n'
    # The complete mapping mirrors the dataset card rather than an invented subset.
    return {
        "task_id": f"Python/{index}",
        "prompt": prompt,
        "declaration": f"def {entry_point}(value):\n",
        "canonical_solution": "    return value  # GOLD_MUST_NOT_LEAK\n",
        "buggy_solution": "    return value + 1\n",
        "bug_type": "operator misuse",
        "failure_symptoms": "incorrect output",
        "entry_point": entry_point,
        "import": "",
        "test_setup": "",
        "test": (f"def check(candidate):\n    assert candidate(7) == 7\ncheck({entry_point})\n"),
        "example_test": "",
        "signature": f"{entry_point}(value)",
        "docstring": "Return value unchanged.",
        "instruction": f"Write a Python function `{entry_point}(value)`.",
    }


def _complete_dataset() -> list[dict[str, str]]:
    """Build the official row count without downloading anything."""
    # Unique indices preserve the benchmark's one-task-per-ID invariant.
    return [_row(index) for index in range(EXPECTED_TASK_COUNT)]


def test_loader_pins_repository_config_split_revision_and_cache(tmp_path: Path) -> None:
    """Prevent an upstream branch or another language from silently changing scores."""
    # The adapter's imported loader is the only network/cache boundary being replaced.
    with patch("minibug_rl.external_eval.load_dataset", return_value=_complete_dataset()) as load:
        # An explicit cache path must flow to Datasets without being created by the adapter.
        tasks = load_humanevalfix(cache_dir=tmp_path / "hf-cache")

    # All 164 records are retained for the frozen external benchmark.
    assert len(tasks) == EXPECTED_TASK_COUNT
    # Exact keyword arguments lock every source selector documented by Datasets.
    load.assert_called_once_with(
        DATASET_ID,
        DATASET_CONFIG,
        split=DATASET_SPLIT,
        revision=DATASET_REVISION,
        cache_dir=str(tmp_path / "hf-cache"),
    )


def test_adapter_matches_official_docs_prompt_without_test_or_gold_leakage() -> None:
    """Reproduce HumanEvalFixDocs context while reserving tests for Docker only."""
    # Mocking the complete corpus lets the public loader's count check remain active.
    with patch("minibug_rl.external_eval.load_dataset", return_value=_complete_dataset()):
        task = load_humanevalfix()[0]

    # The buggy program is the official prompt prefix followed by its buggy body.
    assert task.buggy_source == _row()["prompt"] + _row()["buggy_solution"]
    # The BigCode harness asks for the repair and then repeats the prefix for completion.
    expected_prompt = (
        task.buggy_source + f"\nFix bugs in {task.function_name}.\n\n" + task.completion_prefix
    ).strip()
    assert task.model_prompt == expected_prompt
    # Neither the gold body nor held-out assertion script is visible to generation.
    assert "GOLD_MUST_NOT_LEAK" not in task.model_prompt
    assert "candidate(7)" not in task.model_prompt
    # The typed object itself intentionally drops the canonical solution field.
    assert "GOLD_MUST_NOT_LEAK" not in json.dumps(asdict(task), sort_keys=True)
    # The fixed metadata identifies this score as external evaluation, never training.
    assert task.split == "external_test"
    assert task.prompt_variant == "humanevalfixdocs-python"


def test_generated_suffix_and_script_tests_form_a_docker_json_request() -> None:
    """Carry arbitrary benchmark assertions to isolation without host execution."""
    # Load through the same validated conversion path used by the real dataset.
    with patch("minibug_rl.external_eval.load_dataset", return_value=_complete_dataset()):
        task = load_humanevalfix()[0]
    # The harness generates only the body after the repeated function prefix.
    candidate = task.candidate_source("\n    return value\n")

    # The script-mode Docker runner receives sources as separate JSON fields.
    request = task.tests.build_request(candidate)

    # The versioned mode prevents accidental routing through the JSON-call runner.
    assert request["protocol"] == SCRIPT_PROTOCOL
    assert request["language"] == "python"
    assert request["entry_point"] == task.function_name
    assert request["candidate_source"] == candidate
    assert request["test_source"] == _row()["test"]
    assert request["test_setup_source"] == ""
    # Successful strict serialization proves the object is Docker-stdin compatible.
    assert json.loads(json.dumps(request, allow_nan=False)) == request
    # The interface is explicit that ordinary host-comparison calls cannot represent it.
    assert "assertion scripts" in task.tests.json_call_unsupported_reason


def test_loader_rejects_schema_drift_and_duplicate_ids() -> None:
    """Fail closed instead of producing a misleading score from malformed rows."""
    # Removing a pinned field simulates upstream or cache corruption.
    missing_test = _row()
    del missing_test["test"]
    malformed = _complete_dataset()
    malformed[0] = missing_test
    with (
        patch("minibug_rl.external_eval.load_dataset", return_value=malformed),
        pytest.raises(ExternalEvaluationDataError, match="schema"),
    ):
        load_humanevalfix()

    # Duplicate task IDs would make paired base-versus-RL results ambiguous.
    duplicated = _complete_dataset()
    duplicated[-1]["task_id"] = duplicated[0]["task_id"]
    with (
        patch("minibug_rl.external_eval.load_dataset", return_value=duplicated),
        pytest.raises(ExternalEvaluationDataError, match="Duplicate"),
    ):
        load_humanevalfix()

    # A unique but substituted ID would make resumed evidence describe another corpus.
    substituted = _complete_dataset()
    substituted[-1]["task_id"] = "Python/999"
    with (
        patch("minibug_rl.external_eval.load_dataset", return_value=substituted),
        pytest.raises(ExternalEvaluationDataError, match="ordered task IDs"),
    ):
        load_humanevalfix()


@pytest.mark.skipif(
    os.environ.get("MINIBUG_RUN_NETWORK_TESTS") != "1",
    reason="set MINIBUG_RUN_NETWORK_TESTS=1 to verify the pinned Hub dataset",
)
def test_pinned_hub_dataset_metadata_smoke() -> None:
    """Optionally verify the cached/downloaded public artifact without executing code."""
    # Datasets uses its ordinary local cache and fetches the exact immutable revision.
    tasks = load_humanevalfix()

    # These stable public facts detect accidental config or split changes.
    assert len(tasks) == EXPECTED_TASK_COUNT
    assert tasks[0].id == "Python/0"
    assert tasks[0].function_name == "has_close_elements"
    assert tasks[0].revision == DATASET_REVISION

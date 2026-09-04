"""Global context: verify frozen task loading and model-visible prompt boundaries.

Sources:
- https://huggingface.co/docs/trl/main/en/dataset_formats
- https://huggingface.co/docs/transformers/main/en/chat_templating
"""

import json
from pathlib import Path

import pytest

from minibug_rl.prompts import build_messages
from minibug_rl.task_data import TaskDataError, load_tasks, training_rows


def _write_tasks(path: Path) -> None:
    """Write one minimal valid raw task fixture for loader-focused tests."""
    # Keeping the fixture local makes each test independent from the 60-task curriculum.
    path.write_text(
        json.dumps(
            [
                {
                    "id": "train_increment_01",
                    "split": "train",
                    "family": "arithmetic",
                    "difficulty": "easy",
                    "specification": "Return x plus one.",
                    "function_name": "increment",
                    "buggy_code": "def increment(x):\n    return x - 1",
                    "public_tests": [
                        {"args": [1], "kwargs": {}, "expected": 2},
                        {"args": [-1], "kwargs": {}, "expected": 0},
                    ],
                    "hidden_tests": [
                        {"args": [0], "kwargs": {}, "expected": 1},
                        {"args": [5], "kwargs": {}, "expected": 6},
                        {"args": [-5], "kwargs": {}, "expected": -4},
                        {"args": [99], "kwargs": {}, "expected": 100},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )


def test_loader_creates_immutable_task_and_rejects_duplicate_ids(tmp_path: Path) -> None:
    """Normalize JSON into typed values and fail closed on ambiguous identifiers."""
    task_file = tmp_path / "tasks.json"
    _write_tasks(task_file)

    tasks = load_tasks(task_file)

    assert tasks[0].public_tests[0].args == (1,)
    assert tasks[0].hidden_tests[0].expected == 1
    # Duplicate IDs would make logs and paired evaluation impossible to interpret.
    duplicated = json.loads(task_file.read_text(encoding="utf-8")) * 2
    task_file.write_text(json.dumps(duplicated), encoding="utf-8")
    with pytest.raises(TaskDataError, match="Duplicate"):
        load_tasks(task_file)


def test_prompt_includes_public_evidence_but_never_hidden_answers(tmp_path: Path) -> None:
    """Give the policy the bug and examples while reserving hidden tests for reward."""
    task_file = tmp_path / "tasks.json"
    _write_tasks(task_file)
    task = load_tasks(task_file)[0]

    messages = build_messages(task)
    visible = messages[-1]["content"]

    assert "Return x plus one" in visible
    assert "return x - 1" in visible
    assert '"expected": 2' in visible
    assert '"expected": 100' not in visible
    assert messages[-1]["role"] == "user"


def test_training_rows_keep_reward_metadata_outside_prompt(tmp_path: Path) -> None:
    """Retain hidden tests as dataset columns consumed only by the custom reward function."""
    task_file = tmp_path / "tasks.json"
    _write_tasks(task_file)
    task = load_tasks(task_file)[0]

    rows = training_rows([task])

    assert rows[0]["task_id"] == task.id
    assert rows[0]["function_name"] == task.function_name
    assert json.loads(rows[0]["hidden_tests_json"])[-1]["expected"] == 100
    assert "100" not in json.dumps(rows[0]["prompt"])
